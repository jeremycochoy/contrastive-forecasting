"""Regression tests for PrefetchIterator shutdown behaviour.

The FRESH 167k-step run of #10 crashed at process exit with
``Fatal Python error: PyGILState_Release: thread state must be current
when releasing``. Root cause: the producer thread blocked on a full queue
when the consumer stopped iterating early (training loop reached
total_steps and stopped calling next()). The daemon thread leaked into
interpreter finalisation. The fix introduces a stop-event and a drain in
the consumer's finally block so the producer always returns within ~0.5s
of consumer exit.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataloader import PrefetchIterator  # noqa: E402


class PrefetchEarlyExitTest(unittest.TestCase):
    def test_early_exit_does_not_leak_producer_thread(self):
        """Consumer stops after a few items; producer must finish promptly."""
        before = threading.active_count()
        infinite = iter(range(10_000_000))  # source way larger than what we consume
        it = iter(PrefetchIterator(infinite, prefetch=2))

        # Consume just 5 items, then stop. This is what the training loop
        # does when it hits total_steps before the iterable is exhausted.
        consumed = []
        for i, x in enumerate(it):
            consumed.append(x)
            if i >= 4:
                break
        del it  # trigger GeneratorExit on the inner generator

        # Producer must wind down within a small grace period.
        deadline = time.monotonic() + 6.0
        while threading.active_count() > before and time.monotonic() < deadline:
            time.sleep(0.05)

        self.assertEqual(consumed, list(range(5)))
        self.assertEqual(
            threading.active_count(), before,
            "producer thread leaked after early consumer exit",
        )

    def test_normal_completion_yields_all_items(self):
        """Full iteration of a finite source must yield every item, in order."""
        before = threading.active_count()
        items = list(PrefetchIterator(iter(range(50)), prefetch=2))
        self.assertEqual(items, list(range(50)))
        # Allow the producer thread a moment to wind down on the join path.
        deadline = time.monotonic() + 6.0
        while threading.active_count() > before and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(threading.active_count(), before)

    def test_exception_in_source_propagates_and_no_leak(self):
        """A raise inside the iterable should reach the consumer and clean up."""
        before = threading.active_count()

        def bad_source():
            yield 1
            yield 2
            raise RuntimeError("boom")

        it = PrefetchIterator(bad_source(), prefetch=2)
        with self.assertRaises(RuntimeError) as cm:
            list(it)
        self.assertIn("boom", str(cm.exception))
        deadline = time.monotonic() + 6.0
        while threading.active_count() > before and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(threading.active_count(), before)


if __name__ == "__main__":
    unittest.main()
