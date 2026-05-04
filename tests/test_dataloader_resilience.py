"""Resilience tests for ``HFStreamingLoader._raw_iter`` (PR #120).

The fix being exercised here handles a known failure mode of the HF
streaming stack: the underlying ``httpx`` client can be closed mid-stream
(typically when the HF connection-pool / worker thread is GC'd part-way
through a long run). The next read out of the row iterator then raises::

    RuntimeError: Cannot send a request, as the client has been closed.

The fix in ``src/dataloader.py`` catches this specific RuntimeError,
rebuilds the row iterator at the current absolute row position via
``_iter_stream_with_fast_skip(skip_rows=...)``, and continues. It limits
itself to ``_HF_MAX_REOPENS=5`` consecutive reopens and resets the
counter on any successful row pull.

These tests inject the failure mode by monkey-patching both
``_open_stream`` and ``_iter_stream_with_fast_skip`` with stubs that read
from an in-memory fake dataset and raise the simulated RuntimeError at
configured row positions on configured invocations. No network access.
Single-shot RuntimeError, multi-shot non-consecutive RuntimeError, and
fail-past-the-cap behaviour are each validated.
"""

import unittest
from unittest.mock import patch

import numpy as np

from src.dataloader import HFStreamingLoader, T_RAW


_HTTPX_CLOSED_MSG = "Cannot send a request, as the client has been closed."


class _FakeHFRemote:
    """Stateful fake HF stream backing both monkey-patched methods.

    Holds an in-memory list of N row-dicts (each with a ``"series"`` field
    of length ``T_RAW``). Each call to ``open_stream()`` /
    ``iter_with_skip(K)`` returns a fresh generator that yields rows
    starting from absolute row index 0 / K respectively.

    A list of ``(invocation_index, raise_at_local_position)`` failure
    triggers can be set; when the matching generator has yielded
    ``raise_at_local_position`` rows, it raises a RuntimeError carrying
    the exact httpx-closed message. The list is consumed in order: the
    first trigger applies to the first generator opened, etc. Triggers
    that don't match an open call simply don't fire.

    The generators are independent objects but share counters via this
    fake so the test can assert on "how many invocations did the fix
    make" and "which one raised".
    """

    def __init__(self, n_rows: int, failure_plan):
        # Each row carries one T_RAW-length window — keeps batches
        # straightforward (1 row in -> 1 window out at C=1).
        rng = np.random.default_rng(0)
        self.rows = [
            {"series": rng.standard_normal(T_RAW).astype(np.float32).tolist()}
            for _ in range(n_rows)
        ]
        # failure_plan: list of (invocation_index, fail_after_local_rows)
        # tuples. invocation_index is 0-based across all open_stream /
        # iter_with_skip calls combined.
        self._plan = list(failure_plan)
        self.invocation_count = 0
        # Per-invocation: (skip_rows_at_call, n_rows_yielded_before_raise_or_stop)
        self.invocations = []

    def _gen(self, start: int):
        """Build the generator for the next invocation, capturing its index now."""
        my_invocation = self.invocation_count
        self.invocation_count += 1
        # Look up whether this invocation should raise mid-stream.
        fail_at_local = None
        for inv_idx, fail_at in self._plan:
            if inv_idx == my_invocation:
                fail_at_local = fail_at
                break
        invocation_record = {"start": start, "yielded": 0, "raised": False}
        self.invocations.append(invocation_record)

        def generator():
            # Yield from absolute-position ``start`` onwards. If we run out
            # of rows, just stop (StopIteration) — that's the natural EOF.
            for absolute_idx in range(start, len(self.rows)):
                if (fail_at_local is not None
                        and invocation_record["yielded"] == fail_at_local):
                    invocation_record["raised"] = True
                    raise RuntimeError(_HTTPX_CLOSED_MSG)
                invocation_record["yielded"] += 1
                yield self.rows[absolute_idx]
            # Even if no row was scheduled to fail at a position
            # *before* EOF, a plan entry with fail_at_local == 0 fires
            # immediately (yielded=0). That's handled by the check
            # above on the very first iteration.

        return generator()

    def open_stream(self):
        """Mock for ``HFStreamingLoader._open_stream`` (yields from row 0)."""
        return self._gen(0)

    def iter_with_skip(self, skip_rows=None):
        """Mock for ``HFStreamingLoader._iter_stream_with_fast_skip``.

        Real method takes ``skip_rows`` as a kwarg with default None
        meaning "use ``self.skip_rows``"; the fix passes an explicit
        positive value on every reopen.
        """
        if skip_rows is None:
            skip_rows = 0
        return self._gen(skip_rows)


def _make_loader(n_rows: int, failure_plan):
    """Build a loader + fake remote and patch both stream methods on it.

    Returns (loader, fake, patches) where ``patches`` is a list of
    already-started ``unittest.mock.patch`` objects the caller must stop
    when done.
    """
    fake = _FakeHFRemote(n_rows=n_rows, failure_plan=failure_plan)
    loader = HFStreamingLoader(
        repo_id="fake/repo", batch_size=1, C=1, prefetch=2, skip_rows=0,
    )
    p1 = patch.object(
        HFStreamingLoader, "_open_stream",
        autospec=True, side_effect=lambda self: fake.open_stream(),
    )
    p2 = patch.object(
        HFStreamingLoader, "_iter_stream_with_fast_skip",
        autospec=True,
        side_effect=lambda self, skip_rows=None: fake.iter_with_skip(skip_rows),
    )
    p1.start()
    p2.start()
    return loader, fake, [p1, p2]


class HFStreamingLoaderResilienceTest(unittest.TestCase):
    """Cover ``_raw_iter``'s recovery from httpx-client-closed mid-stream."""

    def test_recovers_from_single_httpx_close(self):
        """One closure mid-stream: all 200 windows still emitted, no double-feed."""
        # Plan: invocation 0 raises after yielding 50 rows. Invocation 1
        # is the reopen, opened at skip_rows=50, completes through EOF.
        loader, fake, patches = _make_loader(
            n_rows=200, failure_plan=[(0, 50)])
        try:
            batches = list(loader._raw_iter())
        finally:
            for p in patches:
                p.stop()

        # 1. Every window made it through.
        self.assertEqual(len(batches), 200,
                         "Expected one batch per row at C=1, batch_size=1")

        # 2. No row was double-fed: the windows from the recovered iter
        #    must match the ground-truth row order exactly.
        for i, batch in enumerate(batches):
            # batch is a [B=1, T_RAW, C=1] tensor.
            self.assertEqual(tuple(batch.shape), (1, T_RAW, 1))
            expected = np.asarray(fake.rows[i]["series"], dtype=np.float32)
            np.testing.assert_array_equal(
                batch[0, :, 0].numpy(), expected,
                err_msg=f"Row {i} content does not match the fake source",
            )

        # 3. Exactly two invocations of the underlying stream:
        #    one for skip_rows=0 (open_stream), one for skip_rows=50
        #    (the reopen via iter_with_skip).
        self.assertEqual(fake.invocation_count, 2)
        self.assertEqual(fake.invocations[0]["start"], 0)
        self.assertEqual(fake.invocations[0]["yielded"], 50)
        self.assertTrue(fake.invocations[0]["raised"])
        self.assertEqual(fake.invocations[1]["start"], 50)
        self.assertEqual(fake.invocations[1]["yielded"], 150)  # 200 - 50
        self.assertFalse(fake.invocations[1]["raised"])

    def test_recovers_from_multiple_non_consecutive_closures(self):
        """Three well-separated closures all recover; counter resets in between."""
        # Plan: invocation 0 raises after 30 rows; invocation 1 (reopen at
        # row 30) raises after 50 more (absolute row 80); invocation 2
        # (reopen at row 80) raises after 70 more (absolute row 150);
        # invocation 3 (reopen at row 150) finishes cleanly.
        # Between each failure we yield well over 1 successful row, so the
        # ``reopens`` counter resets to 0 after each — i.e. the 3 failures
        # never become "consecutive" from the cap's point of view.
        loader, fake, patches = _make_loader(
            n_rows=200, failure_plan=[(0, 30), (1, 50), (2, 70)])
        try:
            batches = list(loader._raw_iter())
        finally:
            for p in patches:
                p.stop()

        self.assertEqual(len(batches), 200,
                         "All rows must reach the consumer despite 3 closures")

        # No double-feed: each window matches its absolute row.
        for i, batch in enumerate(batches):
            expected = np.asarray(fake.rows[i]["series"], dtype=np.float32)
            np.testing.assert_array_equal(
                batch[0, :, 0].numpy(), expected,
                err_msg=f"Row {i} mismatch (would indicate double-feed/skip)",
            )

        # Exactly 4 invocations: 1 initial + 3 reopens.
        self.assertEqual(fake.invocation_count, 4)
        starts = [inv["start"] for inv in fake.invocations]
        self.assertEqual(starts, [0, 30, 80, 150])
        # First three raised, last completed.
        raised = [inv["raised"] for inv in fake.invocations]
        self.assertEqual(raised, [True, True, True, False])
        # Final invocation drained the rest.
        self.assertEqual(fake.invocations[-1]["yielded"], 50)  # 200 - 150

    def test_gives_up_after_max_consecutive_closures(self):
        """6 consecutive closures with zero progress: re-raise, no infinite loop."""
        # Plan: every invocation raises on the very first row (yielded=0)
        # so ``rows_consumed`` never advances and ``reopens`` never resets.
        # The fix's cap is _HF_MAX_REOPENS=5; the 6th re-open attempt
        # is what triggers the give-up branch (``reopens > 5``).
        # We schedule 7 immediate failures so the test is robust even if
        # the cap arithmetic shifts by one in either direction — it would
        # still fail loudly at the right boundary if the cap broke.
        plan = [(i, 0) for i in range(7)]
        loader, fake, patches = _make_loader(
            n_rows=200, failure_plan=plan)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                list(loader._raw_iter())
        finally:
            for p in patches:
                p.stop()

        self.assertIn("client has been closed", str(ctx.exception))

        # Six attempts to open the stream: 1 initial + 5 reopens.
        # The 6th invocation is the "exceeded cap" guard: in the current
        # implementation the cap-check (`reopens > _HF_MAX_REOPENS`) fires
        # AFTER incrementing on the 6th failure, before _open_at() runs.
        # So we expect exactly 6 invocations of the stream.
        self.assertEqual(
            fake.invocation_count, 6,
            f"Expected 1 initial + 5 reopens = 6 stream invocations under "
            f"the _HF_MAX_REOPENS={HFStreamingLoader._HF_MAX_REOPENS} cap, "
            f"got {fake.invocation_count}",
        )
        # Every invocation raised on its very first row (no progress made).
        self.assertTrue(all(inv["raised"] for inv in fake.invocations))
        self.assertTrue(all(inv["yielded"] == 0 for inv in fake.invocations))


if __name__ == "__main__":
    unittest.main()
