"""Regression tests for the dataloader mod-wrap on resume.

When resuming a long training run on a dataset whose total row count is
smaller than ``hf_rows_consumed``, ``HFStreamingLoader._iter_stream_with_fast_skip``
must mod-wrap ``skip_rows`` against ``total_rows`` and continue yielding
rows from the wrapped within-epoch position. Without the mod-wrap, the
generator falls into the "all shards come before target" branch, returns
without yielding anything, and the training loop dies with
``StopIteration`` on the first batch after resume.

The two cases covered:

1. ``skip_rows`` is an exact multiple of ``total_rows``: wraps to 0 and
   yields the very first row.
2. ``skip_rows`` is past one full epoch by a fractional offset: wraps to
   the within-epoch offset and yields the row at that offset.

Both tests monkey-patch ``_list_shard_files``, ``_shard_row_counts`` and
``_pyarrow_stream_from_shards`` so they're deterministic, network-free,
fast.
"""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.dataloader import HFStreamingLoader


def _make_loader(skip_rows: int) -> HFStreamingLoader:
    """Build a minimally-configured loader for unit tests."""
    return HFStreamingLoader(
        repo_id="ignored",
        batch_size=4,
        C=1,
        skip_rows=skip_rows,
        prefetch=0,
    )


def _make_fake_pyarrow_stream(n_rows: int, started_flag: list[bool]):
    """Return a stub for ``_pyarrow_stream_from_shards``.

    The stub honours ``within_shard_skip`` — drops the first N rows of the
    fake stream then yields the remainder. ``started_flag[0]`` is set True
    when the generator is first advanced (so tests can assert the inner
    stream was actually consumed).
    """

    def stub(shards, within_shard_skip: int = 0):
        started_flag[0] = True
        for i in range(within_shard_skip, n_rows):
            yield {"series": [0.0] * 1025, "source_id": 0, "meta": f"r{i}"}

    return stub


class HFStreamingLoaderModWrapTest(unittest.TestCase):
    """Cover ``_iter_stream_with_fast_skip``'s mod-wrap of ``skip_rows``."""

    def test_modwrap_when_skip_exceeds_total(self):
        """``skip_rows`` exactly past N epochs wraps to 0 and yields row 0."""
        # 60_000 total rows, but caller asks to skip 240_000 (4 full epochs
        # plus zero offset). Wrap should pin skip_rows to 0 and yield the
        # first row of the first remaining shard.
        loader = _make_loader(skip_rows=240_000)
        started = [False]
        fake_stream = _make_fake_pyarrow_stream(n_rows=3, started_flag=started)

        buf = io.StringIO()
        with patch.object(loader, "_list_shard_files",
                          return_value=["s0", "s1"]), \
             patch.object(loader, "_shard_row_counts",
                          return_value=[30_000, 30_000]), \
             patch.object(loader, "_pyarrow_stream_from_shards",
                          side_effect=fake_stream), \
             redirect_stdout(buf):
            gen = loader._iter_stream_with_fast_skip()
            first_row = next(gen)

        captured = buf.getvalue()
        self.assertIn(
            "wrapping to 0", captured,
            f"expected wrap log in stdout, got:\n{captured}",
        )
        # And the inner stream must actually have been started, yielding a row.
        self.assertEqual(first_row["meta"], "r0")
        self.assertTrue(started[0])

    def test_modwrap_yields_first_row_at_wrapped_offset(self):
        """``skip_rows`` past one epoch wraps to the within-epoch offset."""
        # 60_000 total rows. skip_rows=75_000 → wraps to 15_000 → halfway into
        # shard 0 (which has 30_000 rows). within_shard_skip=15_000 means the
        # generator must drop the first 15_000 rows of the wrapped stream and
        # then yield row 15_000.
        loader = _make_loader(skip_rows=75_000)
        started = [False]
        fake_stream = _make_fake_pyarrow_stream(n_rows=20_000,
                                                started_flag=started)

        buf = io.StringIO()
        with patch.object(loader, "_list_shard_files",
                          return_value=["s0", "s1"]), \
             patch.object(loader, "_shard_row_counts",
                          return_value=[30_000, 30_000]), \
             patch.object(loader, "_pyarrow_stream_from_shards",
                          side_effect=fake_stream), \
             redirect_stdout(buf):
            gen = loader._iter_stream_with_fast_skip()
            first_row = next(gen)

        captured = buf.getvalue()
        self.assertIn("wrapping to 15000", captured)
        # 15_000 within-shard rows dropped → first yielded row is index 15_000.
        self.assertEqual(first_row["meta"], "r15000")


if __name__ == "__main__":
    unittest.main()
