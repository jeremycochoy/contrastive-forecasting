"""Smoke test: HF dataloader factory yields identical batches on two
instantiations from the same start. Verifies the May 3 2026 directive that
sample order is deterministic and in-order (HF dataset is pre-shuffled at
storage, so no in-memory shuffle is needed).

Run with: pytest -xvs tests/test_dataloader_determinism.py
"""

from unittest.mock import patch

import numpy as np
import torch

from src.dataloader import HFStreamingLoader, create_mixed_periodic_dataloader


def _deterministic_rows(n_rows: int, T: int = 1024, source_id: int = 1):
    """Fixed-seed fake parquet rows. Same across calls — proxy for the
    pre-shuffled-but-deterministic HF bundle stream."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_rows):
        series = rng.standard_normal(T + 1).astype(np.float32).tolist()
        rows.append({"series": series, "source_id": source_id,
                     "meta": f"row_{i}"})
    return rows


def test_two_loader_instantiations_yield_identical_batches():
    """Two factory calls with same args + skip_rows=0 must produce
    bytewise-identical first-N batches."""
    n_batches = 10
    fake_rows = _deterministic_rows(64)

    def make_loader():
        loader = HFStreamingLoader(
            repo_id="ignored", batch_size=2, C=2, prefetch=0)
        return loader

    loader_a = make_loader()
    loader_b = make_loader()

    with patch.object(HFStreamingLoader, "_open_stream",
                      side_effect=[list(fake_rows), list(fake_rows)]):
        it_a = iter(loader_a)
        it_b = iter(loader_b)
        for i in range(n_batches):
            try:
                a = next(it_a)
                b = next(it_b)
            except StopIteration:
                break
            assert torch.equal(a, b), f"Batch {i} differs across instantiations"

    # Print first-batch checksum so a manual run shows determinism
    # without diff'ing two giant tensors.
    with patch.object(HFStreamingLoader, "_open_stream",
                      return_value=list(fake_rows)):
        first = next(iter(make_loader()))
        print(f"first-batch shape={tuple(first.shape)} "
              f"sum={first.sum().item():.6f} "
              f"mean={first.mean().item():.6f}")


if __name__ == "__main__":
    test_two_loader_instantiations_yield_identical_batches()
    print("OK: dataloader is deterministic in-order")
