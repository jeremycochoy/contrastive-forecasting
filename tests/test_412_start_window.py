"""The EWMA's first window (#412 review).

``RevEWMNorm`` starts its mean and variance from the first W values of a
series (the first W real values with ``skip_leading_zeros``). A patch whose
next patch starts in that window reads statistics of its depth-0 target:
at a patch size below W, and when the left padding does not end on a patch
boundary. ``start_window_anchors`` marks these patches, and no training term
reads them. These tests check the mask, and that no other patch reads its
target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.norm import RevEWMNorm, start_window_anchors  # noqa: E402

W, T = 16, 128


def normalised(x, skip_zeros):
    norm = RevEWMNorm(num_features=1, span=128, patch_size=W,
                      skip_leading_zeros=skip_zeros)
    return norm(x, mode="norm"), norm


def series(zeros):
    x = torch.randn(1, T, 1, generator=torch.Generator().manual_seed(0)).abs() + 1.0
    x[:, :zeros] = 0.0
    return x


@pytest.mark.parametrize("P, zeros, marked", [
    (8, 0, [0]),          # patch 1 lies in the first window
    (16, 0, []),          # the first window is patch 0 itself
    (8, 16, [0, 1, 2]),   # padding 0-1, then the first real patch
    (16, 21, [0, 1]),     # padding that ends inside patch 1
    (32, 21, [0]),
])
def test_the_marked_patches(P, zeros, marked):
    _, norm = normalised(series(zeros), skip_zeros=zeros > 0)
    anchors = start_window_anchors(norm.start_mask, P)[0, :, 0]
    assert anchors.nonzero().flatten().tolist() == marked


@pytest.mark.parametrize("P, zeros", [(8, 0), (16, 0), (8, 16), (16, 21),
                                      (32, 21), (8, 37)])
def test_no_unmarked_patch_reads_its_target(P, zeros):
    skip = zeros > 0
    x = series(zeros)
    base, norm = normalised(x, skip)
    anchors = start_window_anchors(norm.start_mask, P)[0, :, 0]
    pad = norm.pad_mask[0, :, 0] if skip else torch.zeros(T, dtype=torch.bool)
    for p in range(T // P - 1):
        if anchors[p] or pad[p * P:(p + 1) * P].all():
            continue
        moved = x.clone()
        moved[:, (p + 1) * P:(p + 2) * P] += 5.0      # its depth-0 target
        out, _ = normalised(moved, skip)
        assert torch.equal(out[:, p * P:(p + 1) * P], base[:, p * P:(p + 1) * P]), p


def test_a_marked_patch_reads_its_target():
    """The control: without the mask, patch 0 at P=8 reads patch 1."""
    x = series(0)
    base, _ = normalised(x, skip_zeros=False)
    moved = x.clone()
    moved[:, 8:16] += 5.0
    out, _ = normalised(moved, skip_zeros=False)
    assert not torch.equal(out[:, :8], base[:, :8])
