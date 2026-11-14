"""src/loss_parts.py: the parts of the training loss over the steps, read
from the losses CSVs of a run and its resumes, and averaged over log-spaced
bins of steps (#462)."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.loss_parts import (depth_columns, log_binned_means,  # noqa: E402
                            read_loss_legs)


def losses_csv(path, steps, scale=1.0):
    """A losses CSV with the columns of a value-space run at depth 1."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "gap", "val_err_d0", "val_err_d1"])
        for step in steps:
            d0, d1 = scale * step, scale * 2 * step
            writer.writerow([step, d0 + d1, "", d0, d1])
    return path


def test_depth_columns_in_depth_order():
    header = ["step", "loss", "val_err_d10", "val_err_d2", "val_err_d0",
              "val_err_d1", "cos_err_d0"]
    assert depth_columns(header) == ["val_err_d0", "val_err_d1",
                                     "val_err_d2", "val_err_d10"]


def test_one_leg(tmp_path):
    path = losses_csv(tmp_path / "a.csv", range(1, 6))
    steps, values = read_loss_legs([path], ["loss", "val_err_d1"])
    assert steps.tolist() == [1, 2, 3, 4, 5]
    assert values["loss"].tolist() == [3.0, 6.0, 9.0, 12.0, 15.0]
    assert values["val_err_d1"].tolist() == [2.0, 4.0, 6.0, 8.0, 10.0]


def test_a_resume_replaces_the_steps_it_logs_again(tmp_path):
    # The run stops at step 8 and resumes from its save at step 5.
    first = losses_csv(tmp_path / "a.csv", range(1, 9))
    resume = losses_csv(tmp_path / "a_r2.csv", range(6, 11), scale=10.0)
    steps, values = read_loss_legs([first, resume], ["val_err_d0"])
    assert steps.tolist() == list(range(1, 11))
    assert values["val_err_d0"].tolist() == [1, 2, 3, 4, 5,
                                             60, 70, 80, 90, 100]


def test_log_binned_means_of_full_bins():
    steps = np.arange(1, 1001)
    centres, means = log_binned_means(steps, {"v": steps * 1.0}, 1, 1000, 3)
    edges = [1, 10, 100, 1000]
    assert centres.tolist() == pytest.approx(
        [math.sqrt(lo * hi) for lo, hi in zip(edges, edges[1:])])
    # The steps 1-9, 10-99 and 100-1000: the last bin holds its top edge.
    assert means["v"].tolist() == pytest.approx([5.0, 54.5, 550.0])


def test_a_partial_last_bin_centres_on_its_logged_steps():
    steps = np.arange(1, 201)
    centres, means = log_binned_means(steps, {"v": steps * 1.0}, 1, 1000, 3)
    assert centres[-1] == pytest.approx(math.sqrt(100 * 200))
    assert means["v"][-1] == pytest.approx(150.0)


def test_a_bin_with_no_logged_step_has_no_point():
    steps = np.array([2, 3, 500])
    centres, means = log_binned_means(steps, {"v": steps * 1.0}, 1, 1000, 3)
    assert len(centres) == 2
    assert means["v"].tolist() == pytest.approx([2.5, 500.0])
