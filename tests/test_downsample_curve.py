"""The artefact downsampler, and the silence that let 18.6 MB through.

`scripts/downsample_curve.py` is what keeps a report's committed CSVs small.
Its rule used to be a test on the step value: keep a row when `step < 500` or
`step % 200 == 0`. The trainer writes attention amplitude every 200 steps, so
every row of `<run>_attn_amplitude.csv` already satisfies the second clause.
Applied to that file the filter kept all of it, printed nothing, and exited 0.
`collect_artefacts.sh` did not even apply it, and the 2026-08-04 report
committed 18.6 MB of raw amplitude — but applying it would not have helped.
A rule that silently does nothing is worse than a missing call, because the
next agent reads the call and believes it.

So the rule counts DISTINCT STEPS instead: every distinct step below the dense
threshold, then every Nth distinct step after it, whatever cadence the writer
used. Rows are grouped by step, because one amplitude step is nine rows (three
layers x encoder/forecaster blocks, plus the encoder logged twice) and a step
cut in half is unreadable.

These tests hold four things:

  * a source already logged at the stride's cadence is still reduced;
  * a kept step keeps every one of its rows;
  * a run that removes nothing says so on stderr and exits non-zero;
  * the training-curve output does not move — it is pinned against the
    committed artefacts the report's figures are drawn from.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "downsample_curve.py"
COLLECT = (REPO_ROOT / "experiments" / "2026-08-01_lalign_teacher" / "scripts"
           / "collect_artefacts.sh")
CURVES = (REPO_ROOT / "reports" / "2026-08-04_lalign_teacher" / "results"
          / "training_curves")

# The defaults, restated here so a change to them fails the suite instead of
# quietly rewriting what the next report commits.
DENSE_UNTIL = 500
STRIDE = 200

# One amplitude step, as the trainer writes it: three encoder layers, three
# forecaster layers, then the encoder again (the second call in the forward).
AMP_ROWS_PER_STEP = 9
AMP_CADENCE = 200


def old_rule(step: int) -> bool:
    """The value test this module used to apply. Kept here as the thing the
    new rule has to beat: on a source logged every 200 steps it is True for
    every row."""
    return step < DENSE_UNTIL or step % STRIDE == 0


# --- sources -------------------------------------------------------------

def write_curve(path: Path, first: int, last: int) -> Path:
    """A raw backbone `_losses.csv`: one row per step, as the trainer writes
    it before any collection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "loss", "gap"])
        for step in range(first, last + 1):
            w.writerow([step, 1.0 / step, 0.5])
    return path


def write_amplitude(path: Path, first: int, last: int,
                    cadence: int = AMP_CADENCE,
                    rows_per_step: int = AMP_ROWS_PER_STEP) -> Path:
    """A raw `_attn_amplitude.csv`: `rows_per_step` rows per logged step, one
    per (layer, block), logged every `cadence` steps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "layer_idx", "block", "qk_logit_maxabs"])
        for step in range(first, last + 1, cadence):
            for i in range(rows_per_step):
                w.writerow([step, i % 3, "enc" if i < 3 else "fcst",
                            2.0 + i + step / 1e5])
    return path


# --- running it ----------------------------------------------------------

def run(src: Path, dst: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(src), str(dst),
                           *args], capture_output=True, text=True)


def rows_of(path: Path) -> list[list[str]]:
    with open(path, newline="") as fh:
        return list(csv.reader(fh))[1:]


def steps_of(path: Path) -> list[int]:
    return [int(float(r[0])) for r in rows_of(path)]


def distinct_steps(path: Path) -> list[int]:
    return sorted(set(steps_of(path)))


# --- the defect ----------------------------------------------------------

def test_a_source_logged_at_the_stride_cadence_is_still_reduced(tmp_path):
    """Attention amplitude, logged every 200 steps to backbone 40 000. The
    old value rule keeps every row of it; this is the reduction that never
    happened and cost the 2026-08-04 report 18.6 MB."""
    src = write_amplitude(tmp_path / "a_attn_amplitude.csv", 200, 40000)
    assert all(old_rule(s) for s in steps_of(src)), (
        "this source must be one the old rule kept whole, or the test is not "
        "about the defect")

    res = run(src, tmp_path / "out.csv", "--stride", "5")
    assert res.returncode == 0, res.stdout + res.stderr

    kept = distinct_steps(tmp_path / "out.csv")
    assert len(kept) < len(distinct_steps(src)) / 4, (
        f"kept {len(kept)} of {len(distinct_steps(src))} steps")
    assert len(rows_of(tmp_path / "out.csv")) < len(rows_of(src)) / 4


def test_a_source_coarser_than_the_stride_is_still_reduced(tmp_path):
    """The general case: one row every 1000 steps, five times coarser than
    the 200-step stride. Under the old rule the stride can only ever remove
    rows from a source finer than itself."""
    src = write_amplitude(tmp_path / "coarse_attn_amplitude.csv", 1000, 200000,
                          cadence=1000)
    assert all(old_rule(s) for s in steps_of(src))

    res = run(src, tmp_path / "out.csv", "--stride", "10")
    assert res.returncode == 0, res.stdout + res.stderr
    assert distinct_steps(tmp_path / "out.csv") == \
        list(range(10000, 200001, 10000))


def test_every_row_of_a_kept_step_is_kept(tmp_path):
    """Nine rows describe one step. A step that arrives with eight of them
    reads as a layer that stopped being logged."""
    src = write_amplitude(tmp_path / "a_attn_amplitude.csv", 200, 40000)
    res = run(src, tmp_path / "out.csv", "--stride", "5")
    assert res.returncode == 0, res.stdout + res.stderr

    src_by_step: dict[int, list[list[str]]] = {}
    for row in rows_of(src):
        src_by_step.setdefault(int(row[0]), []).append(row)
    out_by_step: dict[int, list[list[str]]] = {}
    for row in rows_of(tmp_path / "out.csv"):
        out_by_step.setdefault(int(row[0]), []).append(row)

    assert out_by_step, "nothing survived"
    for step, kept in out_by_step.items():
        assert kept == src_by_step[step], (
            f"step {step} kept {len(kept)} of {len(src_by_step[step])} rows")


def test_a_step_logged_twice_keeps_both_copies(tmp_path):
    """Two of the 78 committed amplitude files carry 18 rows on some steps:
    a resume re-logged a step the previous wave had already written. Dropping
    the copies here would edit the artefact, which is not this script's job."""
    src = tmp_path / "dup_attn_amplitude.csv"
    write_amplitude(src, 200, 40000)
    with open(src, "a", newline="") as fh:
        w = csv.writer(fh)
        for i in range(AMP_ROWS_PER_STEP):
            w.writerow([1000, i % 3, "enc", 9.0 + i])

    res = run(src, tmp_path / "out.csv", "--stride", "5")
    assert res.returncode == 0, res.stdout + res.stderr
    kept = steps_of(tmp_path / "out.csv")
    assert kept.count(1000) == 2 * AMP_ROWS_PER_STEP, (
        f"step 1000 kept {kept.count(1000)} rows, not both copies")


# --- the silence ---------------------------------------------------------

def test_a_no_op_is_reported_and_exits_non_zero(tmp_path):
    """The whole point. A run that removes nothing must be impossible to miss:
    the next agent copies these scripts, sees a clean log, and commits the
    same 18 MB."""
    src = write_amplitude(tmp_path / "a_attn_amplitude.csv", 200, 40000)
    res = run(src, tmp_path / "out.csv", "--stride", "1")

    assert res.returncode != 0, "a no-op exited 0"
    assert "NO-OP" in res.stderr, res.stderr
    assert "cadence: 200 steps" in res.stderr, (
        f"the message does not name the cadence it inferred:\n{res.stderr}")
    # The file is still written whole: the exit code is a signal, not a
    # failure to collect the artefact.
    assert rows_of(tmp_path / "out.csv") == rows_of(src)


def test_a_barely_reduced_source_warns_and_succeeds(tmp_path):
    """A run that stopped inside the dense window keeps almost every row, and
    that is correct rather than broken. Say it, do not fail on it."""
    src = write_curve(tmp_path / "short_losses.csv", 1, 511)
    res = run(src, tmp_path / "out.csv")

    assert res.returncode == 0, res.stdout + res.stderr
    assert "BARELY REDUCED" in res.stderr, res.stderr
    assert len(rows_of(tmp_path / "out.csv")) == 500


def test_an_over_reduced_source_is_reported_too(tmp_path):
    """The other side of the same silence, and one this change opens. The
    curve stride against a source logged every 200 steps keeps three points,
    which is not a curve. The old value rule could not produce that; the
    distinct-step rule can, so it has to say so."""
    src = write_amplitude(tmp_path / "a_attn_amplitude.csv", 200, 40000)
    res = run(src, tmp_path / "out.csv")     # the default curve stride

    assert res.returncode == 0, res.stdout + res.stderr
    assert "THIN" in res.stderr, res.stderr
    assert len(distinct_steps(tmp_path / "out.csv")) < 10


def test_the_stride_the_collector_uses_is_not_thin(tmp_path):
    """And the setting `collect_artefacts.sh` ships with must not trip it, or
    the warning is noise the next agent learns to skip."""
    src = write_amplitude(tmp_path / "a_attn_amplitude.csv", 200, 40000)
    res = run(src, tmp_path / "out.csv", "--stride", "5")
    assert res.stderr == "", res.stderr


def test_a_source_inside_the_dense_window_is_silent(tmp_path):
    """Nothing above the threshold means the stride never had a candidate to
    drop. Warning here would train the reader to ignore the warning."""
    src = write_curve(tmp_path / "smoke_losses.csv", 1, 400)
    res = run(src, tmp_path / "out.csv")

    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stderr == "", res.stderr
    assert len(rows_of(tmp_path / "out.csv")) == 400


# --- the training curves, pinned against what is committed ---------------

@pytest.mark.parametrize("committed,first,last,n_rows", [
    # A first wave: raw steps 1..40 000, one row per step.
    ("bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
     "_alignteacher_losses.csv", 1, 40000, 697),
    # A resumed wave: the raw file starts at the step after the snapshot.
    ("bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
     "_alignteacher_r2_losses.csv", 40001, 100000, 300),
])
def test_the_committed_training_curve_is_reproduced_exactly(
        tmp_path, committed, first, last, n_rows):
    """Non-regression, against the artefact itself. Rebuild the raw curve the
    collector was given, run the new rule over it, and require the same steps
    the report is plotted from — not merely the same count."""
    src = write_curve(tmp_path / "raw_losses.csv", first, last)
    res = run(src, tmp_path / "out.csv")
    assert res.returncode == 0, res.stdout + res.stderr

    got = steps_of(tmp_path / "out.csv")
    assert len(got) == n_rows
    assert got == steps_of(CURVES / committed)


def test_the_last_step_is_never_cut_off(tmp_path):
    """A wave that ends off the stride still has to show where it ended."""
    src = write_curve(tmp_path / "odd_losses.csv", 1, 40137)
    res = run(src, tmp_path / "out.csv")
    assert res.returncode == 0, res.stdout + res.stderr
    assert steps_of(tmp_path / "out.csv")[-1] == 40137


def test_the_stride_counts_steps_not_rows(tmp_path):
    """Nine rows per step, stride 5: five distinct steps per kept step, not
    five rows. Counting rows would tie the output to the layer count."""
    src = write_amplitude(tmp_path / "a_attn_amplitude.csv", 200, 20000)
    res = run(src, tmp_path / "out.csv", "--stride", "5")
    assert res.returncode == 0, res.stdout + res.stderr
    # 200 and 400 are the dense window; the rest is every 5th logged step.
    assert distinct_steps(tmp_path / "out.csv") == \
        [200, 400] + list(range(1000, 20001, 1000))


def test_the_header_survives(tmp_path):
    src = write_amplitude(tmp_path / "a_attn_amplitude.csv", 200, 40000)
    run(src, tmp_path / "out.csv", "--stride", "5")
    with open(tmp_path / "out.csv", newline="") as fh:
        assert next(csv.reader(fh)) == ["step", "layer_idx", "block",
                                        "qk_logit_maxabs"]
    assert not (tmp_path / "out.csv.tmp").exists(), "left a partial file behind"


# --- the collector -------------------------------------------------------

def test_collect_artefacts_routes_amplitude_through_the_downsampler(tmp_path):
    """The other half of the defect: `collect_artefacts.sh` downsampled the
    curves and copied the amplitude raw."""
    runs = tmp_path / "wt" / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    name = "bb_small_arm5_lalign_lrep_enc3l3_b64_200k_x_alignteacher"
    src = write_amplitude(runs / f"{name}_attn_amplitude.csv", 200, 40000)
    write_curve(runs / f"{name}_losses.csv", 1, 40000)

    dst = tmp_path / "results"
    env = {**os.environ, "HOME": str(tmp_path / "home"),
           "WT": str(tmp_path / "wt"), "REPO": str(tmp_path / "repo"),
           "DST": str(dst)}
    res = subprocess.run(["bash", str(COLLECT)], capture_output=True,
                         text=True, env=env)
    assert res.returncode == 0, res.stdout + res.stderr

    out = dst / "attn_amplitude" / f"{name}_attn_amplitude.csv"
    assert out.is_file(), res.stdout + res.stderr
    assert len(rows_of(out)) < len(rows_of(src)) / 4, (
        f"the collector wrote {len(rows_of(out))} of {len(rows_of(src))} rows "
        "— it is still copying the amplitude raw")
    assert len(rows_of(dst / "training_curves" / f"{name}_losses.csv")) == 697
