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

These tests hold ten things:

  * a source already logged at the stride's cadence is still reduced;
  * a kept step keeps every one of its rows;
  * a run that removes nothing says so on stderr and exits non-zero;
  * a run that removes nearly everything says so too, however short the
    source is, and exits with its own code;
  * the highest step of the file survives, including when a resume re-logs an
    earlier one at the end;
  * every way of not writing the file has its own exit code and a message;
  * a source that changes between the two passes over it is refused, not
    half-copied;
  * a settings line never divides by a cadence the source does not have;
  * `collect_artefacts.sh` fails, out loud, when a stage collects no file, and
    counts the files it collected un-reduced and the files it collected too
    thin — the same silence one layer up;
  * the training-curve output does not move — it is pinned against the
    committed artefacts the report's figures are drawn from.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "downsample_curve.py"
COLLECT = (REPO_ROOT / "experiments" / "2026-08-01_lalign_teacher" / "scripts"
           / "collect_artefacts.sh")
RESULTS = REPO_ROOT / "reports" / "2026-08-04_lalign_teacher" / "results"
CURVES = RESULTS / "training_curves"
AMPLITUDE = RESULTS / "attn_amplitude"

# The one committed amplitude file whose steps are not monotone: a resume
# re-logged steps 40200..47800 in the middle of it, so 39 of its steps carry
# 18 rows instead of 9. It is the real shape the "last step" rule has to
# survive.
RE_LOGGED = ("bb_small_arm1_tr1_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk"
             "_aon_cpc_tau090_r2_attn_amplitude.csv")

# The defaults, restated here so a change to them fails the suite instead of
# quietly rewriting what the next report commits.
DENSE_UNTIL = 500
STRIDE = 200

# One amplitude step, as the trainer writes it: three encoder layers, three
# forecaster layers, then the encoder again (the second call in the forward).
AMP_ROWS_PER_STEP = 9
AMP_CADENCE = 200

# Latent drift is written once every 10 000 steps, two rows each time.
DRIFT_CADENCE = 10000

# One exit code per outcome, restated here so a collision fails the suite.
# The collector reads them: it collects 0, 3 and 4, and loses the rest.
EXIT_OK = 0
EXIT_NO_HEADER = 1
EXIT_USAGE = 2          # argparse's own code for a bad argument
EXIT_NO_OP = 3
EXIT_THIN = 4
EXIT_UNREADABLE = 5
EXIT_SOURCE_MOVED = 6


def old_rule(step: int) -> bool:
    """The value test this module used to apply. Kept here as the thing the
    new rule has to beat: on a source logged every 200 steps it is True for
    every row."""
    return step < DENSE_UNTIL or step % STRIDE == 0


# --- sources -------------------------------------------------------------

def write_curve(path: Path, first: int, last: int, cadence: int = 1) -> Path:
    """A raw backbone `_losses.csv`: one row per step, as the trainer writes
    it before any collection. `cadence` coarsens it, for the artefacts the
    trainer writes less often than every step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "loss", "gap"])
        for step in range(first, last + 1, cadence):
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


def write_latent_drift(path: Path, last: int = 70000,
                       cadence: int = DRIFT_CADENCE) -> Path:
    """A raw `_latent_drift.csv`: two rows every 10 000 steps. A whole run is
    14 rows and 1 KB, which is why the collector copies it rather than
    downsampling it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "latent", "kind", "step_ref", "delta_step",
                    "drift_cos", "drift_cos_aligned", "rot_gap", "cka"])
        for step in range(cadence, last + 1, cadence):
            for latent in ("student_h", "teacher_h"):
                w.writerow([step, latent, "adjacent", step - cadence, cadence,
                            0.96, 0.73, 0.23, 0.20])
    return path


def write_steps(path: Path, steps: list[int]) -> Path:
    """A curve whose steps are whatever the caller says, in the caller's
    order. For the shapes a generator cannot make."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "loss", "gap"])
        for step in steps:
            w.writerow([step, 1.0 / max(step, 1), 0.5])
    return path


# --- running it ----------------------------------------------------------

def run(src: Path, dst: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(src), str(dst),
                           *args], capture_output=True, text=True)


def load_script():
    """The script as a module, for the one guard a subprocess cannot exercise:
    a source that changes between the two passes over it."""
    spec = importlib.util.spec_from_file_location("downsample_curve", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["downsample_curve"] = mod
    spec.loader.exec_module(mod)
    return mod


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
    distinct-step rule can, so it has to say so.

    THIN exits 4, not 0. Warning only on stderr is the same silence NO-OP
    exists to remove: on a 33-run collect that line sits in the stderr of 33
    runs while stdout says 33/33 collected. The file is still written whole,
    so the code costs nothing but attention."""
    src = write_amplitude(tmp_path / "a_attn_amplitude.csv", 200, 40000)
    res = run(src, tmp_path / "out.csv")     # the default curve stride

    assert res.returncode == EXIT_THIN, res.stdout + res.stderr
    assert "THIN" in res.stderr, res.stderr
    assert len(distinct_steps(tmp_path / "out.csv")) < 10


@pytest.mark.parametrize("n_steps", [
    14,   # a whole latent-drift run, logged every 10 000 steps
    30,   # and a source three times longer, still cut to one point
])
def test_a_short_source_cut_to_one_point_is_reported(tmp_path, n_steps):
    """THIN has to fire on the sources it was written for. The first rule
    asked the source for 40 distinct steps before it would warn, so a 14-point
    latent-drift file cut to its last point alone stayed silent — the exact
    shape the warning exists to catch."""
    src = write_curve(tmp_path / "coarse_losses.csv", DRIFT_CADENCE,
                      DRIFT_CADENCE * n_steps, cadence=DRIFT_CADENCE)
    assert len(distinct_steps(src)) == n_steps < 40

    res = run(src, tmp_path / "out.csv")     # the default curve stride
    assert res.returncode == EXIT_THIN, res.stdout + res.stderr
    assert len(distinct_steps(tmp_path / "out.csv")) == 1
    assert "THIN" in res.stderr, (
        f"{n_steps} points cut to 1, and the run said nothing:\n{res.stderr}")


def test_thin_and_no_op_do_not_share_an_exit_code(tmp_path):
    """Two different failures. NO-OP means the file came out whole; THIN means
    almost nothing came out of it. A caller that cannot tell them apart cannot
    count them apart, and the file it should look at first is the thin one."""
    src = write_amplitude(tmp_path / "a_attn_amplitude.csv", 200, 40000)
    no_op = run(src, tmp_path / "flat.csv", "--stride", "1")
    thin = run(src, tmp_path / "thin.csv", "--stride", "200")

    assert no_op.returncode == EXIT_NO_OP, no_op.stderr
    assert thin.returncode == EXIT_THIN, thin.stderr
    assert no_op.returncode != thin.returncode


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


# --- what "last" means when the steps are not monotone -------------------

def test_the_highest_step_survives_a_re_log_at_the_end_of_the_file(tmp_path):
    """"Last" is the highest step in the file, not the last row and not the
    last step to first appear. Those three agree only while the steps
    increase, and a resume re-logs earlier steps: two committed amplitude
    files already carry that shape. Here the re-log lands at the end of the
    file on a step the first wave never wrote, so the last step to first
    appear is 39900 while the wave reached 40000."""
    src = write_amplitude(tmp_path / "relog_attn_amplitude.csv", 200, 40000)
    with open(src, "a", newline="") as fh:
        w = csv.writer(fh)
        for i in range(AMP_ROWS_PER_STEP):
            w.writerow([39900, i % 3, "enc" if i < 3 else "fcst", 7.0 + i])

    # Stride 7 puts 40000 off the stride, so only the guarantee can keep it.
    res = run(src, tmp_path / "out.csv", "--stride", "7")
    assert res.returncode == 0, res.stdout + res.stderr

    kept = steps_of(tmp_path / "out.csv")
    assert kept.count(40000) == AMP_ROWS_PER_STEP, (
        "the wave reached 40000 and the output does not show it: "
        f"kept {sorted(set(kept))[-3:]}")
    assert 39900 not in kept, (
        "39900 is the last step to first appear, not the end of the wave — "
        "keeping it instead of 40000 is the bug this pins")


def test_a_step_removed_at_the_end_of_the_file_is_not_a_no_op(tmp_path):
    """NO-OP means the stride removed nothing it could have removed, and
    "could have" has to read "last" the same way: every step at or above the
    dense window except the highest one. Reading it as "except the last step
    to first appear" hides exactly one removal, the re-logged step at the end
    of the file, and calls a run that did reduce a clean NO-OP."""
    src = write_amplitude(tmp_path / "relog_attn_amplitude.csv", 200, 40000)
    with open(src, "a", newline="") as fh:      # a resume re-logs 39900, off
        w = csv.writer(fh)                      # the 200-step grid because the
        for i in range(AMP_ROWS_PER_STEP):      # checkpoint it came from was,
            w.writerow([39900, i % 3, "enc", 7.0 + i])   # then dies

    # The dense window takes every step up to 39800, so the only two above it
    # are the re-logged 39900 and the final 40000. 40000 is the highest step
    # and is kept; 39900 is the one step the stride has to remove.
    res = run(src, tmp_path / "out.csv", "--dense-until", "39900")
    assert res.returncode == 0, res.stdout + res.stderr
    assert 39900 not in distinct_steps(tmp_path / "out.csv")
    assert 40000 in distinct_steps(tmp_path / "out.csv")
    assert "NO-OP" not in res.stderr, (
        "the stride removed 39500 and the run reported removing nothing:\n"
        + res.stderr)


def test_the_committed_re_logged_amplitude_keeps_its_highest_step(tmp_path):
    """The same guarantee against the artefact itself, re-log and all. The
    file is read, never rewritten."""
    src = AMPLITUDE / RE_LOGGED
    res = run(src, tmp_path / "out.csv", "--stride", "5")
    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stderr == "", res.stderr

    src_by_step: dict[int, list[list[str]]] = {}
    for row in rows_of(src):
        src_by_step.setdefault(int(row[0]), []).append(row)
    out_by_step: dict[int, list[list[str]]] = {}
    for row in rows_of(tmp_path / "out.csv"):
        out_by_step.setdefault(int(row[0]), []).append(row)

    top = max(src_by_step)
    assert top == 100000
    assert out_by_step.get(top) == src_by_step[top]
    # Every 5th of its 300 logged steps, counted from the first row: 41000,
    # 42000, ... 100000. Spelled out, so a change to how ranks are counted
    # fails here against the real file rather than only against a fixture.
    assert sorted(out_by_step) == list(range(41000, 100001, 1000))
    # And every kept step is whole, including the re-logged 18-row ones.
    for step, kept in out_by_step.items():
        assert kept == src_by_step[step]
    assert any(len(rows) == 2 * AMP_ROWS_PER_STEP
               for rows in out_by_step.values()), (
        "no re-logged step survived, so this ran against the wrong file")


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


# --- one exit code per outcome -------------------------------------------

def test_a_missing_source_says_which_file_and_does_not_traceback(tmp_path):
    """A path that does not exist is a caller's mistake, and the caller reads
    stderr. A traceback names the line that raised, not the file that is
    missing, and its exit code is Python's, not this script's."""
    res = run(tmp_path / "gone_losses.csv", tmp_path / "out.csv")

    assert res.returncode == EXIT_UNREADABLE, res.stdout + res.stderr
    assert "Traceback" not in res.stderr, res.stderr
    assert "gone_losses.csv" in res.stderr, res.stderr
    assert not (tmp_path / "out.csv").exists()


def test_a_source_with_no_header_has_its_own_exit_code(tmp_path):
    """An empty file is not an unreadable one, and neither is a bad flag."""
    src = tmp_path / "empty_losses.csv"
    src.write_text("")
    res = run(src, tmp_path / "out.csv")

    assert res.returncode == EXIT_NO_HEADER, res.stdout + res.stderr
    assert "no header" in res.stderr, res.stderr
    assert not (tmp_path / "out.csv").exists()


def test_a_stride_below_one_is_a_usage_error(tmp_path):
    """`--stride 0` exited 1, the same code as a source with no header. Two
    different mistakes reported as one, and neither of them tested."""
    src = write_curve(tmp_path / "a_losses.csv", 1, 40000)
    res = run(src, tmp_path / "out.csv", "--stride", "0")

    assert res.returncode == EXIT_USAGE, res.stdout + res.stderr
    assert res.returncode != EXIT_NO_HEADER
    assert "--stride" in res.stderr, res.stderr
    assert not (tmp_path / "out.csv").exists(), (
        "a run that refused its arguments still wrote a file")


# --- the two passes over the source --------------------------------------

def test_a_source_that_grows_between_the_two_passes_is_refused(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    """The script reads the source twice: once for the steps it will keep,
    once to copy the rows of those steps. The two passes are not one atomic
    read. On a file still being written the second pass sees rows the first
    never saw, drops them because their steps are not in the keep set, and the
    highest-step guarantee fails without a word.

    Collect runs after the run today, so this is a precondition rather than a
    live bug. The script checks it instead of trusting it.

    The mtime is put back after the write, so what this pins is the size half
    of the check: a filesystem whose mtime did not tick between the two passes
    still has to catch a file that grew."""
    mod = load_script()
    src = write_curve(tmp_path / "growing_losses.csv", 1, 40000)
    dst = tmp_path / "out.csv"

    read_steps = mod.read_steps

    def read_then_grow(reader, step_col):
        """The trainer writes one more step while pass 1 is reading."""
        steps = read_steps(reader, step_col)
        was = os.stat(src)
        with open(src, "a", newline="") as fh:
            csv.writer(fh).writerow([40001, 0.1, 0.5])
        os.utime(src, ns=(was.st_atime_ns, was.st_mtime_ns))
        return steps

    monkeypatch.setattr(mod, "read_steps", read_then_grow)
    monkeypatch.setattr(sys, "argv",
                        ["downsample_curve.py", str(src), str(dst)])

    assert mod.main() == EXIT_SOURCE_MOVED
    assert "changed while it was read" in capsys.readouterr().err
    assert not dst.exists(), (
        "wrote a curve computed from a source that is no longer that source")
    assert not (tmp_path / "out.csv.tmp").exists(), "left a partial file behind"


def test_a_source_rewritten_at_the_same_length_is_refused(tmp_path,
                                                          monkeypatch,
                                                          capsys):
    """The mtime half. A file rewritten with rows of the same width comes back
    the same length, and the size sees nothing. Here the rewrite moves the
    highest step: the keep set holds 40000, the file on the second pass holds
    49999, and the output would come out with no end to the wave at all."""
    mod = load_script()
    src = write_curve(tmp_path / "rewritten_losses.csv", 1, 40000)
    dst = tmp_path / "out.csv"

    read_steps = mod.read_steps

    def read_then_rewrite(reader, step_col):
        steps = read_steps(reader, step_col)
        was = os.stat(src)
        head, _, tail = src.read_bytes().rpartition(b"40000,")
        src.write_bytes(head + b"49999," + tail)        # same byte count
        os.utime(src, ns=(was.st_atime_ns, was.st_mtime_ns + 10 ** 9))
        assert os.stat(src).st_size == was.st_size, "this rewrite changed size"
        return steps

    monkeypatch.setattr(mod, "read_steps", read_then_rewrite)
    monkeypatch.setattr(sys, "argv",
                        ["downsample_curve.py", str(src), str(dst)])

    assert mod.main() == EXIT_SOURCE_MOVED
    assert "changed while it was read" in capsys.readouterr().err
    assert not dst.exists()


def test_a_source_that_grows_during_the_copy_is_refused(tmp_path, monkeypatch,
                                                        capsys):
    """The second pass has a window of its own. Checking the source before the
    copy starts says nothing about what it is when the copy ends, and the rows
    already written came from whatever the file was while it was read. The
    half-written output goes with it."""
    mod = load_script()
    src = write_curve(tmp_path / "growing_losses.csv", 1, 40000)
    dst = tmp_path / "out.csv"

    copy_kept_rows = mod.copy_kept_rows

    def copy_then_grow(reader, writer, step_col, keep):
        written = copy_kept_rows(reader, writer, step_col, keep)
        with open(src, "a", newline="") as fh:
            csv.writer(fh).writerow([40001, 0.1, 0.5])
        return written

    monkeypatch.setattr(mod, "copy_kept_rows", copy_then_grow)
    monkeypatch.setattr(sys, "argv",
                        ["downsample_curve.py", str(src), str(dst)])

    assert mod.main() == EXIT_SOURCE_MOVED
    assert "changed while it was read" in capsys.readouterr().err
    assert not dst.exists()
    assert not (tmp_path / "out.csv.tmp").exists(), "left a partial file behind"


def test_a_source_that_goes_away_between_the_passes_says_so(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """Gone is a kind of moved, and it lands in the window between the two
    passes: the second one opens the file again. A message, not a traceback —
    the same complaint as a source that was never there."""
    mod = load_script()
    src = write_curve(tmp_path / "vanishing_losses.csv", 1, 40000)
    dst = tmp_path / "out.csv"

    read_steps = mod.read_steps

    def read_then_delete(reader, step_col):
        steps = read_steps(reader, step_col)
        src.unlink()
        return steps

    monkeypatch.setattr(mod, "read_steps", read_then_delete)
    monkeypatch.setattr(sys, "argv",
                        ["downsample_curve.py", str(src), str(dst)])

    assert mod.main() == EXIT_SOURCE_MOVED
    assert "changed while it was read" in capsys.readouterr().err
    assert not dst.exists()
    assert not (tmp_path / "out.csv.tmp").exists()


# --- the settings line ---------------------------------------------------

def test_a_source_whose_steps_only_go_down_has_no_cadence_to_divide_by(
        tmp_path):
    """The stderr block reads the cadence off the source and multiplies it by
    the stride. `infer_cadence` counts the gaps that increase, so a file whose
    steps only ever go down has no gap to count and the cadence is 0 — and the
    block said "one point every 0 steps"."""
    src = write_steps(tmp_path / "backwards_losses.csv",
                      list(range(10000, 0, -1000)))
    res = run(src, tmp_path / "out.csv")

    assert res.returncode == EXIT_THIN, res.stdout + res.stderr
    assert "every 0 steps" not in res.stderr, res.stderr
    assert "cadence: none" in res.stderr, (
        "the block hides that it could not read a cadence:\n" + res.stderr)


# --- the collector -------------------------------------------------------

RUN_NAME = "bb_small_arm5_lalign_lrep_enc3l3_b64_200k_x_alignteacher"


def collect(tmp_path: Path, **env_overrides: str):
    """Lay out one run's raw artefacts the way the trainer leaves them on
    elisa, then run `collect_artefacts.sh` over them into `tmp_path/results`.
    Returns (completed process, destination directory)."""
    runs = tmp_path / "wt" / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    write_amplitude(runs / f"{RUN_NAME}_attn_amplitude.csv", 200, 40000)
    write_curve(runs / f"{RUN_NAME}_losses.csv", 1, 40000)
    write_latent_drift(runs / f"{RUN_NAME}_latent_drift.csv")

    dst = tmp_path / "results"
    env = {**os.environ, "HOME": str(tmp_path / "home"),
           "WT": str(tmp_path / "wt"), "REPO": str(tmp_path / "repo"),
           "DST": str(dst), **env_overrides}
    return subprocess.run(["bash", str(COLLECT)], capture_output=True,
                          text=True, env=env), dst


def test_collect_artefacts_routes_amplitude_through_the_downsampler(tmp_path):
    """The other half of the defect: `collect_artefacts.sh` downsampled the
    curves and copied the amplitude raw."""
    res, dst = collect(tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr

    runs = tmp_path / "wt" / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    src = runs / f"{RUN_NAME}_attn_amplitude.csv"
    out = dst / "attn_amplitude" / f"{RUN_NAME}_attn_amplitude.csv"
    assert out.is_file(), res.stdout + res.stderr
    assert len(rows_of(out)) < len(rows_of(src)) / 4, (
        f"the collector wrote {len(rows_of(out))} of {len(rows_of(src))} rows "
        "— it is still copying the amplitude raw")
    assert len(rows_of(dst / "training_curves" / f"{RUN_NAME}_losses.csv")) == 697


def test_the_collector_fails_when_a_stage_collects_no_file(tmp_path):
    """The same silence, one layer up. The downsampler exiting non-zero says
    nothing if the collector never ran it: point DOWNSAMPLE at nothing and the
    old collector wrote no curve, no amplitude, and exited 0 — which reads as
    a clean collection."""
    res, dst = collect(tmp_path,
                       DOWNSAMPLE=str(tmp_path / "gone" / "downsample_curve.py"))

    assert res.returncode != 0, (
        "collected nothing and reported success:\n" + res.stdout + res.stderr)
    assert res.stdout.count("collected 0 files") == 2, (
        "both stages collected nothing and the log does not say so:\n"
        + res.stdout)
    assert not (dst / "training_curves" / f"{RUN_NAME}_losses.csv").exists()
    assert not (dst / "attn_amplitude"
                / f"{RUN_NAME}_attn_amplitude.csv").exists()


def test_an_un_reduced_file_is_collected_and_says_so(tmp_path):
    """And the other side of that split: exit 3 means the file is on disk and
    came out whole, which is a warning, not a lost artefact. The collector has
    to tell that from a file it never wrote."""
    res, dst = collect(tmp_path, ATTN_STRIDE="1")   # stride 1 removes nothing
    assert res.returncode == 0, res.stdout + res.stderr

    runs = tmp_path / "wt" / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    out = dst / "attn_amplitude" / f"{RUN_NAME}_attn_amplitude.csv"
    assert rows_of(out) == rows_of(runs / f"{RUN_NAME}_attn_amplitude.csv"), (
        "the artefact was not collected whole")
    assert "un-reduced" in res.stdout, res.stdout
    assert "1/1 downsampled" in res.stdout, (
        "a file that was written whole was counted as one that was not "
        "written:\n" + res.stdout)
    assert "collected 0 files" not in res.stdout, res.stdout
    assert "NO-OP" in res.stderr, res.stderr


def test_the_collector_counts_a_thin_file_and_says_so(tmp_path):
    """THIN was the last silence the collector did not count. NO-OP has an
    exit code, a counter and a summary line; THIN exited 0, landed in the ok
    count, and printed on stderr only. Over 33 runs that line sits in the
    stderr of 33 runs while stdout says 33/33 downsampled and nothing else.

    The file is collected, like an un-reduced one: too few points is a warning
    about the stride, not a lost artefact."""
    res, dst = collect(tmp_path, ATTN_STRIDE="200")
    assert res.returncode == 0, res.stdout + res.stderr

    out = dst / "attn_amplitude" / f"{RUN_NAME}_attn_amplitude.csv"
    assert len(distinct_steps(out)) < 10, "this stride does not make it thin"
    assert "1/1 downsampled" in res.stdout, (
        "a thin file was counted as one that was not written:\n" + res.stdout)
    assert "too thin" in res.stdout, (
        "a file cut below a curve was collected and the summary said "
        "nothing:\n" + res.stdout)
    assert "THIN" in res.stderr, res.stderr


def test_the_collector_copies_latent_drift_whole(tmp_path):
    """Latent drift is not downsampled, and that is the decision, not an
    oversight. The trainer writes it once every 10 000 steps, so a whole run
    is 14 rows and 1 KB — 136 KB across the 33 files of the 2026-08-04 report
    — and the curve stride would cut those 14 points to one. The downsampler
    calls that THIN; the collector avoids it by copying."""
    res, dst = collect(tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr

    runs = tmp_path / "wt" / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    src = runs / f"{RUN_NAME}_latent_drift.csv"
    out = dst / "latent_drift" / f"{RUN_NAME}_latent_drift.csv"
    assert out.read_bytes() == src.read_bytes()
