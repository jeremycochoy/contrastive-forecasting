"""Tests for #409's study scaffolding: the arms, the launcher and the AUC watch.

#409 runs ONE configuration at ONE decay shape, over repeat backbone seeds.
The configuration is the EMA momentum sweep's best arm: #373's cell
`arm6_v2_combab_alignT` at k = 32 under the `mean` reduction, with L_align on
the EMA teacher and a momentum of 0.9 that rises to 1.0 at step 100,000. The
decay is one extra factor in front of L_rep. It starts at 1.0 and falls
linearly to 0.0 at step 10,000.

The card runs NO control. The sweep already measured this cell at two seeds,
1.1491 at seed 20260524 and 1.1507 at seed 20260520, and
`reports/2026-08-19_ema_momentum_k32/ema_momentum_k32.md` holds both. Every
backbone of this card goes to the decay arm, at a seed of its own.

It adds two trainer flags, `--rep-loss-weight-end` and
`--rep-loss-weight-ramp-steps`, and no new pipeline. It reuses #373's runner
and supplies the decay and the seed. `tests/test_409_rep_weight_decay.py` holds
the objective itself.

That is the contract these tests hold:

  * the study's constants are the card's: cell `arm6_v2_combab_alignT`,
    k = 32, the `mean` reduction, the teacher target, one stop at 40,000
    backbone steps, a 30,000-step head, and the student encoder.
  * the EMA schedule is the runner's own default, so no arm moves it.
  * ONE decay shape lives in study.sh, not in the arms table, so no row can
    hold a second shape.
  * the arms table holds one row per backbone seed, and every seed differs.
  * every arm's decay flags follow `src.models.linear_schedule_at_step`, so
    the shell table and the trainer agree on the weight at every step.
  * no two arms share a file.
  * the runner is #373's `run_leg_k.sh` — no second trainer invocation
    exists in this study.
  * the AUC watch names the step a run lost the contrastive task.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP = REPO_ROOT / "reports" / "2026-08-22_rep_weight_decay"
PARENT = REPO_ROOT / "reports" / "2026-08-08_rollout_depth"
SWEEP = REPO_ROOT / "reports" / "2026-08-19_ema_momentum_k32"

STUDY_SH = EXP / "scripts" / "study.sh"
RUN_ARM = EXP / "scripts" / "run_arm.sh"
ARMS_TSV = EXP / "scripts" / "arms.tsv"
AUC_WATCH = EXP / "scripts" / "auc_watch.py"
PARENT_LEG = PARENT / "scripts" / "run_leg_k.sh"

CELL = "arm6_v2_combab_alignT"
K = 32
REDUCE = "mean"
ALIGN_TARGET = "teacher"
STOP = 40_000
RAMP = 10_000
# The EMA schedule of the sweep's best arm, which is also the runner's default.
EMA = "--ema-tau 0.9 --ema-tau-end 1.0 --ema-tau-ramp-steps 100000"
# Every arm of the card, in the order it runs them. One row for each backbone
# seed. `launch.sh` deals them round-robin, so the two seeds the sweep already
# scored sit on different cards.
ARMS = ("dec_s20", "dec_s24", "dec_s22", "dec_s23", "dec_s25", "dec_s26")
SEEDS = ("20260520", "20260524", "20260522", "20260523", "20260525",
         "20260526")
# The two seeds the sweep scored on this cell. Their reference numbers are
# read from that report, not measured again here.
SCORED_BY_THE_SWEEP = ("20260520", "20260524")


def study_value(name: str, env=None) -> str:
    """One exported value of study.sh, read by sourcing it."""
    full = dict(os.environ)
    full.update(env or {})
    out = subprocess.run(
        ["bash", "-c", f'. "{STUDY_SH}" >/dev/null && printf "%s" "${{{name}}}"'],
        capture_output=True, text=True, timeout=60, env=full)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def study_call(snippet: str, env=None) -> subprocess.CompletedProcess:
    """Run one function of study.sh and return the process."""
    full = dict(os.environ)
    full.update(env or {})
    return subprocess.run(
        ["bash", "-c", f'. "{STUDY_SH}" >/dev/null && {snippet}'],
        capture_output=True, text=True, timeout=60, env=full)


def study_out(snippet: str, env=None) -> str:
    out = study_call(snippet, env)
    assert out.returncode == 0, f"{snippet}: {out.stderr}"
    return out.stdout.strip()


def dry_run(arm, stop=STOP, env=None):
    full = dict(os.environ)
    full["CF409_DRY_RUN"] = "1"
    full.update(env or {})
    return subprocess.run(["bash", str(RUN_ARM), arm, str(stop)],
                          capture_output=True, text=True, env=full,
                          cwd=str(REPO_ROOT), timeout=120)


class TestTheStudyIsTheCard:

    def test_the_files_exist(self):
        for path in (STUDY_SH, RUN_ARM, ARMS_TSV, AUC_WATCH):
            assert path.is_file(), f"missing {path}"

    def test_the_cell_is_the_sweep_best_arm(self):
        assert study_value("CF409_CELL") == CELL
        assert study_value("CF409_K") == str(K)
        assert study_value("CF409_REDUCE") == REDUCE
        assert study_value("CF409_ALIGN_TARGET") == ALIGN_TARGET

    def test_one_stop_and_one_head_budget(self):
        assert study_value("CF409_STOPS") == str(STOP)
        assert study_value("CF409_HEAD_STEPS") == "30000"
        assert study_value("CF409_HEAD_SEED") == "20260722"
        assert study_value("CF409_ENC") == "student"

    def test_the_decay_is_one_shape_and_the_table_cannot_hold_a_second(self):
        """The card gives ONE decay: 1.0 falling linearly to 0.0 at step
        10,000. It lives in study.sh, so no row of the arms table can hold a
        floor or a second ramp."""
        assert study_value("CF409_REP_W_START") == "1.0"
        assert study_value("CF409_REP_W_END") == "0.0"
        assert study_value("CF409_REP_W_RAMP") == str(RAMP)
        for line in ARMS_TSV.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            assert len(line.split("\t")) == 2, line

    def test_the_durable_root_is_this_study_only(self):
        root = study_value("CF409_ROOT")
        assert root.startswith("/")
        assert "cf-409" in root
        assert "/tmp/" not in root
        assert str(REPO_ROOT) not in root


class TestTheEmaScheduleDoesNotMove:
    """The card holds the EMA schedule fixed at the sweep's best arm, a
    momentum of 0.9 that rises to 1.0 at step 100,000. That is already the
    runner's own default, so this study passes NO momentum flag and no arm
    can carry a different one."""

    def test_the_runner_default_is_the_sweep_best_schedule(self):
        assert EMA in PARENT_LEG.read_text()

    def test_no_script_of_this_study_replaces_the_runner_schedule(self):
        """`EMA_ARGS` is the one variable that replaces it. #404 swept the
        momentum with it. This card writes it nowhere."""
        offenders = []
        for script in sorted(EXP.glob("scripts/*.sh")):
            for n, line in enumerate(script.read_text().splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                if "EMA_ARGS=" in line:
                    offenders.append(f"{script.name}:{n}: {line.strip()}")
        assert not offenders, offenders

    def test_the_leg_flags_carry_no_momentum(self):
        """GAP_ARGS is appended last, so a momentum there would override the
        runner's own. The dry run prints the whole block."""
        out = dry_run("dec_s20")
        assert out.returncode == 0, out.stderr
        assert "--ema-tau" not in out.stdout

    def test_the_launcher_reads_the_momentum_back_off_the_leg_log(self):
        """A leg trained at another momentum is not this card's arm. The
        wrapper reads the trainer's own command line and stops it."""
        assert study_value("CF409_EMA_SIG") == "0.9 1.0 100000"
        got = study_out(
            f'printf "%s" "python train.py {EMA} --seed 1" '
            f'| cf409_ema_of_cmdline')
        assert got == "0.9 1.0 100000"
        assert "cf409_ema_of_cmdline" in RUN_ARM.read_text()


class TestTheArms:

    def test_one_arm_for_each_backbone_seed(self):
        assert study_value("CF409_ARMS").split() == list(ARMS)
        got = tuple(study_out(f'cf409_seed {a}') for a in ARMS)
        assert got == SEEDS
        assert len(set(got)) == len(got), "two arms share a seed"

    def test_every_arm_is_the_decay_arm(self):
        """The card runs no control. An arm that held the weight would spend
        a card on a number the sweep already published."""
        for arm in ARMS:
            args = study_out('cf409_decay_args')
            assert "--rep-loss-weight-end 0.0" in args, arm
            assert f"--rep-loss-weight-ramp-steps {RAMP}" in args, arm

    def test_the_two_sweep_seeds_are_covered(self):
        """Those two seeds carry the reference scores, 1.1507 and 1.1491, so
        their decay arms pair directly against a measured number."""
        seeds = {study_out(f'cf409_seed {a}') for a in ARMS}
        for seed in SCORED_BY_THE_SWEEP:
            assert seed in seeds

    def test_the_collapsed_seed_is_not_reused(self):
        """20260521 lost the contrastive task once in the sweep, so a
        collapse there could not be read as this card's."""
        seeds = {study_out(f'cf409_seed {a}') for a in ARMS}
        assert "20260521" not in seeds

    def test_an_unknown_arm_is_refused(self):
        out = study_call('cf409_require_arm nosucharm')
        assert out.returncode != 0
        assert "not an arm" in out.stderr

    def test_an_unknown_stop_is_refused(self):
        assert study_call('cf409_require_stop 40000').returncode == 0
        out = study_call('cf409_require_stop 12345')
        assert out.returncode != 0


class TestTheDecayFlags:

    def test_an_arm_passes_all_three(self):
        args = study_out('cf409_decay_args')
        assert args == ("--rep-loss-weight 1.0 --rep-loss-weight-end 0.0 "
                        f"--rep-loss-weight-ramp-steps {RAMP}")

    def test_the_signature_matches_the_command_line_reader(self):
        """The launcher compares the two. A reader that disagrees with the
        table would stop every leg, or pass every wrong one."""
        for arm in ARMS:
            args = study_out('cf409_decay_args')
            sig = study_out('cf409_decay_sig')
            got = study_out(
                f'printf "%s" "python train.py {args} --seed 1" '
                f'| cf409_decay_of_cmdline')
            assert got == sig, arm

    def test_the_shell_weight_matches_the_python_schedule(self):
        """`study.sh` repeats the ramp so a table needs no interpreter. The
        two must not drift."""
        sys.path.insert(0, str(REPO_ROOT))
        from src.models import linear_schedule_at_step
        for arm in ARMS:
            for step in (0, 1, 2_500, 5_000, 9_999, 10_000, 25_000, 40_000):
                want = linear_schedule_at_step(step, STOP, 1.0, 0.0, RAMP)
                got = float(study_out(f'cf409_rep_w_at {step}'))
                assert got == pytest.approx(want, abs=5e-4), (arm, step)

    def test_the_weight_reaches_zero_at_the_ramp_and_holds(self):
        assert float(study_out(f'cf409_rep_w_at {RAMP}')) == 0.0
        assert float(study_out(f'cf409_rep_w_at {STOP}')) == 0.0
        assert float(study_out('cf409_rep_w_at 0')) == 1.0


class TestTheCommentsNameFilesThatExist:
    """A comment that names a file the branch does not carry sends the next
    reader to look for it. The reader then trusts less of the rest."""

    PATH = re.compile(
        r"(?:scripts|notes|tests|docs)/[A-Za-z0-9_.-]+\.(?:sh|py|tsv|md)")
    # A `scripts/` or `notes/` path is this study's. A `tests/` or `docs/` path
    # is the repository's.
    AT_REPO_ROOT = ("tests/", "docs/")

    def test_every_path_a_comment_names_is_on_the_branch(self):
        missing = []
        for script in sorted(EXP.glob("scripts/*.sh")) \
                + sorted(EXP.glob("scripts/*.py")):
            for n, line in enumerate(script.read_text().splitlines(), 1):
                if not line.lstrip().startswith("#"):
                    continue
                for path in self.PATH.findall(line):
                    root = (REPO_ROOT if path.startswith(self.AT_REPO_ROOT)
                            else EXP)
                    if not (root / path).exists():
                        missing.append(f"{script.name}:{n}: {path}")
        assert not missing, missing


class TestNoTwoArmsShareAFile:

    def test_the_run_name_carries_the_study_and_the_arm(self):
        names = {a: study_out(f'cf409_run_name {a}') for a in ARMS}
        assert len(set(names.values())) == len(ARMS)
        for arm, name in names.items():
            assert name.endswith(f"_cf409_{arm}")
            assert CELL in name and f"cf373k{K}" in name

    def test_the_checkpoint_root_carries_the_arm(self):
        roots = {study_out(f'cf409_arm_root {a}') for a in ARMS}
        assert len(roots) == len(ARMS)

    def test_the_leg_log_carries_the_arm(self):
        logs = {study_out(f'cf409_leg_log {a}') for a in ARMS}
        assert len(logs) == len(ARMS)

    def test_no_name_can_be_read_as_a_published_one(self):
        """#373 and #404 publish numbers on this cell. A shared path would
        overwrite one of them. #404's own suffix is `_mean_<arm>`."""
        for arm in ARMS:
            name = study_out(f'cf409_run_name {arm}')
            assert "cf409" in name
            assert "_mean_" not in name

    def test_a_trial_writes_nowhere_the_study_writes(self):
        trial = {"CF409_TRIAL": "400"}
        assert study_value("CF409_ROOT", trial) != study_value("CF409_ROOT")
        assert study_value("CF409_RESULTS", trial) != study_value("CF409_RESULTS")
        assert study_value("CF409_ROOT", trial).endswith("-trial")

    def test_a_trial_still_crosses_its_whole_decay(self):
        """A 400-step trial on a 10,000-step ramp would hold the weight at
        0.975 and never reach zero, so the trial would check nothing."""
        trial = {"CF409_TRIAL": "400"}
        assert float(study_out('cf409_rep_w_at 400', trial)) == 0.0


class TestTheLauncher:

    def test_it_reuses_the_parent_runner(self):
        assert PARENT_LEG.is_file()
        body = RUN_ARM.read_text()
        assert "run_leg_k.sh" in body
        # One trainer invocation exists in this study, and it is the parent's.
        assert "train.py" not in body

    def test_the_dry_run_names_the_decay_and_the_seed(self):
        out = dry_run("dec_s20")
        assert out.returncode == 0, out.stderr
        assert "--rep-loss-weight-end 0.0" in out.stdout
        assert f"--rep-loss-weight-ramp-steps {RAMP}" in out.stdout
        assert "seed=20260520" in out.stdout

    def test_the_dry_run_names_the_cell_of_the_sweep_best_arm(self):
        out = dry_run("dec_s24")
        assert out.returncode == 0, out.stderr
        assert f"cell={CELL}" in out.stdout
        assert f"k={K}" in out.stdout
        assert f"reduce={REDUCE}" in out.stdout
        assert f"align_target={ALIGN_TARGET}" in out.stdout
        assert "seed=20260524" in out.stdout

    def test_an_unknown_arm_is_refused(self):
        assert dry_run("nosucharm").returncode != 0

    def test_an_unknown_stop_is_refused(self):
        assert dry_run("dec_s20", stop=12345).returncode != 0


class TestAucWatch:

    def _csv(self, tmp_path, aucs, name="run_losses.csv"):
        path = tmp_path / name
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "loss", "auc", "rep_w"])
            for i, a in enumerate(aucs, start=1):
                w.writerow([i, 1.0, a, 1.0])
        return path

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(AUC_WATCH), *[str(a) for a in args]],
            capture_output=True, text=True, timeout=120)

    def test_a_healthy_run_holds(self, tmp_path):
        path = self._csv(tmp_path, [0.97] * 200)
        out = self._run(path, "--window", "10")
        assert out.returncode == 0, out.stdout
        assert "held" in out.stdout

    def test_a_collapsed_run_names_the_step(self, tmp_path):
        path = self._csv(tmp_path, [0.97] * 100 + [0.50] * 100)
        out = self._run(path, "--window", "10")
        assert out.returncode == 1
        fields = out.stdout.strip().split("\t")
        assert fields[1] == "lost"
        # The median crosses one half-window after the fall starts.
        assert 100 <= int(fields[2]) <= 110

    def test_a_dip_that_recovers_is_not_a_loss(self, tmp_path):
        path = self._csv(tmp_path, [0.97] * 50 + [0.40] * 20 + [0.96] * 100)
        out = self._run(path, "--window", "5")
        assert out.returncode == 0
        assert "held" in out.stdout

    def test_a_missing_auc_column_is_an_error_not_a_verdict(self, tmp_path):
        path = tmp_path / "bad_losses.csv"
        path.write_text("step,loss\n1,2.0\n")
        out = self._run(path)
        assert out.returncode == 2
        assert "error" in out.stdout

    def test_many_runs_read_in_one_call(self, tmp_path):
        good = self._csv(tmp_path, [0.97] * 60, "a_losses.csv")
        bad = self._csv(tmp_path, [0.97] * 20 + [0.4] * 40, "b_losses.csv")
        out = self._run(good, bad, "--window", "5", "--tsv")
        assert out.returncode == 1
        lines = out.stdout.strip().splitlines()
        assert lines[0].startswith("run\t")
        assert len(lines) == 3

    def test_a_nan_hides_no_collapse(self, tmp_path):
        """`float("nan")` succeeds, and `statistics.median` of a list that
        holds a NaN gives an arbitrary element. A NaN median is under no
        threshold, so one NaN row would read a dead arm as healthy."""
        path = self._csv(tmp_path, [0.97] * 20 + [0.40] * 10 + ["nan"] * 2)
        out = self._run(path, "--window", "3")
        assert out.returncode == 1, out.stdout
        assert out.stdout.split("\t")[1] == "lost"

    def test_no_nan_reaches_the_table(self, tmp_path):
        """The report reads this table. `nan` in the floor column is not a
        measurement."""
        path = self._csv(tmp_path, ["nan"] + [0.97] * 30)
        out = self._run(path, "--window", "3")
        assert out.returncode == 0, out.stdout
        assert "nan" not in out.stdout.lower()

    def test_skip_rows_drops_the_rows_of_the_leg_before(self, tmp_path):
        """A re-fired leg appends to the CSV of the leg that crashed, so the
        gate must read the rows above the ones already there."""
        path = self._csv(tmp_path, [0.40] * 50 + [0.97] * 50)
        whole = self._run(path, "--window", "10").stdout.split("\t")
        after = self._run(path, "--window", "10",
                          "--skip-rows", "50").stdout.split("\t")
        assert float(whole[3]) < 0.55
        assert float(after[3]) > 0.90

    def test_a_collapse_in_the_skipped_rows_is_not_this_legs(self, tmp_path):
        path = self._csv(tmp_path, [0.40] * 50 + [0.97] * 20)
        assert self._run(path, "--window", "60").returncode == 1
        out = self._run(path, "--window", "60", "--skip-rows", "50")
        assert out.returncode == 0, out.stdout

    def test_a_skip_past_the_run_is_not_a_verdict(self, tmp_path):
        """This leg has written no row yet. That is a wait, not a verdict."""
        path = self._csv(tmp_path, [0.40] * 20)
        assert self._run(path, "--skip-rows", "20").returncode == 2
