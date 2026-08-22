"""Tests for #409's study scaffolding: the arms, the launcher and the AUC watch.

#409 runs ONE configuration — #373's cell A4, `arm6_v2_combab_alignS` at
k = 3 under the default `sum` reduction — at four L_rep weight floors. It adds
two trainer flags, `--rep-loss-weight-end` and `--rep-loss-weight-ramp-steps`,
and no new pipeline. It reuses #373's runner and supplies the decay, the seed
and the L_align target. `tests/test_409_rep_weight_decay.py` holds the
objective itself.

That is the contract these tests hold:

  * the study's constants are the card's: cell A4, k = 3, the `sum`
    reduction, one stop at 40,000 backbone steps, a 30,000-step head, and
    the student encoder.
  * the arms table holds eight arms, the floors 1.0 / 0.5 / 0.2 / 0.0, and a
    repeat seed on three of them.
  * every arm's decay flags follow `src.models.linear_schedule_at_step`, so
    the shell table and the trainer agree on the weight at every step.
  * the two control arms pass NO decay flag, because the trainer reads "no
    end value" as "the weight is constant".
  * no two arms share a file.
  * the runner is #373's `run_leg_k.sh` — no second trainer invocation
    exists in this study.
  * the AUC watch names the step a run lost the contrastive task.
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
EXP = REPO_ROOT / "reports" / "2026-08-22_rep_weight_decay"
PARENT = REPO_ROOT / "reports" / "2026-08-08_rollout_depth"

STUDY_SH = EXP / "scripts" / "study.sh"
RUN_ARM = EXP / "scripts" / "run_arm.sh"
ARMS_TSV = EXP / "scripts" / "arms.tsv"
AUC_WATCH = EXP / "scripts" / "auc_watch.py"
PARENT_LEG = PARENT / "scripts" / "run_leg_k.sh"

CELL = "arm6_v2_combab_alignS"
K = 3
REDUCE = "sum"
STOP = 40_000
RAMP = 10_000
# Every arm of the card, in the order it runs them.
ARMS = ("ctrl_s20", "ctrl_s24", "dec0_s20", "dec0_s24",
        "flr05_s20", "flr05_s24", "flr02_s20", "dec0T_s20")
# The floors the card walks. 1.0 is the control, which passes no decay flag.
FLOORS = ("-", "-", "0.0", "0.0", "0.5", "0.5", "0.2", "0.0")


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

    def test_the_cell_is_the_project_best(self):
        assert study_value("CF409_CELL") == CELL
        assert study_value("CF409_K") == str(K)
        assert study_value("CF409_REDUCE") == REDUCE

    def test_one_stop_and_one_head_budget(self):
        assert study_value("CF409_STOPS") == str(STOP)
        assert study_value("CF409_HEAD_STEPS") == "30000"
        assert study_value("CF409_HEAD_SEED") == "20260722"
        assert study_value("CF409_ENC") == "student"

    def test_the_weight_starts_at_one(self):
        """Every arm starts on the published objective and walks down."""
        assert study_value("CF409_REP_W_START") == "1.0"

    def test_the_durable_root_is_this_study_only(self):
        root = study_value("CF409_ROOT")
        assert root.startswith("/")
        assert "cf-409" in root
        assert "/tmp/" not in root
        assert str(REPO_ROOT) not in root


class TestTheArms:

    def test_eight_arms_in_the_card_order(self):
        assert study_value("CF409_ARMS").split() == list(ARMS)

    def test_the_floors_walk_one_to_zero(self):
        got = tuple(study_out(f'cf409_rep_end {a}') for a in ARMS)
        assert got == FLOORS

    def test_every_decaying_arm_shares_the_ramp(self):
        for arm, floor in zip(ARMS, FLOORS):
            want = "0" if floor == "-" else str(RAMP)
            assert study_out(f'cf409_ramp {arm}') == want

    def test_three_arms_repeat_at_a_second_seed(self):
        seeds = {a: study_out(f'cf409_seed {a}') for a in ARMS}
        assert sorted(set(seeds.values())) == ["20260520", "20260524"]
        assert sum(1 for s in seeds.values() if s == "20260524") == 3
        # A repeat differs from its partner in the seed alone.
        for a, b in (("ctrl_s20", "ctrl_s24"), ("dec0_s20", "dec0_s24"),
                     ("flr05_s20", "flr05_s24")):
            assert study_out(f'cf409_rep_end {a}') == study_out(f'cf409_rep_end {b}')
            assert seeds[a] != seeds[b]

    def test_the_collapsed_seed_is_not_reused(self):
        """20260521 lost the contrastive task once at full weight, so a
        collapse there could not be read as this card's."""
        seeds = {study_out(f'cf409_seed {a}') for a in ARMS}
        assert "20260521" not in seeds

    def test_one_arm_aligns_on_the_teacher(self):
        targets = {a: study_out(f'cf409_align_target {a}') for a in ARMS}
        assert targets["dec0T_s20"] == "teacher"
        assert [a for a, t in targets.items() if t == "teacher"] == ["dec0T_s20"]
        assert targets["dec0_s20"] == "student"

    def test_an_unknown_arm_is_refused(self):
        out = study_call('cf409_require_arm nosucharm')
        assert out.returncode != 0
        assert "not an arm" in out.stderr

    def test_an_unknown_stop_is_refused(self):
        assert study_call('cf409_require_stop 40000').returncode == 0
        out = study_call('cf409_require_stop 12345')
        assert out.returncode != 0


class TestTheDecayFlags:

    def test_a_control_passes_no_end_value(self):
        """train.py reads "no end value" as "the weight is constant", and no
        value of the flag means the same. A control that passed one would be
        a treated arm."""
        for arm in ("ctrl_s20", "ctrl_s24"):
            args = study_out(f'cf409_decay_args {arm}')
            assert args == "--rep-loss-weight 1.0"
            assert "--rep-loss-weight-end" not in args

    def test_a_treated_arm_passes_all_three(self):
        args = study_out('cf409_decay_args dec0_s20')
        assert args == ("--rep-loss-weight 1.0 --rep-loss-weight-end 0.0 "
                        f"--rep-loss-weight-ramp-steps {RAMP}")

    def test_the_signature_matches_the_command_line_reader(self):
        """The launcher compares the two. A reader that disagrees with the
        table would stop every leg, or pass every wrong one."""
        for arm in ARMS:
            args = study_out(f'cf409_decay_args {arm}')
            sig = study_out(f'cf409_decay_sig {arm}')
            got = study_out(
                f'printf "%s" "python train.py {args} --seed 1" '
                f'| cf409_decay_of_cmdline')
            assert got == sig, arm

    def test_the_shell_weight_matches_the_python_schedule(self):
        """`study.sh` repeats the ramp so a table needs no interpreter. The
        two must not drift."""
        sys.path.insert(0, str(REPO_ROOT))
        from src.models import linear_schedule_at_step
        for arm, floor in zip(ARMS, FLOORS):
            end = None if floor == "-" else float(floor)
            ramp = None if floor == "-" else RAMP
            for step in (0, 1, 2_500, 5_000, 9_999, 10_000, 25_000, 40_000):
                want = linear_schedule_at_step(step, STOP, 1.0, end, ramp)
                got = float(study_out(f'cf409_rep_w_at {arm} {step}'))
                assert got == pytest.approx(want, abs=5e-4), (arm, step)

    def test_the_weight_reaches_the_floor_at_the_ramp(self):
        assert float(study_out(f'cf409_rep_w_at dec0_s20 {RAMP}')) == 0.0
        assert float(study_out(f'cf409_rep_w_at flr05_s20 {RAMP}')) == 0.5
        assert float(study_out(f'cf409_rep_w_at ctrl_s20 {RAMP}')) == 1.0


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
        overwrite one of them."""
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
        0.975 and never reach the floor, so the trial would check nothing."""
        trial = {"CF409_TRIAL": "400"}
        assert float(study_out('cf409_rep_w_at dec0_s20 400', trial)) == 0.0


class TestTheLauncher:

    def test_it_reuses_the_parent_runner(self):
        assert PARENT_LEG.is_file()
        body = RUN_ARM.read_text()
        assert "run_leg_k.sh" in body
        # One trainer invocation exists in this study, and it is the parent's.
        assert "train.py" not in body

    def test_the_dry_run_names_the_decay_and_the_seed(self):
        out = dry_run("dec0_s20")
        assert out.returncode == 0, out.stderr
        assert "--rep-loss-weight-end 0.0" in out.stdout
        assert f"--rep-loss-weight-ramp-steps {RAMP}" in out.stdout
        assert "seed=20260520" in out.stdout
        assert "align_target=student" in out.stdout

    def test_the_dry_run_of_a_control_names_no_end_value(self):
        out = dry_run("ctrl_s20")
        assert out.returncode == 0, out.stderr
        assert "--rep-loss-weight-end" not in out.stdout

    def test_the_teacher_arm_overrides_the_cell_target(self):
        """The cell states `--align-target student`. GAP_ARGS is appended
        last, and argparse keeps the last value."""
        out = dry_run("dec0T_s20")
        assert out.returncode == 0, out.stderr
        assert "align_target=teacher" in out.stdout

    def test_an_unknown_arm_is_refused(self):
        assert dry_run("nosucharm").returncode != 0

    def test_an_unknown_stop_is_refused(self):
        assert dry_run("dec0_s20", stop=12345).returncode != 0


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
