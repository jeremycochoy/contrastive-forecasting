"""Tests for #409's study scaffolding: the arms, the launcher and the AUC watch.

#409 decays the L_rep weight to zero. The cell is #373's
`arm6_v2_combab_alignT` at k = 32 under the `mean` reduction, with L_align on
the EMA teacher. The decay is one extra factor in front of L_rep. It starts at
1.0 and falls linearly to 0.0 at the arm's RAMP.

The card gives a budget of eight backbones, and the search spends them one
round at a time. Each round reads the scores of the round before it, so
`scripts/arms.tsv` is a CATALOGUE of candidates and not a queue. Some rows
never ran. `notes/search_protocol.md` holds the rule.

Two axes carry the rounds. The EMA schedule is one: the align target is the
teacher, so the schedule still acts after the ramp, when L_align is the whole
main loss. The decay ramp is the other. A schedule can also carry a repeat
seed, because a repeat gives the error bar of a headline number.

The card runs NO control. The sweep already measured this cell with no decay,
and `reports/2026-08-19_ema_momentum_k32/ema_momentum_k32.md` holds a score for
seven of the eight schedules.

It adds two trainer flags, `--rep-loss-weight-end` and
`--rep-loss-weight-ramp-steps`, and no new pipeline. It reuses #373's runner
and supplies the schedule, the decay and the seed.
`tests/test_409_rep_weight_decay.py` holds the objective itself.

That is the contract these tests hold:

  * the study's constants are the card's: cell `arm6_v2_combab_alignT`,
    k = 32, the `mean` reduction, the teacher target, one stop at 40,000
    backbone steps, a 30,000-step head, and the student encoder.
  * the arms table is a catalogue, and every row of it is unique by the EMA
    schedule, the decay ramp and the seed.
  * the decay RAMP is a column of that table, so a reader reproduces an arm
    from its row alone. `CF409_REP_W_RAMP` stays an override for a dry run.
  * every arm's decay flags follow `src.models.linear_schedule_at_step`, and
    every arm's momentum follows `src.models.ema_tau_at_step`, so the shell
    table and the trainer agree at every step.
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
# The card's own ramp, which every row takes unless its column says otherwise.
RAMP = 10_000
# The runner's own default schedule, which is the sweep's best arm. It is arm
# 1 of this card, and every other arm replaces it.
EMA = "--ema-tau 0.9 --ema-tau-end 1.0 --ema-tau-ramp-steps 100000"
# Every ROW of the catalogue, in the order `launch.sh` deals them, as
# `(arm, tau, end, ema ramp, decay ramp, seed)`. `-` is a flag the arm does
# NOT pass: train.py reads "no end value" as a fixed momentum.
#
# The three values after the arm name are the EMA schedule. The fourth is the
# decay ramp, and `results/scores.csv` records it for every scored arm.
ARM_ROWS = (
    ("dec_s20",         "0.9",  "1.0", "100000", "10000", "20260520"),
    ("dec_s22",         "0.9",  "1.0", "100000", "10000", "20260522"),
    ("dec_s23",         "0.9",  "1.0", "100000", "10000", "20260523"),
    ("dec_s24",         "0.9",  "1.0", "100000", "10000", "20260524"),
    ("dec_s25",         "0.9",  "1.0", "100000", "10000", "20260525"),
    ("dec_m090_fix",    "0.9",  "-",   "-",      "10000", "20260520"),
    ("dec_m090_r60",    "0.9",  "1.0", "60000",  "10000", "20260520"),
    ("dec_m095_fix",    "0.95", "-",   "-",      "10000", "20260520"),
    ("dec_m099_fix",    "0.99", "-",   "-",      "10000", "20260520"),
    ("dec_m090_r200",   "0.9",  "1.0", "200000", "10000", "20260520"),
    ("dec_m080_r200",   "0.8",  "1.0", "200000", "10000", "20260520"),
    ("dec_m095_r100",   "0.95", "1.0", "100000", "10000", "20260520"),
    ("dec_m070_fix",    "0.7",  "-",   "-",      "10000", "20260520"),
    ("dec_m050_fix",    "0.5",  "-",   "-",      "10000", "20260520"),
    ("dec_ramp30k_m080", "0.8", "1.0", "200000", "30000", "20260520"),
    ("dec_ramp20k_m080", "0.8", "1.0", "200000", "20000", "20260520"),
    ("dec_ramp5k_m080",  "0.8", "1.0", "200000", "5000",  "20260520"),
    ("dec_m080_r200_s24", "0.8", "1.0", "200000", "10000", "20260524"),
    ("dec_m090r100_ramp5k", "0.9", "1.0", "100000", "5000", "20260520"),
    ("dec_m090r100_ramp2k", "0.9", "1.0", "100000", "2000", "20260520"),
    ("dec_m090r100_ramp1k", "0.9", "1.0", "100000", "1000", "20260520"),
)
ARMS = tuple(r[0] for r in ARM_ROWS)
SEEDS = tuple(r[5] for r in ARM_ROWS)
SCHEDULES = {r[0]: (r[1], r[2], r[3]) for r in ARM_ROWS}
RAMPS = {r[0]: r[4] for r in ARM_ROWS}
SEED_OF = {r[0]: r[5] for r in ARM_ROWS}
# The card's own seed. A row that moves it says so in its name.
SEED = "20260520"
# The schedule this card repeated most, at five seeds.
ARM_ONE = ("dec_s20", "dec_s22", "dec_s23", "dec_s24", "dec_s25")
# The momentum each arm holds at the 40,000-step stop. That value ranks the
# arms, and no two SCHEDULES share it.
REACHED = {
    "dec_s20": 0.940, "dec_s22": 0.940, "dec_s23": 0.940,
    "dec_s24": 0.940, "dec_s25": 0.940,
    "dec_m090_fix": 0.900, "dec_m090_r60": 0.967, "dec_m095_fix": 0.950,
    "dec_m099_fix": 0.990, "dec_m090_r200": 0.920, "dec_m080_r200": 0.840,
    "dec_m095_r100": 0.970, "dec_m070_fix": 0.700, "dec_m050_fix": 0.500,
    "dec_ramp30k_m080": 0.840, "dec_ramp20k_m080": 0.840,
    "dec_ramp5k_m080": 0.840, "dec_m080_r200_s24": 0.840,
    "dec_m090r100_ramp5k": 0.940, "dec_m090r100_ramp2k": 0.940,
    "dec_m090r100_ramp1k": 0.940,
}
# One schedule per distinct `(tau, end, ramp)`. Ten of them over the rows.
SCHEDULE_COUNT = 10


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

    def test_the_environment_can_carry_an_arm_past_the_stop(self):
        """A scored arm can run on to a second stop. `CF409_STOPS` was a plain
        assignment, so study.sh dropped the value the environment gave it and
        no script ever saw the second stop."""
        assert study_value("CF409_STOPS",
                           {"CF409_STOPS": "40000 80000"}) == "40000 80000"

    def test_the_decay_ramp_is_a_column_of_the_arms_table(self):
        """The weight falls from 1.0 to 0.0, and those two ends are the
        card's. The RAMP is the arm's, so it is column 5 of the table. Four
        rows hold one schedule and one seed and differ in the ramp alone, so
        a table without that column cannot tell them apart."""
        assert study_value("CF409_REP_W_START") == "1.0"
        assert study_value("CF409_REP_W_END") == "0.0"
        for line in ARMS_TSV.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            assert len(line.split("\t")) == 6, line
        for arm in ARMS:
            assert study_out(f"cf409_decay_ramp_of {arm}") == RAMPS[arm], arm

    def test_the_ramp_column_is_the_ramp_each_arm_ran(self):
        """`results/scores.csv` records the ramp of every scored arm. A row
        that disagrees with it names a run that never happened."""
        path = EXP / "results" / "scores.csv"
        rows = list(csv.DictReader(path.read_text().splitlines()))
        assert rows
        for row in rows:
            assert row["ramp"] == RAMPS[row["arm"]], row["arm"]

    def test_the_environment_ramp_overrides_the_row(self):
        """`CF409_REP_W_RAMP` is the override a dry run of an unlisted ramp
        needs. It moves the LEG, and it never moves the table: a stray value
        in a lane's environment would otherwise rewrite the ramp column of
        every arm in `results/scores.csv`."""
        env = {"CF409_REP_W_RAMP": "7500"}
        assert study_out("cf409_ramp dec_ramp5k_m080", env) == "7500"
        assert study_out("cf409_decay_ramp_of dec_ramp5k_m080", env) == "5000"

    def test_the_durable_root_is_this_study_only(self):
        root = study_value("CF409_ROOT")
        assert root.startswith("/")
        assert "cf-409" in root
        assert "/tmp/" not in root
        assert str(REPO_ROOT) not in root


class TestTheEmaScheduleIsTheAxis:
    """The card's own "The arms" section makes the EMA schedule the axis. The
    align target is the teacher, so the schedule keeps acting after step
    10,000, when the decay has taken L_rep out and L_align is the whole main
    loss.

    `EMA_ARGS` REPLACES the runner's three flags. It cannot append: a fixed
    arm passes `--ema-tau` alone, and no repeated flag can remove
    `--ema-tau-end`."""

    def test_the_runner_takes_a_replacement_schedule(self):
        assert "EMA_ARGS" in PARENT_LEG.read_text()
        assert EMA in PARENT_LEG.read_text()

    def test_the_table_holds_ten_schedules(self):
        got = {study_out(f'cf409_ema_sig {a}') for a in ARMS}
        assert len(got) == SCHEDULE_COUNT, sorted(got)

    def test_a_ramp_arm_passes_all_three_flags(self):
        assert study_out('cf409_ema_args dec_m090_r60') == (
            "--ema-tau 0.9 --ema-tau-end 1.0 --ema-tau-ramp-steps 60000")

    def test_a_fixed_arm_passes_the_momentum_alone(self):
        """train.py reads "no end value" as a constant momentum, and no value
        of `--ema-tau-end` means the same."""
        for arm in ("dec_m090_fix", "dec_m095_fix", "dec_m099_fix"):
            args = study_out(f'cf409_ema_args {arm}')
            assert "--ema-tau-end" not in args, arm
            assert "--ema-tau-ramp-steps" not in args, arm
            assert args.startswith("--ema-tau ")

    def test_arm_one_is_the_schedule_already_run(self):
        for arm in ARM_ONE:
            assert study_out(f'cf409_ema_args {arm}') == EMA

    def test_the_signature_matches_the_command_line_reader(self):
        """The launcher compares the two. A reader that disagrees with the
        table would stop every leg, or pass every wrong one."""
        for arm in ARMS:
            args = study_out(f'cf409_ema_args {arm}')
            sig = study_out(f'cf409_ema_sig {arm}')
            got = study_out(
                f'printf "%s" "python train.py {args} --seed 1" '
                f'| cf409_ema_of_cmdline')
            assert got == sig, arm
            assert sig == " ".join(SCHEDULES[arm]), arm

    def test_the_shell_momentum_matches_the_python_schedule(self):
        """`study.sh` repeats the ramp so a table needs no interpreter. The
        two must not drift."""
        sys.path.insert(0, str(REPO_ROOT))
        from src.models import ema_tau_at_step
        for arm, (tau, end, ramp) in SCHEDULES.items():
            for step in (0, 1_000, 20_000, 40_000, 100_000):
                want = ema_tau_at_step(
                    step, STOP, float(tau),
                    None if end == "-" else float(end),
                    None if ramp == "-" else int(ramp))
                got = float(study_out(f'cf409_momentum_at {arm} {step}'))
                assert got == pytest.approx(want, abs=5e-4), (arm, step)

    def test_the_momentum_at_the_stop_identifies_the_schedule(self):
        """That value ranks the arms on the EMA axis. Rows that share it
        share a schedule and move the decay ramp or the seed instead."""
        reached = {}
        for arm in ARMS:
            reached.setdefault(
                float(study_out(f'cf409_momentum_at {arm} {STOP}')), []
            ).append(arm)
            assert float(study_out(f'cf409_momentum_at {arm} {STOP}')) == \
                pytest.approx(REACHED[arm], abs=5e-4), arm
        # Rows on one schedule share one value, so the count is the count of
        # schedules and not the count of rows.
        assert len(reached) == SCHEDULE_COUNT, reached

    def test_the_leg_flags_carry_the_arm_schedule(self):
        """The dry run prints the whole block."""
        out = dry_run("dec_m095_fix")
        assert out.returncode == 0, out.stderr
        assert "ema=--ema-tau 0.95" in out.stdout
        assert "--ema-tau-end" not in out.stdout

    def test_the_schedule_never_rides_the_appended_block(self):
        """GAP_ARGS is appended last. A momentum there would override the
        arm's own, and could not unset a flag."""
        out = dry_run("dec_m090_fix")
        assert out.returncode == 0, out.stderr
        decay = [ln for ln in out.stdout.splitlines()
                 if ln.strip().startswith("decay=")]
        assert decay and "--ema-tau" not in decay[0]

    def test_the_launcher_reads_the_momentum_back_off_the_leg_log(self):
        """A leg trained at another momentum is not this arm. The wrapper
        reads the trainer's own command line and stops it."""
        got = study_out(
            f'printf "%s" "python train.py {EMA} --seed 1" '
            f'| cf409_ema_of_cmdline')
        assert got == "0.9 1.0 100000"
        body = RUN_ARM.read_text()
        assert "cf409_ema_of_cmdline" in body
        assert "cf409_ema_sig" in body
        assert "EMA_ARGS=" in body


class TestTheArms:

    def test_every_row_is_unique_by_schedule_ramp_and_seed(self):
        """`arms.tsv` is a CATALOGUE. Each round of the search picks its arms
        from it, so a row is a candidate and not a run. Those three values
        ARE the arm: two rows that share all three are one arm under two
        names, and no reader could say which row a score belongs to."""
        assert study_value("CF409_ARMS").split() == list(ARMS)
        keys = []
        for arm in ARMS:
            key = (study_out(f'cf409_ema_sig {arm}'),
                   study_out(f'cf409_decay_ramp_of {arm}'),
                   study_out(f'cf409_seed {arm}'))
            assert key == (" ".join(SCHEDULES[arm]), RAMPS[arm],
                           SEED_OF[arm]), arm
            keys.append(key)
        assert len(set(keys)) == len(ARMS), \
            [k for k in keys if keys.count(k) > 1]

    def test_the_catalogue_holds_rows_that_never_ran(self):
        """A round picks its arms from the catalogue, so a row is a
        candidate. Every scored arm is a row, and rows are left over."""
        path = EXP / "results" / "scores.csv"
        scored = {r["arm"] for r in csv.DictReader(path.read_text()
                                                   .splitlines())}
        assert scored
        for arm in scored:
            assert arm in ARMS, arm
        assert scored < set(ARMS)

    def test_the_ramp_axis_moves_the_ramp_and_nothing_else(self):
        """Four rows hold ONE schedule and ONE seed. They are the ramp axis,
        so the ramp column is the only thing that separates them, and the
        table alone tells them apart."""
        axis = ("dec_ramp5k_m080", "dec_m080_r200", "dec_ramp20k_m080",
                "dec_ramp30k_m080")
        assert len({study_out(f'cf409_ema_sig {a}') for a in axis}) == 1
        assert len({study_out(f'cf409_seed {a}') for a in axis}) == 1
        ramps = [study_out(f'cf409_decay_ramp_of {a}') for a in axis]
        assert ramps == ["5000", "10000", "20000", "30000"]
        assert len({study_out(f'cf409_run_name {a}') for a in axis}) == 4

    def test_a_repeat_seed_moves_the_seed_and_nothing_else(self):
        """A repeat gives the error bar of a headline number, so it must
        differ from its arm in the seed alone."""
        pair = ("dec_m080_r200", "dec_m080_r200_s24")
        assert len({study_out(f'cf409_ema_sig {a}') for a in pair}) == 1
        assert len({study_out(f'cf409_decay_ramp_of {a}') for a in pair}) == 1
        assert len({study_out(f'cf409_seed {a}') for a in pair}) == 2

    def test_a_row_that_moves_the_seed_says_so_in_its_name(self):
        """A repeat seed is a row of its own, so two rows of one schedule can
        differ in the seed. The name carries it, or a reader would take two
        seeds of one arm for two treatments."""
        for arm in ARMS:
            seed = study_out(f'cf409_seed {arm}')
            if seed == SEED:
                continue
            assert arm.endswith("_s" + seed[-2:]) or arm == "dec_s" + seed[-2:], \
                (arm, seed)

    def test_arm_one_carries_the_seed_spread(self):
        """One schedule at five seeds. Three of them reached 40,000 steps and
        hold the only run-to-run spread this card measured."""
        seeds = {study_out(f'cf409_seed {a}') for a in ARM_ONE}
        assert seeds == {"20260520", "20260522", "20260523",
                         "20260524", "20260525"}
        sigs = {study_out(f'cf409_ema_sig {a}') for a in ARM_ONE}
        assert len(sigs) == 1

    def test_every_arm_is_the_decay_arm(self):
        """Every arm decays the weight to 0.0 at its own ramp. An arm that
        held the weight would spend a backbone on a number the sweep
        published."""
        for arm in ARMS:
            args = study_out(f'cf409_decay_args {arm}')
            assert "--rep-loss-weight-end 0.0" in args, arm
            assert f"--rep-loss-weight-ramp-steps {RAMPS[arm]}" in args, arm

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
        assert study_out('cf409_decay_args dec_s20') == (
            "--rep-loss-weight 1.0 --rep-loss-weight-end 0.0 "
            f"--rep-loss-weight-ramp-steps {RAMP}")

    def test_the_ramp_the_row_holds_is_the_ramp_the_trainer_gets(self):
        """Three rows move the ramp. The flag must move with them, or the
        four rows of the ramp axis would train one arm four times."""
        for arm in ("dec_ramp5k_m080", "dec_ramp20k_m080",
                    "dec_ramp30k_m080"):
            assert study_out(f'cf409_decay_args {arm}').endswith(
                f"--rep-loss-weight-ramp-steps {RAMPS[arm]}"), arm

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
            assert sig == f"1.0 0.0 {RAMPS[arm]}", arm

    def test_the_shell_weight_matches_the_python_schedule(self):
        """`study.sh` repeats the ramp so a table needs no interpreter. The
        two must not drift."""
        sys.path.insert(0, str(REPO_ROOT))
        from src.models import linear_schedule_at_step
        for arm in ("dec_s20", "dec_ramp5k_m080", "dec_ramp30k_m080"):
            ramp = int(RAMPS[arm])
            for step in (0, 1, 2_500, 5_000, 9_999, 10_000, 25_000, 40_000):
                want = linear_schedule_at_step(step, STOP, 1.0, 0.0, ramp)
                got = float(study_out(f'cf409_rep_w_at {arm} {step}'))
                assert got == pytest.approx(want, abs=5e-4), (arm, step)

    def test_the_weight_reaches_zero_at_the_ramp_and_holds(self):
        for arm in ARMS:
            ramp = RAMPS[arm]
            assert float(study_out(f'cf409_rep_w_at {arm} {ramp}')) == 0.0, arm
            assert float(study_out(f'cf409_rep_w_at {arm} {STOP}')) == 0.0, arm
            assert float(study_out(f'cf409_rep_w_at {arm} 0')) == 1.0, arm


class TestTheCommentsNameFilesThatExist:
    """A comment that names a file the branch does not carry sends the next
    reader to look for it. The reader then trusts less of the rest."""

    PATH = re.compile(
        r"(?:scripts|notes|tests|docs)/[A-Za-z0-9_.-]+\.(?:sh|py|tsv|md)")
    # A `notes/` path is this study's. A `tests/` or `docs/` path is the
    # repository's. A `scripts/` path is EITHER: the study holds one, and so
    # does the repository — `scripts/hub_gate.sh` is shared by every study.
    AT_REPO_ROOT = ("tests/", "docs/")
    AT_EITHER_ROOT = ("scripts/",)

    def test_every_path_a_comment_names_is_on_the_branch(self):
        missing = []
        for script in sorted(EXP.glob("scripts/*.sh")) \
                + sorted(EXP.glob("scripts/*.py")):
            for n, line in enumerate(script.read_text().splitlines(), 1):
                if not line.lstrip().startswith("#"):
                    continue
                for path in self.PATH.findall(line):
                    if path.startswith(self.AT_EITHER_ROOT):
                        roots = (EXP, REPO_ROOT)
                    elif path.startswith(self.AT_REPO_ROOT):
                        roots = (REPO_ROOT,)
                    else:
                        roots = (EXP,)
                    if not any((root / path).exists() for root in roots):
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
        for arm in ("dec_s20", "dec_ramp5k_m080", "dec_ramp30k_m080"):
            assert float(study_out(f'cf409_rep_w_at {arm} 400', trial)) == 0.0


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

    def test_the_leg_takes_the_ramp_off_the_arm_row(self):
        """The row is what a reader reproduces the arm from, so the leg
        reads its ramp there and not from the lane's environment."""
        out = dry_run("dec_ramp5k_m080")
        assert out.returncode == 0, out.stderr
        assert "--rep-loss-weight-ramp-steps 5000" in out.stdout

    def test_the_environment_ramp_overrides_the_leg(self):
        """One dry run of a ramp no row carries, before the row is added."""
        out = dry_run("dec_s20", env={"CF409_REP_W_RAMP": "7500"})
        assert out.returncode == 0, out.stderr
        assert "--rep-loss-weight-ramp-steps 7500" in out.stdout

    def test_the_dry_run_names_the_schedule_and_what_it_reaches(self):
        out = dry_run("dec_m090_r200")
        assert out.returncode == 0, out.stderr
        assert ("ema=--ema-tau 0.9 --ema-tau-end 1.0 "
                "--ema-tau-ramp-steps 200000") in out.stdout
        assert "0.920" in out.stdout

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
