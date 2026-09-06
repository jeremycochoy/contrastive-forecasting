"""Tests for #409's path from a backbone to a score, and for its AUC gate.

`tests/test_409_rep_weight_decay.py` holds the objective.
`tests/test_409_launcher_shape.py` holds the arms and the backbone leg. This
file holds the four things the implementation review asked for.

1. A path from a backbone to a score. `head_eval.sh` trains this card's
   30,000-step student head on one arm's backbone and runs that head's 97
   GIFT-Eval configs. `collect.sh` writes one score for each arm.
2. An entry point for the arms. `launch.sh` deals them over elisa's two
   cards, and `phase1.sh` re-fires a leg that crashed.
3. The AUC watch, wired. `auc_guard.sh` reads the losses CSV of a running leg
   and stops the leg that lost the contrastive task.
4. The loss-by-term formula of `docs/rep_loss_weight_schedule.md`. This card
   runs k = 32 under the `mean` reduction against the EMA teacher, so the
   `l_align` column carries the depth-0 copy alone and the loss closes over a
   residual. `notes/loss_decomposition.md` states that formula.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.loss import contrastive_latent_loss
from src.metrics import rollout_cos_error

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP = REPO_ROOT / "reports" / "2026-08-22_rep_weight_decay"
PARENT = REPO_ROOT / "reports" / "2026-08-08_rollout_depth"
SCRIPTS = EXP / "scripts"

STUDY_SH = SCRIPTS / "study.sh"
HEAD_EVAL = SCRIPTS / "head_eval.sh"
PHASE1 = SCRIPTS / "phase1.sh"
LAUNCH = SCRIPTS / "launch.sh"
COLLECT = SCRIPTS / "collect.sh"
AUC_GUARD = SCRIPTS / "auc_guard.sh"
AUC_WATCH = SCRIPTS / "auc_watch.py"
RUN_ARM = SCRIPTS / "run_arm.sh"
DOC = REPO_ROOT / "docs" / "rep_loss_weight_schedule.md"
DECOMP = EXP / "notes" / "loss_decomposition.md"

# One list of arms, in ONE place. Two copies drifted apart once already: the
# run added `dec_m070_fix` and `dec_m050_fix` to `arms.tsv` mid-run, and this
# file still named ten arms.
from tests.test_409_launcher_shape import ARMS, RAMPS  # noqa: E402

# One round of the search, as it names its arms to the launcher. An odd count,
# so the two lanes cannot take an equal share.
ROUND = ("dec_s20", "dec_m080_r200", "dec_ramp5k_m080", "dec_m070_fix",
         "dec_m099_fix")

STOP = 40_000
HEAD_STEPS = 30_000


def study_out(snippet: str, env=None) -> str:
    """Run one function of study.sh and return its stdout."""
    full = dict(os.environ)
    full.update(env or {})
    out = subprocess.run(
        ["bash", "-c", f'. "{STUDY_SH}" >/dev/null && {snippet}'],
        capture_output=True, text=True, timeout=60, env=full)
    assert out.returncode == 0, f"{snippet}: {out.stderr}"
    return out.stdout.strip()


def study_call(snippet: str, env=None) -> subprocess.CompletedProcess:
    full = dict(os.environ)
    full.update(env or {})
    return subprocess.run(
        ["bash", "-c", f'. "{STUDY_SH}" >/dev/null && {snippet}'],
        capture_output=True, text=True, timeout=60, env=full)


def script_dry_run(script: Path, *args, env=None) -> subprocess.CompletedProcess:
    full = dict(os.environ)
    full["CF409_DRY_RUN"] = "1"
    full.update(env or {})
    return subprocess.run(["bash", str(script), *[str(a) for a in args]],
                          capture_output=True, text=True, env=full,
                          cwd=str(REPO_ROOT), timeout=120)


def losses_csv(path: Path, aucs, first_step=1):
    """A losses CSV in the trainer's shape, with one row per step."""
    rows = ["step,loss,auc,rep_w,l_rep,l_align"]
    for i, auc in enumerate(aucs):
        rows.append(f"{first_step + i},1.0,{auc},1.0,11.5,0.9")
    path.write_text("\n".join(rows) + "\n")
    return path


# --- 1. from a backbone to a score ---------------------------------------


class TestTheHeadAndEvalLeg:
    """The card's first deliverable is a 97-config score for every arm."""

    def test_the_scripts_exist(self):
        for path in (HEAD_EVAL, PHASE1, LAUNCH, COLLECT, AUC_GUARD):
            assert path.is_file(), f"missing {path}"

    def test_the_head_is_the_parent_protocol(self):
        """#373's `head_eval_bb.sh` takes an explicit backbone path. This
        study must reuse it, not carry a second head trainer."""
        assert (PARENT / "scripts" / "head_eval_bb.sh").is_file()
        body = HEAD_EVAL.read_text()
        assert "head_eval_bb.sh" in body
        assert "train_forecasting_head.py" not in body

    def test_the_card_head_budget_and_seed_reach_the_head(self):
        """Both sat in study.sh and no script read them."""
        out = script_dry_run(HEAD_EVAL, "dec_s20", STOP)
        assert out.returncode == 0, out.stderr
        assert f"steps={HEAD_STEPS}" in out.stdout
        assert "seed=20260722" in out.stdout
        assert "enc=student" in out.stdout

    def test_the_dry_run_names_the_tag_the_eval_and_the_score(self):
        out = script_dry_run(HEAD_EVAL, "dec_s20", STOP)
        assert out.returncode == 0, out.stderr
        tag = study_out(f"cf409_tag dec_s20 {STOP} {HEAD_STEPS}")
        assert tag in out.stdout
        assert study_out("cf409_eval_dir dec_s20 " + tag) in out.stdout
        assert study_out(f"cf409_score_file dec_s20 {STOP}") in out.stdout

    def test_another_head_budget_is_refused(self):
        """One budget, or two tags of this study name two protocols."""
        assert script_dry_run(HEAD_EVAL, "dec_s20", STOP, 15000).returncode != 0
        assert script_dry_run(HEAD_EVAL, "nosucharm", STOP).returncode != 0

    def test_the_tag_carries_the_arm_the_stop_and_the_budget(self):
        tag = study_out(f"cf409_tag dec_s20 {STOP} {HEAD_STEPS}")
        assert tag == "dec_s20_bb40k_h30k_student"
        assert study_out("cf409_steps_of 40k") == str(STOP)
        assert study_out("cf409_steps_of 400") == "400"
        assert study_out("cf409_steps_label 40000") == "40k"
        assert study_out("cf409_steps_label 400") == "400"

    def test_no_two_arms_share_a_score_file_or_an_eval_directory(self):
        scores, evals = set(), set()
        for arm in ARMS:
            tag = study_out(f"cf409_tag {arm} {STOP} {HEAD_STEPS}")
            scores.add(study_out(f"cf409_score_file {arm} {STOP}"))
            evals.add(study_out(f"cf409_eval_dir {arm} {tag}"))
        assert len(scores) == len(ARMS)
        assert len(evals) == len(ARMS)

    def test_a_trial_head_scales_with_the_trial_budget(self):
        trial = {"CF409_TRIAL": "400"}
        assert study_out("printf %s \"$CF409_HEAD_STEPS\"", trial) == "200"
        out = script_dry_run(HEAD_EVAL, "dec_s20", 400, env=trial)
        assert out.returncode == 0, out.stderr
        assert "_bb400_h200_student" in out.stdout


class TestCollect:
    """`collect.sh` turns the score files into the card's one table."""

    def _results(self, tmp_path, scored):
        res = tmp_path / "results"
        res.mkdir()
        for arm, value in scored.items():
            tag = study_out(f"cf409_tag {arm} {STOP} {HEAD_STEPS}")
            (res / f"score_{tag}.txt").write_text(value)
        return res

    def _run(self, res, env=None):
        full = dict(os.environ)
        full["CF409_RESULTS"] = str(res)
        # Never the machine's own checkpoint root. `collect.sh` reads the eval
        # log of every (arm, stop), and the real root would put real scores in
        # a temporary table.
        full["CF409_ROOT"] = str(Path(res).parent / "root")
        full.update(env or {})
        return subprocess.run(["bash", str(COLLECT)], capture_output=True,
                              text=True, env=full, cwd=str(REPO_ROOT),
                              timeout=180)

    def _eval_log(self, root, arm, stop, value):
        """The aggregate line #373's eval writes under the arm's eval root."""
        tag = study_out(f"cf409_tag {arm} {stop} {HEAD_STEPS}")
        out = Path(study_out(f"cf409_eval_dir {arm} {tag}",
                             {"CF409_ROOT": str(root)}))
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_local.log").write_text(
            f"Aggregate GM-Relative MASE (97 configs): {value}\n")

    def _by_stop(self, res):
        rows = list(csv.DictReader(
            (res / "scores.csv").read_text().splitlines()))
        return {r["stop"]: r["score"] for r in rows}

    def test_one_row_for_each_scored_arm(self, tmp_path):
        res = self._results(tmp_path, {"dec_s20": "1.1400\n",
                                       "dec_m095_fix": "2.4100\n"})
        out = self._run(res)
        assert out.returncode == 0, out.stderr
        rows = (res / "scores.csv").read_text().strip().splitlines()
        assert rows[0].startswith("arm,")
        assert len(rows) == 3
        body = {r.split(",")[0]: r for r in rows[1:]}
        assert body["dec_s20"].endswith(",1.1400")
        assert body["dec_m095_fix"].endswith(",2.4100")

    def test_the_row_is_keyed_by_the_schedule(self, tmp_path):
        """The axis is the EMA schedule, so the schedule is what identifies a
        row. The seed rides beside it and does not key it."""
        res = self._results(tmp_path, {"dec_m090_r60": "1.2000\n"})
        assert self._run(res).returncode == 0
        head, row = (res / "scores.csv").read_text().strip().splitlines()
        cell = dict(zip(head.split(","), row.split(",")))
        assert cell["ema_tau"] == "0.9"
        assert cell["ema_end"] == "1.0"
        assert cell["ema_ramp"] == "60000"
        assert cell["ema_at_stop"] == "0.967"
        assert cell["seed"] == "20260520"

    def test_two_seeds_of_one_schedule_share_the_key(self, tmp_path):
        """Arm 1 ran at three seeds. Those rows are one schedule's spread, so
        they must carry one key and differ in the seed alone."""
        res = self._results(tmp_path, {"dec_s20": "1.2670\n",
                                       "dec_s22": "1.2593\n"})
        assert self._run(res).returncode == 0
        rows = list(csv.DictReader(
            (res / "scores.csv").read_text().splitlines()))
        keys = {(r["ema_tau"], r["ema_end"], r["ema_ramp"]) for r in rows}
        assert keys == {("0.9", "1.0", "100000")}
        assert {r["seed"] for r in rows} == {"20260520", "20260522"}

    def test_the_row_carries_the_arm_definition(self, tmp_path):
        """Every arm shares one decay. The table states it, so a reader needs
        no second file open."""
        res = self._results(tmp_path, {"dec_s24": "1.2000\n"})
        assert self._run(res).returncode == 0
        head, row = (res / "scores.csv").read_text().strip().splitlines()
        cell = dict(zip(head.split(","), row.split(",")))
        assert cell["rep_end"] == "0.0"
        assert cell["ramp"] == "10000"
        assert cell["seed"] == "20260524"
        assert cell["rep_w_at_stop"] == "0.000"
        assert cell["stop"] == str(STOP)
        assert cell["head_steps"] == str(HEAD_STEPS)

    def test_two_stops_of_one_arm_are_two_rows(self, tmp_path):
        """The STOP keys a row beside the arm. `dec_m090r100_ramp1k` scored
        1.2322 at the 40,000-step stop and 1.2381 at 80,000. A table keyed by
        the arm alone keeps one of those measured numbers and drops the
        other."""
        arm = "dec_m090r100_ramp1k"
        res = tmp_path / "results"
        res.mkdir()
        for stop, value in ((40_000, "1.2322"), (80_000, "1.2381")):
            tag = study_out(f"cf409_tag {arm} {stop} {HEAD_STEPS}")
            (res / f"score_{tag}.txt").write_text(value + "\n")
        out = self._run(res, {"CF409_STOPS": "40000 80000"})
        assert out.returncode == 0, out.stderr
        assert self._by_stop(res) == {"40000": "1.2322", "80000": "1.2381"}

    def test_a_stop_whose_score_file_is_gone_still_lands(self, tmp_path):
        """`results/` is under git, so a checkout can take a score file away
        while the measurement stays in the eval's own log. That took the
        40,000-step files of three arms, and the table then showed their
        80,000-step scores in place of them."""
        arm = "dec_m090r100_ramp1k"
        res = tmp_path / "results"
        res.mkdir()
        tag = study_out(f"cf409_tag {arm} 80000 {HEAD_STEPS}")
        (res / f"score_{tag}.txt").write_text("1.2381\n")
        self._eval_log(tmp_path / "root", arm, 40_000, "1.2322")
        out = self._run(res, {"CF409_STOPS": "40000 80000"})
        assert out.returncode == 0, out.stderr
        assert self._by_stop(res) == {"40000": "1.2322", "80000": "1.2381"}

    def test_an_empty_score_file_is_not_a_zero(self, tmp_path):
        """An eval killed between opening and writing leaves one, and 0.0
        would be the best GM-Relative MASE the project ever recorded."""
        res = self._results(tmp_path, {"dec_s20": "1.1400\n",
                                       "dec_m099_fix": ""})
        assert self._run(res).returncode == 0
        rows = (res / "scores.csv").read_text().strip().splitlines()
        assert len(rows) == 2
        assert "dec_m099_fix" not in (res / "scores.csv").read_text()

    def test_a_foreign_score_file_is_not_a_row(self, tmp_path):
        res = self._results(tmp_path, {"dec_s20": "1.1400\n"})
        (res / "score_a08_bb40k_h30k_student.txt").write_text("1.1782\n")
        out = self._run(res)
        assert out.returncode == 0
        assert "a08" not in (res / "scores.csv").read_text()

    def test_it_writes_the_auc_verdict_of_every_run(self, tmp_path):
        """The card asks for the contrastive AUC of every run, and the step
        of any loss."""
        res = self._results(tmp_path, {"dec_s20": "1.1400\n"})
        root = tmp_path / "root"
        for arm, aucs in (("dec_s20", [0.97] * 3000),
                          ("dec_m080_r200", [0.97] * 1500 + [0.50] * 1500)):
            csv_path = Path(study_out(
                f"cf409_losses_csv {arm} {STOP}",
                {"CF409_ROOT": str(root)}))
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            losses_csv(csv_path, aucs)
        out = self._run(res, {"CF409_ROOT": str(root)})
        assert out.returncode == 0, out.stderr
        table = (res / "auc_verdicts.tsv").read_text()
        assert table.splitlines()[0].startswith("run\t")
        assert "dec_s20" in table and "held" in table
        assert "dec_m080_r200" in table and "lost" in table


# --- 2. the arms on the two cards ----------------------------------


class TestTheLauncher:

    def _dealt(self, arms=None):
        """`{card: [arm, ...]}` from one dry run of the launcher."""
        env = {"CF409_GPU_COUNT": "2", "GPUS": "0 1"}
        if arms is not None:
            env["ARMS"] = " ".join(arms)
        out = script_dry_run(LAUNCH, env=env)
        assert out.returncode == 0, out.stderr
        lanes = {}
        for line in out.stdout.splitlines():
            if line.startswith("arm "):
                gpu = line.split("gpu=")[1].split()[0]
                lanes.setdefault(gpu, []).append(line.split()[1])
        return lanes

    def test_every_named_arm_is_dealt_exactly_once(self):
        """A round of the search NAMES its arms, and the whole catalogue is
        the default. The launcher deals what it is given, once each: a dealt
        arm twice is two lanes on one set of file names."""
        for arms in (None, ROUND):
            want = list(ARMS) if arms is None else list(arms)
            dealt = [a for lane in self._dealt(arms).values() for a in lane]
            assert sorted(dealt) == sorted(want), arms
            assert len(dealt) == len(set(dealt)), arms

    def test_the_lanes_share_the_arms_they_are_given(self):
        """Round-robin over the cards, so an odd count leaves one card with
        one more. One card must never take a whole end of the ladder."""
        for arms in (None, ROUND):
            want = ARMS if arms is None else arms
            lanes = self._dealt(arms)
            assert sorted(lanes) == ["0", "1"], arms
            counts = sorted(len(v) for v in lanes.values())
            assert sum(counts) == len(want), arms
            assert counts[1] - counts[0] <= 1, (arms, counts)

    def test_the_plan_names_the_ramp_of_each_arm(self):
        """The dealt line is what an operator reads before a round starts.
        One ramp for every arm would hide the ramp axis."""
        env = {"CF409_GPU_COUNT": "2", "GPUS": "0 1",
               "ARMS": "dec_m080_r200 dec_ramp5k_m080 dec_ramp30k_m080"}
        out = script_dry_run(LAUNCH, env=env)
        assert out.returncode == 0, out.stderr
        got = {line.split()[1]: line.split("ramp=")[1].split()[0]
               for line in out.stdout.splitlines() if line.startswith("arm ")}
        assert got == {a: RAMPS[a] for a in env["ARMS"].split()}

    def test_a_card_this_machine_does_not_carry_is_refused(self):
        """A lane on a card that is not there dies inside .to(device), hours
        after the operator has left."""
        out = script_dry_run(LAUNCH, env={"CF409_GPU_COUNT": "1",
                                          "GPUS": "0 1"})
        assert out.returncode != 0
        assert "card" in out.stderr

    def test_a_machine_with_no_card_is_refused(self):
        out = script_dry_run(LAUNCH, env={"CF409_GPU_COUNT": "0"})
        assert out.returncode != 0

    def test_the_lane_runs_the_backbone_then_the_head(self):
        out = script_dry_run(PHASE1, env={"ARMS": "dec_s20"})
        assert out.returncode == 0, out.stderr
        assert "arm dec_s20" in out.stdout
        assert "head dec_s20" in out.stdout


class TestTheCheckoutTheStudyNeeds:
    """A machine bootstrapped from a stale branch trains one copy of the
    published cell for each arm and logs nothing unusual. The launcher asks
    first."""

    def _checkout(self, tmp_path, trainer=True, gap=True, token=True,
                  head=True, head_results=True, head_trainer=True,
                  ema=True):
        wt = tmp_path / "wt"
        train = wt / "experiments" / "2026-04-27_freq-embedding" / "scripts"
        runner = wt / "reports" / "2026-08-08_rollout_depth" / "scripts"
        gift = wt / "experiments" / "2026-04-13_gift-eval" / "scripts"
        train.mkdir(parents=True)
        runner.mkdir(parents=True)
        gift.mkdir(parents=True)
        (train / "train.py").write_text(
            "--rep-loss-weight-end\n" if trainer else "# stale\n")
        (runner / "run_leg_k.sh").write_text(
            ("GAP_ARGS\n" if gap else "# stale\n")
            + ("EMA_ARGS_ARR\n" if ema else "# no schedule\n"))
        if head:
            (runner / "head_eval_bb.sh").write_text(
                'RES="${CF_RESULTS:-x}"\nSCORE_OUT="$RES/score_${TAG}.txt"\n'
                if head_results else 'SCORE_OUT="$RES/score_${TAG}.txt"\n')
        if head_trainer:
            (gift / "train_forecasting_head.py").write_text("# head\n")
        (wt / "experiments" / "hf_token.txt").write_text(
            "hf_abc\n" if token else "")
        return wt

    def test_a_current_checkout_passes(self, tmp_path):
        wt = self._checkout(tmp_path)
        assert study_call(f'cf409_check_checkout "{wt}"').returncode == 0

    def test_a_trainer_without_the_flag_is_refused(self, tmp_path):
        wt = self._checkout(tmp_path, trainer=False)
        out = study_call(f'cf409_check_checkout "{wt}"')
        assert out.returncode != 0
        assert "--rep-loss-weight-end" in out.stderr

    def test_a_runner_without_gap_args_is_refused(self, tmp_path):
        wt = self._checkout(tmp_path, gap=False)
        assert study_call(f'cf409_check_checkout "{wt}"').returncode != 0

    def test_a_runner_without_ema_args_is_refused(self, tmp_path):
        """`EMA_ARGS` is the axis. Without it every arm trains the runner's
        own schedule, so eight arms would be arm 1 eight times."""
        wt = self._checkout(tmp_path, ema=False)
        out = study_call(f'cf409_check_checkout "{wt}"')
        assert out.returncode != 0
        assert "EMA_ARGS" in out.stderr

    def test_an_empty_hf_token_is_refused(self, tmp_path):
        """The anonymous rate limit idles the card at about 20 percent use."""
        wt = self._checkout(tmp_path, token=False)
        out = study_call(f'cf409_check_checkout "{wt}"')
        assert out.returncode != 0
        assert "token" in out.stderr

    def test_a_checkout_without_the_head_script_is_refused(self, tmp_path):
        """Without it the study trains eight backbones for hours, and then
        every head exits 2 and `scores.csv` is empty."""
        wt = self._checkout(tmp_path, head=False)
        out = study_call(f'cf409_check_checkout "{wt}"')
        assert out.returncode != 0
        assert "head_eval_bb.sh" in out.stderr

    def test_a_head_script_that_ignores_cf_results_is_refused(self, tmp_path):
        """`collect.sh` reads `score_<tag>.txt` under THIS study's results/. A
        head script without CF_RESULTS writes them under #373's."""
        wt = self._checkout(tmp_path, head_results=False)
        out = study_call(f'cf409_check_checkout "{wt}"')
        assert out.returncode != 0
        assert "CF_RESULTS" in out.stderr

    def test_a_checkout_without_the_head_trainer_is_refused(self, tmp_path):
        """`head_eval_bb.sh` runs it, and refuses without it — after the
        backbone."""
        wt = self._checkout(tmp_path, head_trainer=False)
        assert study_call(f'cf409_check_checkout "{wt}"').returncode != 0

    def test_this_checkout_carries_the_pieces_on_the_branch(self):
        """The HF token is gitignored, so a worktree has none. The rest are
        on the branch."""
        trainer = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
                   / "scripts" / "train.py").read_text()
        runner = (REPO_ROOT / "reports" / "2026-08-08_rollout_depth"
                  / "scripts" / "run_leg_k.sh").read_text()
        head = (REPO_ROOT / "reports" / "2026-08-08_rollout_depth"
                / "scripts" / "head_eval_bb.sh").read_text()
        assert "--rep-loss-weight-end" in trainer
        assert "GAP_ARGS" in runner
        assert "EMA_ARGS" in runner
        assert "CF_RESULTS" in head
        assert 'score_${TAG}.txt' in head
        assert (REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
                / "train_forecasting_head.py").exists()


class TestALaneRefiresACrashedLeg:
    """The review asked for an entry point that restarts a leg after a
    crash. A refusal is not a crash: it repeats."""

    def _fake(self, tmp_path, exits):
        """A run_arm.sh that gives `exits` in order, and counts its calls."""
        counter = tmp_path / "calls"
        script = tmp_path / "fake_run_arm.sh"
        script.write_text(
            "#!/bin/bash\n"
            f'n=$(cat "{counter}" 2>/dev/null || echo 0)\n'
            'n=$(( n + 1 ))\n'
            f'printf "%s" "$n" >"{counter}"\n'
            f'codes=({" ".join(str(e) for e in exits)})\n'
            'idx=$(( n - 1 ))\n'
            '[ "$idx" -ge "${#codes[@]}" ] && idx=$(( ${#codes[@]} - 1 ))\n'
            'exit "${codes[$idx]}"\n')
        script.chmod(0o755)
        return script, counter

    def _lane(self, tmp_path, exits, env=None):
        run_arm, counter = self._fake(tmp_path, exits)
        head = tmp_path / "fake_head.sh"
        head.write_text("#!/bin/bash\nexit 0\n")
        head.chmod(0o755)
        full = dict(os.environ)
        full.update({"ARMS": "dec_s20",
                     "CF409_RUN_ARM": str(run_arm),
                     "CF409_HEAD_EVAL": str(head),
                     "CF409_RESULTS": str(tmp_path / "results"),
                     # The lane probes the Hub before every arm. `true` is a
                     # Hub that always answers, so no test needs a network.
                     "HUB_GATE_PROBE": "true",
                     "HEAD_BG": "0"})
        full.update(env or {})
        out = subprocess.run(["bash", str(PHASE1)], capture_output=True,
                             text=True, env=full, cwd=str(REPO_ROOT),
                             timeout=180)
        return out, int(counter.read_text())

    def test_a_leg_that_crashes_once_is_refired_and_then_scored(self, tmp_path):
        out, calls = self._lane(tmp_path, [1, 0])
        assert calls == 2
        assert out.returncode == 0, out.stdout + out.stderr

    def test_the_refires_stop_at_the_try_budget(self, tmp_path):
        out, calls = self._lane(tmp_path, [1], {"CF409_LEG_TRIES": "3"})
        assert calls == 3
        assert out.returncode != 0

    def test_a_refused_arm_is_never_refired(self, tmp_path):
        """Exit 2 is a refusal — an unknown arm, an unknown stop, a missing
        runner. Every re-fire would refuse again."""
        _, calls = self._lane(tmp_path, [2])
        assert calls == 1

    def test_a_wrong_objective_is_never_refired(self, tmp_path):
        _, calls = self._lane(tmp_path, [3])
        assert calls == 1

    def test_a_collapsed_arm_is_never_refired(self, tmp_path):
        """The AUC watch stopped it. A re-fire trains the same collapse."""
        _, calls = self._lane(tmp_path, [4])
        assert calls == 1

    def test_a_lane_that_lost_its_leg_runs_no_head(self, tmp_path):
        out, _ = self._lane(tmp_path, [4])
        assert "head dec_s20" not in out.stdout
        assert out.returncode != 0


# --- 3. the AUC watch, wired to the leg ----------------------------------


class TestTheAucWarmup:
    """AUC climbs from about 0.5 at step 0. Without a warmup the gate would
    stop every arm in its first minute."""

    def _watch(self, *args):
        return subprocess.run(
            [sys.executable, str(AUC_WATCH), *[str(a) for a in args]],
            capture_output=True, text=True, timeout=120)

    def test_the_warmup_drops_the_early_rows(self, tmp_path):
        """A run 200 steps into its climb still holds more low rows than high
        ones, so its median reads as a loss."""
        path = losses_csv(tmp_path / "a_losses.csv", [0.50] * 120 + [0.97] * 80)
        assert self._watch(path).returncode == 1
        assert self._watch(path, "--warmup", "120").returncode == 0

    def test_a_collapse_after_the_warmup_is_still_read(self, tmp_path):
        path = losses_csv(tmp_path / "b_losses.csv",
                          [0.50] * 50 + [0.97] * 100 + [0.50] * 100)
        out = self._watch(path, "--window", "10", "--warmup", "50")
        assert out.returncode == 1
        assert int(out.stdout.split("\t")[2]) >= 150

    def test_a_warmup_past_the_run_is_not_a_verdict(self, tmp_path):
        path = losses_csv(tmp_path / "c_losses.csv", [0.50] * 20)
        assert self._watch(path, "--warmup", "500").returncode == 2

    def test_the_default_warmup_reads_the_whole_run(self, tmp_path):
        path = losses_csv(tmp_path / "d_losses.csv", [0.50] * 100)
        assert self._watch(path, "--window", "10").returncode == 1


class TestTheAucGuard:
    """The guard is what makes the watch save GPU time. Four arms decay to
    0.0 and cross the known-dead ratio near step 5,600, and each collapsed
    arm would otherwise burn about 30,000 dead steps.

    Every test writes the rows of THIS leg while the guard runs, because that
    is what the trainer does. Rows already on disk when the guard starts
    belong to the leg before it.
    """

    # A leg the gate must NOT stop ends by itself, so the guard returns in
    # seconds. A leg the gate must stop outlives the test.
    #
    # The victim is an orphan, not a child of this test: `kill -0` succeeds on
    # a zombie, and a child nothing reaps stays one.
    def _victim(self, seconds=120):
        out = subprocess.run(
            ["sh", "-c", f"sleep {seconds} >/dev/null 2>&1 & echo $!"],
            capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip())

    def _alive(self, pid):
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    def _reap(self, pid):
        try:
            os.kill(pid, 9)
        except (OSError, ProcessLookupError):
            pass

    def _env(self, base, env=None):
        full = dict(os.environ)
        full.update({"CF409_ROOT": str(base / "root"),
                     "CF409_RESULTS": str(base / "results"),
                     "CF409_AUC_POLL": "1",
                     "CF409_AUC_WINDOW": "10",
                     "CF409_AUC_WARMUP": "0"})
        full.update(env or {})
        return full

    def _csv_path(self, base, name=None):
        path = Path(study_out(f"cf409_losses_csv dec_s20 {STOP}",
                              {"CF409_ROOT": str(base / "root")}))
        if name:
            path = path.parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _guard(self, base, aucs, pid, env=None):
        """The guard over a CSV that is whole before it starts."""
        base.mkdir(parents=True, exist_ok=True)
        losses_csv(self._csv_path(base), aucs)
        full = self._env(base, env)
        out = subprocess.run(
            ["bash", str(AUC_GUARD), "dec_s20", str(STOP), str(pid)],
            capture_output=True, text=True, env=full, cwd=str(REPO_ROOT),
            timeout=120)
        return out, base / "results"

    def _start(self, base, before, pid, env=None):
        """Start the guard over a CSV that holds `before`, and return once it
        has read its baseline. Rows appended after this are the leg's own."""
        base.mkdir(parents=True, exist_ok=True)
        losses_csv(self._csv_path(base), before)
        full = self._env(base, env)
        proc = subprocess.Popen(
            ["bash", str(AUC_GUARD), "dec_s20", str(STOP), str(pid)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=full, cwd=str(REPO_ROOT))
        log = base / "results" / "auc_guard.log"
        for _ in range(300):
            if log.exists():
                return proc, base / "results"
            time.sleep(0.1)
        proc.kill()
        raise AssertionError("the guard logged no start")

    def _append(self, base, aucs, name=None, first_step=1):
        """The rows this leg writes, in the trainer's shape."""
        path = self._csv_path(base, name)
        if path.exists():
            first_step = sum(1 for _ in path.read_text().splitlines())
            with open(path, "a") as fh:
                for i, auc in enumerate(aucs):
                    fh.write(f"{first_step + i},1.0,{auc},1.0,11.5,0.9\n")
        else:
            losses_csv(path, aucs, first_step)

    def _drain(self, proc, timeout=120):
        out = proc.communicate(timeout=timeout)[0]
        return proc.returncode, out

    def test_it_is_called_from_the_leg(self):
        assert "auc_guard.sh" in RUN_ARM.read_text()

    def test_a_collapsed_leg_is_stopped(self, tmp_path):
        pid = self._victim()
        try:
            proc, res = self._start(tmp_path, [0.97] * 60, pid)
            self._append(tmp_path, [0.42] * 60)
            rc, out = self._drain(proc)
            assert rc == 1, out
            assert not self._alive(pid)
            note = (res / "collapsed_dec_s20.txt").read_text()
            assert "dec_s20" in note
            assert "lost" in note
        finally:
            self._reap(pid)

    def test_a_healthy_leg_runs_on(self, tmp_path):
        pid = self._victim(8)
        try:
            proc, res = self._start(tmp_path, [0.97] * 20, pid)
            self._append(tmp_path, [0.97] * 100)
            rc, out = self._drain(proc)
            assert rc == 0, out
            assert not (res / "collapsed_dec_s20.txt").exists()
        finally:
            self._reap(pid)

    def test_the_guard_stops_when_the_leg_stops(self, tmp_path):
        pid = self._victim()
        self._reap(pid)
        out, res = self._guard(tmp_path, [0.97] * 120, pid)
        assert out.returncode == 0
        assert not (res / "collapsed_dec_s20.txt").exists()

    def test_a_collapse_before_this_leg_does_not_stop_it(self, tmp_path):
        """A re-fired leg writes no CSV row for its first 100 steps. The
        newest CSV is then the dead leg's, and a verdict on it would stop a
        leg that has trained nothing yet."""
        pid = self._victim(6)
        try:
            out, res = self._guard(tmp_path, [0.97] * 30 + [0.40] * 60, pid)
            assert out.returncode == 0, out.stdout + out.stderr
            assert not (res / "collapsed_dec_s20.txt").exists()
        finally:
            self._reap(pid)

    def test_the_gate_reads_the_csv_this_leg_opened(self, tmp_path):
        """train.py branches a re-fired leg's run name to `<name>_r2`, so the
        dead leg's collapsed CSV stays on disk beside the new one."""
        pid = self._victim(8)
        name = study_out("cf409_run_name dec_s20")
        try:
            proc, res = self._start(tmp_path, [0.97] * 20 + [0.40] * 60, pid)
            self._append(tmp_path, [0.97] * 100, f"{name}_r2_losses.csv")
            rc, out = self._drain(proc)
            assert rc == 0, out
            assert not (res / "collapsed_dec_s20.txt").exists()
        finally:
            self._reap(pid)

    def test_a_collapse_this_leg_wrote_is_still_read(self, tmp_path):
        """The skipped rows must not hold the gate off the rows above them."""
        pid = self._victim()
        try:
            proc, _ = self._start(tmp_path, [0.97] * 40, pid)
            self._append(tmp_path, [0.97] * 20 + [0.41] * 60)
            rc, out = self._drain(proc)
            assert rc == 1, out
            assert not self._alive(pid)
        finally:
            self._reap(pid)

    def test_the_warmup_holds_the_gate_off_the_start_of_a_run(self, tmp_path):
        """Without it the gate stops every arm in its first minute: the AUC of
        a fresh run starts near 0.5 and climbs."""
        climb = [0.50] * 120 + [0.97] * 80
        window = {"CF409_AUC_WINDOW": "500"}
        cold = self._victim()
        try:
            proc, _ = self._start(tmp_path / "cold", [], cold, window)
            self._append(tmp_path / "cold", climb)
            rc, out = self._drain(proc)
            assert rc == 1, out
            assert not self._alive(cold)
        finally:
            self._reap(cold)
        warm = self._victim(8)
        try:
            proc, res = self._start(tmp_path / "warm", [], warm,
                                    {**window, "CF409_AUC_WARMUP": "120"})
            self._append(tmp_path / "warm", climb)
            rc, out = self._drain(proc)
            assert rc == 0, out
            assert not (res / "collapsed_dec_s20.txt").exists()
        finally:
            self._reap(warm)

    def test_the_card_warmup_reaches_the_shortest_ramp(self):
        """The gate reads no step below the warmup. So the warmup must not
        outlast the shortest decay ramp, or an arm would run its whole decay
        unwatched.

        `dec_m090r100_ramp1k` sets that floor. Its weight reaches 0.0 at step
        1,000, which is the step the gate turns on. The three short-ramp arms
        held the task, at floors 0.9075, 0.9092 and 0.8688 against the 0.55
        threshold, so the gate read each of them in time."""
        warmup = int(study_out('printf %s "$CF409_AUC_WARMUP"'))
        assert 0 < warmup <= min(int(RAMPS[arm]) for arm in ARMS)

    def test_a_longer_ramp_still_meets_the_gate_under_weight(self):
        """An arm whose ramp outlasts the warmup meets the gate with at least
        half of the repel term in front of it. `dec_m090r100_ramp2k` is the
        tightest, at 0.5."""
        warmup = int(study_out('printf %s "$CF409_AUC_WARMUP"'))
        for arm in ARMS:
            if int(RAMPS[arm]) <= warmup:
                continue
            got = float(study_out(f"cf409_rep_w_at {arm} {warmup}"))
            assert got >= 0.5, (arm, got)


class TestTheLaneHoldsOneHead:
    """`phase1.sh` puts each head in the background on the lane's own card,
    and a lane holds four arms. Without a queue, three heads and one backbone
    share one card."""

    def _lane(self, tmp_path, arms, head_body, env=None):
        marks = tmp_path / "marks"
        run_arm = tmp_path / "fake_run_arm.sh"
        run_arm.write_text("#!/bin/bash\nexit 0\n")
        run_arm.chmod(0o755)
        head = tmp_path / "fake_head.sh"
        head.write_text("#!/bin/bash\n"
                        f'printf "start %s\\n" "$1" >>"{marks}"\n'
                        f"{head_body}\n"
                        f'printf "end %s\\n" "$1" >>"{marks}"\n')
        head.chmod(0o755)
        full = dict(os.environ)
        full.update({"ARMS": " ".join(arms),
                     "CF409_RUN_ARM": str(run_arm),
                     "CF409_HEAD_EVAL": str(head),
                     "CF409_RESULTS": str(tmp_path / "results"),
                     "HUB_GATE_PROBE": "true",
                     "HEAD_BG": "1"})
        full.update(env or {})
        out = subprocess.run(["bash", str(PHASE1)], capture_output=True,
                             text=True, env=full, cwd=str(REPO_ROOT),
                             timeout=300)
        rows = marks.read_text().splitlines() if marks.exists() else []
        return out, rows

    def _peak(self, rows):
        """The most heads that ran at the same time."""
        live = peak = 0
        for row in rows:
            live += 1 if row.startswith("start") else -1
            peak = max(peak, live)
        return peak

    def test_one_head_runs_at_a_time(self, tmp_path):
        out, rows = self._lane(tmp_path, ARMS[:3], "sleep 1")
        assert self._peak(rows) == 1, rows
        assert len([r for r in rows if r.startswith("start")]) == 3
        assert out.returncode == 0, out.stdout + out.stderr

    def test_every_head_ends_before_the_lane_does(self, tmp_path):
        """`collect.sh` runs after the queue, so it must see every score."""
        _, rows = self._lane(tmp_path, ARMS[:3], "sleep 1")
        assert len([r for r in rows if r.startswith("end")]) == 3

    def test_a_head_that_fails_fails_the_lane(self, tmp_path):
        out, rows = self._lane(tmp_path, ARMS[:2], "exit 7")
        assert out.returncode != 0
        assert len([r for r in rows if r.startswith("start")]) == 2

    def test_the_next_backbone_does_not_wait_for_the_head(self, tmp_path):
        """The queue holds the heads apart. It must not hold the card idle
        between two backbones."""
        out, _ = self._lane(tmp_path, ARMS[:2], "sleep 3")
        lines = out.stdout.splitlines()
        second_arm = next(i for i, line in enumerate(lines)
                          if f"arm {ARMS[1]} ->" in line)
        first_head_done = next(i for i, line in enumerate(lines)
                               if f"head {ARMS[0]}" in line and "rc=" in line)
        assert second_arm < first_head_done, out.stdout


# --- 4. the loss-by-term formula of the doc ------------------------------


def _spec(**extra):
    cfg = {"contrastive_divergence_temperature": 0.10,
           "contrastive_latent_noise": None,
           "loss_shape": "cosine_similarity_batch_rep_only",
           "contrastive_latent_delay": 0}
    cfg.update(extra)
    return SimpleNamespace(train_configuration=cfg)


def _rollout_latents(k, B=3, T=8, C=2, H=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    o = torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
    rollouts = [torch.randn(B, T, C, H, generator=g, dtype=torch.float64)
                for _ in range(k)]
    return f, o, rollouts


class TestTheLossByTermFormula:
    """The report rebuilds the loss by term from the CSV columns.

    This card runs k = 32 under the `mean` reduction, against the EMA
    TEACHER. Two things follow, and `notes/loss_decomposition.md` states
    both:

      * the `l_align` column is the depth-0 copy alone, and the loss holds
        the MEAN of k + 1 copies.
      * `l_align` is NOT `2 * cos_err_d0` here. `cos_err_dj` reads the
        student's next latent and the teacher target reads the teacher's,
        so the `cos_err_d*` columns cannot rebuild the align part.

    So the align part of the total is read as a residual. That residual is
    exact on this cell, whose CPC weight is 0.0.
    """

    K = 32
    ALIGN_W = 1.0
    REP_W = 0.7

    # k = 32 needs k + 2 time steps to keep one anchor, so T is 36 here.
    T = 36

    def _loss(self, reduce="mean", align_target="teacher"):
        f, o, rollouts = _rollout_latents(self.K, T=self.T, seed=5)
        g = torch.Generator().manual_seed(11)
        teacher_o = torch.randn(*o.shape, generator=g, dtype=torch.float64)
        terms = {}
        loss = contrastive_latent_loss(
            (f, o), False, _spec(train_rollout_depth=self.K,
                                 train_rollout_reduce=reduce),
            align_loss_weight=self.ALIGN_W, align_target=align_target,
            teacher_original_latent=teacher_o,
            rollout_latents=rollouts, train_rollout_depth=self.K,
            train_rollout_reduce=reduce,
            rep_loss_weight=self.REP_W, term_out=terms)
        return float(loss), terms, rollout_cos_error(f, o, rollouts)

    def test_the_depth_zero_formula_does_not_close_at_k_thirty_two(self):
        loss, terms, _ = self._loss()
        naive = self.REP_W * terms["l_rep"] + self.ALIGN_W * terms["l_align"]
        assert loss != pytest.approx(naive, rel=1e-3)

    def test_the_cos_err_columns_cannot_rebuild_a_teacher_align(self):
        """The identity `l_align = 2 * cos_err_d0` holds on the student
        target only. #404's own plot script used it on a teacher run."""
        _, terms, errors = self._loss()
        assert terms["l_align"] != pytest.approx(2.0 * errors[0], rel=1e-3)

    def test_the_align_part_is_the_residual_of_the_total(self):
        """This is the formula the report uses. On this cell the CPC weight
        is 0.0 and SIGReg is added outside this call, so the residual is the
        align part alone.

        Tripling `align_w` must triple the residual and move nothing else. A
        residual that held another term would not scale."""
        loss, terms, _ = self._loss()
        residual = loss - self.REP_W * terms["l_rep"]
        assert residual > 0.0

        f, o, rollouts = _rollout_latents(self.K, T=self.T, seed=5)
        g = torch.Generator().manual_seed(11)
        teacher_o = torch.randn(*o.shape, generator=g, dtype=torch.float64)
        terms3 = {}
        loss3 = float(contrastive_latent_loss(
            (f, o), False, _spec(train_rollout_depth=self.K,
                                 train_rollout_reduce="mean"),
            align_loss_weight=3.0 * self.ALIGN_W, align_target="teacher",
            teacher_original_latent=teacher_o,
            rollout_latents=rollouts, train_rollout_depth=self.K,
            train_rollout_reduce="mean",
            rep_loss_weight=self.REP_W, term_out=terms3))
        residual3 = loss3 - self.REP_W * terms3["l_rep"]
        assert terms3["l_rep"] == pytest.approx(terms["l_rep"])
        assert residual3 == pytest.approx(3.0 * residual, rel=1e-9)

    def test_the_residual_reads_the_mean_of_every_depth(self):
        """Perturbing a copy the depth-0 column never sees must move the
        residual. Otherwise the report would read one depth as all 33."""
        loss_a, terms_a, _ = self._loss()
        f, o, rollouts = _rollout_latents(self.K, T=self.T, seed=5)
        rollouts[7] = rollouts[7] + 0.5
        g = torch.Generator().manual_seed(11)
        teacher_o = torch.randn(*o.shape, generator=g, dtype=torch.float64)
        terms_b = {}
        loss_b = float(contrastive_latent_loss(
            (f, o), False, _spec(train_rollout_depth=self.K,
                                 train_rollout_reduce="mean"),
            align_loss_weight=self.ALIGN_W, align_target="teacher",
            teacher_original_latent=teacher_o,
            rollout_latents=rollouts, train_rollout_depth=self.K,
            train_rollout_reduce="mean",
            rep_loss_weight=self.REP_W, term_out=terms_b))
        assert terms_a["l_align"] == pytest.approx(terms_b["l_align"])
        assert loss_a != pytest.approx(loss_b, rel=1e-6)

    def test_the_student_target_still_closes_over_the_cos_err_columns(self):
        """The identity the doc states for `--align-target student`. Other
        studies read it, so #409 must not break it."""
        loss, terms, errors = self._loss("mean", "student")
        assert len(errors) == self.K + 1
        closed = (self.REP_W * terms["l_rep"]
                  + self.ALIGN_W * 2.0 * sum(errors) / (self.K + 1))
        assert loss == pytest.approx(closed, rel=1e-6)

    def test_the_doc_states_the_rollout_case(self):
        body = DOC.read_text()
        formula = body.split("loss = rep_w * l_rep")[1]
        assert "--train-rollout-depth" in formula
        assert "cos_err_d" in formula
        assert "k + 1" in formula

    def test_the_note_states_this_cards_own_cell(self):
        """`notes/loss_decomposition.md` must describe the cell this card
        runs, not the one the first attempt ran."""
        body = DECOMP.read_text()
        assert "k = 32" in body
        assert "mean" in body
        assert "teacher" in body
        assert "residual" in body


# --- 5. the Hub outage of 2026-08-23 -------------------------------------


class TestALaneRidesOutAHubOutage:
    """On 08-23 at 18:48 elisa lost DNS. Every leg died in 3 seconds, the
    lane spent its whole ladder in two minutes, declared the arm dead and
    moved to the next one. Three arms went that way in seven minutes and the
    card sat idle for 27 hours.

    A network failure is not a failed arm. `scripts/hub_gate.sh` holds the
    reading and the wait, and this lane holds the policy.
    """

    def _fake(self, tmp_path, exits):
        counter = tmp_path / "calls"
        script = tmp_path / "fake_run_arm.sh"
        script.write_text(
            "#!/bin/bash\n"
            f'n=$(cat "{counter}" 2>/dev/null || echo 0)\n'
            "n=$(( n + 1 ))\n"
            f'printf "%s" "$n" >"{counter}"\n'
            f'codes=({" ".join(str(e) for e in exits)})\n'
            "idx=$(( n - 1 ))\n"
            '[ "$idx" -ge "${#codes[@]}" ] && idx=$(( ${#codes[@]} - 1 ))\n'
            'exit "${codes[$idx]}"\n')
        script.chmod(0o755)
        return script, counter

    def _lane(self, tmp_path, exits, arms="dec_s20", probe="true", env=None):
        run_arm, counter = self._fake(tmp_path, exits)
        head = tmp_path / "fake_head.sh"
        head.write_text("#!/bin/bash\nexit 0\n")
        head.chmod(0o755)
        full = dict(os.environ)
        full.update({"ARMS": arms,
                     "CF409_RUN_ARM": str(run_arm),
                     "CF409_HEAD_EVAL": str(head),
                     "CF409_RESULTS": str(tmp_path / "results"),
                     "HEAD_BG": "0",
                     "HUB_GATE_PROBE": probe,
                     "HUB_GATE_BASE_WAIT": "1",
                     "HUB_GATE_MAX_WAIT": "1",
                     # The wait after a CRASH is not what these tests read.
                     "CF409_LEG_RETRY_WAIT": "0",
                     "CF409_LEG_TRIES": "3"})
        full.update(env or {})
        out = subprocess.run(["bash", str(PHASE1)], capture_output=True,
                             text=True, env=full, cwd=str(REPO_ROOT),
                             timeout=300)
        calls = int(counter.read_text()) if counter.exists() else 0
        return out, calls

    def test_the_network_code_is_the_shared_one(self):
        shared = subprocess.run(
            ["bash", "-c",
             f'. "{REPO_ROOT}/scripts/hub_gate.sh" && printf %s "$HUB_GATE_RC"'],
            capture_output=True, text=True, timeout=60).stdout
        assert study_out('printf %s "$CF409_RC_NETWORK"') == shared

    def test_a_network_failure_never_counts_against_the_ladder(self, tmp_path):
        """Four outage deaths and then a clean leg. At three tries the old
        lane declared the arm dead on the third."""
        rc = int(study_out('printf %s "$CF409_RC_NETWORK"'))
        out, calls = self._lane(tmp_path, [rc, rc, rc, rc, 0])
        assert calls == 5, out.stdout + out.stderr
        assert out.returncode == 0, out.stdout + out.stderr

    def test_a_crash_after_an_outage_still_counts(self, tmp_path):
        rc = int(study_out('printf %s "$CF409_RC_NETWORK"'))
        out, calls = self._lane(tmp_path, [rc, 1, 1, 1])
        assert calls == 4, out.stdout + out.stderr
        assert out.returncode != 0

    def test_the_lane_reads_the_hub_before_it_starts_an_arm(self, tmp_path):
        """Card rule 4: do not advance to the next arm while the network is
        down."""
        probes = tmp_path / "probes"
        probe = tmp_path / "probe.sh"
        probe.write_text("#!/bin/bash\n"
                         f'printf "x" >>"{probes}"\n'
                         "exit 0\n")
        probe.chmod(0o755)
        out, calls = self._lane(tmp_path, [0], arms="dec_s20 dec_s22",
                                probe=f"bash {probe}")
        assert out.returncode == 0, out.stdout + out.stderr
        assert calls == 2
        assert len(probes.read_text()) >= 2

    def test_a_lane_starts_no_arm_while_the_hub_is_down(self, tmp_path):
        out, calls = self._lane(tmp_path, [0], arms="dec_s20 dec_s22",
                                probe="false",
                                env={"CF409_NET_DEADLINE": "2"})
        assert calls == 0, out.stdout + out.stderr
        assert out.returncode != 0
        assert "huggingface.co" in out.stdout

    def test_an_outage_that_never_ends_stops_the_lane(self, tmp_path):
        """A Hub that answers the probe while every leg still dies would
        re-fire for ever and hold the card at zero steps. One leg gets the
        deadline, and no more. Its checkpoints stay, so a later lane
        resumes."""
        rc = int(study_out('printf %s "$CF409_RC_NETWORK"'))
        out, calls = self._lane(tmp_path, [rc], arms="dec_s20 dec_s22",
                                env={"CF409_NET_DEADLINE": "2"})
        assert out.returncode != 0
        assert calls < 20, out.stdout
        assert "deadline" in out.stdout

    def test_the_deadline_is_hours_not_minutes(self):
        """Card rule 2: a DNS outage of 30 minutes must not end a study."""
        assert int(study_out('printf %s "$CF409_NET_DEADLINE"')) >= 3 * 3600


class TestTheLegsSaveEvery5000:
    """Card rule 5. `dec_m080_r200` reached 19,900 steps at the runner's own
    20,000 cadence and saved no step checkpoint, so the outage cost all of
    it. At 5,000 an outage costs at most 5,000 steps."""

    def test_the_study_default_is_5000(self):
        assert study_out('printf %s "$CF409_SAVE_EVERY"') == "5000"

    def test_the_launcher_hands_it_to_the_runner(self):
        out = script_dry_run(RUN_ARM, "dec_s20", STOP)
        assert "save_every=5000" in out.stdout, out.stdout

    def test_the_cadence_divides_the_stop(self):
        every = int(study_out('printf %s "$CF409_SAVE_EVERY"'))
        assert STOP % every == 0

    def test_the_shared_runner_keeps_its_own_default(self):
        """#401 and #404 read the same runner. This card changes its own
        cadence, not theirs."""
        body = (PARENT / "scripts" / "run_leg_k.sh").read_text()
        assert 'SAVE_EVERY="${SAVE_EVERY:-20000}"' in body


# --- 7. the losses CSV a re-fired leg wrote ------------------------------


class TestTheReaderStitchesARefiredLeg:
    """A run of this card can write one step more than once.

    A leg that starts again from step 0 APPENDS to the CSV it already opened:
    `dec_m080_r200` holds 59,900 rows over 40,000 steps. A leg re-fired after
    a crash resumes under a `_rN` name and opens a SECOND file:
    `dec_m099_fix` holds two that overlap from step 15,001 to 19,900.

    The first reader took one ROW in ten before it looked at the step column.
    On those two arms it interleaved two attempts, and it read the last ROW of
    the file as the last STEP of the run. `results/loss_terms_at_stop.csv`
    then stopped at step 2,591 with three arms in it, where the card asks for
    every arm to 40,000 steps.
    """

    STYLE = EXP / "scripts" / "arm_style.py"

    @staticmethod
    def _csv(path, rows, columns=("loss", "auc")):
        head = ",".join(("step",) + tuple(columns))
        body = "\n".join(",".join(str(c) for c in row) for row in rows)
        path.write_text(f"{head}\n{body}\n")
        return str(path)

    @property
    def style(self):
        spec = importlib.util.spec_from_file_location(
            "arm_style_under_test", self.STYLE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_leg_that_started_again_wins_over_the_first_attempt(self, tmp_path):
        S = self.style
        path = self._csv(tmp_path / "a.csv",
                         [(1, 9.0, 0.9), (2, 9.0, 0.9), (3, 9.0, 0.9),
                          (1, 1.0, 0.5), (2, 2.0, 0.5), (3, 3.0, 0.5),
                          (4, 4.0, 0.5)])
        got = S.read_run([path], ["loss"])["loss"]
        assert got == [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)]

    def test_the_last_step_is_the_last_step_and_not_the_last_row(self, tmp_path):
        S = self.style
        path = self._csv(tmp_path / "a.csv",
                         [(1, 9.0, 0.9), (2, 9.0, 0.9), (3, 9.0, 0.9),
                          (1, 1.0, 0.5), (2, 2.0, 0.5)])
        assert S.read_run([path], ["loss"])["loss"][-1][0] == 3

    def test_a_resumed_leg_carries_the_run_past_its_first_file(self, tmp_path):
        """`dec_m099_fix` in the small: a base file to 19,900 and an `_r2`
        that resumes at 15,001 and reaches the stop."""
        S = self.style
        base = self._csv(tmp_path / "base.csv",
                         [(s, 9.0, 0.9) for s in range(1, 6)])
        r2 = self._csv(tmp_path / "r2.csv",
                       [(s, 1.0, 0.5) for s in range(4, 9)])
        got = dict(S.read_run([base, r2], ["loss"])["loss"])
        assert max(got) == 8
        assert got[3] == 9.0      # only the base reached it
        assert got[4] == 1.0      # both did, and the resume wrote it last

    def test_the_file_that_reached_furthest_wins_the_overlap(self, tmp_path):
        """`dec_s23` ran on in its BASE file to 22,900, and its `_r2` gave up
        at 20,300. Taking the `_rN` file every time would splice 100 steps of
        a dead attempt into the middle of a live one."""
        S = self.style
        base = self._csv(tmp_path / "base.csv",
                         [(s, 9.0, 0.9) for s in range(1, 11)])
        r2 = self._csv(tmp_path / "r2.csv",
                       [(s, 1.0, 0.5) for s in range(5, 8)])
        got = dict(S.read_run([base, r2], ["loss"])["loss"])
        assert max(got) == 10
        assert got[6] == 9.0      # the base ran further, so the base wins

    def test_one_row_in_every_n_is_taken_after_the_sort(self, tmp_path):
        S = self.style
        path = self._csv(tmp_path / "a.csv",
                         [(s, float(s), 0.9) for s in (5, 4, 3, 2, 1)])
        got = S.read_run([path], ["loss"], 2)["loss"]
        assert [s for s, _ in got] == [1, 3, 5]

    def test_a_blank_cell_is_a_gap_and_never_a_zero(self, tmp_path):
        """`l_rep` goes blank at step 10,000, where the weight reaches 0.0 and
        the trainer computes no L_rep. A zero there would read as a term that
        collapsed."""
        S = self.style
        path = self._csv(tmp_path / "a.csv",
                         [(1, 11.0, 0.9), (2, "", 0.9), (3, 12.0, 0.9)],
                         columns=("l_rep", "auc"))
        got = S.read_run([path], ["l_rep"])["l_rep"]
        assert got == [(1, 11.0), (3, 12.0)]

    def test_a_later_blank_clears_an_earlier_value(self, tmp_path):
        S = self.style
        path = self._csv(tmp_path / "a.csv",
                         [(1, 11.0, 0.9), (1, "", 0.9)],
                         columns=("l_rep", "auc"))
        assert S.read_run([path], ["l_rep"])["l_rep"] == []

    def test_the_figures_read_one_stop_of_the_score_table(self, tmp_path):
        """Every figure and the rank gate compare arms AT ONE STOP. Two rows
        of one arm reach them, so a reader keyed by the arm alone would draw
        the 80,000-step score under a 40,000-step title."""
        S = self.style
        path = tmp_path / "scores.csv"
        path.write_text("arm,stop,score\n"
                        "dec_m090r100_ramp1k,40000,1.2322\n"
                        "dec_m090r100_ramp1k,80000,1.2381\n")
        assert S.read_scores(path) == {"dec_m090r100_ramp1k": 1.2322}
        assert S.read_scores(path, 80000) == {"dec_m090r100_ramp1k": 1.2381}
        assert S.read_scores(path, None) == {"dec_m090r100_ramp1k": 1.2381}

    def test_a_score_table_without_a_stop_column_is_not_filtered(self, tmp_path):
        """`rank_gate.py` takes any table its caller names."""
        S = self.style
        path = tmp_path / "scores.csv"
        path.write_text("arm,score\na,1.10\nb,1.30\n")
        assert S.read_scores(path) == {"a": 1.10, "b": 1.30}

    def test_the_window_mean_names_the_steps_it_covers(self, tmp_path):
        S = self.style
        series = [(s, float(s)) for s in range(1, 101)]
        assert S.window_mean(series, 100, 10) == pytest.approx(95.5)
        assert S.window_mean(series, 50, 10) == pytest.approx(45.5)
        assert S.window_mean(series, 500, 10) is None


class TestTheLossByTermTableReachesTheStop:
    """The card asks for the training loss to 40,000 steps, by term."""

    TABLE = EXP / "results" / "loss_terms_at_stop.csv"
    TRACK = EXP / "results" / "loss_terms_trajectory.csv"
    TERMS = {"loss", "rep_w", "l_rep", "align_reduced", "cos_err"}
    # Every arm that reached the stop. `results/scores.csv` scores all six.
    AT_STOP = ("dec_s20", "dec_s22", "dec_s24", "dec_m099_fix",
               "dec_m080_r200", "dec_m070_fix")

    def _rows(self, path):
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))

    def test_every_scored_arm_reaches_the_stop(self):
        rows = self._rows(self.TABLE)
        reached = {r["arm"]: int(r["reached"]) for r in rows}
        for arm in self.AT_STOP:
            assert reached.get(arm) == STOP, arm

    def test_every_term_is_there_for_every_arm(self):
        rows = self._rows(self.TABLE)
        by_arm = {}
        for r in rows:
            by_arm.setdefault(r["arm"], set()).add(r["term"])
        for arm in self.AT_STOP:
            assert by_arm.get(arm) == self.TERMS, arm

    def test_the_slope_window_is_thirty_to_forty_thousand(self):
        """`notes/SECOND_ANSWER.md` measures headroom over that window."""
        rows = self._rows(self.TABLE)
        for r in rows:
            if r["arm"] in self.AT_STOP and r["term"] == "loss":
                assert r["value_at_30k"], r
                assert r["change_30k_to_40k"], r

    def test_a_run_that_stopped_early_gets_no_slope(self):
        """`dec_m050_fix` lost the contrastive task at step 10,162. A change
        over 30,000 to 40,000 steps it never ran would be a made-up number."""
        rows = self._rows(self.TABLE)
        early = [r for r in rows if r["arm"] == "dec_m050_fix"]
        assert early
        for r in early:
            assert r["change_30k_to_40k"] == ""
            assert int(r["reached"]) < STOP

    def test_l_rep_ends_where_the_decay_ends(self):
        """The trainer computes no L_rep at weight 0.0, so the column goes
        blank one step before the ramp ends. That is the treatment working.

        Every SCORED arm carries the card's own ramp, 10,000 steps, so its
        `l_rep` ends at 9,999. `dec_ramp30k_m080` carries a 30,000-step ramp,
        and an arm still inside its ramp has not gone blank at all — its
        `l_rep` runs to the last step it reached. So the rule that holds on
        every row is the weaker one: `l_rep` never outlives the run."""
        rows = [r for r in self._rows(self.TABLE) if r["term"] == "l_rep"]
        assert rows
        scored = {int(r["last_step"]) for r in rows if r["arm"] in self.AT_STOP}
        assert scored == {9999}, scored
        for r in rows:
            assert int(r["last_step"]) <= int(r["reached"]), r

    def test_the_trajectory_reads_the_whole_run(self):
        rows = self._rows(self.TRACK)
        steps = {int(r["step"]) for r in rows}
        assert steps == set(range(5000, STOP + 1, 5000))
        for arm in self.AT_STOP:
            got = {int(r["step"]) for r in rows
                   if r["arm"] == arm and r["term"] == "loss"}
            assert got == set(range(5000, STOP + 1, 5000)), arm


# --- 8. what the card can rank, and against what ------------------------


class TestTheRankGate:
    """This card scored six arms and repeated ONE schedule. That schedule's
    range is the whole run-to-run spread the card measured, so a gap under it
    is not a rank."""

    GATE = EXP / "results" / "rank_gate.tsv"
    SCRIPT = EXP / "scripts" / "rank_gate.py"

    def _rows(self):
        with open(self.GATE, newline="") as fh:
            body = [ln for ln in fh if not ln.startswith("#")]
        return list(csv.DictReader(body, delimiter="\t"))

    def _gate(self):
        """The gate value, from the header line that states it."""
        head = self.GATE.read_text().splitlines()[0]
        return float(head.split(":")[1].split(",")[0])

    @staticmethod
    def _verdict(gap, gate):
        """The rule the report states. A gap at or under the gate is noise, a
        gap under twice the gate is a threshold, and a gap of twice the gate or
        more is a rank."""
        if abs(gap) <= gate:
            return "noise"
        return "threshold" if abs(gap) < 2 * gate else "rank"

    def test_the_table_and_the_script_exist(self):
        assert self.SCRIPT.is_file()
        assert self.GATE.is_file()

    def test_the_gate_is_measured_by_this_card(self):
        """Not borrowed. The header names the arms it came from, and every one
        of them is an arm of this card with a score."""
        head = self.GATE.read_text().splitlines()[:3]
        assert head[0].startswith("# the gate:")
        named = head[1].split(":")[1].replace(",", " ").split()
        assert len(named) >= 2
        for arm in named:
            assert arm in ARMS, arm

    def test_every_arm_loses_to_the_card_target_by_more_than_the_gate(self):
        """The card's first question: does the decay give a new best score?
        The `vs target` block answers it. No arm beats the target 1.1491, and
        no arm's loss is inside the seed range. The best arm, at 1.8 gates, is
        a threshold, and the other nine are ranks."""
        gate = self._gate()
        rows = [r for r in self._rows() if r["block"] == "vs target"]
        assert rows
        for r in rows:
            assert float(r["right_score"]) == 1.1491, r
            gap = float(r["gap"])
            assert gap > 0, r                  # the decay costs the score
            assert r["verdict"] != "noise", r  # the loss clears the gate
            assert r["verdict"] == self._verdict(gap, gate), r

    def test_the_no_decay_gaps_read_against_the_comparator_own_range(self):
        """The `vs no-decay` block reads each arm against the sweep's run of
        the SAME schedule, and that comparator carries its own seed range. A
        gap inside that range is not a rank, whatever this card's gate says.
        The schedule `0.8 to 1.0 at 200k` spans 0.1432 over the sweep's seeds,
        so most of its arms give `inside the comparator range`."""
        gate = self._gate()
        rows = [r for r in self._rows() if r["block"] == "vs no-decay"
                and r["gap"] != "-"]
        assert rows
        inside = 0
        for r in rows:
            gap, comparator = float(r["gap"]), float(r["comparator_range"])
            assert gap > 0, r                  # the decay costs the score
            if gap <= comparator:
                assert r["verdict"] == "inside the comparator range", r
                inside += 1
            else:
                assert r["verdict"] == self._verdict(gap, gate), r
        assert inside, "a wide comparator range must hold some gap"

    def test_the_decay_schedules_do_not_all_separate_from_each_other(self):
        """The review's point 5. Some arm-to-arm gaps are under the gate, so
        the report must not order the schedules."""
        rows = [r for r in self._rows() if r["block"] == "arm vs arm"]
        assert rows
        assert any(r["verdict"] == "noise" for r in rows)

    def test_the_gate_refuses_to_run_without_a_replicate(self, tmp_path):
        """A gate taken from a study with no repeated seed would not be this
        treatment's spread. The script exits rather than invent one."""
        arms = tmp_path / "arms.tsv"
        arms.write_text("# arm\ttau\tend\tema_ramp\trep_ramp\tseed\n"
                        "a\t0.9\t1.0\t100000\t10000\t1\n"
                        "b\t0.8\t1.0\t200000\t10000\t1\n")
        scores = tmp_path / "scores.csv"
        scores.write_text("arm,score\na,1.10\nb,1.30\n")
        out = subprocess.run(
            [sys.executable, str(self.SCRIPT), "--scores", str(scores),
             "--arms", str(arms), "--out", str(tmp_path / "gate.tsv")],
            capture_output=True, text=True, timeout=120)
        assert out.returncode != 0
        assert "two seeds" in out.stderr


class TestTheReferenceIsComparable:
    """This card runs NO control. Its headline is a gap against 1.1491, which
    `reports/2026-08-19_ema_momentum_k32/` measured. A gap between two studies
    is a result only when the two measure the same thing."""

    TABLE = EXP / "results" / "reference_match.tsv"
    SCRIPT = EXP / "scripts" / "reference_match.sh"
    # Every item the review asks the report to state, plus the ones that carry
    # them: the cell, the runner and the depth settings.
    WANT = {"cell", "head steps", "head seed", "head encoder", "eval",
            "align target", "backbone stop", "head runner"}

    def _rows(self):
        with open(self.TABLE, newline="") as fh:
            return list(csv.DictReader(fh, delimiter="\t"))

    def test_the_table_and_the_script_exist(self):
        assert self.SCRIPT.is_file()
        assert self.TABLE.is_file()

    def test_it_covers_every_item_the_headline_rests_on(self):
        got = {r["item"] for r in self._rows()}
        assert self.WANT <= got, sorted(self.WANT - got)

    def test_every_item_matches(self):
        for r in self._rows():
            assert r["verdict"] == "match", r

    def test_both_studies_call_one_head_runner(self):
        """Not two settings that agree. One file."""
        row = [r for r in self._rows() if r["item"] == "head runner"]
        assert row and row[0]["this_card"] == row[0]["the_sweep"]
        assert "head_eval_bb.sh" in row[0]["this_card"]

    def test_a_row_says_whether_an_artefact_or_a_script_backs_it(self):
        """The sweep's checkpoint root is deleted, so two rows rest on its
        scripts. A reader must be able to see which."""
        rows = self._rows()
        assert {r["evidence"] for r in rows} <= {"script", "artefact"}
        assert any(r["evidence"] == "artefact" for r in rows)

    def test_the_script_fails_loudly_on_a_mismatch(self):
        """A silent pass would let a protocol drift carry the headline."""
        body = self.SCRIPT.read_text()
        assert "DIFFERS" in body
        assert "exit 1" in body


class TestTheStudyKeepsOneReport:
    """`reports/REPORT_STANDARD.md`, first item: ONE canonical Markdown report
    per experiment. Supporting information lives in scripts, docstrings and
    execution logs, never in additional report files.

    The run phase writes no report. It must also not grow a second one under
    `notes/`, which is what a findings note becomes when it carries a table of
    results and a conclusion.
    """

    NOTES = EXP / "notes"
    # The notes this study carries, each one a decision or an operational
    # record and not a findings summary.
    ALLOWED = {"artefacts.md", "execution_log.md", "loss_decomposition.md",
               "search_protocol.md", "SECOND_ANSWER.md", "one_report.md"}

    def test_no_new_report_file_grew_under_notes(self):
        got = {p.name for p in self.NOTES.glob("*.md")}
        assert got == self.ALLOWED, sorted(got ^ self.ALLOWED)

    def test_the_decision_note_carries_no_result(self):
        """`one_report.md` is the decision that kept this study at one report.
        Agents kept proposing a second one. A note that starts to carry scores
        IS that second report under an allowed name, so this note names the
        one file that holds them and holds none itself."""
        body = (self.NOTES / "one_report.md").read_text()
        assert "rep_weight_decay.md" in body
        assert not re.search(r"\b\d\.\d{4}\b", body), "a score in the note"

    def test_the_findings_live_in_the_script_and_the_table(self):
        """The rank gate and the reference match state their result in the
        docstring that produces it and in the table itself."""
        gate = (EXP / "scripts" / "rank_gate.py").read_text()
        assert "WHAT THE TABLE SAID" in gate
        match = (EXP / "scripts" / "reference_match.sh").read_text()
        assert "WHAT THE TABLE SAID" in match

    def test_the_gate_table_names_its_narrowest_pass(self):
        """A threshold test says `rank` one part in a thousand over the line.
        A reader of the verdict column alone would report that as a result."""
        head = [ln for ln in (EXP / "results" / "rank_gate.tsv")
                .read_text().splitlines() if ln.startswith("#")]
        assert any("narrowest pair" in ln for ln in head), head
        assert any("threshold, not a rank" in ln for ln in head), head


class TestTheRefreshWatchStartsNothing:
    """`scripts/refresh_when_done.sh` waits for the arms in flight and then
    rebuilds the figures and the tables. It must never train, never kill and
    never commit: it runs unattended, beside other sessions, on a shared
    checkpoint root and a shared results directory."""

    SCRIPT = EXP / "scripts" / "refresh_when_done.sh"

    def test_it_exists_and_is_bounded(self):
        body = self.SCRIPT.read_text()
        assert self.SCRIPT.is_file()
        assert "DEADLINE_H" in body, "an unattended wait needs a deadline"

    def test_it_starts_no_backbone(self):
        body = self.SCRIPT.read_text()
        for forbidden in ("launch.sh", "phase1.sh", "run_arm.sh",
                          "train.py --", "lane_when_free"):
            assert forbidden not in body, forbidden

    def test_it_kills_nothing_and_commits_nothing(self):
        """Vast.ai and elisa are shared. A background `git` beside a live
        session is how two sessions lose each other's work."""
        body = self.SCRIPT.read_text()
        for forbidden in ("kill ", "pkill", "git commit", "git add",
                          "git push", "rm -rf"):
            assert forbidden not in body, forbidden

    def test_it_only_rebuilds_artefacts(self):
        body = self.SCRIPT.read_text()
        assert "make_plots.sh" in body
        assert "run_state.sh" in body
