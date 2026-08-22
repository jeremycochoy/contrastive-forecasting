"""Tests for #409's path from a backbone to a score, and for its AUC gate.

`tests/test_409_rep_weight_decay.py` holds the objective.
`tests/test_409_launcher_shape.py` holds the arms and the backbone leg. This
file holds the four things the implementation review asked for.

1. A path from a backbone to a score. `head_eval.sh` trains this card's
   30,000-step student head on one arm's backbone and runs that head's 97
   GIFT-Eval configs. `collect.sh` writes one score for each arm.
2. An entry point for the eight arms. `launch.sh` deals them over elisa's two
   cards, and `phase1.sh` re-fires a leg that crashed.
3. The AUC watch, wired. `auc_guard.sh` reads the losses CSV of a running leg
   and stops the leg that lost the contrastive task.
4. The loss-by-term formula of `docs/rep_loss_weight_schedule.md`, which must
   close for a `--train-rollout-depth` run under the `sum` reduction. This
   card's own arms are such runs.
"""

from __future__ import annotations

import os
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

ARMS = ("ctrl_s20", "ctrl_s24", "dec0_s20", "dec0_s24",
        "flr05_s20", "flr05_s24", "flr02_s20", "dec0T_s20")
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
        out = script_dry_run(HEAD_EVAL, "dec0_s20", STOP)
        assert out.returncode == 0, out.stderr
        assert f"steps={HEAD_STEPS}" in out.stdout
        assert "seed=20260722" in out.stdout
        assert "enc=student" in out.stdout

    def test_the_dry_run_names_the_tag_the_eval_and_the_score(self):
        out = script_dry_run(HEAD_EVAL, "dec0_s20", STOP)
        assert out.returncode == 0, out.stderr
        tag = study_out(f"cf409_tag dec0_s20 {STOP} {HEAD_STEPS}")
        assert tag in out.stdout
        assert study_out("cf409_eval_dir dec0_s20 " + tag) in out.stdout
        assert study_out(f"cf409_score_file dec0_s20 {STOP}") in out.stdout

    def test_another_head_budget_is_refused(self):
        """One budget, or two tags of this study name two protocols."""
        assert script_dry_run(HEAD_EVAL, "dec0_s20", STOP, 15000).returncode != 0
        assert script_dry_run(HEAD_EVAL, "nosucharm", STOP).returncode != 0

    def test_the_tag_carries_the_arm_the_stop_and_the_budget(self):
        tag = study_out(f"cf409_tag dec0_s20 {STOP} {HEAD_STEPS}")
        assert tag == "dec0_s20_bb40k_h30k_student"
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
        out = script_dry_run(HEAD_EVAL, "dec0_s20", 400, env=trial)
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
        full.update(env or {})
        return subprocess.run(["bash", str(COLLECT)], capture_output=True,
                              text=True, env=full, cwd=str(REPO_ROOT),
                              timeout=180)

    def test_one_row_for_each_scored_arm(self, tmp_path):
        res = self._results(tmp_path, {"ctrl_s20": "1.0862\n",
                                       "dec0_s20": "2.4100\n"})
        out = self._run(res)
        assert out.returncode == 0, out.stderr
        rows = (res / "scores.csv").read_text().strip().splitlines()
        assert rows[0].startswith("arm,")
        assert len(rows) == 3
        body = {r.split(",")[0]: r for r in rows[1:]}
        assert body["ctrl_s20"].endswith(",1.0862")
        assert body["dec0_s20"].endswith(",2.4100")

    def test_the_row_carries_the_arm_definition(self, tmp_path):
        """A reader compares two arms by their floor, ramp, seed and target,
        so the table must not need the arms file beside it."""
        res = self._results(tmp_path, {"dec0T_s20": "1.2000\n"})
        assert self._run(res).returncode == 0
        head, row = (res / "scores.csv").read_text().strip().splitlines()
        cell = dict(zip(head.split(","), row.split(",")))
        assert cell["rep_end"] == "0.0"
        assert cell["ramp"] == "10000"
        assert cell["seed"] == "20260520"
        assert cell["align_target"] == "teacher"
        assert cell["stop"] == str(STOP)
        assert cell["head_steps"] == str(HEAD_STEPS)

    def test_an_empty_score_file_is_not_a_zero(self, tmp_path):
        """An eval killed between opening and writing leaves one, and 0.0
        would be the best GM-Relative MASE the project ever recorded."""
        res = self._results(tmp_path, {"ctrl_s20": "1.0862\n", "dec0_s20": ""})
        assert self._run(res).returncode == 0
        rows = (res / "scores.csv").read_text().strip().splitlines()
        assert len(rows) == 2
        assert "dec0_s20" not in (res / "scores.csv").read_text()

    def test_a_foreign_score_file_is_not_a_row(self, tmp_path):
        res = self._results(tmp_path, {"ctrl_s20": "1.0862\n"})
        (res / "score_a08_bb40k_h30k_student.txt").write_text("1.1782\n")
        out = self._run(res)
        assert out.returncode == 0
        assert "a08" not in (res / "scores.csv").read_text()

    def test_it_writes_the_auc_verdict_of_every_run(self, tmp_path):
        """The card asks for the contrastive AUC of every run, and the step
        of any loss."""
        res = self._results(tmp_path, {"ctrl_s20": "1.0862\n"})
        root = tmp_path / "root"
        for arm, aucs in (("ctrl_s20", [0.97] * 3000),
                          ("dec0_s20", [0.97] * 1500 + [0.50] * 1500)):
            csv_path = Path(study_out(
                f"cf409_losses_csv {arm} {STOP}",
                {"CF409_ROOT": str(root)}))
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            losses_csv(csv_path, aucs)
        out = self._run(res, {"CF409_ROOT": str(root)})
        assert out.returncode == 0, out.stderr
        table = (res / "auc_verdicts.tsv").read_text()
        assert table.splitlines()[0].startswith("run\t")
        assert "ctrl_s20" in table and "held" in table
        assert "dec0_s20" in table and "lost" in table


# --- 2. the eight arms on the two cards ----------------------------------


class TestTheLauncher:

    def test_every_arm_is_dealt_exactly_once(self):
        out = script_dry_run(LAUNCH, env={"CF409_GPU_COUNT": "2",
                                          "GPUS": "0 1"})
        assert out.returncode == 0, out.stderr
        dealt = [line.split()[1] for line in out.stdout.splitlines()
                 if line.startswith("arm ")]
        assert sorted(dealt) == sorted(ARMS)

    def test_the_two_cards_take_four_arms_each(self):
        out = script_dry_run(LAUNCH, env={"CF409_GPU_COUNT": "2",
                                          "GPUS": "0 1"})
        lanes = {}
        for line in out.stdout.splitlines():
            if line.startswith("arm "):
                gpu = line.split("gpu=")[1].split()[0]
                lanes.setdefault(gpu, []).append(line.split()[1])
        assert sorted(lanes) == ["0", "1"]
        assert [len(v) for v in lanes.values()] == [4, 4]

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
        out = script_dry_run(PHASE1, env={"ARMS": "dec0_s20"})
        assert out.returncode == 0, out.stderr
        assert "arm dec0_s20" in out.stdout
        assert "head dec0_s20" in out.stdout


class TestTheCheckoutTheStudyNeeds:
    """A machine bootstrapped from a stale branch trains eight copies of the
    control and logs nothing unusual. The launcher asks first."""

    def _checkout(self, tmp_path, trainer=True, gap=True, token=True):
        wt = tmp_path / "wt"
        train = wt / "experiments" / "2026-04-27_freq-embedding" / "scripts"
        runner = wt / "reports" / "2026-08-08_rollout_depth" / "scripts"
        train.mkdir(parents=True)
        runner.mkdir(parents=True)
        (train / "train.py").write_text(
            "--rep-loss-weight-end\n" if trainer else "# stale\n")
        (runner / "run_leg_k.sh").write_text(
            "GAP_ARGS\n" if gap else "# stale\n")
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

    def test_an_empty_hf_token_is_refused(self, tmp_path):
        """The anonymous rate limit idles the card at about 20 percent use."""
        wt = self._checkout(tmp_path, token=False)
        out = study_call(f'cf409_check_checkout "{wt}"')
        assert out.returncode != 0
        assert "token" in out.stderr

    def test_this_checkout_carries_both_pieces(self):
        """The HF token is gitignored, so a worktree has none. The other two
        are on the branch."""
        trainer = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
                   / "scripts" / "train.py").read_text()
        runner = (REPO_ROOT / "reports" / "2026-08-08_rollout_depth"
                  / "scripts" / "run_leg_k.sh").read_text()
        assert "--rep-loss-weight-end" in trainer
        assert "GAP_ARGS" in runner


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
        full.update({"ARMS": "dec0_s20",
                     "CF409_RUN_ARM": str(run_arm),
                     "CF409_HEAD_EVAL": str(head),
                     "CF409_RESULTS": str(tmp_path / "results"),
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
        assert "head dec0_s20" not in out.stdout
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
    arm would otherwise burn about 30,000 dead steps."""

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

    def _guard(self, base, aucs, pid, env=None):
        base.mkdir(parents=True, exist_ok=True)
        root = base / "root"
        csv_path = Path(study_out(f"cf409_losses_csv dec0_s20 {STOP}",
                                  {"CF409_ROOT": str(root)}))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        losses_csv(csv_path, aucs)
        full = dict(os.environ)
        full.update({"CF409_ROOT": str(root),
                     "CF409_RESULTS": str(base / "results"),
                     "CF409_AUC_POLL": "1",
                     "CF409_AUC_WINDOW": "10",
                     "CF409_AUC_WARMUP": "0"})
        full.update(env or {})
        out = subprocess.run(
            ["bash", str(AUC_GUARD), "dec0_s20", str(STOP), str(pid)],
            capture_output=True, text=True, env=full, cwd=str(REPO_ROOT),
            timeout=120)
        return out, base / "results"

    def test_it_is_called_from_the_leg(self):
        assert "auc_guard.sh" in RUN_ARM.read_text()

    def test_a_collapsed_leg_is_stopped(self, tmp_path):
        pid = self._victim()
        try:
            out, res = self._guard(tmp_path, [0.97] * 60 + [0.42] * 60, pid)
            assert out.returncode == 1, out.stdout + out.stderr
            assert not self._alive(pid)
            note = (res / "collapsed_dec0_s20.txt").read_text()
            assert "dec0_s20" in note
            assert "lost" in note
        finally:
            self._reap(pid)

    def test_a_healthy_leg_runs_on(self, tmp_path):
        pid = self._victim(6)
        try:
            out, res = self._guard(tmp_path, [0.97] * 120, pid)
            assert out.returncode == 0, out.stdout + out.stderr
            assert not (res / "collapsed_dec0_s20.txt").exists()
        finally:
            self._reap(pid)

    def test_the_guard_stops_when_the_leg_stops(self, tmp_path):
        pid = self._victim()
        self._reap(pid)
        out, res = self._guard(tmp_path, [0.97] * 120, pid)
        assert out.returncode == 0
        assert not (res / "collapsed_dec0_s20.txt").exists()

    def test_the_warmup_holds_the_gate_off_the_start_of_a_run(self, tmp_path):
        """Without it the gate stops every arm in its first minute: the AUC of
        a fresh run starts near 0.5 and climbs."""
        climb = [0.50] * 120 + [0.97] * 80
        window = {"CF409_AUC_WINDOW": "500"}
        cold = self._victim()
        try:
            out, _ = self._guard(tmp_path / "cold", climb, cold, window)
            assert out.returncode == 1
            assert not self._alive(cold)
        finally:
            self._reap(cold)
        warm = self._victim(6)
        try:
            out, res = self._guard(tmp_path / "warm", climb, warm,
                                   {**window, "CF409_AUC_WARMUP": "120"})
            assert out.returncode == 0, out.stdout + out.stderr
            assert not (res / "collapsed_dec0_s20.txt").exists()
        finally:
            self._reap(warm)

    def test_the_card_warmup_is_inside_the_ramp(self, tmp_path):
        """Every arm holds a weight above 0.9 through the warmup, so no arm
        can collapse from the decay before the gate turns on."""
        warmup = int(study_out('printf %s "$CF409_AUC_WARMUP"'))
        assert 0 < warmup <= 2000
        assert float(study_out(f"cf409_rep_w_at dec0_s20 {warmup}")) >= 0.9


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
    """The report rebuilds the loss by term from the CSV columns. This
    card's arms run k = 3 under `sum`, so the total holds FOUR copies of
    L_align and the `l_align` column holds the depth-0 copy alone."""

    K = 3
    ALIGN_W = 1.0
    REP_W = 0.7

    def _loss(self, reduce="sum"):
        f, o, rollouts = _rollout_latents(self.K, seed=5)
        terms = {}
        loss = contrastive_latent_loss(
            (f, o), False, _spec(train_rollout_depth=self.K,
                                 train_rollout_reduce=reduce),
            align_loss_weight=self.ALIGN_W, align_target="student",
            rollout_latents=rollouts, train_rollout_depth=self.K,
            train_rollout_reduce=reduce,
            rep_loss_weight=self.REP_W, term_out=terms)
        return float(loss), terms, rollout_cos_error(f, o, rollouts)

    def test_the_depth_zero_formula_does_not_close_at_k_three(self):
        loss, terms, _ = self._loss()
        naive = self.REP_W * terms["l_rep"] + self.ALIGN_W * terms["l_align"]
        assert loss != pytest.approx(naive, rel=1e-3)

    def test_the_column_is_the_depth_zero_copy(self):
        _, terms, errors = self._loss()
        assert terms["l_align"] == pytest.approx(2.0 * errors[0])

    def test_the_summed_total_closes_over_the_cos_err_columns(self):
        """`l_align = 2 * cos_err_d0` under the student target, so the k + 1
        copies read off the `cos_err_d*` columns."""
        loss, terms, errors = self._loss()
        assert len(errors) == self.K + 1
        closed = (self.REP_W * terms["l_rep"]
                  + self.ALIGN_W * 2.0 * sum(errors))
        assert loss == pytest.approx(closed, rel=1e-6)

    def test_the_mean_total_divides_by_k_plus_one(self):
        loss, terms, errors = self._loss("mean")
        closed = (self.REP_W * terms["l_rep"]
                  + self.ALIGN_W * 2.0 * sum(errors) / (self.K + 1))
        assert loss == pytest.approx(closed, rel=1e-6)

    def test_the_doc_states_the_rollout_case(self):
        body = DOC.read_text()
        formula = body.split("loss = rep_w * l_rep")[1]
        assert "--train-rollout-depth" in formula
        assert "cos_err_d" in formula
        assert "k + 1" in formula
