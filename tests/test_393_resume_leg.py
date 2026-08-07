"""#393: a resumed leg is the protocol, so a resumed leg has to be tested.

Every stop past 40k is a fresh `train.py` process started with `--resume`.
Three things have to hold for the ladder to mean what the issue says, and
none of them is visible from a single from-scratch leg:

  (a) α keeps climbing across the process boundary. `ema_tau_at_step` is
      called with the loop's `step`; a per-process counter would restart
      every leg at α = 0.9 and the fixed anchor would buy nothing.
  (b) `--total-steps` is an ABSOLUTE global step, not a number of extra
      steps. If it were additional, the 100k leg would land at 140k.
  (c) the optimizer state comes back — the issue says "resume from that
      checkpoint with the saved optimizer state".

Two `train.py` processes on CPU: 6 steps, then resume to 12, both with
`--ema-tau-ramp-steps 30`. The ramp is deliberately longer than either
budget, so a budget-relative or per-process α is a different number at
every step and cannot pass by coincidence.

`--mix-ratio 1.0` (pure synth) keeps the network and the HF token out of
it; d_model=8 / T_raw=64 keeps both processes to a few seconds.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.models import ema_tau_at_step

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")

LEG_ONE = 6
LEG_TWO = 12
RAMP = 30
TAU_START, TAU_END = 0.9, 1.0


def anchored_alpha(step: int) -> float:
    """α the anchored schedule owes at a GLOBAL step."""
    return TAU_START + (TAU_END - TAU_START) * min(step / RAMP, 1.0)


def leg_cmd(save_dir: Path, total_steps: int, resume: str | None):
    """One leg of the ladder, shrunk to the smallest arch that still runs.

    Mirrors `run_leg.sh`: a fixed-anchor ramp, one EMA teacher, resume from
    the previous leg's checkpoint into a FRESH save dir.
    """
    cmd = [
        sys.executable, "-u", str(TRAIN_PY),
        "--device", "cpu",
        "--total-steps", str(total_steps),
        "--save-every", str(LEG_ONE),
        "--batch-size", "2",
        "--lr", "1e-3", "--weight-decay", "0.1",
        "--save-dir", str(save_dir), "--run-name", "legtest",
        "--mix-ratio", "1.0", "--synth-kind", "periodic",
        "--t-raw", "64", "--n-channels", "1",
        "--d-model", "8", "--n-heads", "2",
        "--num-encoder-layers", "1", "--num-layers", "1",
        "--log-every", "1", "--seed", "20260520", "--tau", "0.10",
        "--loss-shape", "cosine_similarity_batch_split_pred_rep",
        "--ema-embedding", "--ema-encoder",
        "--ema-tau", str(TAU_START), "--ema-tau-end", str(TAU_END),
        "--ema-tau-ramp-steps", str(RAMP),
        "--hf-repo", "none", "--hf-path", "none",
    ]
    if resume is not None:
        cmd.extend(["--resume", resume])
    return cmd


def run_leg(save_dir: Path, total_steps: int, resume: str | None, tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(leg_cmd(save_dir, total_steps, resume), env=env,
                          capture_output=True, text=True, cwd=str(tmp_path),
                          timeout=900)
    if proc.returncode != 0:
        pytest.fail(f"train.py rc={proc.returncode} (total_steps={total_steps}, "
                    f"resume={resume})\nstdout:\n{proc.stdout[-2500:]}\n"
                    f"stderr:\n{proc.stderr[-2500:]}")
    return proc


def alpha_by_step(save_dir: Path) -> dict[int, float]:
    """{global step: logged α} from the leg's own losses CSV."""
    csvs = sorted(save_dir.glob("*_losses.csv"))
    assert len(csvs) == 1, f"expected one losses CSV in {save_dir}, got {csvs}"
    with open(csvs[0], newline="") as fh:
        return {int(r["step"]): float(r["ema_tau"])
                for r in csv.DictReader(fh) if r.get("ema_tau")}


def logged_steps(stdout: str) -> list[int]:
    return [int(ln.lstrip().split("]")[0].lstrip("[").strip())
            for ln in stdout.splitlines()
            if ln.lstrip().startswith("[") and "loss=" in ln]


def first_exp_avg(optim_state: dict):
    for _, s in optim_state["optimizer_state_dict"]["state"].items():
        ea = s.get("exp_avg")
        if ea is not None and ea.numel() > 0:
            return ea.detach().cpu().flatten().clone()
    return None


@pytest.fixture(scope="module")
def two_legs(tmp_path_factory):
    """Leg 1 (0→6) then leg 2 (6→12), each in its own save dir.

    A fresh dir per leg is what `run_leg.sh` does, and it is load-bearing:
    train.py branches `--run-name` to `<name>_r2` when the save dir already
    holds `<name>_*.pth`, so reusing one dir would rename every artefact
    past the first leg.
    """
    tmp_path = tmp_path_factory.mktemp("resume_leg")
    one, two = tmp_path / "leg_6", tmp_path / "leg_12"
    one.mkdir()
    two.mkdir()
    p1 = run_leg(one, LEG_ONE, None, tmp_path)
    ckpt = one / f"legtest_{LEG_ONE // 1000}k.pth"
    assert ckpt.exists(), (
        f"leg 1 wrote no {ckpt.name}; files: "
        f"{sorted(p.name for p in one.iterdir())}")
    p2 = run_leg(two, LEG_TWO, str(ckpt), tmp_path)
    return {"dir1": one, "dir2": two, "proc1": p1, "proc2": p2, "ckpt1": ckpt}


class TestAlphaAcrossTheSeam:
    """(a) — the assumption the whole experiment rests on."""

    def test_the_resumed_leg_starts_where_the_first_stopped(self, two_legs):
        steps = sorted(alpha_by_step(two_legs["dir2"]))
        assert steps == list(range(LEG_ONE + 1, LEG_TWO + 1)), (
            f"resumed leg logged steps {steps}; expected "
            f"{LEG_ONE + 1}..{LEG_TWO}. A per-process counter would log "
            f"1..{LEG_TWO - LEG_ONE}.")

    def test_alpha_follows_the_anchor_at_the_global_step(self, two_legs):
        for leg in ("dir1", "dir2"):
            for step, alpha in alpha_by_step(two_legs[leg]).items():
                assert alpha == pytest.approx(anchored_alpha(step), abs=1e-9), (
                    f"{leg} step {step}: α={alpha}, anchored curve owes "
                    f"{anchored_alpha(step)}")

    def test_alpha_is_continuous_and_still_climbing(self, two_legs):
        """The seam: α at the resumed leg's first step must sit one ramp
        increment above the last α of the first leg, not back at 0.9."""
        before = alpha_by_step(two_legs["dir1"])[LEG_ONE]
        after = alpha_by_step(two_legs["dir2"])[LEG_ONE + 1]
        assert after > before, f"α fell across the seam: {before} -> {after}"
        assert after - before == pytest.approx(
            (TAU_END - TAU_START) / RAMP, abs=1e-9), (
            f"α jumped {after - before} across the seam; one anchored step "
            f"is {(TAU_END - TAU_START) / RAMP}")

    def test_the_resumed_leg_never_restarts_the_ramp(self, two_legs):
        assert min(alpha_by_step(two_legs["dir2"]).values()) > TAU_START, (
            "the resumed leg logged α = the start value, so its ramp "
            "restarted from step 0")

    def test_the_schedule_matches_the_shared_function(self, two_legs):
        """The driver and the report quote `ema_tau_at_step`; the trainer
        must be applying that same curve."""
        for step, alpha in alpha_by_step(two_legs["dir2"]).items():
            assert alpha == pytest.approx(
                ema_tau_at_step(step, LEG_TWO, TAU_START, TAU_END, RAMP),
                abs=1e-9)


class TestTotalStepsIsAbsolute:
    """(b) — `--total-steps 12` on a leg resumed at 6 means stop at 12."""

    def test_the_resumed_leg_stops_at_the_target(self, two_legs):
        assert max(logged_steps(two_legs["proc2"].stdout)) == LEG_TWO

    def test_it_runs_the_remainder_not_the_whole_budget(self, two_legs):
        steps = logged_steps(two_legs["proc2"].stdout)
        assert min(steps) == LEG_ONE + 1
        assert len(steps) == LEG_TWO - LEG_ONE, (
            f"resumed leg ran {len(steps)} steps; expected "
            f"{LEG_TWO - LEG_ONE}. Additional-step semantics would run "
            f"{LEG_TWO}.")

    def test_the_target_checkpoint_lands_under_the_expected_name(self, two_legs):
        """`run_leg.sh` and `eval_stop.sh` both look for
        `<name>_<N>k.pth` at the stop. A renamed artefact fails the leg."""
        expected = two_legs["dir2"] / f"legtest_{LEG_TWO // 1000}k.pth"
        assert expected.exists(), (
            f"no {expected.name} after the resumed leg; files: "
            f"{sorted(p.name for p in two_legs['dir2'].iterdir())}")


class TestOptimizerStateComesBack:
    """(c) — `--resume` alone has to restore AdamW, not just the weights."""

    def test_the_banner_reports_the_restore(self, two_legs):
        out = two_legs["proc2"].stdout
        assert "Restored optimizer from" in out, (
            f"no optimizer-restore banner:\n{out[-2000:]}")
        assert f"at step {LEG_ONE}" in out, (
            f"resume banner did not report step {LEG_ONE}:\n{out[-2000:]}")

    def test_the_moments_carried_over_and_kept_moving(self, two_legs):
        before = torch.load(
            two_legs["dir1"] / f"legtest_{LEG_ONE // 1000}k_optimizer.pth",
            map_location="cpu", weights_only=False)
        after = torch.load(
            two_legs["dir2"] / f"legtest_{LEG_TWO // 1000}k_optimizer.pth",
            map_location="cpu", weights_only=False)
        assert before["step"] == LEG_ONE and after["step"] == LEG_TWO
        ea_before, ea_after = first_exp_avg(before), first_exp_avg(after)
        assert ea_before is not None and ea_after is not None, (
            "no AdamW exp_avg in either optimizer file — cannot tell a "
            "restored optimizer from a fresh one")
        assert not torch.equal(ea_before, ea_after)


class TestOnePassSurvivesTheResume:
    """The step cap is 'steps in one pass over the dataset'. A cell is 6+
    processes, so the cap only means that if each leg picks the stream up
    where the previous one left it instead of re-showing the first rows.

    These legs run pure synth (no HF token, no network), so the counter
    they exercise is `synth_rows_consumed`. It is the same counter,
    incremented and checkpointed on the same lines as `hf_rows_consumed`,
    and it is the one that moves when every row is synthetic.
    """

    def optimizer_state(self, two_legs, leg, step):
        return torch.load(
            two_legs[leg] / f"legtest_{step // 1000}k_optimizer.pth",
            map_location="cpu", weights_only=False)

    def test_the_row_counter_is_checkpointed(self, two_legs):
        state = self.optimizer_state(two_legs, "dir1", LEG_ONE)
        assert state["synth_rows_consumed"] == LEG_ONE * 2, (
            f"row counter after {LEG_ONE} steps of batch 2 x 1 channel is "
            f"{state['synth_rows_consumed']}, not the absolute position")

    def test_the_counter_carries_across_the_seam(self, two_legs):
        after = self.optimizer_state(two_legs, "dir2", LEG_TWO)
        assert after["synth_rows_consumed"] == LEG_TWO * 2, (
            "the resumed leg restarted the row counter; the stream would "
            "re-show the head of the dataset and one pass would no longer "
            "be one pass")

    def test_the_trainer_feeds_the_counter_to_the_stream(self):
        """`skip_rows=hf_rows_consumed` is the line that turns the restored
        counter into an offset into the dataset. All three synth-kind
        branches must carry it, or one recipe silently re-reads the head."""
        src = TRAIN_PY.read_text()
        assert src.count("skip_rows=hf_rows_consumed") == 3, (
            "not every dataloader branch is offset by the restored row "
            "counter")
        assert 'restored.get("hf_rows_consumed", 0)' in src

    def test_a_checkpoint_without_the_counter_reconstructs_it(self):
        """The fallback matters: pre-#356 checkpoints carry 0. `start_step *
        hf_rows_per_step` is the same absolute offset the counter holds."""
        src = TRAIN_PY.read_text()
        assert "hf_rows_consumed = start_step * hf_rows_per_step" in src


class TestTheStreamOpensAtTheRowOffset:
    """The other half of the answer: `skip_rows` has to position the stream
    at an absolute row, not be quietly ignored. Stubs the shard listing and
    the parquet reader so this runs with no network."""

    def loader(self, monkeypatch, skip_rows, counts):
        from src.dataloader import HFStreamingLoader
        loader = HFStreamingLoader("repo/none", batch_size=2, C=1,
                                   skip_rows=skip_rows)
        shards = [f"s{i}.parquet" for i in range(len(counts))]
        opened = {}
        monkeypatch.setattr(loader, "_list_shard_files", lambda: shards)
        monkeypatch.setattr(loader, "_shard_row_counts", lambda _: counts)

        def fake_stream(remaining, within_shard_skip=0):
            opened["shards"] = list(remaining)
            opened["within_shard_skip"] = within_shard_skip
            return iter(())

        monkeypatch.setattr(loader, "_pyarrow_stream_from_shards", fake_stream)
        list(loader._iter_stream_with_fast_skip())
        return opened

    def test_zero_starts_at_the_head(self, monkeypatch):
        got = self.loader(monkeypatch, 0, [100, 100, 100])
        assert got["shards"][0] == "s0.parquet"
        assert got["within_shard_skip"] == 0

    def test_a_resumed_offset_starts_mid_dataset(self, monkeypatch):
        """250 rows in: shard 2, 50 rows past its head. A stream that
        restarted at the head would report shard 0 / offset 0."""
        got = self.loader(monkeypatch, 250, [100, 100, 100, 100])
        assert got["shards"][0] == "s2.parquet"
        assert got["within_shard_skip"] == 50

    def test_the_offset_lands_on_a_shard_boundary(self, monkeypatch):
        got = self.loader(monkeypatch, 200, [100, 100, 100, 100])
        assert got["shards"][0] == "s2.parquet"
        assert got["within_shard_skip"] == 0

    def test_past_the_end_wraps_into_a_second_pass(self, monkeypatch):
        """This is what the step cap exists to prevent: past one pass the
        stream wraps modulo the dataset and starts repeating rows."""
        got = self.loader(monkeypatch, 450, [100, 100, 100, 100])
        assert got["shards"][0] == "s0.parquet"
        assert got["within_shard_skip"] == 50
