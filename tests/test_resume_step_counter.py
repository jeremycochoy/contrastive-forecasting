"""#382 review round 2: --resume must restore the step counter.

The `run_arm.sh` launcher in `experiments/2026-07-28_loss_term_isolation/`
picks the newest `<run>_*k.pth` after a vast.ai preemption and passes it via
`--resume`. If `train.py --resume` silently starts from step 0, we lose the
prior work up to that checkpoint and burn a compute-day on a duplicate
warm-up — the failure the reviewer flagged for round-2.

Runs `experiments/2026-04-27_freq-embedding/scripts/train.py` in-subprocess
twice (100 → 200 steps), then verifies:

  (a) the resume banner reports `at step 100` — start_step is populated from
      the checkpoint dict, not left at 0.
  (b) the first post-resume log line reports a step > 100 — if start_step
      were reset to 0, the log lines would run 50..100 (log-every=50) rather
      than 150..200.
  (c) a step-200 checkpoint lands under the branched `_r2` run name (safe_run_name
      protects the pre-existing checkpoints) with `optimizer_state_dict.step == 200`.
  (d) the AdamW `exp_avg` in the step-200 optimizer differs from the step-100
      one — evidence that training actually continued rather than starting
      from a re-initialised optimizer at step 100.

Uses `--mix-ratio 1.0` (pure synth) so no HF token / network call is made,
and the smallest CPU arch we can build (d_model=8, one layer both sides,
batch=2, T_raw=64) so the two subprocess runs finish in seconds.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts" / "train.py"


def _assert_train_deps_available() -> None:
    """Fail loudly (not skip) so a broken env can't hide a silent green."""
    for mod in ("torch", "numpy", "datasets"):
        try:
            __import__(mod)
        except ImportError as e:
            pytest.fail(
                f"train.py dep {mod!r} not importable: {e}. "
                "This test MUST run in CI — do not silently skip; either "
                "install the dep or drop this test from the CI selector."
            )


def _tiny_train_cmd(save_dir: Path, run_name: str, total_steps: int,
                    resume: str | None):
    """The smallest arch that still exercises the resume path end-to-end."""
    cmd = [
        sys.executable, "-u", str(TRAIN_PY),
        "--device", "cpu",
        "--total-steps", str(total_steps),
        "--save-every", "100",
        "--traj-save-every", "100",
        "--batch-size", "2",
        "--lr", "1e-3",
        "--weight-decay", "0.1",
        "--save-dir", str(save_dir),
        "--run-name", run_name,
        "--mix-ratio", "1.0",       # pure synth ⇒ no HF token, no network
        "--synth-kind", "periodic",
        "--t-raw", "64",
        "--n-channels", "1",
        "--d-model", "8",
        "--n-heads", "2",
        "--num-encoder-layers", "1",
        "--num-layers", "1",
        "--log-every", "50",
        "--seed", "20260728",
        "--loss-shape", "cosine_similarity_batch_full_hh_negs_xshh_allt",
        "--tau", "0.10",
        # Required by argparse (no default); pure-synth path never resolves them.
        "--hf-repo", "none",
        "--hf-path", "none",
    ]
    if resume is not None:
        cmd.extend(["--resume", resume])
    return cmd


def _run(cmd, env, cwd, timeout=600):
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(cwd), timeout=timeout)


def _first_adam_exp_avg(optim_state: dict) -> torch.Tensor | None:
    """AdamW keeps per-parameter `exp_avg` in `optimizer_state_dict.state`.
    Grab the first non-empty tensor for a diff-check."""
    state = optim_state["optimizer_state_dict"]["state"]
    for _, s in state.items():
        ea = s.get("exp_avg")
        if ea is not None and ea.numel() > 0:
            return ea.detach().cpu().flatten().clone()
    return None


def test_resume_restores_step_counter(tmp_path):
    _assert_train_deps_available()

    save_dir = tmp_path / "runs"
    save_dir.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    # ── Run 1 ── 100 steps of the tiny backbone; produces `<run>_step100.pth`.
    run_name = "resumetest"
    r1 = _run(_tiny_train_cmd(save_dir, run_name, 100, resume=None),
              env, tmp_path)
    if r1.returncode != 0:
        pytest.fail(
            f"first train.py run rc={r1.returncode}\n"
            f"stdout tail:\n{r1.stdout[-2000:]}\n"
            f"stderr tail:\n{r1.stderr[-2000:]}"
        )

    ckpt = save_dir / f"{run_name}_step100.pth"
    optim_ckpt = save_dir / f"{run_name}_step100_optimizer.pth"
    assert ckpt.exists() and optim_ckpt.exists(), (
        f"first run did not produce {ckpt.name} / {optim_ckpt.name}. "
        f"Files: {sorted(x.name for x in save_dir.iterdir())}"
    )
    optim_before = torch.load(optim_ckpt, map_location="cpu",
                              weights_only=False)
    assert optim_before["step"] == 100, (
        f"pre-resume optimizer file logged step={optim_before['step']}, "
        f"expected 100 (save was triggered at total_steps=100)"
    )

    # ── Run 2 ── resume from step 100, extend to 200. safe_run_name branches
    # to `<run>_r2` because save_dir already holds `<run>_*.pth`.
    r2 = _run(_tiny_train_cmd(save_dir, run_name, 200, resume=str(ckpt)),
              env, tmp_path)
    if r2.returncode != 0:
        pytest.fail(
            f"resumed train.py run rc={r2.returncode}\n"
            f"stdout tail:\n{r2.stdout[-2000:]}\n"
            f"stderr tail:\n{r2.stderr[-2000:]}"
        )

    # (a) The resume banner MUST report `at step 100`. If load_training_state
    # silently returned step=0, the banner would too — and start_step would
    # start the loop from 1 instead of 101.
    assert "Resumed from" in r2.stdout, (
        f"expected `Resumed from ...` banner in stdout; got:\n"
        f"{r2.stdout[-1500:]}"
    )
    assert "at step 100" in r2.stdout, (
        f"resume banner did not report `at step 100` — start_step was NOT "
        f"restored from the checkpoint. stdout tail:\n{r2.stdout[-1500:]}"
    )

    # (b) The first post-resume step-log line MUST be > 100. With log-every=50,
    # a resumed 100→200 range prints [150] and [200]; a broken reset would
    # print [50] and [100].
    step_lines = [ln for ln in r2.stdout.splitlines()
                  if ln.lstrip().startswith("[") and "loss=" in ln]
    assert step_lines, (
        f"no `[<step>] loss=…` lines in resumed run stdout — training loop "
        f"never entered:\n{r2.stdout[-1500:]}"
    )
    first_step = int(step_lines[0].lstrip().split("]")[0].lstrip("[").strip())
    assert first_step > 100, (
        f"first step logged post-resume is {first_step}; expected > 100. "
        f"start_step likely reset to 0 — the resumed run repeated steps 1..100."
    )

    # (c) safe_run_name branched to `<run>_r2`; the max step-tagged file must
    # be exactly 200 (total_steps), with a matching optimizer companion
    # whose `step` field reads 200.
    resumed_ckpts = [p for p in save_dir.glob(f"{run_name}_r2_step*.pth")
                     if "_optimizer" not in p.stem]
    assert resumed_ckpts, (
        f"resumed run wrote no `{run_name}_r2_step*.pth` files. safe_run_name "
        f"did not branch, or the loop never saved. Files: "
        f"{sorted(x.name for x in save_dir.iterdir())}"
    )
    resumed_steps = sorted(int(p.stem.rsplit("_step", 1)[1])
                           for p in resumed_ckpts)
    assert resumed_steps[-1] == 200, (
        f"resumed run wrote step-checkpoints {resumed_steps}; expected 200 "
        f"as the max (the final save at total_steps)."
    )

    optim_after_path = save_dir / f"{run_name}_r2_step200_optimizer.pth"
    assert optim_after_path.exists(), (
        f"missing {optim_after_path.name}. Files: "
        f"{sorted(x.name for x in save_dir.iterdir())}"
    )
    optim_after = torch.load(optim_after_path, map_location="cpu",
                             weights_only=False)
    assert optim_after["step"] == 200, (
        f"resumed optimizer file logged step={optim_after['step']}, expected 200"
    )

    # (d) AdamW `exp_avg` must have moved between the two checkpoints. If the
    # optimizer were re-initialised at resume (all state zeroed and re-stepped
    # from 0), and by any coincidence produced identical exp_avg after 100
    # more steps, this would flag it — the tighter guarantee is "steps
    # actually happened AND used the restored optimizer state."
    ea_before = _first_adam_exp_avg(optim_before)
    ea_after = _first_adam_exp_avg(optim_after)
    assert ea_before is not None and ea_after is not None, (
        "no AdamW exp_avg tensors in either optimizer_state_dict.state — "
        "cannot verify that training actually progressed"
    )
    assert ea_before.shape == ea_after.shape, (
        f"exp_avg shape drift: {ea_before.shape} vs {ea_after.shape}"
    )
    assert not torch.equal(ea_before, ea_after), (
        "AdamW exp_avg identical between step-100 and step-200 optimizer "
        "checkpoints — training either did not step post-resume or the "
        "optimizer was fully re-initialised (both are resume failures)."
    )
