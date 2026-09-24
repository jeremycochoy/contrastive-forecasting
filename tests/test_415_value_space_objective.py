"""Tests for #415: the value-space twin of `arm6_v2_combab_alignT`.

The card keeps the body and the input head of #414's cell and changes ONLY the
objective. The model predicts the FUTURE VALUES, the rollout runs in value
space, and the loss reads the actual values. Six groups of guards follow.

1. The objective itself. The target of depth j must be the patch the rollout
   reaches at depth j, and the rolled input must be the values the model
   predicted. A shift of one patch is invisible in the loss curve and makes
   the model predict the present.
2. The gradient. The value head, the body and the INPUT HEAD must all take
   gradient, or the card trains a decoder on a frozen random encoder.
3. The scoring path. #414's head trainer and GIFT-Eval rebuild the backbone
   and load it strictly. A checkpoint that carries the new `value_head.*`
   keys must still load, or no stop of this card can be scored.
4. The no-op. #414 runs the same `train.py` right now. Without
   `--value-space-objective` every byte of the model, the objective and the
   command line must be what it was.
5. The command line. The run refuses every flag it does not read, and the
   leg's body is #414's cell, flag for flag.
6. The ladder. Every stop is scored under B4 and A2, the checkpoints stay on
   the box's disk, and no score holds up the training.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.checkpoint import prepare_backbone_state_dict  # noqa: E402
from src.forecasting_head import (  # noqa: E402
    QUANTILE_LEVELS,
    median_quantile_index,
    patch_value_targets,
    value_patches_to_series,
    value_space_forward,
    value_space_objective,
)
from src.models import ConfigurableModel  # noqa: E402

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
EXP = REPO_ROOT / "reports" / "2026-09-23_value_space_reference"
RUN_LEG = EXP / "scripts" / "run_leg_value.sh"
HEAD_EVAL = EXP / "scripts" / "head_eval_value.sh"
RUN_SH = EXP / "run.sh"
PARENT = REPO_ROOT / "reports" / "2026-08-08_rollout_depth" / "scripts"
EVAL_LOCAL = PARENT / "eval_local.sh"
RUN_LEG_K = PARENT / "run_leg_k.sh"
RUN_ARM_412 = (REPO_ROOT / "reports" / "2026-09-06_moirai_small_size"
               / "scripts" / "run_arm.sh")

# The cell #414 trains, at the width #412 chose. #415 keeps both.
D_MODEL = 384
NUM_LAYERS = 3
NUM_ENCODER_LAYERS = 3
# The stops the card scores. 665,000 steps is one pass over the data.
STOPS = (40000, 100000, 200000, 300000, 400000, 500000, 600000, 665000)
# #414's best arm. Its command line is the body this card must match.
ARM_414 = "k3_r100_09_lr56_fix09_dec10k_cos200k"
# Where the checkpoints go when nothing names a root: the box's own disk.
BOX_RUNS = "/workspace/ckpt/cf-415"

W = 8
Q = len(QUANTILE_LEVELS)


def tiny_model(value_head=True, C=1, H=16, T_raw=64, **kw):
    """A small ConfigurableModel that runs on the CPU in milliseconds."""
    cfg = dict(
        C=C, H=H, W=W, encoder_type="gru", num_layers=2, nhead=2,
        ffn_mult=2.0, activation="gelu", depthwise_conv=3, dropout=0.0,
        rev_norm_kind="ewma", rev_norm_span=16, num_encoder_layers=2,
        enc_transformer_num_layers=1, enc_transformer_nhead=2,
        enc_transformer_use_grad_checkpoint=False,
    )
    cfg.update(kw)
    if value_head:
        cfg["value_head_quantiles"] = Q
    return ConfigurableModel(**cfg)


def norm_batch(model, B=2, C=1, T_raw=64):
    x = torch.randn(B, T_raw, C)
    return model.rev_norm(x, mode="norm")


# ---------------------------------------------------------------------------
# 1. The objective
# ---------------------------------------------------------------------------

def test_target_of_depth_0_is_the_next_patch():
    """Position t is supervised against patch t + 1, never against patch t."""
    x_norm = torch.randn(2, 64, 1)
    targets, t_valid = patch_value_targets(x_norm, W, shift=0)
    assert t_valid == 64 // W - 1
    assert targets.shape == (2, t_valid, 1, W)
    for t in range(t_valid):
        want = x_norm[:, (t + 1) * W:(t + 2) * W, 0]
        assert torch.equal(targets[:, t, 0, :], want)


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
def test_target_of_depth_j_moves_one_patch_per_depth(shift):
    """Depth j is supervised against patch t + 1 + j — the patch the rollout
    reaches after j further forecast steps."""
    x_norm = torch.randn(2, 64, 1)
    targets, t_valid = patch_value_targets(x_norm, W, shift=shift)
    assert t_valid == 64 // W - 1 - shift
    for t in range(t_valid):
        want = x_norm[:, (t + 1 + shift) * W:(t + 2 + shift) * W, 0]
        assert torch.equal(targets[:, t, 0, :], want)


def test_median_quantile_index_picks_0_5():
    assert median_quantile_index(QUANTILE_LEVELS) == 4
    assert QUANTILE_LEVELS[median_quantile_index(QUANTILE_LEVELS)] == 0.5


def test_rolled_input_is_the_predicted_values():
    """The rollout feeds the model its OWN median forecast, laid back out as
    a value series. A latent would make this a latent-space rollout."""
    B, T, C = 2, 8, 1
    v_hat = torch.randn(B, T, C, Q, W)
    series = value_patches_to_series(v_hat, median_quantile_index(QUANTILE_LEVELS))
    assert series.shape == (B, T * W, C)
    for t in range(T):
        want = v_hat[:, t, 0, 4, :]
        assert torch.equal(series[:, t * W:(t + 1) * W, 0], want)


def test_value_space_forward_shape():
    torch.manual_seed(0)
    model = tiny_model()
    x_norm = norm_batch(model)
    f_lat, o_lat, v_hat = value_space_forward(model, x_norm)
    T = 64 // W
    assert f_lat.shape == (2, T, 1, model.H)
    assert o_lat.shape == (2, T, 1, model.H)
    assert v_hat.shape == (2, T, 1, Q, W)


def test_depth_gives_one_loss_term_per_forecast_step():
    torch.manual_seed(0)
    model = tiny_model()
    x_norm = norm_batch(model)
    for depth in (0, 1, 3):
        loss, _, _, per_depth = value_space_objective(
            model, x_norm, depth=depth, reduce="sum")
        assert len(per_depth) == depth + 1
        assert torch.isfinite(loss)


def test_reduce_sum_and_mean_agree_at_depth_0():
    """Every run reproduces under either reduction at k = 0 — the contract
    `--train-rollout-reduce` already holds for the latent rollout."""
    torch.manual_seed(0)
    model = tiny_model()
    x_norm = norm_batch(model)
    model.eval()
    with torch.no_grad():
        a, _, _, _ = value_space_objective(model, x_norm, depth=0, reduce="sum")
        b, _, _, _ = value_space_objective(model, x_norm, depth=0, reduce="mean")
    assert torch.allclose(a, b)


def test_reduce_divides_the_depth_copies():
    torch.manual_seed(0)
    model = tiny_model()
    model.eval()
    x_norm = norm_batch(model)
    with torch.no_grad():
        s, _, _, terms = value_space_objective(model, x_norm, depth=2, reduce="sum")
        m, _, _, _ = value_space_objective(model, x_norm, depth=2, reduce="mean")
    assert torch.allclose(s, sum(terms))
    assert torch.allclose(m, s / 3)


def test_the_loss_reads_the_actual_values():
    """A model whose median output IS the target scores 0 on the median
    quantile, so the loss falls when the prediction approaches the values."""
    torch.manual_seed(0)
    model = tiny_model()
    model.eval()
    x_norm = norm_batch(model)
    with torch.no_grad():
        far, _, _, _ = value_space_objective(model, x_norm, depth=0)
        near, _, _, _ = value_space_objective(model, x_norm * 0.0, depth=0)
    # x_norm = 0 makes every target 0; a random head is further from 0 than
    # from nothing only by chance, so compare against a shifted series.
    with torch.no_grad():
        shifted, _, _, _ = value_space_objective(model, x_norm * 50.0, depth=0)
    assert shifted > far > 0
    assert near >= 0


# ---------------------------------------------------------------------------
# 2. The gradient
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [0, 2])
def test_gradient_reaches_the_input_head_and_the_value_head(depth):
    """The card trains the whole model on values. A rollout that detached the
    predicted patch would leave the body untrained beyond depth 0."""
    torch.manual_seed(0)
    model = tiny_model()
    x_norm = norm_batch(model)
    loss, _, _, _ = value_space_objective(model, x_norm, depth=depth)
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert any(n.startswith("encoder.") for n in grads), "input head got no gradient"
    assert any(n.startswith("transformer.layers.") for n in grads), "body got none"
    assert "value_head.weight" in grads
    assert all(torch.isfinite(g).all() for g in grads.values())


def test_no_teacher_and_no_cpc_weights_exist():
    """The card drops the teacher, the EMA and the CPC auxiliary. None of
    their parameters may be created, or the count is not the cell's."""
    model = tiny_model()
    names = set(model.state_dict())
    assert not any(n.startswith("teacher_") for n in names)
    assert not any(n.startswith("cpc_w1") for n in names)


# ---------------------------------------------------------------------------
# 3. The scoring path
# ---------------------------------------------------------------------------

def test_the_eval_loads_a_value_trained_checkpoint():
    """#414's head trainer and GIFT-Eval rebuild the backbone WITHOUT the
    value head and load it strictly. `prepare_backbone_state_dict` must drop
    `value_head.*` the way it drops `cpc_w1.*` and `teacher_*`."""
    torch.manual_seed(0)
    trained = tiny_model(value_head=True)
    sd = trained.state_dict()
    assert any(k.startswith("value_head.") for k in sd)

    downstream = tiny_model(value_head=False)
    prepared = prepare_backbone_state_dict(sd)
    assert not any(k.startswith("value_head.") for k in prepared)
    downstream.load_state_dict(prepared)  # strict — raises if a key is left


# ---------------------------------------------------------------------------
# 4. The no-op for #414
# ---------------------------------------------------------------------------

def test_the_value_head_is_off_by_default():
    plain = tiny_model(value_head=False)
    assert plain.value_head is None
    assert not any(k.startswith("value_head") for k in plain.state_dict())
    with pytest.raises(RuntimeError):
        plain.value_forward(torch.zeros(1, 2, 1, plain.H))


def test_a_plain_model_state_dict_is_unchanged_by_the_new_kwarg():
    torch.manual_seed(7)
    a = tiny_model(value_head=False)
    torch.manual_seed(7)
    b = ConfigurableModel(
        C=1, H=16, W=W, encoder_type="gru", num_layers=2, nhead=2,
        ffn_mult=2.0, activation="gelu", depthwise_conv=3, dropout=0.0,
        rev_norm_kind="ewma", rev_norm_span=16, num_encoder_layers=2,
        enc_transformer_num_layers=1, enc_transformer_nhead=2,
        enc_transformer_use_grad_checkpoint=False,
        value_head_quantiles=0,
    )
    assert list(a.state_dict()) == list(b.state_dict())


# ---------------------------------------------------------------------------
# The trainer flag
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_py():
    """train.py as a module, for its parser, its refusal and its schedule."""
    spec = importlib.util.spec_from_file_location("train_py_415", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_trainer(*extra):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run(
        [sys.executable, str(TRAIN_PY), "--device", "cpu",
         "--weight-decay", "0.1", "--total-steps", "1", *extra],
        capture_output=True, text=True, env=env, timeout=600)


# The smallest value-space run the CPU trains in seconds.
TINY_VALUE_RUN = (
    "--value-space-objective",
    "--t-raw", "128", "--n-channels", "1", "--d-model", "16",
    "--n-heads", "2", "--num-layers", "1",
    "--num-encoder-layers", "1", "--enc-num-layers", "1",
    "--enc-nhead", "2", "--batch-size", "2",
    "--mix-ratio", "1.0", "--synth-kind", "periodic",
    "--freq-emb-dim", "0", "--seasonality-emb-dim", "0",
    "--log-every", "1", "--save-every", "1000000")


# The flags the review of PR #416 named, and the other knobs of the terms
# the card removes. Each one is given at its own default, or at a value that
# changes nothing, so a check that read VALUES would let most of them pass.
# The run refuses the NAME: it does not read the flag, so a line that names
# one was copied from a contrastive run, and it is not this card's twin.
UNREAD_FLAGS = [
    ["--loss-shape", "cosine_similarity_batch_no_time_neg"],
    ["--rep-loss-weight", "1.0"],
    ["--tau", "0.07"],
    ["--align-target", "student"],
    ["--ema-tau", "0.99"],
    ["--ema-tau-end", "1.0"],
    ["--ema-tau-ramp-steps", "100000"],
    ["--grad-clip", "1.0"],
    ["--align-loss-weight", "0.0"],
    ["--align-moco-loss-weight", "0.0"],
    ["--cpc-infonce-weight", "0.0"],
    ["--cpc-infonce-negs", "matched"],
    ["--cpc-k-steps", "12"],
    ["--sigreg-embedding-weight", "0.1"],
    ["--sigreg-n-chunk", "2048"],
    ["--tau-rep", "1.0"],
    ["--pred-loss-weight", "1.0"],
    ["--rep-loss-weight-ramp-steps", "10000"],
    ["--shard-loss-on-batch"],
    ["--ema-embedding"],
    ["--moco-rep-keys"],
    ["--learnable-tau"],
]


@pytest.mark.parametrize("flags", UNREAD_FLAGS, ids=lambda f: f[0])
def test_the_value_run_refuses_a_flag_it_does_not_read(train_py, flags):
    clash = train_py.value_space_conflicts(
        ["--value-space-objective", "--weight-decay", "0.1", *flags])
    assert clash == [flags[0]]


def test_every_listed_flag_exists(train_py):
    """A misspelt name in the list would allow nothing, and the real flag
    would then be refused on the leg's own line."""
    dests = {a.dest for a in train_py.build_parser()._actions}
    assert train_py.VALUE_SPACE_FLAGS <= dests


def test_an_abbreviated_flag_is_refused_too(train_py):
    """argparse takes a unique prefix, so `--grad-c 1` sets the clip."""
    clash = train_py.value_space_conflicts(
        ["--value-space-objective", "--weight-decay", "0.1", "--grad-c", "1"])
    assert clash == ["--grad-clip"]


@pytest.mark.parametrize("flags", [
    ["--ema-embedding", "--ema-encoder"],
    ["--loss-shape", "cosine_similarity_batch_rep_only"],
    ["--grad-clip", "1.0"],
    ["--sigreg-embedding"],
])
def test_the_trainer_refuses_before_it_builds_anything(tmp_path, flags):
    """The refusal runs first in `main`, so a copied line costs seconds and
    leaves nothing on disk: not even the save directory."""
    never = tmp_path / "never"
    r = run_trainer("--value-space-objective", "--save-dir", str(never),
                    *flags)
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert "--value-space-objective" in out and flags[0] in out
    assert not never.exists()


def test_a_warmup_cannot_reach_the_run(train_py):
    """The trainer has no warmup, so no flag names one, and argparse stops a
    command line that names an unknown flag."""
    parser = train_py.build_parser()
    assert not [a.dest for a in parser._actions if "warmup" in a.dest]
    with pytest.raises(SystemExit):
        train_py.parse_args(
            ["--weight-decay", "0.1", "--warmup-steps", "1000"])


def test_the_trainer_defaults_give_no_clip_and_no_warmup(train_py):
    """The Moirai recipe clips nothing and has no warmup here. Unnamed, the
    trainer clips nothing, and a schedule starts at its peak rate."""
    args = train_py.parse_args(["--weight-decay", "0.1"])
    assert args.grad_clip is None
    assert args.lr_final is None
    peak, final, steps = 5e-4, 1e-6, 665000
    assert train_py.cosine_lr(0, peak, final, steps) == peak
    assert train_py.cosine_lr(1, peak, final, steps) == pytest.approx(
        peak, rel=1e-9)
    assert train_py.cosine_lr(steps, peak, final, steps) == pytest.approx(
        final)


def test_the_trainer_accepts_a_value_space_depth(tmp_path):
    """`--train-rollout-depth` rides on the terms that tie f to h. The value
    objective is a new consumer, so the refusal must not fire on it."""
    r = run_trainer(*TINY_VALUE_RUN, "--train-rollout-depth", "3",
                    "--save-dir", str(tmp_path), "--run-name", "cf415_smoke")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "value-space objective" in r.stdout


def test_a_value_leg_resumes_its_own_checkpoint(tmp_path):
    """The ladder resumes every stop from the one below it. The value head is
    a new parameter in the same AdamW group, so the optimizer state has to
    round-trip with it — this project has lost days to a resume that did not.
    The leg names the same schedule at every stop, so the resume must read
    it back with no warning.
    """
    def leg(total, *extra):
        return subprocess.run(
            [sys.executable, str(TRAIN_PY), "--device", "cpu",
             "--weight-decay", "0.1", "--total-steps", str(total),
             *TINY_VALUE_RUN, "--train-rollout-depth", "2",
             "--lr", "5e-4", "--lr-final", "1e-6", "--lr-cosine-steps", "8",
             "--save-dir", str(tmp_path), "--run-name", "leg", *extra],
            capture_output=True, text=True, timeout=600,
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))

    first = leg(2)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    ckpt = tmp_path / "leg_final.pth"
    assert ckpt.is_file()

    resumed = leg(4, "--resume", str(ckpt))
    assert resumed.returncode == 0, resumed.stdout[-3000:] + resumed.stderr[-3000:]
    assert "Resumed from" in resumed.stdout
    assert "Restored optimizer" in resumed.stdout
    assert "[lr] WARNING" not in resumed.stdout
    # The resumed leg trains steps 3 and 4, not 1 and 2.
    assert "[      3]" in resumed.stdout and "[      1]" not in resumed.stdout


def test_the_contrastive_path_still_refuses_a_depth_with_no_consumer():
    """The refusal #373 added stays live for every run that is not this one."""
    r = run_trainer("--train-rollout-depth", "3",
                    "--loss-shape", "cosine_similarity_batch_rep_only")
    assert r.returncode != 0
    assert "has no term to enter" in r.stdout + r.stderr


# ---------------------------------------------------------------------------
# The launcher
#
# The runner builds the trainer command line from variables, so the SOURCE
# text says little. Its dry run resolves that line and creates nothing, so
# every guard below reads what would actually reach the trainer — the check
# #412's `run_arm.sh` makes at run time, made once here.
# ---------------------------------------------------------------------------


def leg_command_line(*args):
    r = subprocess.run(["bash", str(RUN_LEG), *(args or ("40000",))],
                       capture_output=True, text=True,
                       env=dict(os.environ, CF415_DRY_RUN="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    line = [l for l in r.stdout.splitlines() if l.startswith("Command line: ")]
    assert len(line) == 1, r.stdout
    return line[0]


def test_every_script_parses():
    for script in (EXP / "scripts" / "paths.sh", RUN_LEG, HEAD_EVAL, RUN_SH):
        assert script.is_file(), f"missing {script}"
        r = subprocess.run(["bash", "-n", str(script)], capture_output=True)
        assert r.returncode == 0, r.stderr.decode()


def test_the_leg_runs_the_value_objective_at_the_cells_shape():
    line = leg_command_line()
    assert "--value-space-objective" in line
    assert f"--d-model {D_MODEL}" in line
    assert f"--num-layers {NUM_LAYERS}" in line
    assert f"--num-encoder-layers {NUM_ENCODER_LAYERS}" in line
    assert "--train-rollout-depth 3" in line


def test_the_leg_drops_every_term_the_card_removes():
    """Not one of these may reach the trainer, or the twin is not a twin.
    The trainer refuses them too — this catches the runner first."""
    line = leg_command_line()
    for flag in ("--ema-embedding", "--ema-encoder", "--align-loss-weight",
                 "--align-target", "--sigreg-embedding", "--sigreg-encoding",
                 "--moco-rep-keys", "--moco-negatives", "--rep-loss-weight",
                 "--cpc-infonce-weight", "--grad-clip", "--loss-shape",
                 "--tau ", "--learnable-tau"):
        assert flag not in line, f"{flag} reaches the trainer"


def test_the_leg_carries_the_moirai_recipe():
    """lr 1e-3, weight decay 0.1, betas (0.9, 0.98), flat, no warmup."""
    line = leg_command_line()
    assert "--lr 1e-3" in line
    assert "--weight-decay 0.1" in line
    assert "--adam-beta1 0.9" in line
    assert "--adam-beta2 0.98" in line
    assert "--lr-final" not in line
    assert "--lr-cosine-steps" not in line


def test_the_leg_keeps_the_data_and_the_clock_of_414():
    """The twin must see the same corpus and the same rows per step, or
    665,000 steps is not one pass in both runs."""
    line = leg_command_line()
    assert "--hf-repo jeremycochoy/gift-pretrain-full-4096" in line
    assert "--hf-path small_v1" in line
    assert "--t-raw 4096" in line
    assert "--n-channels 1" in line
    assert "--batch-size 64" in line
    assert "--seed 20260520" in line


def test_the_last_stop_gets_its_own_checkpoint():
    """665,000 is not a multiple of the 20,000 save cadence, so the periodic
    save never lands it. --extra-save-steps has to."""
    line = leg_command_line("665000")
    assert "--total-steps 665000" in line
    assert "--extra-save-steps 665000" in line


def test_one_stop_list_feeds_the_ladder_and_the_score():
    """`head_eval_value.sh` validates the stop it is given, so a second list
    in `run.sh` would let a stop train and then fail to score."""
    paths = (EXP / "scripts" / "paths.sh").read_text()
    listed = re.search(r'CF415_STOPS="\$\{CF415_STOPS:-([^}]*)\}"', paths)
    assert listed, "paths.sh names no stop list"
    assert [int(s) for s in listed.group(1).split()] == list(STOPS)
    assert 'STOPS="$CF415_STOPS"' in RUN_SH.read_text()


def test_the_ladder_scores_every_stop_it_trains():
    r = subprocess.run(["bash", str(RUN_SH)], capture_output=True, text=True,
                       env=dict(os.environ, CF415_DRY_RUN="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    for stop in STOPS:
        assert f"--total-steps {stop}" in r.stdout, f"{stop} never trains"
        assert f"head stop={stop}" in r.stdout, f"{stop} never scores"


def test_the_head_and_the_eval_are_414s():
    """The scoring path must be the one #414 uses, so the numbers compare."""
    r = subprocess.run(["bash", str(HEAD_EVAL), "40000"],
                       capture_output=True, text=True,
                       env=dict(os.environ, CF415_DRY_RUN="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "head_eval_bb.sh" in r.stdout
    assert "2026-08-08_rollout_depth" in r.stdout
    assert f"--d-model {D_MODEL}" in r.stdout
    assert "enc=student" in r.stdout


def test_the_head_refuses_a_step_count_that_is_not_a_stop():
    r = subprocess.run(["bash", str(HEAD_EVAL), "123000"],
                       capture_output=True, text=True,
                       env=dict(os.environ, CF415_DRY_RUN="1"))
    assert r.returncode != 0
    assert "not a stop" in r.stdout + r.stderr
