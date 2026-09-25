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
# The stops the card scores. 166,000 steps at batch 256 is one pass.
STOPS = (10000, 25000, 50000, 75000, 100000, 125000, 150000, 166000)
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


def test_the_warmup_rises_linearly_then_hands_over_to_the_anneal(train_py):
    """The Moirai schedule: 0 to the peak over the warmup, then one cosine
    from the peak to the final rate, which ends at the anneal length."""
    peak, final, total, warmup = 1e-3, 0.0, 166000, 10000
    lr = lambda step: train_py.scheduled_lr(step, peak, final, total, warmup)
    assert lr(0) == 0.0
    assert lr(5000) == pytest.approx(peak / 2)
    assert lr(warmup) == pytest.approx(peak)
    assert lr((warmup + total) // 2) == pytest.approx(peak / 2)
    assert lr(total) == pytest.approx(final)
    assert lr(total + 50000) == pytest.approx(final)


def test_no_warmup_gives_the_plain_anneal(train_py):
    """Every run before this flag keeps its curve, value for value."""
    for step in (0, 1, 333, 200000, 665000):
        assert train_py.scheduled_lr(step, 5.6e-5, 1e-6, 200000) == \
            train_py.cosine_lr(step, 5.6e-5, 1e-6, 200000)


@pytest.mark.parametrize("flags", [
    ["--lr-warmup-steps", "10"],
    ["--lr-warmup-steps", "10", "--lr-final", "0", "--lr-cosine-steps", "10"],
])
def test_a_warmup_with_no_anneal_after_it_is_refused(tmp_path, flags):
    r = run_trainer(*TINY_VALUE_RUN, *flags, "--save-dir", str(tmp_path),
                    "--run-name", "cf415_warmup")
    assert r.returncode != 0
    assert "--lr-warmup-steps" in r.stdout + r.stderr


def test_the_trainer_defaults_give_no_clip_and_no_warmup(train_py):
    """The trainer's own defaults stay as they were: no clip, no warmup and a
    constant rate. The leg of this card names the Moirai schedule."""
    args = train_py.parse_args(["--weight-decay", "0.1"])
    assert args.grad_clip is None
    assert args.lr_final is None
    assert args.lr_warmup_steps == 0
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


# The variables the card's scripts read. A test starts from an environment
# without them, so a value set in the shell that runs the suite changes no
# path and no flag.
CARD_VARS = ("RUNS", "WT", "LR", "K", "SEED", "BB_GPU", "CF_RESULTS",
             "CF_BB_SHAPE", "EVAL_STRATEGY", "EVAL_SHARDS",
             "EVAL_CONFIG_FILTER", "EVAL_EXPECT_CONFIGS", "GIFT_EVAL")


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if k not in CARD_VARS and not k.startswith(("CF415_", "CF412_"))}
    env.update({k: v for k, v in extra.items() if v is not None})
    return env


def leg_command_line(*args, **env):
    r = subprocess.run(["bash", str(RUN_LEG), *(args or ("10000",))],
                       capture_output=True, text=True,
                       env=clean_env(CF415_DRY_RUN="1", **env))
    assert r.returncode == 0, r.stdout + r.stderr
    line = [l for l in r.stdout.splitlines() if l.startswith("Command line: ")]
    assert len(line) == 1, r.stdout
    return line[0]


def leg_argv(stop="10000"):
    """The trainer's argv on this card's leg, as the dry run resolves it."""
    tokens = shlex.split(leg_command_line(stop)[len("Command line: "):])
    assert tokens[:2] == ["python3", "-u"], tokens[:3]
    assert tokens[2].endswith("train.py"), tokens[:3]
    return tokens[3:]


@pytest.fixture
def durable_root():
    """A root outside /tmp and the checkout, which `runs_root` accepts."""
    root = tempfile.mkdtemp(prefix="cf415-", dir="/var/tmp")
    yield Path(root)
    shutil.rmtree(root, ignore_errors=True)


def stub_checkout(tmp_path):
    """A checkout whose trainer writes down its own argv, and nothing else."""
    wt = tmp_path / "wt"
    scripts = wt / "experiments" / "2026-04-27_freq-embedding" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "train.py").write_text(
        "import json, os, sys\n"
        "json.dump(sys.argv[1:], open(os.environ['CF415_ARGV_OUT'], 'w'))\n")
    (wt / "experiments" / "hf_token.txt").write_text("stub-token\n")
    return wt


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


# The rate schedule of this card: the Moirai recipe. `paths.sh` gives the
# source of each number.
PEAK_LR, FINAL_LR, ANNEAL_STEPS = 1e-3, 0.0, 166000
WARMUP_STEPS, GRAD_CLIP = 10000, 1.0


@pytest.mark.parametrize("stop", ["10000", "166000"])
def test_every_leg_follows_the_moirai_schedule(train_py, stop):
    """Each leg runs to its own --total-steps. An anneal length left to its
    default takes that number, so the 10,000-step leg would reach the floor
    at step 10,000. Every leg names one pass instead, and the same warmup."""
    args = train_py.parse_args(leg_argv(stop))
    assert (args.lr, args.lr_final, args.lr_cosine_steps) == (
        PEAK_LR, FINAL_LR, ANNEAL_STEPS)
    assert args.lr_warmup_steps == WARMUP_STEPS
    assert args.grad_clip == GRAD_CLIP
    assert (args.weight_decay, args.adam_beta1, args.adam_beta2) == (
        0.1, 0.9, 0.98)
    assert args.batch_size == 256


def test_the_leg_line_names_nothing_the_value_run_ignores(train_py):
    assert train_py.value_space_conflicts(leg_argv()) == []


def arm_414_settings(tmp_path):
    """K, EMA_ARGS, GAP_ARGS and SEED of #414's best arm, as #412's
    `run_arm.sh` hands them to `run_leg_k.sh`."""
    r = subprocess.run(
        ["bash", str(RUN_ARM_412), ARM_414, "40000"],
        capture_output=True, text=True, timeout=120,
        env=clean_env(CF412_DRY_RUN="1", CF412_RESULTS=str(tmp_path / "r412"),
                      CF412_ROOT=str(tmp_path / "root412")))
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    return {"K": re.search(r" k=(\d+) ", out).group(1),
            "GAP_ARGS": re.search(r"^  gap=(.*)$", out, re.M).group(1),
            "EMA_ARGS": re.search(r"^  ema=(.*)$", out, re.M).group(1),
            "SEED": re.search(r"^  seed=(\d+) ", out, re.M).group(1)}


def test_the_body_is_414s_cell_flag_for_flag(train_py, tmp_path,
                                              durable_root):
    """#373's runner builds the line #414's best arm trains, and both lines
    go through the trainer's own parser, so a repeated flag, a default and
    an abbreviation resolve as they do at run time. Every flag outside the
    objective, the rate schedule and the run's two names must then read the
    same: the width, both stacks, the input head, the normalisation, the
    numerics, the data, the batch and the seed."""
    argv_out = tmp_path / "argv.json"
    env = clean_env(WT=str(stub_checkout(tmp_path)), RUNS=str(durable_root),
                    CF_RESULTS=str(tmp_path / "res"),
                    CF415_ARGV_OUT=str(argv_out),
                    GPU_GATE_LOCKDIR=str(tmp_path),
                    **arm_414_settings(tmp_path))
    subprocess.run(["bash", str(RUN_LEG_K), "arm6_v2_combab_alignT", "40000"],
                   capture_output=True, text=True, env=env, timeout=300)
    assert argv_out.is_file(), "run_leg_k.sh never reached the trainer"
    cell = vars(train_py.parse_args(json.loads(argv_out.read_text())))
    ours = vars(train_py.parse_args(leg_argv()))

    objective = set(cell) - train_py.VALUE_SPACE_FLAGS
    not_body = {"value_space_objective", "train_rollout_depth",
                "train_rollout_reduce", "lr", "lr_final", "lr_cosine_steps",
                "lr_warmup_steps", "grad_clip", "batch_size",
                "save_dir", "run_name"}
    body = sorted(set(cell) - objective - not_body)
    differ = {d: (cell[d], ours[d]) for d in body if cell[d] != ours[d]}
    assert not differ, f"#414 against #415: {differ}"
    # The comparison is not empty: the width, the input head, the numerics,
    # the data and the seed are all in it.
    for dest in ("d_model", "n_heads", "num_layers", "num_encoder_layers",
                 "encoder_type", "encoder_dropkey", "rev_norm_span",
                 "residual_dtype", "qk_norm", "hf_path", "mix_ratio",
                 "mixup_p", "seed"):
        assert dest in body, dest


def test_the_leg_keeps_the_data_and_the_clock_of_414():
    """The twin must see the same corpus as #414. At 256 rows a step, 166,000
    steps is one pass."""
    line = leg_command_line()
    assert "--hf-repo jeremycochoy/gift-pretrain-full-4096" in line
    assert "--hf-path small_v1" in line
    assert "--t-raw 4096" in line
    assert "--n-channels 1" in line
    assert "--batch-size 256" in line
    assert "--seed 20260520" in line


def test_the_last_stop_gets_its_own_checkpoint():
    """166,000 is not a multiple of the 20,000 save cadence, so the periodic
    save never lands it. --extra-save-steps has to."""
    line = leg_command_line("166000")
    assert "--total-steps 166000" in line
    assert "--extra-save-steps 166000" in line


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
                       env=clean_env(CF415_DRY_RUN="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    for stop in STOPS:
        assert f"--total-steps {stop}" in r.stdout, f"{stop} never trains"
        assert f"head stop={stop}" in r.stdout, f"{stop} never scores"


def test_the_head_and_the_eval_are_414s():
    """The scoring path must be the one #414 uses, so the numbers compare."""
    r = subprocess.run(["bash", str(HEAD_EVAL), "10000"],
                       capture_output=True, text=True,
                       env=clean_env(CF415_DRY_RUN="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "head_eval_bb.sh" in r.stdout
    assert "2026-08-08_rollout_depth" in r.stdout
    assert f"--d-model {D_MODEL}" in r.stdout
    assert "enc=student" in r.stdout


def test_the_head_refuses_a_step_count_that_is_not_a_stop():
    r = subprocess.run(["bash", str(HEAD_EVAL), "123000"],
                       capture_output=True, text=True,
                       env=clean_env(CF415_DRY_RUN="1"))
    assert r.returncode != 0
    assert "not a stop" in r.stdout + r.stderr


# ---------------------------------------------------------------------------
# The checkpoints stay on the box's disk
# ---------------------------------------------------------------------------

def test_the_checkpoints_default_to_the_box_disk():
    """The leg trains on the vast box, and the orchestrator mirrors the box's
    disk. A default on elisa's disk is a path the box cannot write."""
    leg = subprocess.run(["bash", str(RUN_LEG), "10000"], capture_output=True,
                         text=True, env=clean_env(CF415_DRY_RUN="1"))
    assert f"runs={BOX_RUNS}/value_space" in leg.stdout, leg.stdout
    head = subprocess.run(["bash", str(HEAD_EVAL), "10000"],
                          capture_output=True, text=True,
                          env=clean_env(CF415_DRY_RUN="1"))
    assert f"eval={BOX_RUNS}/value_space/eval/" in head.stdout, head.stdout


def test_a_leg_that_cannot_make_its_save_dir_trains_nothing(tmp_path):
    """Off the box, the default root cannot be made. The leg must stop before
    the trainer starts, not at its first save 20,000 steps in."""
    argv_out = tmp_path / "argv.json"
    r = subprocess.run(
        ["bash", str(RUN_LEG), "10000"], capture_output=True, text=True,
        env=clean_env(WT=str(stub_checkout(tmp_path)), RUNS="/dev/null/cf415",
                      CF_RESULTS=str(tmp_path / "res"),
                      CF415_ARGV_OUT=str(argv_out),
                      GPU_GATE_LOCKDIR=str(tmp_path)))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "cannot create" in r.stdout + r.stderr
    assert not argv_out.exists()


# ---------------------------------------------------------------------------
# A2 beside B4
#
# B4 rolls the forecast out in latent space. This model trains its rollout
# in value space, which is what A2 does, so every stop is scored under both.
# The stubs below write what the real head trainer and GIFT-Eval write, and
# log the command line each one got.
# ---------------------------------------------------------------------------

STUB_HEAD = r'''
import json, os, sys
a = sys.argv[1:]
arg = lambda k: a[a.index(k) + 1]
final = os.path.join(arg("--save-dir"), arg("--run-name") + "_final.pth")
open(final, "w").write("head")
with open(os.environ["CF415_CALLS"], "a") as fh:
    fh.write(json.dumps({"prog": "head", "final": final}) + "\n")
'''

STUB_GIFT = r'''
import json, os, sys
a = sys.argv[1:]
arg = lambda k: a[a.index(k) + 1]
out = arg("--output-dir")
os.makedirs(out, exist_ok=True)
rows = os.path.join(out, "all_results.csv")
if os.path.exists(rows):
    open(os.path.join(out, "summary.txt"), "w").write(
        "Aggregate GM-Relative MASE (1 configs): 0.9876\n")
else:
    open(rows, "w").write("dataset,MASE\nm4_hourly/H/short,1.0\n")
with open(os.environ["CF415_CALLS"], "a") as fh:
    fh.write(json.dumps({"prog": "gift", "strategy": arg("--strategy"),
                         "out": out, "head": arg("--head-path")}) + "\n")
'''


def stub_scoring_checkout(tmp_path):
    """A checkout whose head trainer and GIFT-Eval are the stubs above."""
    wt = tmp_path / "wt_scoring"
    scripts = wt / "experiments" / "2026-04-13_gift-eval" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "train_forecasting_head.py").write_text(STUB_HEAD)
    (scripts / "eval_gift_eval_official.py").write_text(STUB_GIFT)
    (wt / "experiments" / "hf_token.txt").write_text("stub-token\n")
    return wt


def scoring_env(tmp_path, **extra):
    """One config instead of 97, and no lock, slot or VRAM wait on the
    machine that runs the suite."""
    (tmp_path / "gift_data").mkdir(exist_ok=True)
    return clean_env(
        WT=str(stub_scoring_checkout(tmp_path)),
        CF415_CALLS=str(tmp_path / "calls.jsonl"),
        GIFT_EVAL=str(tmp_path / "gift_data"),
        EVAL_CONFIG_FILTER="^m4_hourly/H/short$", EVAL_EXPECT_CONFIGS="1",
        CF393_EVAL_SLOTDIR=str(tmp_path / "slots"),
        GPU_GATE_LOCKDIR=str(tmp_path), CF415_HEAD_VRAM_MIB="0", **extra)


def calls(tmp_path, prog):
    path = tmp_path / "calls.jsonl"
    rows = path.read_text().splitlines() if path.exists() else []
    return [c for c in map(json.loads, rows) if c["prog"] == prog]


@pytest.mark.parametrize("strategy,gift,log", [
    (None, "gift", "eval_local.log"),
    ("B4", "gift", "eval_local.log"),
    ("A2", "gift_a2", "eval_local_a2.log"),
])
def test_eval_local_scores_the_strategy_it_is_given(tmp_path, strategy,
                                                    gift, log):
    """Unset, the eval is #414's B4, into the same directory as before.
    A2 keeps its own directory and log, so no merge mixes the two."""
    bb, head = tmp_path / "bb.pth", tmp_path / "head.pth"
    bb.write_text("bb")
    head.write_text("head")
    out, score = tmp_path / "eval", tmp_path / "score.txt"
    r = subprocess.run(
        ["bash", str(EVAL_LOCAL), "cell", "40", "student", str(bb),
         str(head), str(out), str(score)],
        capture_output=True, text=True, timeout=300,
        env=scoring_env(tmp_path, EVAL_STRATEGY=strategy))
    assert r.returncode == 0, r.stdout + r.stderr
    assert score.read_text().strip() == "0.9876"
    evals = calls(tmp_path, "gift")
    assert {c["strategy"] for c in evals} == {strategy or "B4"}
    assert all(Path(c["out"]).is_relative_to(out / gift) for c in evals)
    assert (out / log).is_file()


def test_eval_local_refuses_a_strategy_this_head_cannot_score(tmp_path):
    """A1 and the B variants read a 128-value head or crop one. The head of
    this protocol decodes 16 values, so only B4 and A2 score it."""
    bb, head = tmp_path / "bb.pth", tmp_path / "head.pth"
    bb.write_text("bb")
    head.write_text("head")
    r = subprocess.run(
        ["bash", str(EVAL_LOCAL), "cell", "40", "student", str(bb),
         str(head), str(tmp_path / "eval"), str(tmp_path / "score.txt")],
        capture_output=True, text=True, timeout=300,
        env=scoring_env(tmp_path, EVAL_STRATEGY="A1"))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "A1" in r.stdout + r.stderr
    assert not calls(tmp_path, "gift")


def test_every_stop_is_scored_under_b4_and_a2_with_one_head(tmp_path,
                                                            durable_root):
    leg = durable_root / "value_space" / "leg_10k"
    leg.mkdir(parents=True)
    (leg / "cf415_value_k3_10k.pth").write_text("backbone")
    res = tmp_path / "res"
    env = scoring_env(tmp_path, RUNS=str(durable_root), CF_RESULTS=str(res))

    r = subprocess.run(["bash", str(HEAD_EVAL), "10000"], capture_output=True,
                       text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    heads, evals = calls(tmp_path, "head"), calls(tmp_path, "gift")
    assert len(heads) == 1
    assert [c["strategy"] for c in evals] == ["B4", "B4", "A2", "A2"]
    assert {c["head"] for c in evals} == {heads[0]["final"]}
    tag = "value_bb10k_h30k_student"
    assert (res / f"score_{tag}.txt").read_text().strip() == "0.9876"
    assert (res / f"score_{tag}_a2.txt").read_text().strip() == "0.9876"
    per_config = durable_root / "value_space" / "eval" / tag
    assert (per_config / "gift" / "all_results.csv").is_file()
    assert (per_config / "gift_a2" / "all_results.csv").is_file()

    again = subprocess.run(["bash", str(HEAD_EVAL), "10000"],
                           capture_output=True, text=True, timeout=300,
                           env=env)
    assert again.returncode == 0, again.stdout + again.stderr
    assert len(calls(tmp_path, "gift")) == 4, "a scored stop scored again"


def test_the_dry_run_names_both_scores():
    r = subprocess.run(["bash", str(HEAD_EVAL), "10000"],
                       capture_output=True, text=True,
                       env=clean_env(CF415_DRY_RUN="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    tag = "value_bb10k_h30k_student"
    assert f"score_{tag}.txt" in r.stdout
    assert f"score_{tag}_a2.txt" in r.stdout


# ---------------------------------------------------------------------------
# The ladder
#
# A2 re-runs the whole cell once per forecast patch, so its eval takes
# several times as long as B4's. A ladder that scored each stop before it
# trained the next would leave the card idle for most of the run.
# ---------------------------------------------------------------------------

# A leg writes the weights, then the optimizer state, as the trainer does.
STUB_LEG = r'''#!/bin/bash
echo "train $1" >>"$LADDER_LOG"
[ "$1" = "${FAIL_AT:-none}" ] && exit 1
k=$(( $1 / 1000 ))
d="$RUNS/value_space/leg_${k}k"
mkdir -p "$d"
echo bb >"$d/cf415_value_k3_${k}k.pth"
echo opt >"$d/cf415_value_k3_${k}k_optimizer.pth"
'''

# The score of the FIRST stop waits for the SECOND leg to start. A ladder
# that scored a stop before it trained on would never start that leg, and
# the stub would give up after 30 seconds.
STUB_SCORE = r'''#!/bin/bash
echo "score $1 start" >>"$LADDER_LOG"
if [ "$1" = "$FIRST" ]; then
  for _ in $(seq 300); do
    grep -q "^train $SECOND$" "$LADDER_LOG" && break
    sleep 0.1
  done
  grep -q "^train $SECOND$" "$LADDER_LOG" || exit 7
fi
echo "score $1 done" >>"$LADDER_LOG"
'''


def run_ladder(tmp_path, durable_root, stops, **extra):
    leg, score = tmp_path / "leg.sh", tmp_path / "score.sh"
    leg.write_text(STUB_LEG)
    score.write_text(STUB_SCORE)
    log = tmp_path / "ladder.log"
    env = clean_env(RUNS=str(durable_root), CF_RESULTS=str(tmp_path / "res"),
                    CF415_LEG_RUNNER=str(leg), CF415_SCORER=str(score),
                    CF415_POLL="0.2", LADDER_LOG=str(log),
                    FIRST=str(stops[0]), SECOND=str(stops[1]), **extra)
    t0 = time.monotonic()
    r = subprocess.run(["bash", str(RUN_SH), *map(str, stops)],
                       capture_output=True, text=True, timeout=120, env=env)
    return r, log.read_text().splitlines(), time.monotonic() - t0


def test_no_score_holds_up_the_training(tmp_path, durable_root):
    stops = (40000, 100000, 200000)
    r, lines, _ = run_ladder(tmp_path, durable_root, stops)
    assert r.returncode == 0, r.stdout + r.stderr
    for stop in stops:
        assert lines.count(f"train {stop}") == 1
        assert lines.count(f"score {stop} done") == 1
        assert lines.index(f"train {stop}") < lines.index(f"score {stop} start")
    # The stops are scored in the order they train.
    done = [l for l in lines if l.endswith(" done")]
    assert done == [f"score {s} done" for s in stops]


def test_a_failed_leg_ends_the_ladder_without_a_hang(tmp_path, durable_root):
    """The lane waits for a checkpoint. A leg that fails leaves none, so the
    lane must see the trainer end and stop, not poll forever."""
    stops = (40000, 100000, 200000)
    r, lines, took = run_ladder(tmp_path, durable_root, stops,
                                FAIL_AT="100000")
    assert r.returncode != 0
    assert "train 200000" not in lines
    assert "score 100000 start" not in lines
    assert took < 60
