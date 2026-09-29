"""Tests for #421: --gru-input-bound, the bound on the values the GRU reads.

The NaN of the #421z run came from the size-128 GRU patch encoder: on patches
with values above about 20, its backward pass multiplied the gradient by 1e7
to 1e13, and the value-space rollout compounded it to 1e21 (the dump of step
15392). With the bound c the GRU reads c * tanh(x / c), and the skip layer
keeps x. Five groups:

1. Off, nothing changes: no buffer, the same forward, the same run.
2. On, the GRU never reads a value above c, the skip layer reads the raw
   values, and the gradient stays finite on inputs of any size.
3. The checkpoint records c, and the eval and the loaders rebuild it.
4. The trainer: the refusals, a run with the bound, and a resume of a
   checkpoint from before the bound.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.checkpoint import (gru_input_bound_of,  # noqa: E402
                            load_backbone_from_checkpoint)
from src.encoders import GRUEncoder, create_encoder  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
EVAL_PY = (REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
           / "eval_gift_eval_official.py")
SIZES = (8, 16, 32, 64, 128)
BOUND = 10.0


def model(bound=0.0, **kw):
    torch.manual_seed(0)
    cfg = dict(C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
               ffn_mult=4.0, dropout=0.0, rev_norm_kind="meanstd",
               rev_norm_skip_leading_zeros=True, num_encoder_layers=1,
               freq_emb_dim=3, num_freqs=15, seasonality_emb_dim=3,
               value_head_quantiles=9, multi_patch_sizes=SIZES,
               gru_input_bound=bound,
               enc_transformer_use_grad_checkpoint=False)
    cfg.update(kw)
    return ConfigurableModel(**cfg)


def gru_inputs(encoder, x):
    """What the GRU and the skip layer of ``encoder`` read for ``x``."""
    seen = {}
    hooks = [encoder.gru.register_forward_pre_hook(
                 lambda m, a: seen.__setitem__("gru", a[0].detach())),
             encoder.skip.register_forward_pre_hook(
                 lambda m, a: seen.__setitem__("skip", a[0].detach()))]
    try:
        encoder(x)
    finally:
        for hook in hooks:
            hook.remove()
    return seen


# ---------------------------------------------------------------------------
# 1. Off
# ---------------------------------------------------------------------------

def test_off_the_encoder_has_no_buffer_and_reads_the_raw_values():
    enc = GRUEncoder(134, 16)
    assert enc.input_bound is None
    assert not any("input_bound" in k for k in enc.state_dict())
    x = 50 * torch.randn(3, 2, 1, 134)
    seen = gru_inputs(enc, x)
    assert torch.equal(seen["gru"], x.reshape(-1, 134, 1))


def test_off_the_model_state_dict_is_unchanged():
    assert list(model().state_dict()) == list(
        model(bound=0.0).state_dict())
    assert not any("input_bound" in k for k in model().state_dict())


# ---------------------------------------------------------------------------
# 2. On
# ---------------------------------------------------------------------------

def test_on_the_gru_reads_at_most_the_bound_and_the_skip_the_raw_values():
    enc = GRUEncoder(134, 16, input_bound=BOUND)
    x = 1e6 * torch.randn(3, 2, 1, 134)
    seen = gru_inputs(enc, x)
    assert seen["gru"].abs().max() <= BOUND
    assert torch.equal(seen["skip"], x)
    small = 0.01 * torch.randn(1, 1, 1, 134)
    close = gru_inputs(enc, small)["gru"]
    assert torch.allclose(close, small.reshape(-1, 134, 1), rtol=1e-5)


def test_on_the_gradient_stays_finite_on_any_input():
    enc = GRUEncoder(134, 16, input_bound=BOUND)
    for scale in (1.0, 1e3, 1e6):
        x = (scale * torch.randn(4, 2, 1, 134)).requires_grad_(True)
        enc(x).square().sum().backward()
        assert torch.isfinite(x.grad).all()
        assert all(torch.isfinite(p.grad).all() for p in enc.parameters())
        enc.zero_grad()


def test_every_patch_size_takes_the_bound():
    m = model(bound=BOUND)
    for size in SIZES:
        assert float(m.encoder.encoders[str(size)].input_bound) == BOUND


def test_only_the_gru_encoder_takes_a_bound():
    with pytest.raises(ValueError):
        create_encoder("mlp", 16, 8, gru_input_bound=BOUND)
    create_encoder("mlp", 16, 8)


# ---------------------------------------------------------------------------
# 3. The checkpoint
# ---------------------------------------------------------------------------

def test_the_checkpoint_records_the_bound():
    sd = model(bound=BOUND).state_dict()
    assert float(sd["encoder.encoders.128.input_bound"]) == BOUND
    assert gru_input_bound_of(sd) == BOUND
    assert gru_input_bound_of(model().state_dict()) == 0.0


def test_the_backbone_loader_rebuilds_the_bound(tmp_path):
    path = tmp_path / "bb.pth"
    torch.save(model(bound=BOUND).state_dict(), path)
    backbone, cfg = load_backbone_from_checkpoint(
        str(path), "cpu", C=1, H=16, W=16, nhead=2, num_layers=1)
    assert cfg["gru_input_bound"] == BOUND
    assert float(backbone.encoder.encoders["128"].input_bound) == BOUND


def test_the_eval_rebuilds_the_bound(tmp_path):
    pytest.importorskip("gift_eval")
    path = tmp_path / "bb.pth"
    torch.save(model(bound=BOUND).state_dict(), path)
    spec = importlib.util.spec_from_file_location("eval_421_bound", EVAL_PY)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    argv = ["eval", "--backbone-path", str(path), "--native-value-head",
            "--strategy", "A2", "--device", "cpu", "--n-channels", "1",
            "--d-model", "16", "--n-heads", "2", "--num-layers", "1",
            "--encoder-type", "gru"]
    from unittest.mock import patch
    with patch.object(sys, "argv", argv):
        args = ev.parse_args()
    backbone, _ = ev.load_models(args, torch.device("cpu"))
    assert float(backbone.encoder.encoders["128"].input_bound) == BOUND


# ---------------------------------------------------------------------------
# 4. The trainer
# ---------------------------------------------------------------------------

TINY_RUN = (
    "--value-space-objective", "--rev-norm-kind", "meanstd",
    "--multi-patch-sizes", "8,16,32,64,128", "--t-raw", "512",
    "--n-channels", "1", "--d-model", "16", "--n-heads", "2",
    "--num-layers", "1", "--num-encoder-layers", "1", "--batch-size", "4",
    "--mix-ratio", "1.0", "--synth-kind", "periodic", "--freq-emb-dim", "3",
    "--seasonality-emb-dim", "3", "--mixup-p", "1.0",
    "--train-rollout-depth", "2", "--log-every", "1",
    "--save-every", "1000000")


def run(root, save_dir, *extra):
    env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4")
    script = (root / "experiments" / "2026-04-27_freq-embedding" / "scripts"
              / "train.py")
    return subprocess.run(
        [sys.executable, str(script), "--device", "cpu", "--weight-decay",
         "0.1", *TINY_RUN, "--save-dir", str(save_dir), "--run-name", "r",
         *extra], capture_output=True, text=True, env=env, timeout=900)


@pytest.mark.parametrize("extra,why", [
    (("--gru-input-bound", "-1"), "0 (off) or a positive bound"),
    (("--gru-input-bound", "10", "--encoder-type", "mlp"),
     "--encoder-type gru"),
])
def test_the_trainer_refuses_a_bound_it_cannot_use(tmp_path, extra, why):
    r = run(REPO_ROOT, tmp_path / "never", "--total-steps", "1", *extra)
    assert r.returncode != 0 and why in r.stdout + r.stderr
    assert not (tmp_path / "never").exists()


def test_a_run_with_the_bound_records_it_and_resumes(tmp_path):
    """A run from a checkpoint without the bound takes it, and its own
    checkpoint then records it. A resume with another bound is refused."""
    first = run(REPO_ROOT, tmp_path / "a", "--total-steps", "2")
    assert first.returncode == 0, first.stdout[-2000:] + first.stderr
    plain = tmp_path / "a" / "r_final.pth"
    assert gru_input_bound_of(torch.load(plain, weights_only=True)) == 0.0
    bounded = run(REPO_ROOT, tmp_path / "b", "--total-steps", "4",
                  "--resume", str(plain), "--gru-input-bound", "10")
    assert bounded.returncode == 0, bounded.stdout[-2000:] + bounded.stderr
    assert "NaN/Inf DETECTED" not in bounded.stdout
    sd = torch.load(tmp_path / "b" / "r_final.pth", weights_only=True)
    assert gru_input_bound_of(sd) == BOUND
    other = run(REPO_ROOT, tmp_path / "c", "--total-steps", "5", "--resume",
                str(tmp_path / "b" / "r_final.pth"), "--gru-input-bound", "8")
    assert other.returncode != 0
    assert "Resume it with the same bound" in other.stdout + other.stderr


def git_tree(commit, tmp_path):
    """The src and trainer of ``commit``, or a skip without git history."""
    if subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e",
                       f"{commit}^{{commit}}"], capture_output=True).returncode:
        pytest.skip(f"{commit} is not in this checkout's history")
    tree = tmp_path / commit
    tree.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", commit, "src",
         "experiments/2026-04-27_freq-embedding/scripts"],
        capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    return tree


def test_off_the_run_writes_the_csv_of_the_commit_before_the_bound(tmp_path):
    old_root = git_tree("9eb14ced", tmp_path)
    old = run(old_root, tmp_path / "old", "--total-steps", "3")
    new = run(REPO_ROOT, tmp_path / "new", "--total-steps", "3")
    assert old.returncode == 0 and new.returncode == 0, new.stderr[-2000:]
    assert ((tmp_path / "new" / "r_losses.csv").read_bytes()
            == (tmp_path / "old" / "r_losses.csv").read_bytes())
    rows = list(csv.DictReader(open(tmp_path / "new" / "r_losses.csv")))
    assert len(rows) == 3
