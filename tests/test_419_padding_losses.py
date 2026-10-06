"""Tests for #419: the contrastive terms and the B4 head skip zero padding.

With ``--gift-pretrain`` a window of a short series starts with zero padding.
No padded patch may enter a loss term: as an anchor, as a key (negative or
MoCo key) or as a positive. The check for every masked term is the same: a
batch whose rows share k padded patches gives the loss of the same batch
with those k patches cut off, and no gradient reaches a padded position.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.forecasting_head import (QUANTILE_LEVELS, compute_valid_targets,  # noqa: E402
                                  masked_quantile_loss, quantile_loss,
                                  valid_target_keep)
from src.freq_embedding import FREQ_NAMES_V2  # noqa: E402
from src.loss import (align_loss, contrastive_latent_loss,  # noqa: E402
                      cpc_infonce_aux_loss, masked_mean, sigreg_loss)
from src.models import ConfigurableModel  # noqa: E402
from src.norm import RevEWMNorm, patch_padding  # noqa: E402

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
HEAD_PY = (REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
           / "train_forecasting_head.py")
B, T, H, K = 4, 12, 8, 3          # batch, patches, width, padded patches


def latents(C=1, seed=0, n=3):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, T, C, H, generator=g, requires_grad=True)
            for _ in range(n)]


def pad_of(C=1, k=K):
    pad = torch.zeros(B, T, C, dtype=torch.bool)
    pad[:, :k] = True
    return pad


def cut(*tensors, k=K):
    return [t[:, k:] for t in tensors]


def spec(**extra):
    cfg = {"loss_shape": "cosine_similarity_batch_rep_only",
           "contrastive_divergence_temperature": 0.1,
           "contrastive_divergence_temperature_rep": 1.0}
    cfg.update(extra)
    return SimpleNamespace(train_configuration=cfg)


def rep_loss(f, h, teacher=None, pad=None, **extra):
    return contrastive_latent_loss(
        (f, h), validation=False, spec=spec(**extra),
        teacher_original_latent=teacher, pad_patches=pad)


# ---------------------------------------------------------------------------
# L_rep of the rep_only shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("C", [1, 2])
def test_l_rep_equals_the_loss_with_the_padding_removed(C):
    f, h, _ = latents(C)
    masked = rep_loss(f, h, pad=pad_of(C))
    assert torch.allclose(masked, rep_loss(*cut(f, h)), atol=1e-5)


@pytest.mark.parametrize("C", [1, 2])
def test_l_rep_with_moco_keys_equals_the_loss_with_the_padding_removed(C):
    f, h, teacher = latents(C)
    masked = rep_loss(f, h, teacher, pad_of(C), moco_rep_keys=True)
    plain = rep_loss(*cut(f, h), cut(teacher)[0], moco_rep_keys=True)
    assert torch.allclose(masked, plain, atol=1e-5)


def test_no_gradient_reaches_a_padded_position():
    f, h, teacher = latents()
    pad = torch.zeros(B, T, 1, dtype=torch.bool)
    pad[0, :5], pad[2, :9] = True, True          # rows padded unevenly
    rep_loss(f, h, teacher, pad, moco_rep_keys=True, align_loss_weight=1.0,
             align_target="teacher").backward()
    for grad in (f.grad, h.grad):
        assert torch.isfinite(grad).all()
        assert (grad[pad] == 0).all() and (grad[~pad] != 0).any()


def test_no_padding_gives_the_unmasked_loss():
    f, h, teacher = latents()
    none = torch.zeros(B, T, 1, dtype=torch.bool)
    for extra in ({}, {"moco_rep_keys": True}):
        t = teacher if extra else None
        assert torch.allclose(rep_loss(f, h, t, none, **extra),
                              rep_loss(f, h, t, **extra), atol=1e-6)


# ---------------------------------------------------------------------------
# L_align, in the loss and alone, with rollout depths
# ---------------------------------------------------------------------------

def cell_loss(f, h, teacher, rollouts, pad=None, reduce="sum"):
    """The cos200k cell's loss terms: L_rep with MoCo keys, L_align on the
    teacher target, rollout depth k (sum or mean)."""
    return contrastive_latent_loss(
        (f, h), validation=False,
        spec=spec(moco_rep_keys=True, align_loss_weight=1.0,
                  align_target="teacher", train_rollout_depth=len(rollouts),
                  train_rollout_reduce=reduce),
        teacher_original_latent=teacher, rollout_latents=rollouts,
        pad_patches=pad)


@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_the_cells_loss_with_rollout_depths_equals_the_cut_batch(reduce):
    f, h, teacher, f1, f2 = latents(n=5)
    masked = cell_loss(f, h, teacher, [f1, f2], pad_of(), reduce)
    fc, hc, tc, f1c, f2c = cut(f, h, teacher, f1, f2)
    assert torch.allclose(masked, cell_loss(fc, hc, tc, [f1c, f2c],
                                            reduce=reduce), atol=1e-5)


def test_align_alone_with_depths_equals_the_cut_batch():
    f, h, teacher, f1 = latents(n=4)
    masked = align_loss(f, h, 1.0, target_latent=teacher, rollout_latents=[f1],
                        pad_patches=pad_of())
    fc, hc, tc, f1c = cut(f, h, teacher, f1)
    plain = align_loss(fc, hc, 1.0, target_latent=tc, rollout_latents=[f1c])
    assert torch.allclose(masked, plain, atol=1e-6)


# ---------------------------------------------------------------------------
# The CPC auxiliary and SIGReg
# ---------------------------------------------------------------------------

def test_cpc_auxiliary_with_a_depth_equals_the_cut_batch():
    torch.manual_seed(0)
    w1 = torch.nn.Linear(H, H, bias=False)
    f, e, f1 = latents(n=3)
    masked = cpc_infonce_aux_loss(f, e, w1, rollout_latents=[f1],
                                  pad_patches=pad_of())
    fc, ec, f1c = cut(f, e, f1)
    plain = cpc_infonce_aux_loss(fc, ec, w1, rollout_latents=[f1c])
    assert torch.allclose(masked, plain, atol=1e-5)
    masked.backward()
    assert (f.grad[pad_of()] == 0).all() and (e.grad[pad_of()] == 0).all()


def test_sigreg_reads_the_real_positions_only(train_py):
    z = latents(n=1)[0]
    proj = torch.nn.functional.normalize(torch.randn(16, H), dim=-1)
    real = train_py.real_positions(z, pad_of())
    assert real.shape == (B * (T - K), H)
    assert torch.allclose(sigreg_loss(real, projections=proj),
                          sigreg_loss(cut(z)[0], projections=proj), atol=1e-6)
    assert train_py.real_positions(z, None) is z


# ---------------------------------------------------------------------------
# Refusals and the patch mask
# ---------------------------------------------------------------------------

def test_a_shape_without_a_mask_refuses_padding():
    f, h, _ = latents()
    with pytest.raises(NotImplementedError, match="rep_only"):
        contrastive_latent_loss(
            (f, h), validation=False,
            spec=spec(loss_shape="cosine_similarity_batch"), pad_patches=pad_of())


def test_the_fused_kernel_refuses_padding(monkeypatch):
    monkeypatch.setenv("XSHH_ALLT_FUSED", "1")
    f, h, _ = latents()
    with pytest.raises(NotImplementedError, match="XSHH_ALLT_FUSED"):
        rep_loss(f, h, pad=pad_of())


def test_the_floor_refuses_padding():
    f, h, _ = latents()
    with pytest.raises(NotImplementedError, match="subtract_contrastive_floor"):
        rep_loss(f, h, pad=pad_of(), subtract_contrastive_floor=True)


def test_a_patch_is_padding_only_when_every_value_is():
    x = torch.cat([torch.zeros(1, 40, 1), torch.randn(1, 24, 1) + 5], dim=1)
    norm = RevEWMNorm(1, span=16, patch_size=8, skip_leading_zeros=True)
    norm(x, "norm")
    assert patch_padding(norm.pad_mask, 8)[0, :, 0].tolist() == [True] * 5 + [False] * 3
    x[0, 39, 0] = 1.0                            # patch 4 now ends on a value
    norm(x, "norm")
    assert patch_padding(norm.pad_mask, 8)[0, :, 0].tolist() == [True] * 4 + [False] * 4


# ---------------------------------------------------------------------------
# The B4 head
# ---------------------------------------------------------------------------

def test_the_head_target_mask_follows_the_target_layout():
    pad = torch.zeros(2, 64, 1, dtype=torch.bool)
    pad[0, :20] = True
    keep = valid_target_keep(pad, W=8, forecast_len=8)
    targets, t_valid = compute_valid_targets(torch.randn(2, 64, 1), 8, 8)
    assert keep.shape == targets.shape and t_valid == 7
    assert keep[0, :, 0].tolist() == [False, False] + [True] * 5
    assert keep[0, 1].tolist() == [False] * 4 + [True] * 4 and keep[1].all()


def test_the_head_loss_skips_padded_targets():
    preds = torch.randn(3, 5, len(QUANTILE_LEVELS), 16)
    targets = torch.randn(3, 5, 16)
    keep = torch.ones(3, 5, 16, dtype=torch.bool)
    keep[0, :2] = False
    moved = targets + 50.0 * (~keep)
    assert torch.allclose(masked_quantile_loss(preds, targets, keep),
                          masked_quantile_loss(preds, moved, keep))
    keep_all = torch.ones_like(keep)
    assert torch.allclose(masked_quantile_loss(preds, targets, keep_all),
                          quantile_loss(preds, targets))
    assert torch.allclose(masked_mean((preds[..., 0, :] - targets) ** 2, keep_all),
                          torch.nn.functional.mse_loss(preds[..., 0, :], targets))


# ---------------------------------------------------------------------------
# The scripts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_py():
    import importlib.util
    s = importlib.util.spec_from_file_location("train_py_419pad", TRAIN_PY)
    module = importlib.util.module_from_spec(s)
    s.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """The GiftEvalPretrain-layout corpus of test_419_gift_pretrain.py."""
    from tests.test_419_gift_pretrain import build_corpus
    return build_corpus(tmp_path_factory.mktemp("gep_pad"))


def run(script, *args):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, env=env, timeout=900)


def gift_args(corpus, tmp_path):
    from tests.test_419_gift_pretrain import write_index
    root, index, _ = corpus
    return ["--gift-pretrain-root", str(root), "--gift-pretrain-index",
            str(write_index(tmp_path, index))]


# The cos200k cell's terms at a size the CPU trains in seconds.
TINY_COS200K = (
    "--gift-pretrain", "--freq-vocab", "v2", "--device", "cpu",
    "--weight-decay", "0.1", "--t-raw", "4096", "--n-channels", "1",
    "--d-model", "16", "--n-heads", "2", "--num-layers", "1",
    "--num-encoder-layers", "1", "--batch-size", "4",
    "--loss-shape", "cosine_similarity_batch_rep_only",
    "--align-loss-weight", "1.0", "--moco-rep-keys", "--tau-rep", "1.0",
    "--align-target", "teacher", "--ema-embedding", "--ema-encoder",
    "--ema-tau", "0.9", "--sigreg-embedding", "--sigreg-encoding",
    "--sigreg-m", "16", "--sigreg-embedding-weight", "1.0",
    "--sigreg-encoding-weight", "1.0", "--tau", "0.10",
    "--rev-norm-kind", "ewma", "--rev-norm-span", "128",
    "--synth-kind", "forked-arma", "--mix-ratio", "0.25",
    "--crossfade-triplets", "1", "--mixup-p", "0.3",
    "--freq-emb-dim", "3", "--seasonality-emb-dim", "3",
    "--train-rollout-depth", "1", "--train-rollout-reduce", "sum",
    "--rep-loss-weight", "1.0", "--rep-loss-weight-end", "0.0",
    "--rep-loss-weight-ramp-steps", "2", "--log-every", "1",
    "--save-every", "1000000", "--no-latent-drift-probe")


def test_a_few_contrastive_steps_on_the_stream(corpus, tmp_path):
    r = run(TRAIN_PY, *TINY_COS200K, *gift_args(corpus, tmp_path),
            "--total-steps", "3", "--save-dir", str(tmp_path),
            "--run-name", "c419")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "Salesforce/GiftEvalPretrain (#419)" in r.stdout
    assert "[      3]" in r.stdout and "nan" not in r.stdout.lower()


@pytest.mark.parametrize("extra,why", [
    # Since #412 the split shape takes a mask (tests/test_412_bimoco.py).
    (("--loss-shape", "cosine_similarity_batch_full_hh_negs_xshh_allt"),
     "--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt"),
    (("--align-moco-loss-weight", "1.0"), "--align-moco-loss-weight"),
    (("--cpc-infonce-weight", "1.0", "--cpc-infonce-negs", "all"),
     "--cpc-infonce-negs all")])
def test_a_contrastive_term_without_a_mask_is_refused(tmp_path, extra, why):
    r = run(TRAIN_PY, "--gift-pretrain", "--device", "cpu",
            "--weight-decay", "0.1", "--total-steps", "1",
            "--save-dir", str(tmp_path),
            "--loss-shape", "cosine_similarity_batch_rep_only", *extra)
    assert r.returncode != 0 and why in r.stdout + r.stderr


def backbone_file(tmp_path, zero_pad):
    torch.manual_seed(0)
    model = ConfigurableModel(
        C=1, H=32, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        freq_emb_dim=3, seasonality_emb_dim=3, rev_norm_kind="ewma",
        rev_norm_span=8, num_freqs=len(FREQ_NAMES_V2) if zero_pad else 10,
        rev_norm_skip_leading_zeros=zero_pad)
    path = tmp_path / f"bb_{zero_pad}.pth"
    torch.save(model.state_dict(), path)
    return str(path)


def head_args(tmp_path, backbone, name):
    return ["--backbone-path", backbone, "--device", "cpu", "--quantile-head",
            "--forecast-len", "16", "--batch-size", "2", "--lr", "1e-3",
            "--total-steps", "3", "--save-every", "100", "--log-every", "1",
            "--save-dir", str(tmp_path / name), "--run-name", name,
            "--n-channels", "1", "--d-model", "32", "--n-heads", "2",
            "--num-layers", "1", "--encoder-type", "gru",
            "--rev-norm-kind", "ewma", "--rev-norm-span", "8"]


def test_the_head_of_a_zero_pad_backbone_trains_on_the_stream(corpus, tmp_path):
    bb = backbone_file(tmp_path, zero_pad=True)
    r = run(HEAD_PY, *head_args(tmp_path, bb, "h_new"),
            "--hf-repo", "none", "--hf-path", "none",
            *gift_args(corpus, tmp_path))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "the head trains on the same stream, vocabulary v2" in r.stdout
    assert "[      3]" in r.stdout


def test_the_head_of_any_other_backbone_trains_as_before(tmp_path):
    bb = backbone_file(tmp_path, zero_pad=False)
    r = run(HEAD_PY, *head_args(tmp_path, bb, "h_old"), "--mix-ratio", "1.0",
            "--hf-repo", "none", "--hf-path", "none")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "GiftEvalPretrain" not in r.stdout
    assert "[      3]" in r.stdout


def test_off_ddp_the_padding_gather_is_a_no_op():
    from src.dist_utils import gather_mask
    pad = pad_of()
    assert gather_mask(pad) is pad
