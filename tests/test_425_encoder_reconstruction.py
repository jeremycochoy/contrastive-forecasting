"""Tests for #425: the reconstruction head on the encoder output, and its
GM-Relative MASE.

A reconstruction head reads the encoder latent e_t and gives the quantiles of
the values of patch t. The score encodes the context AND the true horizon,
and the head decodes the latents of the horizon patches. So the score
measures the encoding, not a forecast.

Groups, all on the CPU:

1. The loss: each position is scored on the values of its own patch, the
   padding counts in no term, and each row of a head bank trains the head of
   its patch size on the encoder latents at that size.
2. The score of one window (strategy R): the encoder reads the context and
   the horizon at the head's patch size, the statistics are those of the B4
   context, and a head that decodes its latents exactly gives back the true
   horizon.
3. The eval script: strategy R reads the label of each window, and a perfect
   reconstruction scores a MASE of 0.
4. The head trainer: `--reconstruction encoder` trains the B4 head on a bank
   checkpoint, on a zero-padding checkpoint and on an old checkpoint. The
   shared trainer gives each job of one data stream the losses and the head
   of its solo run, bit for bit. After a crash during a save, no cut head
   stays. A new try of a job starts a clean loss CSV.
5. The shell runners: the reconstruction mode reaches the head trainer and
   the eval, and the forecast mode stays as it was. The floor of R (R0) and
   the flags for a shared trainer.
6. The scripts of the report: the job table, the waves and the retries of
   the queue on the box, the prune rule, the floors and the figures.

Group 2b: the floor of R, a head that gives the normalised value 0.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.forecasting_head as fh  # noqa: E402
from src.forecasting_head import (QUANTILE_LEVELS,  # noqa: E402
                                  ForecastingHeadBank,
                                  TransformerQuantileForecastingHead,
                                  ZeroReconstructionHead,
                                  bank_quantile_loss, bank_training_inputs,
                                  compute_reconstruction_targets,
                                  extract_encoder_latents, forecast_B4,
                                  head_bank_sizes, quantile_loss,
                                  reconstruct_horizon, reconstruct_windows,
                                  reconstruction_quantile_loss)
from src.freq_embedding import FREQ_NAMES_V2  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402

GIFT_SCRIPTS = REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
HEAD_PY = GIFT_SCRIPTS / "train_forecasting_head.py"
EVAL_PY = GIFT_SCRIPTS / "eval_gift_eval_official.py"
B4_SCRIPTS = REPO_ROOT / "reports" / "2026-08-08_rollout_depth" / "scripts"
STUDY = REPO_ROOT / "reports" / "2026-10-06_encoder_reconstruction"
SIZES = (8, 16, 32, 64, 128)
Q = len(QUANTILE_LEVELS)
T = 1024
CPU = torch.device("cpu")


def bank_model():
    """A #412 patch-size backbone at d_model 16: one GRU patch encoder per
    size, the mean/std scaling, zero padding. The shape the eval and the
    head trainer rebuild from the flags of TINY_SHAPE."""
    torch.manual_seed(0)
    return ConfigurableModel(
        C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        rev_norm_kind="meanstd", rev_norm_skip_leading_zeros=True,
        num_encoder_layers=1, freq_emb_dim=3, num_freqs=len(FREQ_NAMES_V2),
        seasonality_emb_dim=3, multi_patch_sizes=SIZES).eval()


def ewma_model(zero_pad=False):
    """A one-size EWMA backbone at d_model 16: the old runs (no padding,
    vocabulary v1) and BLK (zero padding, vocabulary v2)."""
    torch.manual_seed(0)
    return ConfigurableModel(
        C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        rev_norm_kind="ewma", rev_norm_span=128, num_encoder_layers=1,
        freq_emb_dim=3, seasonality_emb_dim=3,
        num_freqs=len(FREQ_NAMES_V2) if zero_pad else 10,
        rev_norm_skip_leading_zeros=zero_pad).eval()


def ewma_bank_model():
    """OEF: patch sizes 8 to 128 with the EWMA and zero padding."""
    torch.manual_seed(2)
    return ConfigurableModel(
        C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        rev_norm_kind="ewma", rev_norm_span=128,
        rev_norm_skip_leading_zeros=True, num_encoder_layers=1,
        freq_emb_dim=3, num_freqs=len(FREQ_NAMES_V2), seasonality_emb_dim=3,
        multi_patch_sizes=SIZES).eval()


def meanstd_model():
    """BMS: one patch size with the mean/std scaling and zero padding."""
    torch.manual_seed(3)
    return ConfigurableModel(
        C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        rev_norm_kind="meanstd", rev_norm_skip_leading_zeros=True,
        num_encoder_layers=1, freq_emb_dim=3, num_freqs=len(FREQ_NAMES_V2),
        seasonality_emb_dim=3).eval()


def quantile_head(size=16, seed=1):
    torch.manual_seed(seed)
    return TransformerQuantileForecastingHead(
        H=16, num_layers=1, nhead=2, forecast_len=size).eval()


def bank_of(sizes=SIZES):
    return ForecastingHeadBank({p: quantile_head(p) for p in sizes}).eval()


LENGTHS = (T, 700, 400, T, 900, 300, T, 600)
FREQS = ("1h", "1d", "5min", "1M", "10s", "1h", "1Y", "1d")


def windows(lengths=LENGTHS, seed=0):
    """``[B, T, 1]``: one random walk per length, left zero padded."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(len(lengths), T, 1)
    for b, n in enumerate(lengths):
        x[b, T - n:, 0] = 100.0 + torch.randn(n, generator=g).cumsum(0)
    return x


def walk(n, seed=0, level=50.0):
    g = torch.Generator().manual_seed(seed)
    return level + torch.randn(n, generator=g).cumsum(0)


def patches(x_norm, size):
    """``[B, T_raw, C]`` to ``[B*C, T_raw // size, size]``, the layout of
    the encoder latents and of the reconstruction targets."""
    B, T_raw, C = x_norm.shape
    return (x_norm.view(B, T_raw // size, size, C).permute(0, 3, 1, 2)
            .reshape(B * C, T_raw // size, size))


def oracle_latents(backbone, x, freq_ids=None, seasonality_ids=None,
                   patch_size=None, normalised=False):
    """An encoder whose latent IS the normalised values of its patch."""
    size = patch_size or backbone.W
    x_norm = x if normalised else backbone.rev_norm(x, mode="norm")
    return patches(x_norm, size), x_norm


class OracleHead(nn.Module):
    """Each quantile of position t is the latent of position t: with
    :func:`oracle_latents`, the exact values of patch t."""

    def __init__(self, size, bank_member=True):
        super().__init__()
        self.forecast_len = size
        self.quantile_levels = list(QUANTILE_LEVELS)
        if bank_member:
            self.patch_size = size

    def forward(self, e):
        return e.unsqueeze(-2).expand(-1, -1, Q, -1)


# ---------------------------------------------------------------------------
# 1. The loss
# ---------------------------------------------------------------------------

def test_the_latents_of_a_normalised_window_are_the_latents_of_the_window():
    """``normalised=True`` skips the normaliser, and changes nothing else."""
    m = ewma_model()
    x = walk(T)[None, :, None]
    e_raw, x_norm = extract_encoder_latents(m, x)
    e_norm, same = extract_encoder_latents(m, x_norm, normalised=True)
    assert torch.equal(same, x_norm)
    assert torch.allclose(e_raw, e_norm, atol=1e-6)


def test_each_position_is_scored_on_the_values_of_its_own_patch():
    """A head that gives every quantile of position t the values of patch t
    has no pinball loss. The values of patch t + 1 (the forecast target)
    have one."""
    x_norm = torch.randn(3, 256, 2)
    own = patches(x_norm, 16).unsqueeze(-2).expand(-1, -1, Q, -1)
    assert reconstruction_quantile_loss(own, x_norm, 16).item() == 0.0
    nxt = torch.roll(own, shifts=-1, dims=1)
    assert reconstruction_quantile_loss(nxt, x_norm, 16).item() > 0.1


def test_the_loss_is_the_pinball_loss_of_the_patch_values():
    x_norm = torch.randn(2, 128, 1)
    preds = torch.randn(2, 8, Q, 16)
    targets, t_valid = compute_reconstruction_targets(
        x_norm, W=16, output_len=16, mode="encoder")
    assert t_valid == 8
    assert torch.allclose(reconstruction_quantile_loss(preds, x_norm, 16),
                          quantile_loss(preds, targets))


def test_padded_values_count_in_no_reconstruction_term():
    """The values a mask drops leave the loss unchanged, whatever the head
    gives them. The head sees padding, but no term scores it."""
    x_norm = torch.randn(2, 128, 1)
    keep = torch.ones_like(x_norm, dtype=torch.bool)
    keep[0, :40] = False
    preds = torch.randn(2, 8, Q, 16)
    other = preds.clone()
    other[0, :2] += 100.0                      # patches 0 and 1: all padding
    other[0, 2, :, :8] -= 100.0                # patch 2: values 32 to 39
    base = reconstruction_quantile_loss(preds, x_norm, 16, keep=keep)
    assert torch.allclose(base,
                          reconstruction_quantile_loss(other, x_norm, 16, keep))
    other[0, 2, :, 8] += 1.0                   # value 40 is real
    assert not torch.allclose(
        base, reconstruction_quantile_loss(other, x_norm, 16, keep))


def labels_of(freqs=FREQS):
    freq = torch.tensor([FREQ_NAMES_V2.index(f) for f in freqs])
    return freq, torch.zeros_like(freq)


def test_each_row_trains_the_reconstruction_head_of_its_size():
    """Rows at sizes 8 and 32 only: those two heads get a gradient, and the
    loss is the two group losses weighted by their share of the rows."""
    m = bank_model()
    for p in m.parameters():
        p.requires_grad_(False)
    bank = bank_of()
    freq, seas = labels_of()
    torch.manual_seed(5)
    x_norm, _, keep = bank_training_inputs(m, windows(), freq, SIZES)
    sizes = torch.tensor([8, 8, 32, 32, 8, 32, 32, 32])
    loss = bank_quantile_loss(m, bank, x_norm, sizes, keep, freq, seas,
                              reconstruction=True)
    loss.backward()
    for size in SIZES:
        grads = [p.grad for p in bank.head_for(size).parameters()]
        assert all((g is not None) == (size in (8, 32)) for g in grads), size
    parts = []
    for size, rows in ((8, [0, 1, 4]), (32, [2, 3, 5, 6, 7])):
        e_bc, _ = extract_encoder_latents(
            m, x_norm[rows], freq_ids=freq[rows], seasonality_ids=seas[rows],
            patch_size=size, normalised=True)
        preds = bank.head_for(size)(e_bc)
        parts.append(len(rows) / 8 * reconstruction_quantile_loss(
            preds, x_norm[rows], size, keep=keep[rows]))
    assert torch.allclose(loss, parts[0] + parts[1])


def test_a_bank_reconstruction_reads_the_encoder_at_each_size(monkeypatch):
    """Each group reads the encoder latents of its already normalised rows,
    at its size. No forecaster runs."""
    seen = []
    real = fh.extract_encoder_latents

    def spy(backbone, x, **kw):
        seen.append((kw.get("patch_size"), kw.get("normalised")))
        return real(backbone, x, **kw)

    def no_forecaster(*_, **__):
        raise AssertionError("a reconstruction head reads no forecaster")

    monkeypatch.setattr(fh, "extract_encoder_latents", spy)
    monkeypatch.setattr(fh, "extract_forecaster_latents", no_forecaster)
    m = bank_model()
    freq, seas = labels_of()
    torch.manual_seed(5)
    x_norm, _, keep = bank_training_inputs(m, windows(), freq, SIZES)
    sizes = torch.tensor([8, 16, 16, 64, 128, 8, 16, 64])
    bank_quantile_loss(m, bank_of(), x_norm, sizes, keep, freq, seas,
                       reconstruction=True)
    assert seen == [(8, True), (16, True), (64, True), (128, True)]


def test_a_perfect_reconstruction_has_no_bank_loss(monkeypatch):
    monkeypatch.setattr(fh, "extract_encoder_latents", oracle_latents)
    m = bank_model()
    freq, seas = labels_of()
    torch.manual_seed(5)
    x_norm, sizes, keep = bank_training_inputs(m, windows(), freq, SIZES)
    oracle = ForecastingHeadBank({p: OracleHead(p) for p in SIZES})
    loss = bank_quantile_loss(m, oracle, x_norm, sizes, keep, freq, seas,
                              reconstruction=True)
    assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# 2. The score of one window: strategy R
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make,size,n_ctx,horizon", [
    (ewma_model, 16, T, 45),             # an old run: EWMA, one size
    (lambda: ewma_model(True), 16, T, 30),  # BLK: EWMA with zero padding
    (bank_model, 32, T, 40),             # a patch-size run, mean/std
    (bank_model, 128, T, 300),           # a horizon over several patches
])
def test_a_perfect_head_gives_back_the_true_horizon(monkeypatch, make, size,
                                                    n_ctx, horizon):
    """The window holds the context, then the horizon, then the last value
    up to a whole patch. Undoing the normalisation of each position gives
    back the true values of the horizon, quantile by quantile."""
    monkeypatch.setattr(fh, "extract_encoder_latents", oracle_latents)
    m = make()
    series = walk(n_ctx + horizon, seed=3)
    ctx, future = series[:n_ctx, None], series[n_ctx:]
    out = reconstruct_horizon(m, OracleHead(size), ctx, future.numpy(), CPU)
    assert out.shape == (Q, horizon, 1)
    for q in range(Q):
        assert np.allclose(out[q, :, 0], future.numpy(), atol=1e-3)


def test_a_zero_padded_context_gives_back_the_true_horizon(monkeypatch):
    monkeypatch.setattr(fh, "extract_encoder_latents", oracle_latents)
    m = ewma_model(zero_pad=True)
    ctx = torch.zeros(T, 1)
    ctx[-200:, 0] = walk(200, seed=4)
    future = walk(24, seed=5, level=float(ctx[-1, 0]))
    out = reconstruct_horizon(m, OracleHead(16, bank_member=False), ctx,
                              future.numpy(), CPU)
    assert np.allclose(out[4, :, 0], future.numpy(), atol=1e-3)


def test_the_encoder_reads_the_context_and_the_horizon_at_the_head_size(
        monkeypatch):
    seen = []
    real = fh.extract_encoder_latents

    def spy(backbone, x, **kw):
        seen.append((x.shape[1], kw.get("patch_size"), kw.get("normalised")))
        return real(backbone, x, **kw)

    monkeypatch.setattr(fh, "extract_encoder_latents", spy)
    m = bank_model()
    out = reconstruct_horizon(m, bank_of().head_for(64), walk(T)[:, None],
                              walk(100, seed=1).numpy(), CPU)
    assert out.shape == (Q, 100, 1) and np.isfinite(out).all()
    # 1,024 + 100 values, then 28 copies of the last value: 18 patches of 64.
    assert seen == [(1152, 64, True)]


def test_the_mean_std_statistics_read_the_context_only():
    """loc and scale are those of the B4 forecast of the same context: the
    horizon changes neither them nor the latents of the context."""
    m = bank_model()
    head = bank_of().head_for(32)
    ctx = walk(T)[:, None]
    m.rev_norm(ctx[None], mode="norm")
    loc, scale = m.rev_norm.mean.clone(), m.rev_norm.stdev.clone()
    e_b4, _ = extract_encoder_latents(m, ctx[None], patch_size=32)
    for seed in (1, 2):
        future = (1e3 * walk(64, seed=seed)).numpy()
        reconstruct_horizon(m, head, ctx, future, CPU)
        assert torch.equal(m.rev_norm.mean, loc)
        assert torch.equal(m.rev_norm.stdev, scale)
    x = torch.cat([ctx, torch.as_tensor(future)[:, None]])[None]
    x_norm = fh.normalise_window(m, x, T)
    e_r, _ = extract_encoder_latents(m, x_norm, patch_size=32,
                                     normalised=True)
    assert torch.allclose(e_r[:, :T // 32], e_b4, atol=1e-5)


def test_the_ewma_context_latents_are_the_b4_latents():
    """A causal EWMA and a causal encoder: the horizon leaves the context
    latents as the B4 forecast reads them."""
    m = ewma_model()
    ctx = walk(T)[:, None]
    e_b4, _ = extract_encoder_latents(m, ctx[None])
    x = torch.cat([ctx, walk(48, seed=9)[:, None]])[None]
    e_r, _ = extract_encoder_latents(m, fh.normalise_window(m, x, T),
                                     normalised=True)
    assert torch.allclose(e_r[:, :T // 16], e_b4, atol=1e-5)


def test_a_horizon_value_moves_its_own_patch_and_the_later_ones_only():
    """The model reads the horizon. With a causal encoder and a causal head,
    a value of horizon patch 1 changes nothing in patch 0."""
    m = ewma_model()
    head = quantile_head(16)
    ctx, future = walk(T)[:, None], walk(48, seed=2).numpy()
    base = reconstruct_horizon(m, head, ctx, future, CPU)
    moved = future.copy()
    moved[20] += 50.0
    out = reconstruct_horizon(m, head, ctx, moved, CPU)
    assert np.allclose(out[:, :16], base[:, :16], atol=1e-5)
    assert not np.allclose(out[:, 16:32], base[:, 16:32], atol=1e-3)


def test_a_head_of_another_length_is_refused():
    m = ewma_model()
    torch.manual_seed(0)
    head = TransformerQuantileForecastingHead(H=16, num_layers=1, nhead=2,
                                              forecast_len=128)
    with pytest.raises(ValueError, match="decodes 128 values"):
        reconstruct_horizon(m, head, walk(T)[:, None], np.ones(16), CPU)


@pytest.mark.parametrize("make,size", [(ewma_model, 16), (bank_model, 64)])
def test_a_batch_of_windows_is_the_windows_one_by_one(make, size):
    """The statistics, the latents and the unscaling are per window, so a
    batch of the windows of one config gives each window's own result."""
    m = make()
    head = (quantile_head(16) if size == 16 else bank_of().head_for(size))
    series = [walk(T + 50, seed=s) for s in range(3)]
    series[1] = 1e3 + 20.0 * series[1]                # another level and scale
    contexts = torch.stack([s[:T, None] for s in series])
    futures = torch.stack([s[T:, None] for s in series])
    batch = reconstruct_windows(m, head, contexts, futures, CPU)
    assert batch.shape == (3, Q, 50, 1)
    for i in range(3):
        one = reconstruct_horizon(m, head, contexts[i], futures[i, :, 0], CPU)
        assert np.allclose(batch[i], one, rtol=1e-4, atol=1e-3)


def test_the_b4_forecast_is_unchanged_by_the_new_keyword():
    """The B4 forecast of a bank head still reads its context alone."""
    m = bank_model()
    head = bank_of().head_for(32)
    out = forecast_B4(m, head, walk(T)[:, None], 40, "cpu")
    assert out.shape == (Q, 40, 1) and np.isfinite(out).all()


# ---------------------------------------------------------------------------
# 2b. The floor of strategy R: the statistics alone
# ---------------------------------------------------------------------------

def horizon_means(m, ctx, future, size):
    """The mean that normalised each horizon value of the window of R."""
    n_ctx, h = ctx.shape[0], future.shape[0]
    tail = future[-1:].expand((-(n_ctx + h)) % size, -1)
    window = torch.cat([ctx, future, tail])[None]
    fh.normalise_window(m, window, n_ctx)
    return m.rev_norm.mean.expand(*window.shape)[0, n_ctx:n_ctx + h, 0]


@pytest.mark.parametrize("make", [ewma_model, bank_model, meanstd_model])
def test_the_zero_head_gives_the_mean_of_each_horizon_value(make):
    """Unscaled, the normalised value 0 is the mean that normalised each
    value: the EWMA at that value, or the loc of the context."""
    m = make()
    ctx, future = walk(T)[:, None], walk(48, seed=3, level=80.0)[:, None]
    out = reconstruct_windows(m, ZeroReconstructionHead(16), ctx[None],
                              future[None], CPU)
    want = horizon_means(m, ctx, future, 16).numpy()
    assert out.shape == (1, Q, 48, 1)
    for q in range(Q):
        assert np.allclose(out[0, q, :, 0], want, rtol=1e-6, atol=1e-4)


def test_the_floor_reads_the_scaling_and_no_patch_size_or_weight():
    """One floor serves every run of a scaling setup: BLK (one patch size)
    and OEF (patch sizes 8 to 128) share the EWMA with zero padding."""
    ctx, future = walk(T)[:, None], walk(100, seed=6)[:, None]
    blk = reconstruct_windows(ewma_model(zero_pad=True),
                              ZeroReconstructionHead(16), ctx[None],
                              future[None], CPU)
    oef = ewma_bank_model()
    for size in SIZES:
        out = reconstruct_windows(oef, ZeroReconstructionHead(size),
                                  ctx[None], future[None], CPU)
        assert np.allclose(out, blk, rtol=1e-6, atol=1e-4), size
    bms = reconstruct_windows(meanstd_model(), ZeroReconstructionHead(16),
                              ctx[None], future[None], CPU)
    omb = reconstruct_windows(bank_model(), ZeroReconstructionHead(64),
                              ctx[None], future[None], CPU)
    assert np.allclose(omb, bms, rtol=1e-6, atol=1e-4)
    assert not np.allclose(bms, blk, atol=1e-2)


# ---------------------------------------------------------------------------
# 3. The eval script
# ---------------------------------------------------------------------------

def load_eval_module():
    pytest.importorskip("gluonts")
    pytest.importorskip("gift_eval")
    spec = importlib.util.spec_from_file_location("eval_425", EVAL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_missing_horizon_value_takes_the_value_before_it():
    module = load_eval_module()
    out = module.fill_horizon(np.array([np.nan, 2.0, np.nan, np.nan, 5.0],
                                       dtype=np.float32), last=7.0)
    assert out.tolist() == [7.0, 2.0, 2.0, 2.0, 5.0]
    full = np.arange(4, dtype=np.float32)
    assert module.fill_horizon(full, last=9.0).tolist() == full.tolist()


def loop_fill(target, context_pad):
    """The forward fill of the eval before #425, one value at a time."""
    if not np.isnan(target).any():
        return target
    target = target.copy()
    mask = np.isnan(target)
    if mask.all():
        target[:] = 0.0
        return target
    first_valid = np.where(~mask)[0][0]
    if context_pad == "zeros":
        target = target[first_valid:]
    else:
        target[:first_valid] = target[first_valid]
    for i in range(1, len(target)):
        if np.isnan(target[i]):
            target[i] = target[i - 1]
    return target


@pytest.mark.parametrize("context_pad", ["first", "zeros"])
def test_the_fill_of_a_context_gives_the_values_of_the_loop(context_pad):
    """The eval fills a missing context value in one numpy pass now. It must
    give every B4 score the values it had."""
    module = load_eval_module()
    predictor = module.ContrastiveForecasterPredictor(
        backbone=None, head=None, prediction_length=8, device=CPU,
        context_pad=context_pad)
    rng = np.random.default_rng(0)
    for n in (1, 5, 300):
        for share in (0.0, 0.05, 0.5, 0.95, 1.0):
            for _ in range(5):
                x = rng.standard_normal(n).astype(np.float32)
                x[rng.random(n) < share] = np.nan
                if share == 0.5:
                    x[: n // 3] = np.nan               # a missing start
                got = predictor._fill_missing(x.copy())
                want = loop_fill(x.copy(), context_pad)
                assert got.dtype == want.dtype == np.float32
                assert np.array_equal(got, want), (n, share)


def gluonts_test_data(h=24, windows=2, n=3, length=900):
    import pandas as pd
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.split import split
    rng = np.random.default_rng(0)
    items = [{"target": (60.0 + rng.standard_normal(length).cumsum())
              .astype(np.float32),
              "start": pd.Period("2020-01-01 00:00", freq="h"),
              "item_id": f"s{i}"} for i in range(n)]
    dataset = ListDataset(items, freq="h")
    _, template = split(dataset, offset=-h * windows)
    return template.generate_instances(prediction_length=h, windows=windows)


def predictor_of(module, m, head, test_data, h=24, labels=None):
    """Six windows in batches of four: a full batch, then a partial one."""
    return module.ReconstructionPredictor(
        backbone=m, head=head, prediction_length=h, device=CPU,
        strategy="R", context_pad="first", batch_size=4,
        labels=test_data.label if labels is None else labels)


def test_a_perfect_reconstruction_scores_a_mase_of_zero(monkeypatch):
    from gluonts.ev.metrics import MASE
    from gluonts.model import evaluate_model
    module = load_eval_module()
    monkeypatch.setattr(fh, "extract_encoder_latents", oracle_latents)
    test_data = gluonts_test_data()
    predictor = predictor_of(module, ewma_model(), OracleHead(16, False),
                             test_data)
    res = evaluate_model(predictor, test_data=test_data, metrics=[MASE()],
                         axis=None, mask_invalid_label=True,
                         allow_nan_forecast=False, seasonality=24)
    assert res["MASE[0.5]"][0] < 1e-4


def test_a_real_head_scores_a_finite_mase():
    from gluonts.ev.metrics import MASE
    from gluonts.model import evaluate_model
    module = load_eval_module()
    test_data = gluonts_test_data()
    predictor = predictor_of(module, bank_model(), bank_of().head_for(32),
                             test_data)
    res = evaluate_model(predictor, test_data=test_data, metrics=[MASE()],
                         axis=None, mask_invalid_label=True,
                         allow_nan_forecast=False, seasonality=24)
    assert np.isfinite(res["MASE[0.5]"][0])


def test_a_label_of_another_window_is_refused():
    module = load_eval_module()
    test_data = gluonts_test_data()
    shifted = list(test_data.label)[1:] + list(test_data.label)[:1]
    predictor = predictor_of(module, ewma_model(), quantile_head(16),
                             test_data, labels=shifted)
    with pytest.raises(ValueError, match="does not start where"):
        list(predictor.predict(test_data.input))


# The flags eval_local.sh gives the GIFT-Eval script, at the tiny shape.
TINY_SHAPE = ("--d-model", "16", "--n-heads", "2", "--num-layers", "1")
EVAL_PROTOCOL = (
    "--forecast-len", "16", "--device", "cpu", "--t-raw", "4096",
    "--n-channels", "1", *TINY_SHAPE, "--encoder-type", "gru",
    "--rev-norm-kind", "ewma", "--rev-norm-span", "128", "--head-nhead", "2",
    "--head-causal", "true")


def eval_args(module, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["eval", *EVAL_PROTOCOL, *extra])
    return module.parse_args()


def saved(tmp_path, name, module_or_sd):
    path = tmp_path / name
    sd = (module_or_sd.state_dict() if isinstance(module_or_sd, nn.Module)
          else module_or_sd)
    torch.save(sd, path)
    return str(path)


def test_the_eval_loads_a_bank_under_r(tmp_path, monkeypatch):
    module = load_eval_module()
    bb = saved(tmp_path, "bb.pth", bank_model())
    head = saved(tmp_path, "head.pth", bank_of())
    args = eval_args(module, monkeypatch, "--backbone-path", bb,
                     "--head-path", head, "--strategy", "R")
    backbone, bank = module.load_models(args, CPU)
    assert isinstance(bank, ForecastingHeadBank) and bank.sizes == SIZES
    args = eval_args(module, monkeypatch, "--backbone-path", bb,
                     "--head-path", head, "--strategy", "A2")
    with pytest.raises(SystemExit, match="scores under --strategy B4 or R"):
        module.load_models(args, CPU)


def test_the_eval_builds_a_zero_head_for_each_patch_size(tmp_path,
                                                         monkeypatch):
    """--zero-head (the floor of R) reads no head file."""
    module = load_eval_module()
    bank = saved(tmp_path, "bank.pth", bank_model())
    args = eval_args(module, monkeypatch, "--backbone-path", bank,
                     "--strategy", "R", "--zero-head")
    _, heads = module.load_models(args, CPU)
    assert isinstance(heads, ForecastingHeadBank) and heads.sizes == SIZES
    assert all(isinstance(h, ZeroReconstructionHead)
               for h in heads.heads.values())
    one = saved(tmp_path, "one.pth", ewma_model(zero_pad=True))
    args = eval_args(module, monkeypatch, "--backbone-path", one,
                     "--strategy", "R", "--zero-head")
    _, head = module.load_models(args, CPU)
    assert isinstance(head, ZeroReconstructionHead)
    assert head.patch_size == head.forecast_len == 16


def test_the_zero_head_scores_under_r_only(monkeypatch):
    module = load_eval_module()
    with pytest.raises(SystemExit):
        eval_args(module, monkeypatch, "--backbone-path", "bb.pth",
                  "--strategy", "B4", "--zero-head")


def test_the_floor_forecast_holds_one_value_for_every_quantile():
    module = load_eval_module()
    test_data = gluonts_test_data()
    predictor = predictor_of(module, ewma_model(),
                             ZeroReconstructionHead(16), test_data)
    for forecast in predictor.predict(test_data.input):
        arrays = forecast.forecast_array
        assert np.isfinite(arrays).all()
        assert np.allclose(arrays, arrays[:1], rtol=1e-6)


# ---------------------------------------------------------------------------
# 4. The head trainer: --reconstruction encoder with the B4 head
# ---------------------------------------------------------------------------

# The flags head_eval_bb.sh gives the head trainer, at the tiny shape, with
# the reconstruction flag of #425.
HEAD_PROTOCOL = (
    "--device", "cpu", "--quantile-head", "--grad-clip", "1.0",
    "--forecast-len", "16", "--batch-size", "4", "--lr", "1e-3",
    "--total-steps", "3", "--save-every", "1000000", "--log-every", "1",
    "--seed", "20260722", "--head-arch", "transformer",
    "--head-num-layers", "2", "--head-nhead", "2", "--head-ffn-mult", "4.0",
    "--head-causal", "true", "--head-train-input", "e_then_f",
    "--head-dropout", "0.1", "--t-raw", "4096", "--n-channels", "1",
    *TINY_SHAPE, "--encoder-type", "gru", "--rev-norm-kind", "ewma",
    "--rev-norm-span", "128", "--freq-emb-dim", "3",
    "--seasonality-emb-dim", "3", "--reconstruction", "encoder")


@pytest.fixture(scope="module")
def corpus_flags(tmp_path_factory):
    """The #419 test corpus: the stream of a zero-padding backbone."""
    pytest.importorskip("pyarrow")
    from tests.test_419_gift_pretrain import build_corpus, write_index
    root = tmp_path_factory.mktemp("corpus_425")
    folder, index, _ = build_corpus(root / "corpus")
    return ("--gift-pretrain-root", str(folder),
            "--gift-pretrain-index", str(write_index(root, index)))


def train_head(tmp_path, backbone, *extra):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="2")
    return subprocess.run(
        [sys.executable, str(HEAD_PY), "--backbone-path", backbone,
         *HEAD_PROTOCOL, "--save-dir", str(tmp_path / "head"),
         "--run-name", "qrecon", *extra],
        capture_output=True, text=True, env=env, timeout=900)


def losses_of(tmp_path):
    rows = list(csv.DictReader(open(tmp_path / "head" / "qrecon_losses.csv")))
    return [float(r["loss"]) for r in rows]


@pytest.fixture(scope="module")
def bank_head(tmp_path_factory, corpus_flags):
    tmp = tmp_path_factory.mktemp("bank_head")
    bb = saved(tmp, "bb.pth", bank_model())
    return tmp, bb, train_head(tmp, bb, *corpus_flags)


def test_the_trainer_trains_a_reconstruction_bank(bank_head):
    tmp, _, r = bank_head
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "RECONSTRUCTION mode: encoder" in r.stdout
    assert "Head bank (#412)" in r.stdout
    sd = torch.load(tmp / "head" / "qrecon_final.pth", map_location="cpu",
                    weights_only=True)
    assert head_bank_sizes(sd) == SIZES
    losses = losses_of(tmp)
    assert len(losses) == 3 and np.isfinite(losses).all()


def test_the_eval_reconstructs_with_the_trained_bank(bank_head, monkeypatch):
    import pandas as pd
    tmp, bb, r = bank_head
    assert r.returncode == 0, r.stderr[-3000:]
    module = load_eval_module()
    args = eval_args(module, monkeypatch, "--backbone-path", bb,
                     "--head-path", str(tmp / "head" / "qrecon_final.pth"),
                     "--strategy", "R")
    backbone, bank = module.load_models(args, CPU)
    item = {"target": walk(700).numpy(),
            "start": pd.Period("2020-01-01 00:00", freq="h")}
    from gluonts.dataset.util import forecast_start
    label = {"target": walk(48, seed=7).numpy(), "start": forecast_start(item)}
    predictor = module.ReconstructionPredictor(
        backbone=backbone, head=bank.for_frequency(backbone, "h"),
        prediction_length=48, device=CPU, strategy="R",
        context_pad=args.context_pad, labels=[label])
    (forecast,) = list(predictor.predict([item]))
    assert forecast.forecast_array.shape == (1 + Q, 48)
    assert np.isfinite(forecast.forecast_array).all()


def test_the_trainer_trains_the_head_of_a_zero_padding_backbone(
        tmp_path, corpus_flags):
    bb = saved(tmp_path, "bb.pth", ewma_model(zero_pad=True))
    r = train_head(tmp_path, bb, *corpus_flags)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "RECONSTRUCTION mode: encoder" in r.stdout
    sd = torch.load(tmp_path / "head" / "qrecon_final.pth",
                    map_location="cpu", weights_only=True)
    assert sd["forecast_head.weight"].shape[0] == Q * 16
    assert np.isfinite(losses_of(tmp_path)).all()


def test_the_trainer_trains_the_head_of_an_old_backbone(tmp_path):
    bb = saved(tmp_path, "bb.pth", ewma_model(zero_pad=False))
    r = train_head(tmp_path, bb, "--mix-ratio", "1.0", "--hf-repo", "none",
                   "--hf-path", "none")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert np.isfinite(losses_of(tmp_path)).all()


def test_a_forecaster_reconstruction_of_a_bank_is_still_refused(tmp_path):
    bb = saved(tmp_path, "bb.pth", bank_model())
    flags = [f if f != "encoder" else "forecaster" for f in HEAD_PROTOCOL]
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, str(HEAD_PY), "--backbone-path", bb,
                        *flags, "--save-dir", str(tmp_path / "h")],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode != 0
    assert "--reconstruction forecaster" in r.stdout + r.stderr


# ---------------------------------------------------------------------------
# 4b. The shared trainer: the heads of several jobs on one data stream
# ---------------------------------------------------------------------------

SHARED_PY = GIFT_SCRIPTS / "train_forecasting_heads_shared.py"
OLD_DATA = ("--mix-ratio", "1.0", "--hf-repo", "none", "--hf-path", "none")


def shifted(model, by):
    """``model`` with each weight moved by ``by``: another backbone of the
    same kind."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(by)
    return model


def job_argv(folder, backbone, name, *extra):
    """The flags of one solo run of the head trainer, four steps."""
    return ["--backbone-path", backbone, *HEAD_PROTOCOL, "--total-steps", "4",
            "--save-dir", str(folder / name), "--run-name", name, *extra]


def trainer_env():
    return dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="2")


def train_solo(argv):
    return subprocess.run([sys.executable, str(HEAD_PY), *argv],
                          capture_output=True, text=True, env=trainer_env(),
                          timeout=900)


def train_shared(folder, argvs):
    jobs = folder / "jobs.jsonl"
    jobs.parent.mkdir(parents=True, exist_ok=True)
    jobs.write_text("".join(json.dumps(a) + "\n" for a in argvs))
    return subprocess.run([sys.executable, str(SHARED_PY), "--jobs", str(jobs)],
                          capture_output=True, text=True, env=trainer_env(),
                          timeout=900)


def run_files(folder, name):
    """The loss rows and the final head of one run."""
    rows = list(csv.reader(open(folder / name / f"{name}_losses.csv")))
    head = torch.load(folder / name / f"{name}_final.pth", map_location="cpu",
                      weights_only=True)
    return rows, head


def assert_same_run(a, b, name):
    rows_a, head_a = run_files(a, name)
    rows_b, head_b = run_files(b, name)
    assert len(rows_a) == 5 and rows_a == rows_b, name
    assert head_a.keys() == head_b.keys()
    for key in head_a:
        assert torch.equal(head_a[key], head_b[key]), (name, key)


def solo_and_shared_argvs(tmp_path, models, *extra):
    """Train each model's head alone. Returns the flags of the same runs in
    the folder of the shared run."""
    argvs = []
    for name, model in models.items():
        bb = saved(tmp_path, f"{name}.pth", model)
        r = train_solo(job_argv(tmp_path / "solo", bb, name, *extra))
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        argvs.append(job_argv(tmp_path / "shared", bb, name, *extra))
    return argvs


def test_a_shared_run_gives_each_job_the_steps_of_its_solo_run(
        tmp_path, corpus_flags):
    """One job of each kind on the stream of #419: a bank with the mean/std
    scaling, one size with the EWMA, one size with the mean/std scaling. In
    one process, each job writes the losses and the head of its solo run,
    to the last bit: the same batches, and the same draws of patch size,
    split and dropout."""
    models = {"bank": bank_model(), "ewma": ewma_model(zero_pad=True),
              "meanstd": meanstd_model()}
    argvs = solo_and_shared_argvs(tmp_path, models, *corpus_flags)
    r = train_shared(tmp_path, argvs)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for name in models:
        assert_same_run(tmp_path / "solo", tmp_path / "shared", name)
    assert "job steps/s" in r.stdout and "GiB" in r.stdout


def test_a_shared_run_of_old_jobs_gives_their_solo_steps(tmp_path):
    """Two old-data jobs (no padding, vocabulary v1) on one stream."""
    models = {"old_a": ewma_model(), "old_b": shifted(ewma_model(), 0.01)}
    argvs = solo_and_shared_argvs(tmp_path, models, *OLD_DATA)
    r = train_shared(tmp_path, argvs)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for name in models:
        assert_same_run(tmp_path / "solo", tmp_path / "shared", name)


@pytest.mark.parametrize("other", [OLD_DATA, ("--total-steps", "5")])
def test_jobs_of_two_streams_are_refused(tmp_path, corpus_flags, other):
    """An old-data job reads other batches than a job of the stream of #419,
    and a job of five steps reads one batch more."""
    new = saved(tmp_path, "new.pth", ewma_model(zero_pad=True))
    old = saved(tmp_path, "old.pth", ewma_model())
    second = (job_argv(tmp_path, old, "old", *other) if other == OLD_DATA
              else job_argv(tmp_path, new, "old", *corpus_flags, *other))
    r = train_shared(tmp_path, [job_argv(tmp_path, new, "new", *corpus_flags),
                                second])
    assert r.returncode != 0
    assert "one data stream" in r.stdout + r.stderr
    assert not (tmp_path / "new" / "new_losses.csv").exists()


# ---------------------------------------------------------------------------
# 4c. The crash path of the trainer: no cut head, one try in a loss CSV
# ---------------------------------------------------------------------------

def load_head_trainer():
    spec = importlib.util.spec_from_file_location("head_trainer_425", HEAD_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cut_save(obj, f, *args, **kwargs):
    """A torch.save that stops halfway: some bytes, and no end."""
    if isinstance(f, (str, os.PathLike)):
        with open(f, "wb") as out:
            out.write(b"PK\x03\x04")
    else:
        f.write(b"PK\x03\x04")
    raise OSError("the box stops during the save")


def test_a_save_that_stops_halfway_leaves_no_cut_head(tmp_path, monkeypatch):
    """queue.sh takes a job with a `*_final.pth` as trained, and the sync
    copies that file to elisa. A save that stops halfway (a box crash, a
    full disk) must not put a cut file at the path of a head. It must keep
    the earlier head at that path."""
    trainer = load_head_trainer()
    head = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(head.parameters())
    best, final = tmp_path / "q_best.pth", tmp_path / "q_final.pth"
    trainer._save_head(head, optimizer, str(best), 2, 0.5, 2, "student")
    kept = {k: v.clone() for k, v in head.state_dict().items()}
    with torch.no_grad():
        head.weight.add_(1.0)
    with monkeypatch.context() as m:
        m.setattr(torch, "save", cut_save)
        for path in (best, final):
            with pytest.raises(OSError):
                trainer._save_head(head, optimizer, str(path), 3, 0.4, 3,
                                   "student")
    assert not final.exists()
    sd = torch.load(best, map_location="cpu", weights_only=True)
    assert sd.keys() == kept.keys()
    assert all(torch.equal(sd[k], kept[k]) for k in kept)
    meta = torch.load(tmp_path / "q_best_optimizer.pth", weights_only=False)
    assert meta["step"] == 2


def steps_of(tmp_path):
    rows = csv.DictReader(open(tmp_path / "head" / "qrecon_losses.csv"))
    return [int(r["step"]) for r in rows]


def test_a_new_try_of_a_job_starts_a_clean_loss_csv(tmp_path):
    """queue.sh trains a failed job again from step 1: its loss CSV holds
    the new try only. A run that continues from a checkpoint adds its
    rows."""
    bb = saved(tmp_path, "bb.pth", ewma_model(zero_pad=False))
    for _ in range(2):
        r = train_head(tmp_path, bb, *OLD_DATA)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert steps_of(tmp_path) == [1, 2, 3]
    final = tmp_path / "head" / "qrecon_final.pth"
    r = train_head(tmp_path, bb, *OLD_DATA, "--resume", str(final),
                   "--total-steps", "5")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert steps_of(tmp_path) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# 5. The shell runners: head_eval_bb.sh and eval_local.sh
# ---------------------------------------------------------------------------

STUB_HEAD = r'''
import json, os, sys
argv = sys.argv[1:]
out = argv[argv.index("--save-dir") + 1]
name = argv[argv.index("--run-name") + 1]
os.makedirs(out, exist_ok=True)
json.dump(argv, open(os.path.join(out, "head_argv.json"), "w"))
open(os.path.join(out, name + "_final.pth"), "w").write("head")
'''

STUB_EVAL = r'''
import json, os, sys
argv = sys.argv[1:]
out = argv[argv.index("--output-dir") + 1]
os.makedirs(out, exist_ok=True)
json.dump(argv, open(os.path.join(out, "argv.json"), "w"))
with open(os.path.join(out, "all_results.csv"), "w") as f:
    f.write("dataset,model,mase\nm4_yearly/A/short,x,1.0\n")
with open(os.path.join(out, "summary.txt"), "w") as f:
    f.write("Aggregate GM-Relative MASE (1 configs): 0.5000\n")
'''


@pytest.fixture
def stub_checkout(tmp_path):
    """A checkout whose head trainer and eval record their flags, and a PATH
    whose nvidia-smi reports nothing."""
    wt = tmp_path / "wt"
    scripts = wt / "experiments" / "2026-04-13_gift-eval" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "train_forecasting_head.py").write_text(STUB_HEAD)
    (scripts / "eval_gift_eval_official.py").write_text(STUB_EVAL)
    (wt / "experiments" / "hf_token.txt").write_text("hf_test\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    (tmp_path / "gift").mkdir()
    bb = tmp_path / "bb.pth"
    bb.write_text("backbone")
    env = dict(os.environ, WT=str(wt), CF373_ROOT=str(tmp_path / "root"),
               CF_RESULTS=str(tmp_path / "res"), GIFT_EVAL=str(tmp_path / "gift"),
               PATH=f"{bin_dir}:{os.environ['PATH']}",
               GPU_GATE_LOCKDIR=str(tmp_path), HEAD_VRAM_MIB="0",
               CF393_EVAL_SLOTDIR=str(tmp_path / "slots"),
               EVAL_CONFIG_FILTER="^m4_yearly/short$", EVAL_EXPECT_CONFIGS="1")
    for key in ("CF_RECONSTRUCTION", "CF_SKIP_EVAL", "HEAD_SAVE_EVERY",
                "EVAL_STRATEGY", "EVAL_DEVICE"):
        env.pop(key, None)
    return tmp_path, bb, env


def head_eval(stub, tag, **extra):
    tmp_path, bb, env = stub
    return subprocess.run(
        ["bash", str(B4_SCRIPTS / "head_eval_bb.sh"), tag, str(bb), "student",
         "30000"], capture_output=True, text=True, env=dict(env, **extra),
        timeout=300)


def recorded(path):
    import json
    return json.load(open(path))


def test_the_reconstruction_mode_reaches_the_head_and_the_eval(stub_checkout):
    tmp_path = stub_checkout[0]
    r = head_eval(stub_checkout, "arm_bb40k_h30k_recon",
                  CF_RECONSTRUCTION="encoder", HEAD_SAVE_EVERY="1000000")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = tmp_path / "root" / "eval" / "arm_bb40k_h30k_recon"
    head = recorded(out / "head_argv.json")
    assert head[head.index("--reconstruction") + 1] == "encoder"
    assert head[head.index("--save-every") + 1] == "1000000"
    assert head[head.index("--total-steps") + 1] == "30000"
    shard = recorded(out / "gift_r" / "shard_0" / "argv.json")
    assert shard[shard.index("--strategy") + 1] == "R"
    assert (tmp_path / "res" / "score_arm_bb40k_h30k_recon.txt"
            ).read_text().strip() == "0.5000"
    assert not (out / "gift").exists()


def test_the_forecast_mode_is_unchanged(stub_checkout):
    tmp_path = stub_checkout[0]
    r = head_eval(stub_checkout, "arm_bb40k_h30k_student")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = tmp_path / "root" / "eval" / "arm_bb40k_h30k_student"
    head = recorded(out / "head_argv.json")
    assert "--reconstruction" not in head
    assert head[head.index("--save-every") + 1] == "5000"
    shard = recorded(out / "gift" / "shard_0" / "argv.json")
    assert shard[shard.index("--strategy") + 1] == "B4"
    assert (tmp_path / "res" / "score_arm_bb40k_h30k_student.txt").exists()


def test_the_shards_run_on_the_cpu_unless_the_caller_asks(stub_checkout):
    tmp_path = stub_checkout[0]
    r = head_eval(stub_checkout, "cpu_bb40k_h30k_recon",
                  CF_RECONSTRUCTION="encoder")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    shard = recorded(tmp_path / "root" / "eval" / "cpu_bb40k_h30k_recon"
                     / "gift_r" / "shard_0" / "argv.json")
    assert shard[shard.index("--device") + 1] == "cpu"
    r = head_eval(stub_checkout, "gpu_bb40k_h30k_recon",
                  CF_RECONSTRUCTION="encoder", EVAL_DEVICE="cuda")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    shard = recorded(tmp_path / "root" / "eval" / "gpu_bb40k_h30k_recon"
                     / "gift_r" / "shard_0" / "argv.json")
    assert shard[shard.index("--device") + 1] == "cuda"


def test_skip_eval_trains_the_head_and_scores_nothing(stub_checkout):
    tmp_path = stub_checkout[0]
    r = head_eval(stub_checkout, "arm_bb40k_h30k_recon",
                  CF_RECONSTRUCTION="encoder", CF_SKIP_EVAL="1")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = tmp_path / "root" / "eval" / "arm_bb40k_h30k_recon"
    assert (out / "head_argv.json").exists()
    assert not (out / "gift_r").exists()
    assert not (tmp_path / "res" / "score_arm_bb40k_h30k_recon.txt").exists()
    again = head_eval(stub_checkout, "arm_bb40k_h30k_recon",
                      CF_RECONSTRUCTION="encoder")
    assert again.returncode == 0, again.stdout[-3000:]
    assert "head-train SKIP (final exists)" in again.stdout
    assert (tmp_path / "res" / "score_arm_bb40k_h30k_recon.txt").exists()


def test_a_reconstruction_tag_must_say_so(stub_checkout):
    """A forecast tag in the reconstruction mode would read the forecast
    head and its score file."""
    r = head_eval(stub_checkout, "arm_bb40k_h30k_student",
                  CF_RECONSTRUCTION="encoder")
    assert r.returncode != 0
    assert "_recon" in r.stdout + r.stderr


def test_an_unknown_reconstruction_mode_is_refused(stub_checkout):
    r = head_eval(stub_checkout, "arm_bb40k_h30k_recon",
                  CF_RECONSTRUCTION="forecaster")
    assert r.returncode != 0


def test_the_argv_mode_hands_the_solo_flags_to_a_shared_trainer(
        stub_checkout):
    """CF_HEAD_ARGV_TO: the runner adds the flags of its head trainer to the
    file, as one JSON line, and trains and scores nothing. A head that
    exists adds no line."""
    tmp_path = stub_checkout[0]
    jobs = tmp_path / "jobs.jsonl"
    knobs = dict(CF_RECONSTRUCTION="encoder", HEAD_SAVE_EVERY="1000000")
    r = head_eval(stub_checkout, "arm_bb40k_h30k_recon",
                  CF_HEAD_ARGV_TO=str(jobs), **knobs)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = tmp_path / "root" / "eval" / "arm_bb40k_h30k_recon"
    assert not (out / "head_argv.json").exists() and not (out / "gift_r").exists()
    (argv,) = [json.loads(line) for line in jobs.read_text().splitlines()]
    solo = head_eval(stub_checkout, "arm_bb40k_h30k_recon", CF_SKIP_EVAL="1",
                     **knobs)
    assert solo.returncode == 0, solo.stdout[-3000:]
    assert argv == recorded(out / "head_argv.json")
    again = head_eval(stub_checkout, "arm_bb40k_h30k_recon",
                      CF_HEAD_ARGV_TO=str(jobs), **knobs)
    assert again.returncode == 0 and len(jobs.read_text().splitlines()) == 1
    assert not (out / "gift_r").exists()


def eval_local(stub, strategy, head, **extra):
    tmp_path, bb, env = stub
    out = tmp_path / "out"
    return subprocess.run(
        ["bash", str(B4_SCRIPTS / "eval_local.sh"), "floor_x", "0", "student",
         str(bb), head, str(out), str(tmp_path / "res" / "score_floor_x.txt")],
        capture_output=True, text=True,
        env=dict(env, EVAL_STRATEGY=strategy, **extra), timeout=300), out


def test_the_floor_strategy_scores_r_with_the_zero_head(stub_checkout):
    """EVAL_STRATEGY=R0: R with --zero-head, no head file, its own folder."""
    tmp_path = stub_checkout[0]
    r, out = eval_local(stub_checkout, "R0", "none")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    shard = recorded(out / "gift_r0" / "shard_0" / "argv.json")
    assert shard[shard.index("--strategy") + 1] == "R"
    assert "--zero-head" in shard and "--head-path" not in shard
    assert (tmp_path / "res" / "score_floor_x.txt").read_text().strip() == "0.5000"
    r, _ = eval_local(stub_checkout, "R", "none")
    assert r.returncode != 0 and "no head" in r.stdout + r.stderr


# ---------------------------------------------------------------------------
# 6. The scripts of the report
# ---------------------------------------------------------------------------

SCRIPTS = STUDY / "scripts"
JOBS = SCRIPTS / "jobs.tsv"

# The card's table: each run of ours and its checkpoints on disk on 10-06.
CARD = {
    "BLK": (100, 200, 300, 400),
    "OMB": (10, 25, 50, 75, 100, 125, 150, 166), "OCB": (40,),
    "OCF": (40, 100, 200, 300, 400), "OEF": (40, 100, 200),
    "OMF": (10, 25, 50, 75, 100, 125), "OAF": (40, 100, 200), "OWF": (40,),
    "OWR": (40, 100, 200), "OWL": (40,), "OBM": (10, 25, 50),
    "OBW": (10, 25, 50), "OAL": (10, 25, 50), "BMS": (40, 100, 140),
    "CYN": (665, 1140, 1330), "MIN": (1000, 1080), "TWN": (100, 420),
    "LOW": (665,), "LNG": (665,),
}


OLD_RUNS = {"CYN", "MIN", "TWN", "LOW", "LNG"}


def job_rows(path=JOBS):
    return [line.rstrip("\n").split("\t") for line in open(path)
            if line.strip() and not line.startswith("#")]


def test_the_job_table_holds_the_card_checkpoints():
    sys.path.insert(0, str(REPO_ROOT / "reports" / "2026-09-06_moirai_small_size"
                           / "scripts"))
    from run_style import CODE
    rows = job_rows()
    assert len(rows) == 56
    stops = {}
    for code, arm, stop_k, tier, ckpt, size, data in rows:
        assert CODE[arm] == code
        stops.setdefault(code, []).append(int(stop_k))
        assert int(size) > 50_000_000 and ckpt.endswith(f"_{stop_k}k.pth")
        first_or_last = int(stop_k) in (CARD[code][0], CARD[code][-1])
        assert tier == ("1" if first_or_last else "2")
        assert data == ("old" if code in OLD_RUNS else "gift_pretrain")
    assert {c: tuple(s) for c, s in stops.items()} == CARD


QUEUE_STUB = r'''#!/bin/bash
# A runner that records each call. CF_HEAD_ARGV_TO: the flags of the head.
# Else: the score.
tag="$1"; out="$CF373_ROOT/eval/$tag"; mkdir -p "$out"
if [ -n "${CF_HEAD_ARGV_TO:-}" ]; then
  echo "$tag argv $CF_RECONSTRUCTION $4 $HEAD_SAVE_EVERY" >>"$CF_RESULTS/calls.log"
  printf '["--save-dir", "%s", "--run-name", "qhead_%s"]\n' "$out" "$tag" \
    >>"$CF_HEAD_ARGV_TO"
  exit 0
fi
echo "$tag score $CF_RECONSTRUCTION" >>"$CF_RESULTS/calls.log"
sleep 0.2
echo 0.1234 >"$CF_RESULTS/score_$tag.txt"
'''

TRAINER_STUB = r'''
import json, os, sys, time
jobs = [json.loads(line) for line in open(sys.argv[sys.argv.index("--jobs") + 1])]
res = os.environ["CF425_TEST_RES"]
names = [a[a.index("--run-name") + 1][len("qhead_"):] for a in jobs]
with open(os.path.join(res, "calls.log"), "a") as f:
    f.write("wave " + " ".join(names) + "\n")
with open(os.path.join(res, "gpu.log"), "a") as f:
    f.write(os.environ.get("CUDA_VISIBLE_DEVICES", "unset") + "\n")
print("[shared] 1 steps: a stub", flush=True)
time.sleep(float(os.environ.get("CF425_TEST_TRAIN_SLEEP", "0")))
for a, name in zip(jobs, names):
    if "bad" not in name:
        out = a[a.index("--save-dir") + 1]
        open(os.path.join(out, f"qhead_{name}_final.pth"), "w").write("head")
with open(os.path.join(res, "calls.log"), "a") as f:
    f.write("wave_end " + " ".join(names) + "\n")
sys.exit(1 if any("bad" in name for name in names) else 0)
'''


@pytest.fixture
def queue_box(tmp_path):
    """A box of five jobs on two data streams, with sparse input files, a
    stub runner and a stub shared trainer."""
    ck, res = tmp_path / "ckpt", tmp_path / "res"
    rows = [("AAA", "arm_a", "40", "2", "gift_pretrain"),
            ("AAA", "arm_a", "10", "1", "gift_pretrain"),
            ("BAD", "arm_bad", "40", "1", "gift_pretrain"),
            ("CCC", "arm_c", "100", "1", "old"),
            ("DDD", "arm_d", "50", "2", "old")]
    table = ["#code\tarm\tstop_k\ttier\tckpt\tbytes\tdata"]
    for code, arm, stop, tier, data in rows:
        rel = f"{arm}/leg/{arm}_{stop}k.pth"
        (ck / rel).parent.mkdir(parents=True, exist_ok=True)
        with open(ck / rel, "wb") as f:
            f.truncate(1000 + int(stop))
        table.append("\t".join([code, arm, stop, tier, rel,
                                str(1000 + int(stop)), data]))
    (tmp_path / "jobs.tsv").write_text("\n".join(table) + "\n")
    stub = tmp_path / "runner.sh"
    stub.write_text(QUEUE_STUB)
    (tmp_path / "trainer.py").write_text(TRAINER_STUB)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "hf_token.txt").write_text("hf_test\n")
    with open(tmp_path / "seasonal_naive.csv", "wb") as f:
        f.truncate(24831)
    res.mkdir()
    env = dict(os.environ, CF425_JOBS=str(tmp_path / "jobs.tsv"),
               CF425_CK=str(ck), CF425_ROOT=str(ck / "cf-425" / "recon"),
               CF425_RES=str(res), CF425_RUNNER=str(stub),
               CF425_TRAINER=str(tmp_path / "trainer.py"),
               CF425_CODE=str(tmp_path), CF425_WAVE_SIZE="2",
               CF425_OLD_WAVE_SIZE="2", CF425_MIN_FREE_GB="0",
               CF425_WAVE_VRAM_MIB="0", CF425_LANE_STAGGER="0",
               CF425_GPU_LOCK=str(tmp_path / "gpu.lock"),
               CF425_SN_REF=str(tmp_path / "seasonal_naive.csv"),
               CF425_AFTER=str(tmp_path / "forecast_scores.sh"),
               CF425_AFTER_POLL="1", CF425_GPU_POLL="0.1",
               CF425_TEST_RES=str(res))
    for key in ("CF425_SCORE", "CF425_DRY_RUN", "CF425_TRIES"):
        env.pop(key, None)
    return tmp_path, res, env


def run_queue(env, timeout=120):
    return subprocess.run(["bash", str(SCRIPTS / "queue.sh")],
                          capture_output=True, text=True, env=env,
                          timeout=timeout)


def calls(res):
    return [line.split() for line in open(res / "calls.log")]


def waves(res):
    return [c[1:] for c in calls(res) if c[0] == "wave"]


def job_calls(res):
    """The calls of the runner: "<tag> argv ..." and "<tag> score ..."."""
    return [c for c in calls(res) if c[0] not in ("wave", "wave_end")]


GOOD = ["arm_a_bb10k_h30k_recon", "arm_a_bb40k_h30k_recon",
        "arm_c_bb100k_h30k_recon", "arm_d_bb50k_h30k_recon"]
BAD = "arm_bad_bb40k_h30k_recon"


def test_the_queue_trains_in_waves_and_scores_every_job_once(queue_box):
    _, res, env = queue_box
    r = run_queue(env)
    assert r.returncode == 0, r.stdout + r.stderr
    done = sorted(p.name[6:-4] for p in res.glob("score_*.txt"))
    assert done == GOOD
    scored = [c[0] for c in job_calls(res) if c[1] == "score"]
    assert sorted(scored) == GOOD
    trained = [tag for wave in waves(res) for tag in wave]
    assert sorted(t for t in trained if t != BAD) == GOOD
    assert all(c[2] == "encoder" for c in job_calls(res))


def test_each_wave_reads_one_data_stream(queue_box):
    _, res, env = queue_box
    r = run_queue(env)
    assert r.returncode == 0, r.stdout + r.stderr
    old = {"arm_c_bb100k_h30k_recon", "arm_d_bb50k_h30k_recon"}
    for wave in waves(res):
        assert 1 <= len(wave) <= 2
        assert set(wave) <= old or not set(wave) & old, wave


def test_the_queue_takes_tier_1_first(queue_box):
    _, res, env = queue_box
    r = run_queue(dict(env, CF425_WAVE_SIZE="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    new = [w[0] for w in waves(res) if not w[0].startswith(("arm_c", "arm_d"))]
    assert new[-1] == "arm_a_bb40k_h30k_recon"
    assert waves(res)[0] != ["arm_d_bb50k_h30k_recon"]


def test_a_failing_head_stops_after_its_tries(queue_box):
    _, res, env = queue_box
    r = run_queue(dict(env, CF425_TRIES="2"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert sum(BAD in wave for wave in waves(res)) == 2
    assert len(list((res / "failed").glob(f"{BAD}.*"))) == 2
    again = run_queue(env)
    assert again.returncode == 0
    assert sum(BAD in wave for wave in waves(res)) == 2


# The first CF425_TEST_FAILS scores of the job CF425_TEST_FLAKY fail, late.
FLAKY_SCORE = r'''sleep 0.2
tries=$(grep -c "^$tag score" "$CF_RESULTS/calls.log")
if [ "$tag" = "$CF425_TEST_FLAKY" ] && [ "$tries" -le "$CF425_TEST_FAILS" ]; then
  sleep 1; exit 1
fi'''


@pytest.mark.parametrize("fails", [1, 2])
def test_a_score_that_fails_after_the_last_wave_gets_its_next_try(queue_box,
                                                                  fails):
    """The score of the last wave of a lane is not complete when the lane
    finds no job to lock. If that score fails, the lane gives it its next
    try, and the lane stops after the last try."""
    tmp_path, res, env = queue_box
    (tmp_path / "runner.sh").write_text(
        QUEUE_STUB.replace("sleep 0.2", FLAKY_SCORE))
    flaky = GOOD[1]   # tier 2: the last wave of its lane
    r = run_queue(dict(env, CF425_TEST_FLAKY=flaky,
                       CF425_TEST_FAILS=str(fails)))
    assert r.returncode == 0, r.stdout + r.stderr
    lane = [w for w in waves(res) if not w[0].startswith(("arm_c", "arm_d"))]
    assert flaky in lane[-1]
    scored = [c[0] for c in job_calls(res) if c[1] == "score"]
    assert scored.count(flaky) == 2
    assert len(list((res / "failed").glob(f"{flaky}.*"))) == fails
    assert (res / f"score_{flaky}.txt").exists() == (fails == 1)


def test_a_job_with_a_head_is_scored_and_not_trained_again(queue_box):
    tmp_path, res, env = queue_box
    out = tmp_path / "ckpt" / "cf-425" / "recon" / "eval" / GOOD[2]
    out.mkdir(parents=True)
    (out / f"qhead_{GOOD[2]}_s20260722_final.pth").write_text("head")
    r = run_queue(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not any(GOOD[2] in wave for wave in waves(res))
    assert not any(c[:2] == [GOOD[2], "argv"] for c in job_calls(res))
    assert (res / f"score_{GOOD[2]}.txt").exists()


def test_a_runner_that_reads_stdin_takes_no_job_of_the_list(queue_box):
    """A lane reads the job list on stdin, and the runner gets /dev/null.
    A runner that reads its stdin changes no job of the queue."""
    tmp_path, res, env = queue_box
    stub = tmp_path / "runner.sh"
    stub.write_text(QUEUE_STUB.replace("tag=\"$1\";", "cat >/dev/null; tag=\"$1\";"))
    r = run_queue(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(list(res.glob("score_*.txt"))) == 4


def test_a_score_that_outlives_its_queue_leaves_the_queue_lock_free(queue_box):
    """Kill the queue while a score runs. A new queue starts at once, and it
    leaves the running job to the score that holds its lock."""
    import signal
    import time
    tmp_path, res, env = queue_box
    stub = tmp_path / "runner.sh"
    stub.write_text(QUEUE_STUB.replace("sleep 0.2", "sleep 4"))
    first = subprocess.Popen(["bash", str(SCRIPTS / "queue.sh")], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            log = res / "calls.log"
            if log.exists() and " score " in log.read_text():
                break
            time.sleep(0.1)
        first.send_signal(signal.SIGKILL)
        first.wait()
        again = run_queue(env, timeout=120)
    finally:
        subprocess.run(["pkill", "-f", str(stub)])
    assert again.returncode == 0, again.stdout + again.stderr
    assert "another queue holds" not in again.stdout
    scored = [c[0] for c in job_calls(res) if c[1] == "score"]
    assert len(scored) == len(set(scored))


def test_a_job_another_process_holds_is_skipped(queue_box):
    """A wave holds the lock of each job and passes it to the job's score,
    so a second queue never runs a job that an earlier one still runs."""
    _, res, env = queue_box
    (res / "locks").mkdir(parents=True)
    lock = res / "locks" / "arm_c_bb100k_h30k_recon.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "30"])
    try:
        import time
        time.sleep(0.5)
        r = run_queue(env)
    finally:
        holder.kill()
    assert r.returncode == 0, r.stdout + r.stderr
    assert not any(c[0] == "arm_c_bb100k_h30k_recon" for c in job_calls(res))
    assert not any("arm_c_bb100k_h30k_recon" in w for w in waves(res))
    assert not (res / "score_arm_c_bb100k_h30k_recon.txt").exists()


def test_the_queue_refuses_a_missing_input(queue_box):
    tmp_path, res, env = queue_box
    (tmp_path / "ckpt" / "arm_c" / "leg" / "arm_c_100k.pth").unlink()
    r = run_queue(env)
    assert r.returncode == 3
    assert "arm_c_100k.pth" in r.stdout and "stage_inputs.sh" in r.stdout
    assert not (res / "calls.log").exists()


def test_the_queue_refuses_a_box_with_no_seasonal_naive_reference(queue_box):
    """Each score reads the reference, and git does not hold it: with no
    reference, no head may train."""
    tmp_path, res, env = queue_box
    (tmp_path / "seasonal_naive.csv").write_text("cut")
    r = run_queue(env)
    assert r.returncode == 3
    assert "seasonal-naive reference" in r.stdout
    assert not (res / "calls.log").exists()


def test_the_queue_starts_after_the_forecast_scores(queue_box):
    """forecast_scores.sh uses the same GPU with other locks."""
    import time
    tmp_path, res, env = queue_box
    script = tmp_path / "forecast_scores.sh"
    script.write_text("sleep 3\n")
    other = subprocess.Popen(["bash", str(script)])
    try:
        time.sleep(0.3)
        r = run_queue(env)
        assert other.poll() is not None
    finally:
        other.kill()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "waiting" in r.stdout
    assert len(list(res.glob("score_*.txt"))) == 4


def test_the_gpu_lock_frees_at_the_first_step_of_a_wave(queue_box):
    """One lock serialises the start of the waves of both lanes, and the
    trainer does not keep it: the wave of the other lane starts while the
    first one still trains."""
    _, res, env = queue_box
    r = run_queue(dict(env, CF425_TEST_TRAIN_SLEEP="2", CF425_WAVE_SIZE="3"))
    assert r.returncode == 0, r.stdout + r.stderr
    order = [c[0] for c in calls(res) if c[0] in ("wave", "wave_end")]
    assert order[:3] == ["wave", "wave", "wave_end"]


def test_the_dry_run_lists_the_waves_and_runs_nothing(queue_box):
    _, res, env = queue_box
    r = run_queue(dict(env, CF425_DRY_RUN="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    plan = [line.split() for line in r.stdout.splitlines()]
    assert [(p[0], p[1], p[5]) for p in plan] == [
        ("old", "1", "arm_c_bb100k_h30k_recon"),
        ("old", "1", "arm_d_bb50k_h30k_recon"),
        ("gift_pretrain", "1", "arm_a_bb10k_h30k_recon"),
        ("gift_pretrain", "1", "arm_bad_bb40k_h30k_recon"),
        ("gift_pretrain", "2", "arm_a_bb40k_h30k_recon")]
    assert not (res / "calls.log").exists()


def test_the_queue_hands_the_runner_the_b4_head_steps(queue_box):
    """Each job gets the reconstruction mode, no snapshot every 5,000
    steps, and the 30,000 head steps."""
    _, res, env = queue_box
    r = run_queue(env)
    assert r.returncode == 0, r.stdout + r.stderr
    argv = {tuple(c[2:]) for c in job_calls(res) if c[1] == "argv"}
    assert argv == {("encoder", "30000", "1000000")}


def test_the_waves_train_on_the_gpu_of_the_queue(queue_box):
    _, res, env = queue_box
    r = run_queue(dict(env, CF425_GPU="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert set((res / "gpu.log").read_text().split()) == {"1"}


def test_the_score_knob_stops_each_lane_after_its_heads(queue_box):
    """CF425_SCORE=0 (the probe of one wave): the heads train, and no job
    is scored."""
    _, res, env = queue_box
    r = run_queue(dict(env, CF425_SCORE="0", CF425_WAVE_SIZE="3"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert not any(c[1] == "score" for c in job_calls(res))
    assert len(waves(res)) == 3


def prune_tree(tmp_path):
    """A box tree: a finished and scored head, a finished head with no
    score yet, a head that still trains, and an input checkpoint."""
    root, res = tmp_path / "cf-425", tmp_path / "res"
    files = {"recon/eval/a_recon/qhead_a_final.pth": 10,
             "recon/eval/a_recon/qhead_a_final_optimizer.pth": 20,
             "recon/eval/a_recon/qhead_a_best.pth": 10,
             "recon/eval/a_recon/qhead_a_losses.csv": 5,
             "recon/eval/b_recon/qhead_b_final.pth": 10,
             "recon/eval/b_recon/qhead_b_best.pth": 10,
             "recon/eval/c_recon/qhead_c_best.pth": 10}
    for rel, size in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x" * size)
    res.mkdir()
    (res / "score_a_recon.txt").write_text("0.2\n")
    return root, res, files


def run_prune(tmp_path, root, res, manifest_rows, *extra):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("".join(f"{size}\t{rel}\n" for rel, size in manifest_rows))
    env = dict(os.environ, CF425_PRUNE_ROOT=str(root), CF425_RES=str(res))
    return subprocess.run(["bash", str(SCRIPTS / "prune.sh"), str(manifest),
                           *extra], capture_output=True, text=True, env=env,
                          timeout=60)


def test_prune_deletes_only_what_elisa_holds_at_its_size(tmp_path):
    root, res, files = prune_tree(tmp_path)
    held = dict(files)
    held["recon/eval/b_recon/qhead_b_best.pth"] = 11       # another size
    r = run_prune(tmp_path, root, res, held.items())
    assert r.returncode == 0, r.stdout + r.stderr
    left = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert left == {
        "recon/eval/a_recon/qhead_a_losses.csv",     # not a checkpoint
        "recon/eval/b_recon/qhead_b_final.pth",      # no score yet
        "recon/eval/b_recon/qhead_b_best.pth",       # elisa: another size
        "recon/eval/c_recon/qhead_c_best.pth",       # the head still trains
    }


def test_prune_without_a_manifest_line_deletes_nothing(tmp_path):
    root, res, files = prune_tree(tmp_path)
    r = run_prune(tmp_path, root, res, [("../outside.pth", 10)])
    assert r.returncode == 0
    assert all((root / rel).exists() for rel in files)


def test_the_prune_dry_run_deletes_nothing(tmp_path):
    root, res, files = prune_tree(tmp_path)
    r = run_prune(tmp_path, root, res, files.items(), "--dry-run")
    assert r.returncode == 0 and "would delete" in r.stdout
    assert all((root / rel).exists() for rel in files)


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"cf425_{name}",
                                                  SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_writes_the_two_tables(tmp_path, monkeypatch):
    collect = load_script("collect")
    box = tmp_path / "box"
    box.mkdir()
    (box / "score_cf412om_bb10k_h30k_recon.txt").write_text("0.3100\n")
    (box / "score_cf412om_bb25k_h30k_recon.txt").write_text("")   # no score
    (box / "score_k3_x_lr30x_bb665k_h30k_student.txt").write_text("1.1700\n")
    (box / "score_smoke_bank_s500_recon.txt").write_text("0.5\n")  # no stop
    found = collect.scores(box)
    assert found == {"recon": [("cf412om", 10, 0.31)],
                     "student": [("k3_x_lr30x", 665, 1.17)]}
    out = tmp_path / "t.tsv"
    collect.write_table(found["recon"], out)
    assert out.read_text() == "cf412om\t10\t0.3100\n"


def test_collect_copies_the_raw_artefacts_and_no_head(tmp_path):
    """collect.py copies the score file, the per-config table, the logs and
    the head loss CSV of each scored job, and never a head file."""
    import gzip
    collect = load_script("collect")
    box, mirror = tmp_path / "box", tmp_path / "mirror" / "cf-425"
    tag = "cf412om_bb10k_h30k_recon"
    (box / "waves" / "gift_x").mkdir(parents=True)
    (box / f"score_{tag}.txt").write_text("0.3100\n")
    (box / "queue.log").write_text("queue\n")
    (box / "waves" / "gift_x" / "train.log").write_text("[shared] done\n")
    job = mirror / "recon" / "eval" / tag
    (job / "gift_r").mkdir(parents=True)
    (job / "gift_r" / "all_results.csv").write_text("dataset,mase\n")
    (job / "gift_r" / "summary.txt").write_text("summary\n")
    (job / "stop.log").write_text("stop\n")
    (job / f"qhead_{tag}_s20260722_final.pth").write_bytes(b"head")
    (job / f"qhead_{tag}_s20260722_losses.csv").write_text("step,loss\n1,0.5\n")
    results = tmp_path / "results"
    collect.BOX_RESULTS, collect.MIRROR, collect.RESULTS = box, mirror, results
    collect.SYNC_LOG, collect.LINEAR = tmp_path / "no_sync.log", tmp_path / "lin"
    collect.ELISA_SNAPSHOTS = tmp_path / "no_snap"
    collect.main()
    assert (results / "recon_trajectories.tsv").read_text() == "cf412om\t10\t0.3100\n"
    assert (results / "scores" / f"score_{tag}.txt").read_text() == "0.3100\n"
    assert (results / "per_config" / f"{tag}.csv").is_file()
    assert (results / "logs" / "jobs" / tag / "summary.txt").is_file()
    assert (results / "logs" / "waves" / "gift_x" / "train.log").is_file()
    assert (results / "logs" / "queue.log").is_file()
    packed = results / "head_losses" / f"{tag}_losses.csv.gz"
    assert gzip.decompress(packed.read_bytes()) == b"step,loss\n1,0.5\n"
    first = packed.read_bytes()
    collect.main()   # the same CSV gives the same bytes
    assert packed.read_bytes() == first
    assert not list(results.rglob("*.pth"))


def test_collect_gives_the_snapshot_scores_their_own_table(tmp_path):
    """The scores of snapshot_score.sh stay out of the table of the jobs.
    Their own table has one row for each score, from the box and from
    elisa: the run, the scaling, the head step of the snapshot (the step of
    the best head from the log of its wave, and 30,000 for the final head),
    the machine of the eval, the R of the snapshot, the R of the final head
    in the queue, and the ratio of the two. A snapshot with a score from the
    two machines keeps the score of the box. A job with no score in the
    queue has no R of the final head."""
    collect = load_script("collect")
    box, mirror = tmp_path / "box", tmp_path / "mirror" / "cf-425"
    snap = tmp_path / "snap"
    (box / "snapshots").mkdir(parents=True)
    (box / "waves" / "gift_x").mkdir(parents=True)
    (snap / "results").mkdir(parents=True)
    (box / "score_cf412om_bb10k_h30k_recon.txt").write_text("0.3100\n")
    (box / "waves" / "gift_x" / "train.log").write_text(
        "[qhead_cf412om_bb10k_h30k_recon_s20260722] Done in 4.5h. "
        "Best loss=0.007695 at step 28500\n"
        "[qhead_cf412om_bb25k_h30k_recon_s20260722] Done in 4.5h. "
        "Best loss=0.5 at step 25000\n")
    for snapshot, score in (("best", "0.3000\n"), ("final", "0.3100\n")):
        tag = f"cf412om_bb10k_h30k_{snapshot}_recon"
        (box / "snapshots" / f"score_{tag}.txt").write_text(score)
        gift = mirror / "snapshots" / "eval" / tag / "gift_r"
        gift.mkdir(parents=True)
        (gift / "all_results.csv").write_text("dataset,mase\n")
    for tag, score in (("cf412om_bb25k_h30k_best_recon", "0.5000\n"),
                       ("cf412om_bb10k_h30k_final_recon", "0.9999\n")):
        (snap / "results" / f"score_{tag}.txt").write_text(score)
        gift = snap / "ckpt" / "eval" / tag / "gift_r"
        gift.mkdir(parents=True)
        (gift / "all_results.csv").write_text("dataset,mase\nelisa,1\n")
        (gift / "summary.txt").write_text("summary of elisa\n")
    results = tmp_path / "results"
    results.mkdir()
    (results / "floors.tsv").write_text(FLOORS_TSV)
    collect.BOX_RESULTS, collect.MIRROR, collect.RESULTS = box, mirror, results
    collect.SYNC_LOG, collect.LINEAR = tmp_path / "no_sync.log", tmp_path / "lin"
    collect.ELISA_SNAPSHOTS = snap
    collect.main()
    assert (results / "recon_trajectories.tsv").read_text() == (
        "cf412om\t10\t0.3100\n")
    assert (results / "snapshots" / "scores.tsv").read_text() == (
        "run\tarm\tstop_k\tscaling\tsnapshot\thead_step\tmachine"
        "\tr_snapshot\tr_final\tratio\n"
        "OMB\tcf412om\t10\tmean/std\tbest\t28500\tbox\t0.3000\t0.3100\t1.03\n"
        "OMB\tcf412om\t10\tmean/std\tfinal\t30000\tbox\t0.3100\t0.3100\t1.00\n"
        "OMB\tcf412om\t25\tmean/std\tbest\t25000\telisa\t0.5000\t\t\n")
    per_config = results / "snapshots" / "per_config"
    assert (per_config / "cf412om_bb10k_h30k_best_recon.csv").is_file()
    assert (per_config / "cf412om_bb10k_h30k_final_recon.csv"
            ).read_text() == "dataset,mase\n"                 # of the box
    assert (per_config / "cf412om_bb25k_h30k_best_recon.csv"
            ).read_text() == "dataset,mase\nelisa,1\n"
    assert (results / "snapshots" / "logs" / "cf412om_bb25k_h30k_best_recon"
            / "summary.txt").read_text() == "summary of elisa\n"


def snapshot_on_elisa(stub_checkout, snapshot="best", **extra):
    """snapshot_score.sh with CF425_SNAP_GPU, for the first job of jobs.tsv.
    elisa's mirror holds the checkpoint of the job and its two heads. A call
    of ssh or scp leaves a line in ``box_calls``. Returns the result, the
    mirror, the folder of the snapshots, the checkpoint and the tag."""
    tmp_path, _, env = stub_checkout
    code, arm, stop, _, ckpt = job_rows()[0][:5]
    mirror, base = tmp_path / "mirror", tmp_path / "snap"
    (mirror / ckpt).parent.mkdir(parents=True, exist_ok=True)
    (mirror / ckpt).write_text("backbone")
    job = f"{arm}_bb{stop}k_h30k_recon"
    heads = mirror / "cf-425" / "recon" / "eval" / job
    heads.mkdir(parents=True, exist_ok=True)
    for name in ("best", "final"):
        (heads / f"qhead_{job}_s20260722_{name}.pth").write_text(f"{name} head")
    for tool in ("ssh", "scp"):
        path = tmp_path / "bin" / tool
        path.write_text(f"#!/bin/sh\necho {tool} >>{tmp_path}/box_calls\n"
                        "exit 1\n")
        path.chmod(0o755)
    env = {**env, "CF425_SNAP_GPU": "1", "CF425_SNAP_BASE": str(base),
           "CF425_MIRROR": str(mirror), "CF425_CODE": env["WT"],
           "CF425_RUNNER": str(B4_SCRIPTS / "head_eval_bb.sh"), **extra}
    r = subprocess.run(["bash", str(SCRIPTS / "snapshot_score.sh"), code, stop,
                        snapshot], capture_output=True, text=True, env=env,
                       timeout=60)
    return r, mirror, base, ckpt, f"{arm}_bb{stop}k_h30k_{snapshot}_recon"


def test_a_snapshot_score_runs_on_a_gpu_of_elisa(stub_checkout):
    """With CF425_SNAP_GPU, snapshot_score.sh scores a snapshot on elisa. It
    copies the best head of the job, under the name of a final head, into
    the folders of the snapshots. It trains no head, and it scores strategy
    R on the GPU with the checkpoint of elisa's mirror. It does not use the
    box."""
    import time
    tmp_path = stub_checkout[0]
    r, mirror, base, ckpt, tag = snapshot_on_elisa(stub_checkout)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "on GPU 1 of elisa" in r.stdout
    score = base / "results" / f"score_{tag}.txt"
    deadline = time.time() + 60
    while not score.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert score.read_text().strip() == "0.5000"
    out = base / "ckpt" / "eval" / tag
    head = out / f"qhead_{tag}_s20260722_final.pth"
    assert head.read_text() == "best head"
    assert not (out / "head_argv.json").exists()        # no head training
    shard = recorded(out / "gift_r" / "shard_0" / "argv.json")
    assert shard[shard.index("--strategy") + 1] == "R"
    assert shard[shard.index("--device") + 1] == "cuda"
    assert shard[shard.index("--backbone-path") + 1] == str(mirror / ckpt)
    assert shard[shard.index("--head-path") + 1] == str(head)
    assert shard[shard.index("--d-model") + 1] == "384"
    assert (base / "results" / "evalslots_gpu1").is_dir()
    assert not (tmp_path / "box_calls").exists()


def test_a_snapshot_score_on_elisa_needs_its_code_and_a_gpu_number(
        stub_checkout):
    tmp_path = stub_checkout[0]
    r, _, base, _, tag = snapshot_on_elisa(stub_checkout, CF425_SNAP_GPU="x")
    assert r.returncode == 2 and "GPU number" in r.stderr
    r, _, base, _, tag = snapshot_on_elisa(
        stub_checkout, CF425_RUNNER=str(tmp_path / "no_runner.sh"))
    assert r.returncode == 2 and "deploy_elisa.sh" in r.stderr
    assert not (base / "ckpt" / "eval" / tag).exists()
    assert not (tmp_path / "box_calls").exists()


def check_tree(tmp_path, score="0.5000\n", steps=30000):
    """check_scores.py on the artefacts of one job: 97 configs with a MASE
    of 1 and a seasonal-naive MASE of 2, so the score is 0.5."""
    import gzip
    check = load_script("check_scores")
    tag = "cf412om_bb10k_h30k_recon"
    results, mirror = tmp_path / "results", tmp_path / "mirror"
    logs = results / "logs" / "jobs" / tag
    head = mirror / "recon" / "eval" / tag
    for folder in (results / "scores", results / "per_config", logs,
                   results / "head_losses", head):
        folder.mkdir(parents=True)
    jobs = tmp_path / "jobs.tsv"
    jobs.write_text("#code\tarm\tstop_k\nOMB\tcf412om\t10\t1\tx.pth\t1\tgift\n")
    (results / "scores" / f"score_{tag}.txt").write_text(score)
    configs = [f"data_{i}/H/short" for i in range(97)]
    (results / "per_config" / f"{tag}.csv").write_text(
        f"dataset,{check.MASE}\n" + "".join(f"{c},1.0\n" for c in configs))
    (logs / "summary.txt").write_text(
        "Config      MASE  SN_MASE   Relative\n"
        + "".join(f"{c}    1.0000   2.0000     0.5000\n" for c in configs))
    (logs / "stop.log").write_text(
        "[10-07] [x] eval start (97 configs, R, forecast-len 16, cuda)\n")
    with gzip.open(results / "head_losses" / f"{tag}_losses.csv.gz",
                   "wt") as out:
        out.write(f"step,loss\n1,0.5\n{steps},0.1\n")
    (head / "q_final.pth").write_bytes(b"head")
    check.RESULTS, check.JOBS, check.MIRROR = results, jobs, mirror
    return check, results


def test_the_check_passes_a_complete_job(tmp_path):
    check, results = check_tree(tmp_path)
    assert check.main() == 0
    row, = csv.DictReader(open(results / "checks.tsv"), delimiter="\t")
    assert row == {"code": "OMB", "stop_k": "10", "score": "0.5000",
                   "configs": "97", "gm": "0.5000", "strategy": "R",
                   "head_steps": "30000", "head_bytes": "4", "result": "ok"}


def test_the_check_names_what_a_job_lacks(tmp_path):
    """A score that is not the geometric mean of its configs, and a head
    that stopped before step 30,000."""
    check, results = check_tree(tmp_path, score="0.4000\n", steps=29000)
    assert check.main() == 1
    row, = csv.DictReader(open(results / "checks.tsv"), delimiter="\t")
    assert row["result"] == "FAIL gm,head_steps"


def test_the_check_reads_each_snapshot_score(tmp_path, capsys):
    """check_scores.py also checks each row of snapshots/scores.tsv: 97
    configs, the geometric mean and strategy R. It writes
    snapshots/checks.tsv, and a snapshot score that fails gives exit 1."""
    check, results = check_tree(tmp_path)
    folder = results / "snapshots"
    configs = [f"data_{i}/H/short" for i in range(97)]
    rows = []
    for snapshot, score, mase in (("best", "0.5000", "1.0"),
                                  ("final", "0.4000", "1.0")):
        tag = f"cf412om_bb10k_h30k_{snapshot}_recon"
        (folder / "per_config").mkdir(parents=True, exist_ok=True)
        (folder / "logs" / tag).mkdir(parents=True)
        (folder / "per_config" / f"{tag}.csv").write_text(
            f"dataset,{check.MASE}\n" + "".join(f"{c},{mase}\n"
                                                for c in configs))
        (folder / "logs" / tag / "summary.txt").write_text(
            "".join(f"{c}    1.0000   2.0000     0.5000\n" for c in configs))
        (folder / "logs" / tag / "stop.log").write_text(
            "[10-08] [x] eval start (97 configs, R, forecast-len 16, cuda)\n")
        rows.append(f"OMB\tcf412om\t10\tmean/std\t{snapshot}\t30000\telisa"
                    f"\t{score}\t0.5000\t1.00\n")
    header = ("run\tarm\tstop_k\tscaling\tsnapshot\thead_step\tmachine"
              "\tr_snapshot\tr_final\tratio\n")
    (folder / "scores.tsv").write_text(header + rows[0])
    assert check.main() == 0
    row, = csv.DictReader(open(folder / "checks.tsv"), delimiter="\t")
    assert row == {"code": "OMB", "stop_k": "10", "snapshot": "best",
                   "machine": "elisa", "score": "0.5000", "configs": "97",
                   "gm": "0.5000", "strategy": "R", "result": "ok"}
    # A score that is not the geometric mean of its configs.
    (folder / "scores.tsv").write_text(header + "".join(rows))
    assert check.main() == 1
    best, final = csv.DictReader(open(folder / "checks.tsv"), delimiter="\t")
    assert (best["result"], final["result"]) == ("ok", "FAIL gm")
    assert "OMB 10k final: FAIL gm" in capsys.readouterr().out


def test_the_figures_pair_the_two_scores_of_a_checkpoint(tmp_path,
                                                         monkeypatch):
    pytest.importorskip("matplotlib")
    plot = load_script("plot_recon")
    recon = tmp_path / "recon.tsv"
    recon.write_text("cf412om\t10\t0.3100\ncf412om\t25\t0.2900\n"
                     "k3_r100_09_lr56_fix09_dec10k_lr30x\t665\t0.2000\n")
    forecast = plot.load(plot.FORECAST[:1])
    points = plot.load([recon])
    assert points["cf412om"] == {40000: 0.31, 100000: 0.29}   # batch 256: x4
    assert forecast["cf412om"][40000] == 1.3782
    out = tmp_path / "overlay.png"
    assert plot.draw_figure(plot.GRAPHS["all"], forecast, points, out, True)
    assert out.stat().st_size > 10_000
    moirai = [g for g in plot.base.GROUPS if g not in plot.OURS]
    assert not plot.draw_figure(moirai, forecast, points, tmp_path / "m.png",
                                False)
    assert not (tmp_path / "m.png").exists()


FLOORS_TSV = ("setup\tlabel\tarms\tgm_relative_mase\n"
              "meanstd\tmean/std\tcf412om,cf419ms\t0.9500\n"
              "ewma_old\tEWMA, old data\tk3_r100_09_lr56_fix09_dec10k_lr30x"
              "\t0.1500\n")


def legend_texts(fig):
    """The text of each legend entry of a figure: the key of the axes and
    the runs under them."""
    legends = [ax.get_legend() for ax in fig.axes] + list(fig.legends)
    return [text.get_text() for legend in legends if legend is not None
            for text in legend.get_texts()]


def test_a_y_axis_holds_its_scores_between_two_ticks():
    """The range of a log y axis goes from the tick under its scores to the
    tick above them. A narrow range of scores takes a ladder with more
    ticks, so a reader can still read a point."""
    pytest.importorskip("matplotlib")
    plot = load_script("plot_recon")
    low, high, ticks = plot.y_axis([0.0274, 0.0543])
    assert ticks == pytest.approx([0.025, 0.03, 0.04, 0.05, 0.06])
    assert low < 0.025 and high > 0.06
    assert plot.y_axis([1.1369, 1.2878])[2] == pytest.approx([1.1, 1.2, 1.3])
    assert plot.y_axis([0.03, 2.6])[2] == pytest.approx(
        [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    assert plot.y_axis([0.05])[2] == pytest.approx([0.05])


def test_the_legend_gives_the_floor_of_the_runs(tmp_path):
    """The legend of a figure gives the floor of each scaling setup of its
    runs, and no other floor. The chart holds a floor, as a thin grey line,
    only when an R score of its setup is near it: a floor far above the R
    curves would take the height of the chart from them."""
    pytest.importorskip("matplotlib")
    plot = load_script("plot_recon")
    (tmp_path / "floors.tsv").write_text(FLOORS_TSV)
    floors = plot.load_floors(tmp_path / "floors.tsv")
    assert [f["setup"] for f in floors] == ["meanstd", "ewma_old"]

    def figure(points):
        fig = plot.draw_figure(plot.GRAPHS["ours_patch_sizes"], {}, points,
                               tmp_path / "f.png", False, floors)
        ax, = fig.axes
        flat = [line for line in ax.get_lines()
                if len(set(line.get_ydata())) == 1]
        return legend_texts(fig), flat, ax.get_ylim()

    # 0.31 is more than 3 times under the floor: no line, and a tight range.
    texts, flat, (low, high) = figure({"cf412om": {40000: 0.31, 100000: 0.29}})
    assert "mean/std: 0.9500" in texts
    assert not any("EWMA, old data" in text for text in texts)
    assert any("Each floor is above the chart" in text for text in texts)
    assert not flat and low < 0.29 and 0.31 < high < 0.95
    # 0.60 is near the floor: the chart holds the floor line.
    texts, flat, (low, high) = figure({"cf412om": {40000: 0.31, 100000: 0.60}})
    assert [float(line.get_ydata()[0]) for line in flat] == [0.95]
    assert flat[0].get_color() == plot.FLOOR_COLOUR
    assert flat[0].get_linewidth() <= 1.0
    assert low < 0.31 and high > 0.95
    assert not any("above the chart" in text for text in texts)


def test_the_facts_are_in_the_plot_and_the_legend(tmp_path):
    """The report holds the figures only. A figure has a short title of one
    line and no annotation. Its legends give the first and the last R score
    of each run with their ratio, the floor of its runs and the key. An
    overlay breaks its y axis: the forecast curve at 50% opacity in the top
    panel, the R curve in the bottom panel, and one line for each
    checkpoint with both scores."""
    pytest.importorskip("matplotlib")
    from matplotlib.patches import ConnectionPatch
    plot = load_script("plot_recon")
    floors = [{"setup": "meanstd", "label": "mean/std floor",
               "arms": {"cf412om"}, "score": 1.5721}]
    points = {"cf412om": {40000: 0.3100, 100000: 0.2900}}
    forecast = {"cf412om": {40000: 1.3782, 100000: 1.3345, 200000: 1.45}}
    hue = plot.colour("cf412om")
    for overlay in (False, True):
        fig = plot.draw_figure(plot.GRAPHS["ours_patch_sizes"], forecast,
                               points, tmp_path / f"{overlay}.png", overlay,
                               floors, "Reconstruction (R): ours")
        assert all(len(ax.texts) == 0 for ax in fig.axes)
        title = fig.axes[0].get_title()
        assert title and "\n" not in title and len(title) <= 60
        texts = legend_texts(fig)
        assert any(text.endswith("R 0.3100 → 0.2900, ×0.94") for text in texts)
        assert "How to read" in texts
        assert any(text.endswith("mean/std: 1.5721") for text in texts)
        curves = [[line for line in ax.get_lines() if line.get_color() == hue]
                  for ax in fig.axes]
        links = [a for a in fig.artists if isinstance(a, ConnectionPatch)]
        if not overlay:
            (curve,), = curves
            assert list(curve.get_ydata()) == [0.31, 0.29]
            assert not links
            continue
        (b4,), (r,) = curves
        assert list(b4.get_ydata()) == [1.3782, 1.3345, 1.45]
        assert b4.get_alpha() == plot.FORECAST_ALPHA
        assert list(r.get_ydata()) == [0.31, 0.29] and r.get_alpha() == 1.0
        assert len(links) == 2                    # 200k has no R score
        top, bottom = fig.axes
        assert top.get_ylim()[0] > bottom.get_ylim()[1]


SNAPSHOTS_TSV = (
    "run\tarm\tstop_k\tscaling\tsnapshot\thead_step\tmachine"
    "\tr_snapshot\tr_final\tratio\n"
    "OMB\tcf412om\t10\tmean/std\tbest\t28500\telisa\t0.2000\t0.3100\t1.55\n"
    "OMB\tcf412om\t10\tmean/std\tfinal\t30000\tbox\t0.3100\t0.3100\t1.00\n")


def test_a_hollow_marker_shows_an_earlier_snapshot_of_a_head(tmp_path):
    """snapshot_score.sh scores an earlier snapshot of a head. A figure
    shows that score as a hollow marker at the checkpoint of the head, and
    its y range holds it. The key names the marker in one line. The scores
    are in the table of collect.py, and the legend holds none of them. The
    control score of a final head is no snapshot."""
    pytest.importorskip("matplotlib")
    plot = load_script("plot_recon")
    table = tmp_path / "scores.tsv"
    table.write_text(SNAPSHOTS_TSV)
    snapshots = plot.load_snapshots(table)
    assert snapshots == {"cf412om": {40000: (28500, 0.2)}}   # batch 256: x4
    points = {"cf412om": {40000: 0.3100, 100000: 0.2900}}
    fig = plot.draw_figure(plot.GRAPHS["ours_patch_sizes"], {}, points,
                           tmp_path / "s.png", False, snapshots=snapshots)
    ax, = fig.axes
    hollow = [line for line in ax.get_lines()
              if line.get_markerfacecolor() == "white"]
    assert [list(line.get_ydata()) for line in hollow] == [[0.2]]
    assert ax.get_ylim()[0] < 0.2
    # No curve hides the hollow marker: it lies above each of them.
    assert all(hollow[0].get_zorder() > line.get_zorder()
               for line in ax.get_lines() if line is not hollow[0])
    texts = legend_texts(fig)
    assert "The same head at an earlier head step" in texts
    assert not any("0.2000" in text or "28,500" in text for text in texts)
    plain = plot.draw_figure(plot.GRAPHS["ours_patch_sizes"], {}, points,
                             tmp_path / "p.png", False)
    assert "The same head at an earlier head step" not in legend_texts(plain)


def test_each_entry_of_the_key_is_one_short_line(tmp_path):
    """The report holds the figures only, so a reader reads the key of each
    figure. Each entry of the key is one line of 52 characters or less, for
    each kind of entry: the two heads, an earlier snapshot, the floors with
    and without a line in the chart, and in an overlay, the forecast."""
    pytest.importorskip("matplotlib")
    plot = load_script("plot_recon")
    (tmp_path / "floors.tsv").write_text(FLOORS_TSV)
    (tmp_path / "scores.tsv").write_text(SNAPSHOTS_TSV)
    floors = plot.load_floors(tmp_path / "floors.tsv")
    drawn = plot.Line2D([], [])
    for overlay in (False, True):
        for lines, note in (
                ({}, "Each floor is above the chart"),
                ({"meanstd": drawn}, "A floor with no line is above the chart"),
                ({"meanstd": drawn, "ewma_old": drawn}, None)):
            labels = [label for _, label in plot.key_entries(
                overlay, floors, True, lines, True)]
            assert len(labels) == 8 + 3 * overlay + (note is not None)
            assert all("\n" not in label and len(label) <= 52
                       for label in labels)
            assert (note is None) or note in labels
            assert ("B4: the forecast of the run" in labels) == overlay
    # The key of a figure holds these entries, under its name.
    snapshots = plot.load_snapshots(tmp_path / "scores.tsv")
    points = {"cf412om": {40000: 0.3100, 100000: 0.6000}}
    linear = {"cf412om": {40000: 0.5000, 100000: 0.7000}}
    fig = plot.draw_figure(plot.GRAPHS["ours_patch_sizes"], {}, points,
                           tmp_path / "k.png", False, floors,
                           snapshots=snapshots, linear=linear)
    (key,) = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    assert [text.get_text() for text in key.get_texts()] == [
        "How to read",
        "R: a head decodes the true horizon from its latents",
        "One dot: one checkpoint, one head, one seed",
        "R a → b, ×c: first and last checkpoint, c = b / a",
        "R with a linear head",
        "The same head at an earlier head step",
        "Floor of R: a head that reads no latent",
        "mean/std: 0.9500"]


def floor_checkout(stub_checkout):
    """The stub checkout, and an elisa mirror that holds the checkpoint of
    each floor as a small file at its path in jobs.tsv."""
    tmp_path, _, env = stub_checkout
    mirror = tmp_path / "mirror"
    for row in job_rows():
        (mirror / row[4]).parent.mkdir(parents=True, exist_ok=True)
        (mirror / row[4]).write_text("backbone")
    return tmp_path, dict(env, CF425_MIRROR=str(mirror),
                          CF425_FLOOR_ROOT=str(tmp_path / "floors"),
                          CF425_FLOOR_RESULTS=str(tmp_path / "results"),
                          EVAL_CONFIG_FILTER="^m4_yearly/short$",
                          EVAL_EXPECT_CONFIGS="1")


def test_floors_score_one_checkpoint_of_each_scaling_setup(stub_checkout):
    tmp_path, env = floor_checkout(stub_checkout)
    r = subprocess.run(["bash", str(SCRIPTS / "floors.sh")], capture_output=True,
                       text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rows = list(csv.DictReader(open(tmp_path / "results" / "floors.tsv"),
                               delimiter="\t"))
    assert [row["setup"] for row in rows] == ["ewma_zero_pad", "ewma_old",
                                              "meanstd"]
    assert all(row["gm_relative_mase"] == "0.5000" for row in rows)
    arms = {row["setup"]: set(row["arms"].split(",")) for row in rows}
    jobs = job_rows()
    assert set().union(*arms.values()) == {row[1] for row in jobs}
    assert sum(len(a) for a in arms.values()) == len({row[1] for row in jobs})
    assert arms["ewma_old"] == {row[1] for row in jobs if row[6] == "old"}
    for setup in arms:
        shard = recorded(tmp_path / "floors" / f"floor_{setup}" / "gift_r0"
                         / "shard_0" / "argv.json")
        assert "--zero-head" in shard
        assert (tmp_path / "results" / "per_config"
                / f"floor_{setup}.csv").exists()


FAKE_SSH = """#!/bin/bash
# The box is this machine: run the remote command here.
exec bash -c "${@: -1}"
"""


@pytest.fixture
def sync_box(tmp_path):
    """A box with a scored head, a head that still trains, a head that
    ended after another loop copied its first `*_best.pth`, and a job whose
    score came after a tick deleted its final head from the box."""
    box, res = tmp_path / "box" / "cf-425", tmp_path / "box" / "results"
    files = {
        "recon/eval/done_recon/q_final.pth": b"F" * 10,
        "recon/eval/done_recon/q_best.pth": b"B" * 10,
        "recon/eval/done_recon/q_losses.csv": b"step,loss\n",
        "recon/eval/done_recon/gift_r/summary.txt": b"Aggregate 0.2\n",
        "recon/eval/train_recon/q_best.pth": b"T" * 10,
        "recon/eval/late_recon/q_final.pth": b"L" * 10,
        "recon/eval/late_recon/q_best.pth": b"new best!!",
        "recon/eval/pruned_recon/gift_r/all_results.csv": b"dataset\n",
    }
    for rel, data in files.items():
        (box / rel).parent.mkdir(parents=True, exist_ok=True)
        (box / rel).write_bytes(data)
    res.mkdir(parents=True)
    (res / "score_done_recon.txt").write_text("0.2000\n")
    (res / "score_pruned_recon.txt").write_text("0.3000\n")
    (res / "queue.log").write_text("started\n")
    mirror = tmp_path / "elisa" / "vast_lr100x"
    stale = mirror / "cf-425" / "recon/eval/late_recon/q_best.pth"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old best!!")
    os.utime(stale, (1_000_000_000, 1_000_000_000))
    ssh = tmp_path / "fake_ssh.sh"
    ssh.write_text(FAKE_SSH)
    env = dict(os.environ, CF425_SSH=f"bash {ssh}", CF425_BOX_ROOT=str(box),
               CF425_RES=str(res), CF425_PRUNE_ROOT=str(box),
               CF425_MIRROR=str(mirror),
               CF425_RESULTS_MIRROR=str(tmp_path / "elisa" / "results"),
               CF425_PRUNE=str(SCRIPTS / "prune.sh"))
    return tmp_path, box, mirror / "cf-425", env


def test_a_sync_tick_brings_the_ended_heads_and_frees_the_box(sync_box):
    tmp_path, box, mirror, env = sync_box
    r = subprocess.run(["bash", str(SCRIPTS / "sync_box.sh")],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    held = {str(p.relative_to(mirror)) for p in mirror.rglob("*") if p.is_file()}
    assert held == {"recon/eval/done_recon/q_final.pth",
                    "recon/eval/done_recon/q_best.pth",
                    "recon/eval/done_recon/q_losses.csv",
                    "recon/eval/done_recon/gift_r/summary.txt",
                    "recon/eval/late_recon/q_final.pth",
                    "recon/eval/late_recon/q_best.pth",
                    # no final head on the box, but its job has a score
                    "recon/eval/pruned_recon/gift_r/all_results.csv"}
    assert (mirror / "recon/eval/late_recon/q_best.pth").read_bytes() == b"new best!!"
    results = tmp_path / "elisa" / "results"
    assert (results / "score_done_recon.txt").read_text() == "0.2000\n"
    left = {str(p.relative_to(box)) for p in box.rglob("*") if p.is_file()}
    assert left == {"recon/eval/done_recon/q_losses.csv",
                    "recon/eval/done_recon/gift_r/summary.txt",
                    "recon/eval/pruned_recon/gift_r/all_results.csv",
                    "recon/eval/train_recon/q_best.pth",     # still trains
                    "recon/eval/late_recon/q_final.pth"}     # no score yet
