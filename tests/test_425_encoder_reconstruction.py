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
   checkpoint, on a zero-padding checkpoint and on an old checkpoint.
5. The shell runners: the reconstruction mode reaches the head trainer and
   the eval, and the forecast mode stays as it was.
6. The scripts of the report: the job table, the queue on the box, the
   prune rule, and the figures.
"""

from __future__ import annotations

import csv
import importlib.util
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
                                  bank_quantile_loss, bank_training_inputs,
                                  compute_reconstruction_targets,
                                  extract_encoder_latents, forecast_B4,
                                  head_bank_sizes, quantile_loss,
                                  reconstruct_horizon,
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


def test_the_b4_forecast_is_unchanged_by_the_new_keyword():
    """The B4 forecast of a bank head still reads its context alone."""
    m = bank_model()
    head = bank_of().head_for(32)
    out = forecast_B4(m, head, walk(T)[:, None], 40, "cpu")
    assert out.shape == (Q, 40, 1) and np.isfinite(out).all()


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
    return module.ReconstructionPredictor(
        backbone=m, head=head, prediction_length=h, device=CPU,
        strategy="R", context_pad="first",
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
                "EVAL_STRATEGY"):
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
