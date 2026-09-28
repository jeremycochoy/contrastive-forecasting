"""Tests for #421: the patch heads of #417 on the #419 stream, and the
mean/std scaling of Moirai 1.0.

Six groups, all on the CPU:

1. The merge: every v2 frequency reads its Moirai 1.0 size range, and the
   zero padding stays out of the loss at every patch size.
2. The scaler: loc and scale equal uni2ts ``PackedStdScaler`` on the same
   context values, the padding stays out, and a value after the split
   changes nothing the context feeds.
3. The split: a target fraction of U[0.15, 0.5] of the real patches, in
   patches of each sample's size, and the uni2ts length rule.
4. The loss counts no term whose predicted patch lies before the split.
5. The eval: loc and scale stay fixed through the A2 rollout, and the eval
   and the backbone loader read the kind from the checkpoint.
6. The trainer: its refusal, and one step plus one A2V forecast on the
   stream with its frequency labels.
"""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.forecasting_head as fh  # noqa: E402
from src.checkpoint import load_backbone_from_checkpoint  # noqa: E402
from src.forecasting_head import (  # noqa: E402
    QUANTILE_LEVELS,
    TARGET_RATIO_RANGE,
    draw_context_ends,
    forecast_A2,
    mean_std_inputs,
    median_quantile_index,
    multi_patch_value_objective,
    native_value_head,
    value_space_forward,
    value_space_objective,
)
from src.freq_embedding import FREQ_NAMES_V2, freq_to_id  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402
from src.norm import (  # noqa: E402
    MEAN_STD_MINIMUM_SCALE,
    RevMeanStdNorm,
    mean_std_statistics,
    safe_div,
)
from src.patch_size import draw_patch_sizes, patch_size_choices  # noqa: E402

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
EVAL_PY = (REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
           / "eval_gift_eval_official.py")

Q = len(QUANTILE_LEVELS)
SIZES = (8, 16, 32, 64, 128)
T = 1024
V2 = {name: i for i, name in enumerate(FREQ_NAMES_V2)}


def model(kind="meanstd", sizes=SIZES, **kw):
    """A small value-space model on the v2 vocabulary, with zero padding."""
    cfg = dict(C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
               ffn_mult=2.0, activation="gelu", depthwise_conv=3, dropout=0.0,
               rev_norm_kind=kind, num_encoder_layers=1, freq_emb_dim=3,
               seasonality_emb_dim=3, num_freqs=len(FREQ_NAMES_V2),
               value_head_quantiles=Q, multi_patch_sizes=sizes,
               rev_norm_skip_leading_zeros=True,
               enc_transformer_use_grad_checkpoint=False)
    if kind == "ewma":
        cfg["rev_norm_span"] = 128
    cfg.update(kw)
    return ConfigurableModel(**cfg)


def windows(lengths, level=500.0, seed=0):
    """``[B, T, 1]``: one random walk per length, left zero padded."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(len(lengths), T, 1)
    for b, n in enumerate(lengths):
        walk = level + (3.0 * torch.randn(n, generator=g)).cumsum(0)
        x[b, T - n:, 0] = walk
    return x


def ids(freqs):
    """The label ids of a batch of v2 frequency names."""
    freq = torch.tensor([V2[f] for f in freqs])
    return dict(freq_ids=freq, seasonality_ids=torch.zeros_like(freq))


# ---------------------------------------------------------------------------
# 1. The merge: v2 frequencies and the padding at every size
# ---------------------------------------------------------------------------

# uni2ts DefaultPatchSizeConstraints, with Q and Y at 8 (#417), for every v2
# class, and the inference size of each (#417).
V2_TABLE = [
    ("10s", (64, 128), 128), ("4s", (64, 128), 128),
    ("1min", (32, 64, 128), 64), ("5min", (32, 64, 128), 64),
    ("10min", (32, 64, 128), 64), ("15min", (32, 64, 128), 64),
    ("30min", (32, 64, 128), 64),
    ("1h", (32, 64), 32), ("6h", (32, 64), 32),
    ("1d", (16, 32), 16), ("1w", (16, 32), 16),
    ("1M", (8, 16, 32), 16), ("1Q", (8,), 8), ("1Y", (8,), 8),
]


def test_the_table_covers_every_v2_class():
    assert sorted(name for name, _, _ in V2_TABLE) == sorted(FREQ_NAMES_V2[1:])


@pytest.mark.parametrize("name,train,infer", V2_TABLE)
def test_every_v2_id_reads_its_moirai_range(name, train, infer):
    assert patch_size_choices(V2[name], SIZES) == train
    assert patch_size_choices(V2[name], SIZES, training=False) == (infer,)
    rows = torch.full((500,), V2[name])
    assert set(draw_patch_sizes(rows, SIZES, 500).tolist()) == set(train)


# The anchored aliases v2 maps (#419), and the strings of the stream.
ALIASES = [("W-SUN", "1w"), ("W-FRI", "1w"), ("W-TUE", "1w"),
           ("Q-DEC", "1Q"), ("A-DEC", "1Y"), ("4S", "4s"), ("6H", "6h"),
           ("M", "1M"), ("MS", "1M"), ("5T", "5min"), ("H", "1h")]


@pytest.mark.parametrize("freq,name", ALIASES)
def test_an_alias_takes_the_range_of_its_class(freq, name):
    assert freq_to_id(freq, "v2") == V2[name]
    assert (patch_size_choices(freq, SIZES)
            == patch_size_choices(V2[name], SIZES))
    assert (patch_size_choices(freq, SIZES, training=False)
            == patch_size_choices(V2[name], SIZES, training=False))


def test_a_window_with_no_label_still_draws_every_size():
    drawn = draw_patch_sizes(torch.zeros(2000, dtype=torch.long), SIZES, 2000)
    assert set(drawn.tolist()) == set(SIZES)


def constant_forward(model, x, patch_size=None, **_):
    """A stand-in for the cell whose forecast ignores its input, so the
    loss reads the targets alone."""
    B, T_raw, C = x.shape
    n = T_raw // patch_size
    latent = torch.zeros(B, n, C, model.H)
    return latent, latent, torch.zeros(B, n, C, Q, patch_size)


@pytest.mark.parametrize("size", SIZES)
def test_padded_targets_stay_out_of_the_loss_at_every_size(size, monkeypatch):
    """EWMA + multi-patch (#417 on #419): a padded target moves no term."""
    monkeypatch.setattr(fh, "value_space_forward", constant_forward)
    m = model("ewma")
    x_norm = m.rev_norm(windows([300, 1000, T]), "norm")
    pad = m.rev_norm.pad_mask

    def loss(series):
        return multi_patch_value_objective(
            m, series, torch.full((3,), size), depth=3, pad_mask=pad,
            **ids(["1h"] * 3))[0]

    assert torch.equal(loss(x_norm), loss(x_norm + 50.0 * pad))
    assert not torch.equal(loss(x_norm), loss(x_norm + 50.0 * ~pad))


@pytest.mark.parametrize("size", SIZES)
def test_the_rolled_inputs_keep_their_padding_at_every_size(size, monkeypatch):
    seen, real = [], fh.value_space_forward

    def spy(m, x, **kw):
        seen.append(x.detach().clone())
        return real(m, x, **kw)

    monkeypatch.setattr(fh, "value_space_forward", spy)
    torch.manual_seed(0)
    m = model("ewma").eval()
    x = windows([300])
    with torch.no_grad():
        value_space_objective(m, m.rev_norm(x, "norm"), depth=3,
                              patch_size=size, pad_mask=m.rev_norm.pad_mask,
                              **ids(["1h"]))
    assert len(seen) == 4
    for j, x_in in enumerate(seen):
        assert (x_in[0, :max(T - 300 - j * size, 0)] == 0).all()


# ---------------------------------------------------------------------------
# 2. The scaler
# ---------------------------------------------------------------------------

def uni2ts_packed_std_scaler(target, observed_mask, sample_id, dimension_id,
                             correction=1, minimum_scale=1e-5):
    """uni2ts ``PackedStdScaler``: ``forward`` then ``_get_loc_scale``, copied
    from uni2ts/module/packed_scaler.py (commit edeb1fa)."""
    from einops import reduce
    target = target.double()
    id_mask = torch.logical_and(
        torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
        torch.eq(dimension_id.unsqueeze(-1), dimension_id.unsqueeze(-2)),
    )
    tobs = reduce(
        id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
        "... seq1 seq2 -> ... seq1 1",
        "sum",
    )
    loc = reduce(
        id_mask * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
        "... seq1 seq2 -> ... seq1 1",
        "sum",
    )
    loc = safe_div(loc, tobs)
    var = reduce(
        id_mask
        * reduce(
            ((target - loc) ** 2) * observed_mask,
            "... seq dim -> ... 1 seq",
            "sum",
        ),
        "... seq1 seq2 -> ... seq1 1",
        "sum",
    )
    var = safe_div(var, (tobs - correction))
    scale = torch.sqrt(var + minimum_scale)
    loc[sample_id == 0] = 0
    scale[sample_id == 0] = 1
    return loc.float(), scale.float()


def moirai_loc_scale(x, lengths, size, targets):
    """uni2ts's loc and scale of each window, the way Moirai sees it: the
    series without its padding, cut in patches of ``size`` padded to 128,
    and the last ``targets`` patches masked as the prediction range."""
    out = []
    for b, n in enumerate(lengths):
        n_patch = math.ceil(n / size)
        values = torch.zeros(n_patch * size)
        values[n_patch * size - n:] = x[b, T - n:, 0]
        observed = torch.zeros(n_patch * size, dtype=torch.bool)
        observed[n_patch * size - n:] = True
        pred = torch.zeros(n_patch, dtype=torch.bool)
        pred[n_patch - targets[b]:] = True
        target = torch.zeros(1, n_patch, 128)
        obs = torch.zeros(1, n_patch, 128, dtype=torch.bool)
        target[0, :, :size] = values.view(n_patch, size)
        obs[0, :, :size] = observed.view(n_patch, size)
        ones = torch.ones(1, n_patch, dtype=torch.long)
        loc, scale = uni2ts_packed_std_scaler(
            target, obs * ~pred.view(1, -1, 1), ones, ones)
        out.append((loc[0, 0, 0], scale[0, 0, 0]))
    return out


@pytest.mark.parametrize("size", SIZES)
def test_loc_and_scale_equal_uni2ts_on_the_same_context(size):
    pytest.importorskip("einops")
    lengths = [T, 700, 5 * size + 3, 2 * size]
    targets = [3, 2, 2, 1]
    x = windows(lengths, seed=size)
    ends = torch.tensor([T - k * size for k in targets]).view(-1, 1, 1)
    norm = RevMeanStdNorm(1, skip_leading_zeros=True)
    out = norm(x, "norm", context_end=ends)
    for b, (loc, scale) in enumerate(moirai_loc_scale(x, lengths, size,
                                                      targets)):
        assert torch.allclose(norm.mean[b, 0, 0], loc, rtol=1e-6)
        assert torch.allclose(norm.stdev[b, 0, 0], scale, rtol=1e-6)
        real = x[b, T - lengths[b]:, 0]
        assert torch.allclose(out[b, T - lengths[b]:, 0],
                              (real - loc) / scale, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("values", [[], [7.5]])
def test_too_few_observed_values_follow_uni2ts(values):
    """No observed value: loc 0, scale sqrt(1e-5). One: loc = that value,
    scale sqrt(1e-5). uni2ts's safe_div gives both, and so do we."""
    pytest.importorskip("einops")
    x = torch.zeros(1, 8, 1)
    observed = torch.zeros(1, 8, 1, dtype=torch.bool)
    for i, v in enumerate(values):
        x[0, 5 + i, 0], observed[0, 5 + i, 0] = v, True
    loc, scale = mean_std_statistics(x, observed)
    ones = torch.ones(1, 1, dtype=torch.long)
    want = uni2ts_packed_std_scaler(x.view(1, 1, 8), observed.view(1, 1, 8),
                                    ones, ones)
    assert loc.item() == pytest.approx(want[0].item())
    assert scale.item() == pytest.approx(want[1].item())
    assert loc.item() == (values[0] if values else 0.0)
    assert scale.item() == pytest.approx(math.sqrt(MEAN_STD_MINIMUM_SCALE))


def test_the_padding_stays_out_of_the_statistics():
    """A padded window scales as its series alone, and its padding stays 0."""
    x = windows([200])
    padded = RevMeanStdNorm(1, skip_leading_zeros=True)
    alone = RevMeanStdNorm(1)
    out = padded(x, "norm")
    ref = alone(x[:, T - 200:], "norm")
    assert torch.allclose(padded.mean, alone.mean)
    assert torch.allclose(padded.stdev, alone.stdev)
    assert torch.allclose(out[:, T - 200:], ref)
    assert (out[:, :T - 200] == 0).all()
    assert int(padded.pad_mask.sum()) == T - 200


def test_values_after_the_split_change_nothing_the_context_feeds():
    """Any change after the split moves neither loc, scale nor a normalised
    context value, through the whole training path."""
    torch.manual_seed(3)
    m = model()
    x = windows([T, 600, 300, 150])
    freqs = ids(["5min", "1h", "1d", "1M"])
    torch.manual_seed(11)
    x_norm, sizes, target = mean_std_inputs(m, x, freqs["freq_ids"], SIZES)
    loc, scale = m.rev_norm.mean.clone(), m.rev_norm.stdev.clone()
    changed = x + target * (1e4 * torch.rand(x.shape) + 1.0)
    torch.manual_seed(11)
    x_norm2, sizes2, target2 = mean_std_inputs(m, changed, freqs["freq_ids"],
                                               SIZES)
    assert torch.equal(sizes, sizes2) and torch.equal(target, target2)
    assert target.any(dim=1).all()
    assert torch.equal(m.rev_norm.mean, loc)
    assert torch.equal(m.rev_norm.stdev, scale)
    assert torch.equal(x_norm[~target], x_norm2[~target])
    assert not torch.equal(x_norm[target], x_norm2[target])


def test_the_kind_is_one_buffer_in_the_state_dict():
    ewma = set(model("ewma").state_dict())
    meanstd = set(model("meanstd").state_dict())
    assert meanstd - ewma == {"rev_norm.mean_std_scaling"}
    assert "rev_norm.leading_zero_pad" in meanstd
    plain = model("meanstd", rev_norm_skip_leading_zeros=False).state_dict()
    assert "rev_norm.leading_zero_pad" not in plain


def test_denorm_gives_back_the_real_values():
    x = windows([90])
    norm = RevMeanStdNorm(1, skip_leading_zeros=True)
    back = norm(norm(x, "norm"), "denorm")
    assert torch.allclose(back[:, T - 90:], x[:, T - 90:], rtol=1e-5)


def test_the_kind_refuses_patch_statistics():
    with pytest.raises(ValueError):
        model("meanstd", sizes=(), patch_stats_kind="diff")


# ---------------------------------------------------------------------------
# 3. The split
# ---------------------------------------------------------------------------

def test_the_target_is_a_uniform_fraction_of_the_real_patches():
    """uni2ts MaskedPrediction: max(1, round(r n)) of n patches, r in
    [0.15, 0.5]. Over many draws the fraction spans the range."""
    torch.manual_seed(0)
    n_windows = 4000
    lengths = torch.full((n_windows, 1, 1), T)
    ends = draw_context_ends(lengths, torch.full((n_windows,), 8), T)
    fraction = (T - ends).float().view(-1) / T
    low, high = TARGET_RATIO_RANGE
    assert fraction.min() >= low - 1 / 128 and fraction.max() <= high + 1 / 128
    assert fraction.min() < low + 0.01 and fraction.max() > high - 0.01
    assert abs(fraction.mean().item() - (low + high) / 2) < 0.01


@pytest.mark.parametrize("size", SIZES)
def test_the_split_falls_on_a_patch_of_each_sample_size(size):
    torch.manual_seed(size)
    lengths = torch.randint(2 * size, T + 1, (500, 1, 1))
    ends = draw_context_ends(lengths, torch.full((500,), size), T)
    assert (ends % size == 0).all()
    targets = (T - ends) // size
    assert (targets >= 1).all()
    assert (targets <= (lengths // size // 2).clamp(min=1)).all()
    # The context keeps at least one whole patch of real values.
    assert (ends - (T - lengths) >= size).all()


def test_the_split_counts_whole_patches_only():
    """uni2ts crops a series to whole patches. 3 whole patches and one
    value more give one target patch at every r below 0.5."""
    torch.manual_seed(0)
    lengths = torch.full((2000, 1, 1), 3 * 16 + 1)
    ends = draw_context_ends(lengths, torch.full((2000,), 16), T)
    assert (ends == T - 16).all()


def test_the_channels_of_a_sample_share_one_ratio():
    """uni2ts draws one ratio per sample, for all its variates."""
    torch.manual_seed(0)
    ends = draw_context_ends(torch.full((300, 1, 2), T),
                             torch.full((300,), 8), T)
    assert torch.equal(ends[:, :, 0], ends[:, :, 1])
    assert len(ends.unique()) > 10


def test_a_sample_draws_a_size_that_fits_two_patches():
    """uni2ts GetPatchSize: a size P needs 2 P real values, when one fits."""
    torch.manual_seed(0)
    minutely = torch.full((3000,), V2["5min"])
    lengths = torch.tensor([T, 200, 100, 40] * 750)
    drawn = draw_patch_sizes(minutely, SIZES, 3000, lengths=lengths)
    assert set(drawn[0::4].tolist()) == {32, 64, 128}
    assert set(drawn[1::4].tolist()) == {32, 64}
    assert set(drawn[2::4].tolist()) == {32}
    assert set(drawn[3::4].tolist()) == {32, 64, 128}  # none fits


def test_a_window_too_short_for_two_patches_has_no_target():
    lengths = torch.tensor([T, 15, 16, 0]).view(-1, 1, 1)
    ends = draw_context_ends(lengths, torch.full((4,), 8), T)
    assert ends[0].item() < T and ends[2].item() < T
    assert ends[1].item() == T and ends[3].item() == T


def test_without_padding_every_window_splits():
    torch.manual_seed(0)
    m = model(rev_norm_skip_leading_zeros=False)
    x = windows([T, T]) + 1.0
    _, _, target = mean_std_inputs(m, x, ids(["1h", "1d"])["freq_ids"], SIZES)
    assert target.any(dim=1).all() and (~target).any(dim=1).all()


# ---------------------------------------------------------------------------
# 4. The loss
# ---------------------------------------------------------------------------

def test_no_term_predicts_a_patch_before_the_split(monkeypatch):
    """At every depth j, position t predicts patch t + 1 + j. It counts only
    when that patch lies after the split, and then on its real values."""
    kept, real = [], fh.masked_quantile_loss

    def spy(predicted, target, keep, *a, **kw):
        kept.append(keep.clone())
        return real(predicted, target, keep, *a, **kw)

    monkeypatch.setattr(fh, "masked_quantile_loss", spy)
    torch.manual_seed(0)
    m = model()
    size, depth = 32, 3
    x = windows([T, 500, 90])
    ends = torch.tensor([640, 832, 960]).view(-1, 1, 1)
    x_norm = m.rev_norm(x, "norm", context_end=ends)
    pad = m.rev_norm.pad_mask
    target = torch.arange(T).view(1, -1, 1) >= ends
    with torch.no_grad():
        value_space_objective(m, x_norm, depth=depth, patch_size=size,
                              pad_mask=pad, target_mask=target,
                              **ids(["1h"] * 3))
    assert len(kept) == depth + 1
    real_patch = ~pad.view(3, T // size, size)
    for j, keep in enumerate(kept):
        for b in range(3):
            split = ends[b].item() // size
            for t in range(keep.shape[1]):
                if t + 1 + j < split:
                    assert not keep[b, t].any()
                else:
                    assert torch.equal(keep[b, t, 0], real_patch[b, t + 1 + j])


@pytest.mark.parametrize("size", SIZES)
def test_targets_before_the_split_do_not_move_the_loss(size, monkeypatch):
    """End to end, every depth: a target value before the split moves no
    loss term, and one after it does."""
    monkeypatch.setattr(fh, "value_space_forward", constant_forward)
    m = model()
    x = windows([T, 700])
    ends = torch.tensor([T - 3 * size, T - 2 * size]).view(-1, 1, 1)
    x_norm = m.rev_norm(x, "norm", context_end=ends)
    target = torch.arange(T).view(1, -1, 1) >= ends

    def loss(series):
        return multi_patch_value_objective(
            m, series, torch.full((2,), size), depth=3,
            pad_mask=m.rev_norm.pad_mask, target_mask=target,
            **ids(["1h", "1h"]))[0]

    assert torch.equal(loss(x_norm), loss(x_norm + 50.0 * ~target))
    assert not torch.equal(loss(x_norm), loss(x_norm + 50.0 * target))


def test_a_window_with_no_target_gives_no_gradient():
    torch.manual_seed(0)
    m = model()
    x = windows([T, 12])
    x_norm, sizes, target = mean_std_inputs(
        m, x, ids(["1Y", "1Y"])["freq_ids"], SIZES)
    assert target[0].any() and not target[1].any()
    x_norm = x_norm.clone().requires_grad_(True)
    loss = multi_patch_value_objective(
        m, x_norm, sizes, depth=2, pad_mask=m.rev_norm.pad_mask,
        target_mask=target, **ids(["1Y", "1Y"]))[0]
    loss.backward()
    assert x_norm.grad[0].abs().sum() > 0
    assert (x_norm.grad[1] == 0).all()


# ---------------------------------------------------------------------------
# 5. The eval
# ---------------------------------------------------------------------------

def manual_rollout(m, context, size, horizon):
    """A2 by hand: loc and scale from the observed context values, then the
    normalised median fed back, and each patch de-normalised."""
    t = torch.arange(T).view(1, -1, 1)
    first = int((context != 0).float().argmax())
    loc, scale = mean_std_statistics(context.view(1, T, 1), t >= first)
    x_norm = ((context.view(1, T, 1) - loc) / scale).masked_fill(t < first, 0)
    labels = dict(freq_ids=torch.tensor([m._eval_freq_id]),
                  seasonality_ids=torch.tensor([m._eval_seasonality_id]))
    out = []
    for _ in range(math.ceil(horizon / size)):
        v = value_space_forward(m, x_norm, patch_size=size, **labels)[2]
        last = v[0, -1, 0]                                      # (Q, P)
        out.append(last * scale.view(1, 1) + loc.view(1, 1))
        median = last[median_quantile_index()].view(1, size, 1)
        x_norm = torch.cat([x_norm[:, size:], median], dim=1)
    return torch.cat(out, dim=-1)[:, :horizon]


def test_the_eval_statistics_stay_fixed_through_the_rollout(monkeypatch):
    torch.manual_seed(0)
    m = model().eval()
    m._eval_freq_id, m._eval_seasonality_id = V2["1h"], 4
    context = windows([700], seed=5)[0]
    calls, real = [], m.rev_norm.forward

    def spy(x, mode, context_end=None):
        calls.append(mode)
        return real(x, mode, context_end)

    monkeypatch.setattr(m.rev_norm, "forward", spy)
    with torch.no_grad():
        out = forecast_A2(m, native_value_head(m, "H"), context, 100, "cpu")
        want = manual_rollout(m, context, 32, 100)
    assert calls == ["norm"]
    observed = torch.arange(T).view(1, -1, 1) >= T - 700
    loc, scale = mean_std_statistics(context.view(1, T, 1), observed)
    assert torch.equal(m.rev_norm.mean, loc)
    assert torch.equal(m.rev_norm.stdev, scale)
    assert out.shape == (Q, 100, 1)
    assert torch.allclose(torch.as_tensor(out[:, :, 0]), want, rtol=1e-4,
                          atol=1e-3)


def test_an_ewma_backbone_rolls_out_as_before(monkeypatch):
    """The fixed scale is the meanstd kind's alone: EWMA normalises every
    step, as #415, #417 and #419 do."""
    calls, real = [], fh.extract_forecaster_latents

    def spy(backbone, x, **kw):
        calls.append(kw.get("normalised", False))
        return real(backbone, x, **kw)

    monkeypatch.setattr(fh, "extract_forecaster_latents", spy)
    torch.manual_seed(0)
    m = model("ewma").eval()
    with torch.no_grad():
        forecast_A2(m, native_value_head(m, "H"), windows([700])[0], 70, "cpu")
    assert calls == [False, False, False]


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_in_eval(path, tag):
    """``(args, backbone, head)`` of the GIFT-Eval script on a checkpoint."""
    pytest.importorskip("gift_eval")
    ev = load_script(EVAL_PY, f"eval_421_{tag}")
    argv = ["eval", "--backbone-path", str(path), "--native-value-head",
            "--strategy", "A2", "--device", "cpu", "--n-channels", "1",
            "--d-model", "16", "--n-heads", "2", "--num-layers", "1",
            "--encoder-type", "gru", "--rev-norm-kind", "ewma",
            "--rev-norm-span", "128"]
    from unittest.mock import patch
    with patch.object(sys, "argv", argv):
        args = ev.parse_args()
    backbone, head = ev.load_models(args, torch.device("cpu"))
    return ev, args, backbone, head


def test_the_eval_reads_the_kind_from_the_checkpoint(tmp_path):
    path = tmp_path / "bb.pth"
    torch.save(model(ffn_mult=4.0).state_dict(), path)
    _, args, backbone, _ = load_in_eval(path, "kind")
    assert isinstance(backbone.rev_norm, RevMeanStdNorm)
    assert backbone.rev_norm.skip_leading_zeros
    assert args.rev_norm_kind == "meanstd" and args.context_pad == "zeros"
    assert backbone._freq_vocab == "v2"
    assert backbone.multi_patch_sizes == SIZES


def test_the_backbone_loader_rebuilds_the_kind(tmp_path):
    path = tmp_path / "bb.pth"
    torch.save(model(ffn_mult=4.0).state_dict(), path)
    backbone, cfg = load_backbone_from_checkpoint(
        str(path), "cpu", C=1, H=16, W=16, nhead=2, num_layers=1)
    assert cfg["rev_norm_kind"] == "meanstd"
    assert isinstance(backbone.rev_norm, RevMeanStdNorm)
    assert backbone.rev_norm.skip_leading_zeros
    assert backbone.freq_embedding.embedding.weight.shape[0] == len(FREQ_NAMES_V2)


# ---------------------------------------------------------------------------
# 6. The trainer
# ---------------------------------------------------------------------------

def run_trainer(*extra):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    return subprocess.run(
        [sys.executable, str(TRAIN_PY), "--device", "cpu",
         "--weight-decay", "0.1", *extra],
        capture_output=True, text=True, env=env, timeout=900)


def test_the_trainer_refuses_meanstd_without_the_value_objective(tmp_path):
    never = tmp_path / "never"
    r = run_trainer("--rev-norm-kind", "meanstd", "--total-steps", "1",
                    "--save-dir", str(never))
    assert r.returncode != 0
    assert "--value-space-objective" in r.stdout + r.stderr
    assert not never.exists()


# The #419 Moirai cell with the two #421 flags, at a size the CPU trains in
# seconds.
TINY_421_RUN = (
    "--value-space-objective", "--gift-pretrain", "--freq-vocab", "v2",
    "--multi-patch-sizes", "8,16,32,64,128", "--rev-norm-kind", "meanstd",
    "--t-raw", "4096", "--n-channels", "1", "--d-model", "16",
    "--n-heads", "2", "--num-layers", "1", "--num-encoder-layers", "1",
    "--batch-size", "8", "--synth-kind", "forked-arma",
    "--mix-ratio", "0.25", "--crossfade-triplets", "1", "--mixup-p", "0.3",
    "--freq-emb-dim", "3", "--seasonality-emb-dim", "3",
    "--train-rollout-depth", "3", "--train-rollout-reduce", "sum",
    "--lr", "1e-3", "--lr-final", "0", "--lr-cosine-steps", "166000",
    "--lr-warmup-steps", "10000", "--grad-clip", "1.0",
    "--log-every", "1", "--save-every", "1000000")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Two steps on a small copy of the stream: the save dir and the run."""
    pytest.importorskip("pyarrow")
    from tests.test_419_gift_pretrain import build_corpus, write_index
    root = tmp_path_factory.mktemp("gep421")
    corpus, index, _ = build_corpus(root / "corpus")
    result = run_trainer(*TINY_421_RUN, "--total-steps", "2",
                         "--gift-pretrain-root", str(corpus),
                         "--gift-pretrain-index", str(write_index(root, index)),
                         "--save-dir", str(root), "--run-name", "m421")
    return root, result


def test_a_training_step_on_the_stream(trained):
    root, r = trained
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "Scaling (#421)" in r.stdout and "Multi-patch (#417)" in r.stdout
    assert "Salesforce/GiftEvalPretrain (#419)" in r.stdout
    assert "NaN/Inf DETECTED" not in r.stdout
    sd = torch.load(root / "m421_final.pth", map_location="cpu",
                    weights_only=True)
    assert "rev_norm.mean_std_scaling" in sd
    assert "rev_norm.leading_zero_pad" in sd
    assert all(f"value_heads.{s}.weight" in sd for s in SIZES)
    assert sd["freq_embedding.embedding.weight"].shape[0] == len(FREQ_NAMES_V2)


def test_an_a2v_forecast_of_the_trained_model(trained):
    import pandas as pd
    root, r = trained
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    ev, args, backbone, head = load_in_eval(root / "m421_final.pth", "a2v")
    assert isinstance(backbone.rev_norm, RevMeanStdNorm)
    backbone._eval_freq_id = freq_to_id("H", backbone._freq_vocab)
    backbone._eval_seasonality_id = 4
    predictor = ev.ContrastiveForecasterPredictor(
        backbone=backbone, head=native_value_head(backbone, "H"),
        prediction_length=48, device=torch.device("cpu"), strategy="A2",
        context_pad=args.context_pad)
    rng = np.random.default_rng(0)
    item = {"target": (40.0 + rng.standard_normal(300).cumsum()).astype(
                np.float32),
            "start": pd.Period("2020-01-01 00:00", freq="h"), "item_id": "s"}
    forecast = predictor.predict_item(item)
    assert forecast.forecast_array.shape == (Q + 1, 48)
    assert np.isfinite(forecast.forecast_array).all()
