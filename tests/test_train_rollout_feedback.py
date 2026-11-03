"""--train-rollout-feedback: how the training rollout depth (#373) feeds each
copy of the forecaster back in.

L_align is a cosine: it trains the direction of f only, and a raw f is about
1.45 times longer than an encoder latent. 'normalised' (default) feeds each
copy back at the mean norm of the encoder latents of its window, as the
eval rollout does (rollout_latent). 'raw' keeps the old training rollout.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.forecasting_head import (rollout_forecaster_latents,  # noqa: E402
                                  window_latent_norm)
from src.freq_embedding import FREQ_NAMES_V2  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding" / "scripts"
            / "train.py")
B, T, C, H = 3, 8, 1, 16


def tiny_backbone():
    torch.manual_seed(0)
    return ConfigurableModel(
        C=C, H=H, W=16, encoder_type="gru", num_layers=2, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.0,
        rev_norm_kind="ewma", rev_norm_span=128, num_encoder_layers=1,
        freq_emb_dim=3, seasonality_emb_dim=3, num_freqs=len(FREQ_NAMES_V2),
        rev_norm_skip_leading_zeros=True).eval()


def latents(seed=0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    f = (3.0 * torch.randn(B, T, C, H, generator=g)).requires_grad_(requires_grad)
    o = 2.0 * torch.randn(B, T, C, H, generator=g)
    return f, o


def old_rollout(backbone, f, depth):
    """The training rollout of every run before the flag: the raw copy back in."""
    seq = f.permute(0, 2, 1, 3).reshape(B * C, T, H)
    out = []
    for _ in range(depth):
        seq = backbone.transformer.forecaster_forward(seq)
        out.append(seq.reshape(B, C, T, H).permute(0, 2, 1, 3))
    return out


def fed_sequences(backbone, f, depth, scale):
    seen, forward = [], backbone.transformer.forecaster_forward

    def record(seq, **kwargs):
        seen.append(seq)
        return forward(seq, **kwargs)
    backbone.transformer.forecaster_forward = record
    out = rollout_forecaster_latents(backbone, f, depth, scale=scale)
    backbone.transformer.forecaster_forward = forward
    return out, seen


def test_the_window_norm_is_the_mean_norm_over_the_patches_with_data():
    _, o = latents()
    pad = torch.zeros(B, T, C, dtype=torch.bool)
    pad[0, :3] = True
    norm = window_latent_norm(o, pad)
    assert norm.shape == (B, 1, C, 1)
    assert torch.allclose(norm[0, 0, 0, 0], o[0, 3:, 0].norm(dim=-1).mean())
    assert torch.allclose(norm[1, 0, 0, 0], o[1, :, 0].norm(dim=-1).mean())


def test_a_window_of_padding_only_takes_the_mean_of_all_its_patches():
    _, o = latents()
    pad = torch.ones(B, T, C, dtype=torch.bool)
    assert torch.allclose(window_latent_norm(o, pad), window_latent_norm(o))


def test_the_window_norm_carries_no_gradient():
    o = latents()[1].requires_grad_(True)
    assert not window_latent_norm(o).requires_grad


def test_no_scale_is_the_old_training_rollout():
    backbone, (f, _) = tiny_backbone(), latents()
    new = rollout_forecaster_latents(backbone, f, 3)
    for a, b in zip(new, old_rollout(backbone, f, 3)):
        assert torch.equal(a, b)


def test_each_copy_goes_back_in_at_the_norm_of_its_window():
    backbone, (f, o) = tiny_backbone(), latents()
    scale = window_latent_norm(o)
    out, seen = fed_sequences(backbone, f, 3, scale)
    want = scale.permute(0, 2, 1, 3).reshape(B * C, 1)
    for fed in seen:
        assert torch.allclose(fed.norm(dim=-1), want.expand(-1, T), rtol=1e-5)
    # The copies keep the direction of the outputs they come from.
    for prev, fed in zip([f] + out[:-1], seen):
        prev = prev.permute(0, 2, 1, 3).reshape(B * C, T, H)
        cos = torch.nn.functional.cosine_similarity(prev, fed, dim=-1)
        assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)


def test_the_gradient_flows_through_the_normalised_chain():
    backbone, (f, o) = tiny_backbone(), latents(requires_grad=True)
    out = rollout_forecaster_latents(backbone, f, 2, scale=window_latent_norm(o))
    out[-1].sum().backward()
    assert f.grad is not None and f.grad.abs().sum() > 0


def train_module():
    spec = importlib.util.spec_from_file_location("train_rollout_feedback", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train():
    return train_module()


def test_the_flag_defaults_to_normalised(train, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--weight-decay", "0.1"])
    assert train.parse_args().train_rollout_feedback == "normalised"


@pytest.mark.parametrize("feedback,depth,scaled", [
    ("normalised", 2, True), ("raw", 2, False), ("normalised", 0, False)])
def test_forecaster_depths_gives_a_scale_under_normalised_only(
        train, monkeypatch, feedback, depth, scaled):
    seen = {}

    def record(model, f_lat, k, scale=None):
        seen.update(k=k, scale=scale)
        return []
    monkeypatch.setattr(train, "rollout_forecaster_latents", record)
    f, o = latents()
    args = SimpleNamespace(train_rollout_feedback=feedback, train_rollout_depth=depth)
    train.forecaster_depths(None, f, o, args)
    assert seen["k"] == depth and (seen["scale"] is not None) == scaled
    if scaled:
        assert torch.equal(seen["scale"], window_latent_norm(o))


def test_zero_padding_is_none_when_the_run_pads_nothing(train):
    model = SimpleNamespace(rev_norm=SimpleNamespace(pad_mask=None), W=16)
    assert train.zero_padding(model) is None
