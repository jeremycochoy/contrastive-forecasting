"""rollout_latent: how the B rollout feeds each forecaster output back in.

L_align is a cosine: it trains the direction of f only. A raw output is
about 1.45 times longer than an encoder latent, so a raw feedback grew at
each step and the rollout froze on one latent (electricity/H/long: the
change per patch fell from 0.83 to 0.018 over 45 patches). 'normalised'
feeds back the direction of the output at the norm of the context latents.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.forecasting_head import (ROLLOUT_FEEDBACKS,  # noqa: E402
                                  context_latent_norm, data_patches,
                                  extract_encoder_latents, rollout_latent)
from src.freq_embedding import FREQ_NAMES_V2  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402

EVAL_PY = (REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
           / "eval_gift_eval_official.py")
T_RAW, W = 512, 16


def tiny_backbone():
    torch.manual_seed(0)
    return ConfigurableModel(
        C=1, H=16, W=W, encoder_type="gru", num_layers=2, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        rev_norm_kind="ewma", rev_norm_span=128, num_encoder_layers=1,
        freq_emb_dim=3, seasonality_emb_dim=3, num_freqs=len(FREQ_NAMES_V2),
        rev_norm_skip_leading_zeros=True).eval()


def series(n=3, pad=0):
    g = torch.Generator().manual_seed(0)
    x = 50 + torch.randn(n, T_RAW, 1, generator=g).cumsum(1)
    x[:, :pad] = 0.0
    return x


def context(pad=0):
    backbone = tiny_backbone()
    e_ctx, _ = extract_encoder_latents(backbone, series(pad=pad))
    return backbone, e_ctx


def reference_raw_rollout(backbone, seq, n):
    """The rollout of every score before the option: the raw output back in."""
    out = []
    with torch.no_grad():
        for _ in range(n):
            token = backbone.transformer.forecaster_forward(
                seq, fp32_tail=False, cache_mask=False)[:, -1:]
            out.append(token)
            seq = torch.cat([seq, token], dim=1)
    return torch.cat(out, dim=1)


def fed_tokens(backbone, e_ctx, n, feedback):
    """The latents that the rollout appends to the context, in order."""
    seen, forward = [], backbone.transformer.forecaster_forward

    def record(seq, **kwargs):
        seen.append(seq)
        return forward(seq, **kwargs)
    backbone.transformer.forecaster_forward = record
    rolled = rollout_latent(backbone, e_ctx, n, feedback=feedback)
    return rolled, seen[-1][:, e_ctx.shape[1]:]


def test_raw_is_the_rollout_of_every_score_before():
    backbone, e_ctx = context()
    assert torch.equal(rollout_latent(backbone, e_ctx, 5, feedback="raw"),
                       reference_raw_rollout(backbone, e_ctx, 5))


def test_normalised_feeds_back_the_direction_at_the_context_norm():
    backbone, e_ctx = context()
    rolled, fed = fed_tokens(backbone, e_ctx, 5, "normalised")
    norm = context_latent_norm(backbone, e_ctx)
    assert torch.allclose(fed.norm(dim=-1, keepdim=True), norm.expand(-1, 4, 1), rtol=1e-5)
    cos = torch.nn.functional.cosine_similarity(fed, rolled[:, :4], dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)


def test_the_head_reads_the_raw_outputs():
    """Step 1 reads the context only, so both feedbacks give one first output."""
    backbone, e_ctx = context()
    raw = rollout_latent(backbone, e_ctx, 3, feedback="raw")
    normalised = rollout_latent(backbone, e_ctx, 3, feedback="normalised")
    assert torch.equal(raw[:, 0], normalised[:, 0])
    assert not torch.allclose(raw[:, 1:], normalised[:, 1:])


def test_normalised_is_the_default_and_the_backbone_can_set_raw():
    backbone, e_ctx = context()
    assert torch.equal(rollout_latent(backbone, e_ctx, 3),
                       rollout_latent(backbone, e_ctx, 3, feedback="normalised"))
    backbone._rollout_feedback = "raw"
    assert torch.equal(rollout_latent(backbone, e_ctx, 3),
                       reference_raw_rollout(backbone, e_ctx, 3))


def test_an_unknown_feedback_is_refused():
    backbone, e_ctx = context()
    with pytest.raises(ValueError, match="normalised"):
        rollout_latent(backbone, e_ctx, 2, feedback="unit")


def test_the_context_norm_skips_the_patches_of_left_padding():
    pad_patches = 10
    backbone, e_ctx = context(pad=pad_patches * W)
    real = data_patches(backbone, e_ctx.shape[:2])
    assert real[:, :pad_patches].sum() == 0 and real[:, pad_patches:].all()
    want = e_ctx[:, pad_patches:].norm(dim=-1).mean(1)
    assert torch.allclose(context_latent_norm(backbone, e_ctx).flatten(), want)


def test_a_context_of_padding_only_takes_the_mean_of_all_its_patches():
    backbone, e_ctx = context(pad=T_RAW)
    assert torch.allclose(context_latent_norm(backbone, e_ctx).flatten(),
                          e_ctx.norm(dim=-1).mean(1))


def test_no_pad_mask_of_the_shape_gives_no_data_patches():
    backbone, e_ctx = context()
    assert data_patches(backbone, (e_ctx.shape[0] + 1, e_ctx.shape[1])) is None
    backbone.rev_norm.pad_mask = None
    assert data_patches(backbone, e_ctx.shape[:2]) is None


def eval_module():
    spec = importlib.util.spec_from_file_location("gift_eval_official", EVAL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_eval_flag_takes_each_feedback_and_defaults_to_normalised(monkeypatch):
    module = eval_module()
    base = ["eval", "--backbone-path", "unused.pth", "--head-path", "unused.pth"]
    monkeypatch.setattr(sys, "argv", base)
    assert module.parse_args().rollout_feedback == "normalised"
    for feedback in ROLLOUT_FEEDBACKS:
        monkeypatch.setattr(sys, "argv", base + ["--rollout-feedback", feedback])
        assert module.parse_args().rollout_feedback == feedback
