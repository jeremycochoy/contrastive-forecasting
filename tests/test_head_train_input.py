"""--head-train-input: the input layout of a forecast head at training.

B4 decodes [the encoder latents of the context, the rolled forecaster
latents]. 'e_then_f' trains the head on that layout: [e_0..e_{T-1},
f_0..f_{T-1}] under build_e_then_f_mask. Before this fix the prediction
branch (the forecast head and a frequency family) and the head bank ignored
the flag and trained on f only.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.forecasting_head import (TransformerQuantileForecastingHead,  # noqa: E402
                                  build_e_then_f_mask, decode_forecast_head,
                                  extract_encoder_latents,
                                  extract_forecaster_latents,
                                  forecast_head_input)
from src.freq_embedding import FREQ_NAMES_V2  # noqa: E402
from src.freq_family import FAMILY_MEMBERS, FrequencyFamilyHead  # noqa: E402
from src.models import ConfigurableModel  # noqa: E402

HEAD_PY = (REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
           / "train_forecasting_head.py")
T_RAW, PATCHES = 512, 32


def tiny_backbone():
    torch.manual_seed(0)
    return ConfigurableModel(
        C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
        ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
        rev_norm_kind="ewma", rev_norm_span=128, num_encoder_layers=1,
        freq_emb_dim=3, seasonality_emb_dim=3, num_freqs=len(FREQ_NAMES_V2),
        rev_norm_skip_leading_zeros=True).eval()


def head(seed=1):
    torch.manual_seed(seed)
    return TransformerQuantileForecastingHead(
        H=16, num_layers=2, nhead=2, forecast_len=16, dropout=0.0).eval()


def batch(n=3):
    g = torch.Generator().manual_seed(0)
    return 50 + torch.randn(n, T_RAW, 1, generator=g).cumsum(1)


def layout(name, backbone=None, x=None):
    backbone = backbone or tiny_backbone()
    x = batch() if x is None else x
    f_bc, _ = extract_forecaster_latents(backbone, x)
    return backbone, x, f_bc, forecast_head_input(backbone, x, f_bc, name)


def test_f_only_is_the_forecaster_latents_from_position_0():
    _, _, f_bc, (seq, mask, start) = layout("f_only")
    assert seq is f_bc and mask is None and start == 0


def test_e_then_f_is_the_layout_that_b4_decodes():
    backbone, x, f_bc, (seq, mask, start) = layout("e_then_f")
    e_bc, _ = extract_encoder_latents(backbone, x)
    assert torch.equal(seq, torch.cat([e_bc, f_bc], dim=1))
    assert torch.equal(mask, build_e_then_f_mask(PATCHES, PATCHES))
    assert start == PATCHES


def test_an_unknown_layout_is_refused():
    backbone, x = tiny_backbone(), batch()
    f_bc, _ = extract_forecaster_latents(backbone, x)
    with pytest.raises(ValueError, match="f_only"):
        forecast_head_input(backbone, x, f_bc, "f_then_e")


def test_the_head_decodes_one_output_per_forecaster_latent():
    _, _, f_bc, (seq, mask, start) = layout("e_then_f")
    out = decode_forecast_head(head(), seq, mask, start)
    assert out.shape[:2] == f_bc.shape[:2]


def test_f_only_decodes_as_the_plain_head():
    _, _, f_bc, (seq, mask, start) = layout("f_only")
    h = head()
    assert torch.equal(decode_forecast_head(h, seq, mask, start), h(f_bc))


def test_the_output_of_patch_t_never_reads_the_encoder_latent_of_its_target():
    """f-position t decodes patch t + 1: e_{t+1} encodes that patch."""
    _, _, _, (seq, mask, start) = layout("e_then_f")
    h, t = head(), 10
    moved = seq.clone()
    moved[:, t + 1] += 5.0
    before = decode_forecast_head(h, seq, mask, start)
    after = decode_forecast_head(h, moved, mask, start)
    assert torch.equal(before[:, :t + 1], after[:, :t + 1])
    assert not torch.allclose(before[:, t + 1], after[:, t + 1])


@pytest.mark.parametrize("body", ["heads", "shared"])
def test_a_family_gives_the_mask_to_each_member(body):
    _, _, _, (seq, mask, start) = layout("e_then_f")
    torch.manual_seed(2)
    family = FrequencyFamilyHead(head, FAMILY_MEMBERS, body).eval()
    members = torch.tensor([FAMILY_MEMBERS[0], FAMILY_MEMBERS[-1], FAMILY_MEMBERS[0]])
    out = decode_forecast_head(family, seq, mask, start, members)
    for row, key in enumerate(members.tolist()):
        alone = family.member(key)(seq[row:row + 1], src_mask=mask)[:, start:]
        assert torch.allclose(out[row:row + 1], alone, atol=1e-6)


def trainer():
    spec = importlib.util.spec_from_file_location("head_trainer", HEAD_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def head_args(module, *flags):
    return module.parse_args(["--backbone-path", "unused.pth", *flags])


@pytest.mark.parametrize("flags,why", [
    (["--head-arch", "gru"], "takes no mask"),
    (["--head-arch", "linear"], "takes no mask"),
    (["--head-arch", "transformer", "--mixed-rollout", "4"], "--mixed-rollout")])
def test_the_trainer_refuses_an_e_then_f_head_that_it_cannot_train(flags, why):
    module = trainer()
    args = head_args(module, "--head-train-input", "e_then_f", *flags)
    with pytest.raises(SystemExit, match=why):
        module.refuse_head_input(args)


@pytest.mark.parametrize("flags", [
    ["--head-train-input", "e_then_f", "--head-arch", "transformer"],
    ["--head-train-input", "e_then_f", "--head-arch", "transformer-gaussian"],
    ["--head-train-input", "e_then_f", "--head-arch", "linear",
     "--reconstruction", "encoder"],
    ["--head-train-input", "f_only", "--head-arch", "gru"]])
def test_the_trainer_takes_the_other_layouts(flags):
    module = trainer()
    module.refuse_head_input(head_args(module, *flags))
