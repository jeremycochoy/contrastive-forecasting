"""#482: the score code builds the backbone of each paper ablation from its
checkpoint.

The head trainer and the GIFT-Eval script build the frozen backbone from
fixed flags (--encoder-type gru) and defaults (a depthwise convolution of
kernel 3), plus what the checkpoint names. Two ablations change these: the
residual block of TimesFM as the patch encoder, and no depthwise
convolution. Their checkpoints name both, so the score code reads them
from the checkpoint. Every other checkpoint builds the same backbone as
before.
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.checkpoint import depthwise_conv_of, encoder_type_of
from src.freq_embedding import FREQ_NAMES_V2
from src.models import ConfigurableModel

REPO_ROOT = Path(__file__).resolve().parent.parent
GIFT = REPO_ROOT / "experiments" / "2026-04-13_gift-eval" / "scripts"
# The flags of a head job of follow_multi.sh (head_eval_bb.sh), at d_model 16.
SHAPE = ["--t-raw", "4096", "--n-channels", "1", "--d-model", "16",
         "--n-heads", "2", "--num-layers", "1", "--encoder-type", "gru",
         "--rev-norm-kind", "ewma", "--rev-norm-span", "128"]


def backbone(**extra):
    """A reference-like backbone at d_model 16: GRU, conv 3, sphere."""
    torch.manual_seed(0)
    config = dict(C=1, H=16, W=16, encoder_type="gru", num_layers=1, nhead=2,
                  ffn_mult=4.0, activation="gelu", dropout=0.1,
                  num_encoder_layers=1, freq_emb_dim=3, seasonality_emb_dim=3,
                  num_freqs=len(FREQ_NAMES_V2), rev_norm_kind="ewma",
                  rev_norm_span=128, rev_norm_skip_leading_zeros=True,
                  qk_norm=True, attn_out_norm=True, latent_sphere=True,
                  ema_embedding=True, ema_encoder=True)
    config.update(extra)
    return ConfigurableModel(**config)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VARIANTS = {"reference": {}, "timesfm": {"encoder_type": "timesfm"},
            "no_conv": {"depthwise_conv": 0}}


def test_the_encoder_type_comes_from_the_keys():
    assert encoder_type_of(backbone().state_dict(), "gru") == "gru"
    sd = backbone(encoder_type="timesfm").state_dict()
    assert encoder_type_of(sd, "gru") == "timesfm"


def test_the_conv_kernel_comes_from_the_keys():
    assert depthwise_conv_of(backbone().state_dict(), 3) == 3
    assert depthwise_conv_of(backbone(depthwise_conv=5).state_dict(), 3) == 5
    assert depthwise_conv_of(backbone(depthwise_conv=0).state_dict(), 3) == 0


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_the_head_trainer_builds_the_backbone_of_the_checkpoint(name,
                                                                tmp_path):
    path = tmp_path / "bb.pth"
    torch.save(backbone(**VARIANTS[name]).state_dict(), path)
    head = load(GIFT / "train_forecasting_head.py", f"head_482_{name}")
    args = head.parse_args(["--backbone-path", str(path), "--device", "cpu",
                            "--freq-emb-dim", "3", "--seasonality-emb-dim",
                            "3", *SHAPE])
    sd = torch.load(path, weights_only=True)
    config, _ = head.backbone_config(args, sd)
    assert config["encoder_type"] == VARIANTS[name].get("encoder_type", "gru")
    assert config["depthwise_conv"] == VARIANTS[name].get("depthwise_conv", 3)
    head.load_frozen_backbone(args, config, sd, "cpu")


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_the_eval_builds_the_backbone_of_the_checkpoint(name, tmp_path):
    pytest.importorskip("gift_eval")
    path = tmp_path / "bb.pth"
    torch.save(backbone(**VARIANTS[name]).state_dict(), path)
    ev = load(GIFT / "eval_gift_eval_official.py", f"eval_482_{name}")
    argv = ["eval", "--backbone-path", str(path), "--zero-head",
            "--strategy", "R", "--device", "cpu", *SHAPE]
    with patch.object(sys, "argv", argv):
        args = ev.parse_args()
    built, _ = ev.load_models(args, torch.device("cpu"))
    want = backbone(**VARIANTS[name]).state_dict()
    got = built.state_dict()
    assert set(got) == {k for k in want if not k.startswith("teacher_")}
    assert all(torch.equal(got[k], want[k]) for k in got)
