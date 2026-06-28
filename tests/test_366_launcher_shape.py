"""Shape test for the #366 SIGReg λ × EMA-τ cross launchers.

The arms are picked from prior single-axis winners (#363 arm 2 / arm 6,
#357 τ=0.90). This test guards that the launcher driver still pairs the
selected λ pair with τ=0.90 — a tweak to either side without updating the
other would silently break the cross.
"""

from pathlib import Path

import pytest


EXP_DIR = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "2026-06-28_sigreg_lambda_tau_cross"
)


def read(path):
    return path.read_text()


def test_launch_arms_pairs_both_axes():
    text = read(EXP_DIR / "scripts" / "launch_arms.sh")
    # Arm A: #363 best-at-best (λ_e=10, λ_h=1) × #357 best τ=0.90.
    assert '"10.0   1.0 0.90 lA_emb100_enc10_tau090"' in text
    # Arm B: #363 best-at-last (λ_e=1000, λ_h=1) × #357 best τ=0.90.
    assert '"1000.0 1.0 0.90 lB_emb10000_enc10_tau090"' in text


@pytest.mark.parametrize(
    "flag",
    [
        "--ema-tau",
        "--sigreg-embedding-weight",
        "--sigreg-encoding-weight",
        "--sigreg-embedding",
        "--sigreg-encoding",
        "--ema-embedding",
        "--ema-encoder",
        "--cpc-infonce-weight",
        "--batch-size 512",
    ],
)
def test_backbone_launcher_carries_recipe_flag(flag):
    text = read(EXP_DIR / "scripts" / "train_backbone_sigreg.sh")
    assert flag in text, f"missing {flag!r} in train_backbone_sigreg.sh"


def test_backbone_launcher_parameterises_lambda_and_tau():
    text = read(EXP_DIR / "scripts" / "train_backbone_sigreg.sh")
    assert '--ema-tau "$TAU"' in text
    assert '--sigreg-embedding-weight "$LAMBDA_E"' in text
    assert '--sigreg-encoding-weight "$LAMBDA_H"' in text
    # Positional args order: gpu lambda_e lambda_h tau suffix [steps] [save_every].
    assert 'GPU="${1:?gpu}"; LAMBDA_E="${2:?lambda_e}"; LAMBDA_H="${3:?lambda_h}"' in text
    assert 'TAU="${4:?tau}"; SUFFIX="${5:?suffix}"' in text


def test_downstream_tag_matches_backbone_name_pattern():
    bb = read(EXP_DIR / "scripts" / "train_backbone_sigreg.sh")
    dl = read(EXP_DIR / "scripts" / "downstream_sigreg.sh")
    bb_pattern = "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_"
    dl_pattern = "allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_"
    assert bb_pattern in bb
    assert dl_pattern in dl


def test_launch_downstream_blocks_on_sibling_run_names():
    text = read(EXP_DIR / "scripts" / "launch_downstream.sh")
    # Suffix-aware wait — both #366 arm names must be in the pgrep alternation.
    assert "lA_emb100_enc10_tau090" in text
    assert "lB_emb10000_enc10_tau090" in text
