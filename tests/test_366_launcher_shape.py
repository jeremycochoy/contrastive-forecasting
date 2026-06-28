"""Consistency tests for the #366 SIGReg λ × EMA-τ cross launchers.

The arms are picked at launch time from prior single-axis winners (#363
best-at-best / best-at-last, #357 best τ) via the `winners.sh` manifest;
the launcher derives each arm's run-name suffix from the manifest values
using the encoding ``lX_emb<10·λ_e>_enc<10·λ_h>_tau<100·τ>``.

These tests guard the *relationship* between the README arm table, the
example manifest, and the bash suffix derivation — not their literal
contents. A reformat or value change should still leave them green as
long as the three sources agree.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "2026-06-28_sigreg_lambda_tau_cross"
SCRIPTS = EXP_DIR / "scripts"


def read(path: Path) -> str:
    return path.read_text()


# Suffix the launcher emits is `lX_emb<10·λ_e>_enc<10·λ_h>_tau<100·τ>`.
SUFFIX_RE = re.compile(r"^(l[AB])_emb(\d+)_enc(\d+)_tau(\d+)$")


def parse_suffix(suffix: str) -> tuple[str, float, float, float]:
    """Decode a run-name suffix back into (prefix, λ_e, λ_h, τ)."""
    m = SUFFIX_RE.match(suffix)
    assert m, f"unparseable suffix {suffix!r}"
    prefix, le10, lh10, tau100 = m.groups()
    return prefix, int(le10) / 10, int(lh10) / 10, int(tau100) / 100


def derive_suffix_via_bash(prefix: str, le: str, lh: str, tau: str) -> str:
    """Invoke the same awk expression `launch_arms.sh::suffix_for` uses."""
    cmd = (
        'awk -v p="%s" -v le="%s" -v lh="%s" -v t="%s" '
        "'BEGIN { printf \"%%s_emb%%d_enc%%d_tau%%03d\\n\", p, le*10, lh*10, t*100 }'"
    ) % (prefix, le, lh, tau)
    return subprocess.check_output(["bash", "-c", cmd], text=True).strip()


def parse_winners_manifest(path: Path) -> dict[str, str]:
    """Return the `KEY=value` pairs from a bash-style manifest (no quotes)."""
    pairs: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        pairs[key.strip()] = val.strip().strip('"').strip("'")
    return pairs


def parse_readme_arm_table(text: str) -> list[tuple[str, str, str, str]]:
    """Extract `(suffix, λ_e, λ_h, τ)` from each row of the arm table."""
    pattern = re.compile(
        r"\|\s*Arm [AB]\s*\(`([^`]+)`\)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    )
    return pattern.findall(text)


# --- suffix encoding (bash ↔ Python) --------------------------------------


@pytest.mark.parametrize(
    "prefix,le,lh,tau,expected",
    [
        ("lA", "10.0", "1.0", "0.90", "lA_emb100_enc10_tau090"),
        ("lB", "1000.0", "1.0", "0.90", "lB_emb10000_enc10_tau090"),
        ("lA", "0.1", "0.1", "0.99", "lA_emb1_enc1_tau099"),
    ],
)
def test_bash_suffix_derivation_matches_encoding(prefix, le, lh, tau, expected):
    assert derive_suffix_via_bash(prefix, le, lh, tau) == expected
    p, dle, dlh, dtau = parse_suffix(expected)
    assert p == prefix
    assert dle == float(le)
    assert dlh == float(lh)
    assert dtau == float(tau)


# --- README arm table internal consistency --------------------------------


def test_readme_arm_table_self_consistent():
    """Each row's suffix must decode to the same (λ_e, λ_h, τ) as the row."""
    rows = parse_readme_arm_table(read(EXP_DIR / "README.md"))
    assert len(rows) == 2, f"expected 2 arm rows, got {len(rows)}: {rows}"
    for suffix, le, lh, tau in rows:
        _, dle, dlh, dtau = parse_suffix(suffix)
        assert dle == float(le), f"{suffix}: suffix-decoded λ_e {dle} != row {le}"
        assert dlh == float(lh), f"{suffix}: suffix-decoded λ_h {dlh} != row {lh}"
        assert dtau == float(tau), f"{suffix}: suffix-decoded τ {dtau} != row {tau}"


def test_readme_table_matches_winners_example():
    """Example manifest values must match the README arm rows."""
    rows = parse_readme_arm_table(read(EXP_DIR / "README.md"))
    manifest = parse_winners_manifest(SCRIPTS / "winners.sh.example")
    arm_a, arm_b = rows
    assert float(arm_a[1]) == float(manifest["ARM_A_LAMBDA_E"])
    assert float(arm_a[2]) == float(manifest["ARM_A_LAMBDA_H"])
    assert float(arm_b[1]) == float(manifest["ARM_B_LAMBDA_E"])
    assert float(arm_b[2]) == float(manifest["ARM_B_LAMBDA_H"])
    tau = float(manifest["BEST_TAU"])
    assert float(arm_a[3]) == tau
    assert float(arm_b[3]) == tau


# --- launch_arms.sh: launch-time gate + suffix derivation ----------------


def test_launch_arms_requires_winners_manifest():
    text = read(SCRIPTS / "launch_arms.sh")
    assert 'WINNERS_FILE="${WINNERS_FILE:-$OUT/winners.sh}"' in text
    # Hard-aborts when missing or unstamped.
    assert 'ABORT: winners manifest not found' in text
    for v in (
        "ARM_A_LAMBDA_E",
        "ARM_A_LAMBDA_H",
        "ARM_B_LAMBDA_E",
        "ARM_B_LAMBDA_H",
        "BEST_TAU",
        "WINNERS_VERIFIED_BY",
        "WINNERS_VERIFIED_AT",
    ):
        assert v in text, f"launch_arms.sh does not validate {v}"


def test_launch_arms_does_not_bake_lambda_or_tau_literals():
    """Stale-λ guard: no decimal literal numerics in the ARMS table."""
    text = read(SCRIPTS / "launch_arms.sh")
    arms_block = re.search(r"ARMS=\((.*?)\)", text, re.DOTALL)
    assert arms_block, "ARMS=(...) block missing"
    body = arms_block.group(1)
    # The arms entries must reference variables, not literal floats.
    assert "$ARM_A_LAMBDA_E" in body
    assert "$ARM_B_LAMBDA_E" in body
    assert "$BEST_TAU" in body
    # No bare floats (e.g. 10.0, 0.90) leaked into the table.
    assert not re.search(r"\b\d+\.\d+\b", body), \
        f"ARMS block contains hardcoded float literals: {body!r}"


def test_launch_arms_suffix_derivation_present():
    text = read(SCRIPTS / "launch_arms.sh")
    assert "suffix_for()" in text
    # Encoding constants used in the awk expression.
    assert "le*10" in text and "lh*10" in text and "t*100" in text


# --- train_backbone_sigreg.sh: parameterised recipe flags -----------------


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
    text = read(SCRIPTS / "train_backbone_sigreg.sh")
    assert flag in text, f"missing {flag!r} in train_backbone_sigreg.sh"


def test_backbone_launcher_parameterises_lambda_and_tau():
    text = read(SCRIPTS / "train_backbone_sigreg.sh")
    assert '--ema-tau "$TAU"' in text
    assert '--sigreg-embedding-weight "$LAMBDA_E"' in text
    assert '--sigreg-encoding-weight "$LAMBDA_H"' in text


# --- downstream_sigreg.sh: hard-error if final.pth missing ---------------


def test_downstream_hard_errors_on_missing_last_checkpoint():
    text = read(SCRIPTS / "downstream_sigreg.sh")
    # train.py emits `<run>_final.pth`; downstream must NOT silently skip.
    assert 'BBLAST="$RUNS/bb_${TAG}_final.pth"' in text
    assert 'ABORT: last-checkpoint backbone missing' in text
    # The prior silent-skip log line must be gone.
    assert "skipping last head" not in text


# --- downstream_sigreg.sh tag matches backbone run-name pattern ----------


def test_downstream_tag_matches_backbone_name_pattern():
    bb = read(SCRIPTS / "train_backbone_sigreg.sh")
    dl = read(SCRIPTS / "downstream_sigreg.sh")
    bb_pattern = "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_"
    dl_pattern = "allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_"
    assert bb_pattern in bb
    assert dl_pattern in dl
