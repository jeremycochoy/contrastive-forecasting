"""Consistency tests for the #369 B=1024 retrain launchers.

The arm identity (λ_e, λ_h, τ), the parent's best-loss step, and the
verifier stamps are read from a `winners.sh` manifest written at launch
time. These tests guard the relationship between the README arm table,
the example manifest, and the bash suffix derivation.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "2026-07-03_b1024_traj_ckpts"
SCRIPTS = EXP_DIR / "scripts"


def read(path: Path) -> str:
    return path.read_text()


# Suffix the launcher emits: `l_emb<10·λ_e>_enc<10·λ_h>_tau<100·τ>_b1024`.
SUFFIX_RE = re.compile(r"^l_emb(\d+)_enc(\d+)_tau(\d+)_b1024$")


def parse_suffix(suffix: str) -> tuple[float, float, float]:
    m = SUFFIX_RE.match(suffix)
    assert m, f"unparseable suffix {suffix!r}"
    le10, lh10, tau100 = m.groups()
    return int(le10) / 10, int(lh10) / 10, int(tau100) / 100


def derive_suffix_via_bash(le: str, lh: str, tau: str) -> str:
    """Extract `launch_experiment.sh::suffix_for` and call it."""
    script_text = (SCRIPTS / "launch_experiment.sh").read_text()
    m = re.search(r"^suffix_for\(\).*?^\}", script_text, re.DOTALL | re.MULTILINE)
    assert m, "suffix_for function not found in launch_experiment.sh"
    func = m.group(0)
    cmd = f'{func}\nsuffix_for "{le}" "{lh}" "{tau}"\n'
    return subprocess.check_output(["bash", "-c", cmd], text=True).strip()


def parse_manifest(path: Path) -> dict[str, str]:
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


def parse_readme_arm_table(text: str) -> list[tuple[str, str, str, str, str]]:
    """Extract `(suffix, λ_e, λ_h, τ, batch)` from each row of the arm table."""
    pattern = re.compile(
        r"\|\s*Arm \d+\s*\(`([^`]+)`\)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|"
    )
    return pattern.findall(text)


# --- suffix encoding (bash ↔ Python) --------------------------------------


@pytest.mark.parametrize(
    "le,lh,tau,expected",
    [
        ("1.0", "1.0", "0.90", "l_emb10_enc10_tau090_b1024"),
        ("10.0", "1.0", "0.90", "l_emb100_enc10_tau090_b1024"),
        ("1000.0", "1.0", "0.90", "l_emb10000_enc10_tau090_b1024"),
        ("0.1", "0.1", "0.99", "l_emb1_enc1_tau099_b1024"),
    ],
)
def test_bash_suffix_derivation_matches_encoding(le, lh, tau, expected):
    assert derive_suffix_via_bash(le, lh, tau) == expected
    dle, dlh, dtau = parse_suffix(expected)
    assert dle == float(le)
    assert dlh == float(lh)
    assert dtau == float(tau)


def test_bash_suffix_rounds_not_truncates():
    """awk `%.0f` rounds; `%d` truncates and would silently mis-encode.

    Same guard as `test_366_launcher_shape.test_bash_suffix_rounds_not_truncates`.
    """
    assert derive_suffix_via_bash("0.58", "0.58", "0.58") == "l_emb6_enc6_tau058_b1024"
    dle, dlh, dtau = parse_suffix("l_emb6_enc6_tau058_b1024")
    assert (dle, dlh, dtau) == (0.6, 0.6, 0.58)


# --- README arm table internal consistency --------------------------------


def test_readme_arm_table_self_consistent():
    rows = parse_readme_arm_table(read(EXP_DIR / "README.md"))
    assert len(rows) >= 1, f"expected ≥ 1 arm row, got {len(rows)}"
    for suffix, le, lh, tau, batch in rows:
        dle, dlh, dtau = parse_suffix(suffix)
        assert dle == float(le), f"{suffix}: λ_e {dle} != row {le}"
        assert dlh == float(lh), f"{suffix}: λ_h {dlh} != row {lh}"
        assert dtau == float(tau), f"{suffix}: τ {dtau} != row {tau}"
        assert batch == "1024", f"{suffix}: batch column must be 1024"


def test_readme_table_matches_manifest_example():
    rows = parse_readme_arm_table(read(EXP_DIR / "README.md"))
    manifest = parse_manifest(SCRIPTS / "winners.sh.example")
    # The single row (or first row) must line up with the manifest values.
    suffix, le, lh, tau, batch = rows[0]
    assert float(le) == float(manifest["LAMBDA_E"])
    assert float(lh) == float(manifest["LAMBDA_H"])
    assert float(tau) == float(manifest["TAU"])


def test_readme_procedure_names_winners_example():
    """The launch procedure section must tell the user to update the
    committed example manifest, not just the local one."""
    text = read(EXP_DIR / "README.md")
    m = re.search(r"##\s*Launch-time gate(.*?)(?:^##\s|\Z)", text, re.DOTALL | re.MULTILINE)
    assert m, "README is missing the `## Launch-time gate` section"
    gate = m.group(1)
    assert "winners.sh.example" in gate, \
        "Launch-time gate procedure does not name `winners.sh.example`"
    assert re.search(r"(update|edit)[^.\n]*winners\.sh\.example", gate, re.IGNORECASE), \
        "Launch-time gate does not tell the user to update `winners.sh.example`"


# --- launch_experiment.sh: launch-time gate + suffix derivation ----------


def test_launch_experiment_requires_manifest():
    text = read(SCRIPTS / "launch_experiment.sh")
    assert 'WINNERS_FILE="${WINNERS_FILE:-$OUT/winners.sh}"' in text
    assert 'ABORT: winners manifest not found' in text
    for v in (
        "LAMBDA_E",
        "LAMBDA_H",
        "TAU",
        "PARENT_BEST_LOSS_STEP",
        "WINNERS_VERIFIED_BY",
        "WINNERS_VERIFIED_AT",
    ):
        assert v in text, f"launch_experiment.sh does not validate {v}"


def test_launch_experiment_suffix_derivation_present():
    text = read(SCRIPTS / "launch_experiment.sh")
    assert "suffix_for()" in text
    assert "le*10" in text and "lh*10" in text and "t*100" in text
    # The `_b1024` marker is what distinguishes this run's suffix from
    # the parent's B=512 suffix.
    assert "_b1024" in text, "suffix must carry the _b1024 marker"


# --- train_backbone_b1024.sh: batch=1024 + traj-save wire-up -------------


def test_backbone_launcher_uses_batch_1024():
    text = read(SCRIPTS / "train_backbone_b1024.sh")
    assert "--batch-size 1024" in text, "backbone must run at B=1024"


def test_backbone_launcher_wires_traj_save():
    text = read(SCRIPTS / "train_backbone_b1024.sh")
    assert "--traj-save-every" in text, \
        "backbone must pass --traj-save-every to train.py"
    assert 'TRAJ_SAVE_EVERY="${8:-500}"' in text, \
        "trajectory cadence default must be 500 steps (matches #369 scope)"


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
    ],
)
def test_backbone_carries_recipe_flag(flag):
    """Recipe must match #366 byte-for-byte except for batch size."""
    text = read(SCRIPTS / "train_backbone_b1024.sh")
    assert flag in text, f"missing {flag!r} in train_backbone_b1024.sh"


def test_backbone_parameterises_lambda_and_tau():
    text = read(SCRIPTS / "train_backbone_b1024.sh")
    assert '--ema-tau "$TAU"' in text
    assert '--sigreg-embedding-weight "$LAMBDA_E"' in text
    assert '--sigreg-encoding-weight "$LAMBDA_H"' in text


def test_backbone_no_batch_512_leaked():
    """Guard against a stale copy — the batch-size 512 line from #366 must
    not survive in the b1024 script."""
    text = read(SCRIPTS / "train_backbone_b1024.sh")
    assert "--batch-size 512" not in text, \
        "stale `--batch-size 512` found; this is the B=1024 retrain"


# --- downstream_b1024.sh: two step-tagged loci ---------------------------


def test_downstream_reads_parent_step_locus():
    """The parent-best-step cell must read the retrained backbone at the
    step number the parent's best-loss head landed on. That's the whole
    point of the trajectory checkpoints (per #369 §Scope 2.1)."""
    text = read(SCRIPTS / "downstream_b1024.sh")
    assert 'BB_PARENT="$RUNS/bb_${TAG}_step${PARENT_STEP}.pth"' in text, \
        "downstream must resolve parent-best-step to `_step${PARENT_STEP}.pth`"


def test_downstream_reads_last_step_locus():
    """The last cell must read the retrained backbone at TOTAL_STEPS."""
    text = read(SCRIPTS / "downstream_b1024.sh")
    assert 'BB_LAST="$RUNS/bb_${TAG}_step${TOTAL_STEPS}.pth"' in text, \
        "downstream must resolve last to `_step${TOTAL_STEPS}.pth`"


def test_downstream_hard_errors_on_missing_checkpoints():
    """Both loci are load-bearing; a missing checkpoint at either must
    abort, not silently skip. Same principle as #366's downstream."""
    text = read(SCRIPTS / "downstream_b1024.sh")
    assert "ABORT: parent-best-step backbone missing" in text
    assert "ABORT: last-step backbone missing" in text


def test_downstream_propagates_cell_failures():
    """Failed cells must change the script's exit code — same guard as
    #366's downstream_sigreg.sh."""
    text = read(SCRIPTS / "downstream_b1024.sh")
    assert "|| true" not in text, \
        "downstream_b1024.sh swallows failures via `|| true`"
    assert re.search(r"^exit\s+\"?\$\{?fail\b", text, re.MULTILINE), \
        "downstream_b1024.sh must `exit $fail` after the cell loop"


def test_downstream_tag_matches_backbone_run_name():
    bb = read(SCRIPTS / "train_backbone_b1024.sh")
    dl = read(SCRIPTS / "downstream_b1024.sh")
    bb_pattern = "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b1024_cpc_"
    dl_pattern = "allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b1024_cpc_"
    assert bb_pattern in bb
    assert dl_pattern in dl
