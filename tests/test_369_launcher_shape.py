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


# --- PARENT_BEST_LOSS_STEP × TRAJ_SAVE_EVERY invariant --------------------
#
# The retrained backbone emits `_step<N>.pth` only for N that are
# multiples of --traj-save-every (=500 by default). If the operator
# stamps a raw best_loss_step (e.g. 8237) from the parent's losses CSV
# without snapping it, the parent-best-step cell in downstream_b1024.sh
# hard-aborts on a missing checkpoint. Guard: (1) invariant on the
# committed example manifest; (2) launch_experiment.sh fails fast with
# a clear msg.


TRAJ_SAVE_EVERY_DEFAULT = 500


def test_winners_example_parent_step_is_traj_multiple():
    """The committed manifest example must satisfy the invariant so
    a naive copy-and-fill flow doesn't blow up. Snap the placeholder
    if you change TRAJ_SAVE_EVERY_DEFAULT."""
    manifest = parse_manifest(SCRIPTS / "winners.sh.example")
    step = int(manifest["PARENT_BEST_LOSS_STEP"])
    assert step > 0, "PARENT_BEST_LOSS_STEP must be positive"
    assert step % TRAJ_SAVE_EVERY_DEFAULT == 0, (
        f"PARENT_BEST_LOSS_STEP={step} in winners.sh.example is not a "
        f"multiple of TRAJ_SAVE_EVERY={TRAJ_SAVE_EVERY_DEFAULT}; the "
        "backbone's `_step<N>.pth` cadence would never emit it and "
        "downstream_b1024.sh would hard-abort."
    )


def test_launch_experiment_validates_parent_step_multiple():
    """The launcher must enforce the invariant with a clear ABORT."""
    text = read(SCRIPTS / "launch_experiment.sh")
    assert "PARENT_BEST_LOSS_STEP % TRAJ_SAVE_EVERY" in text, (
        "launch_experiment.sh does not validate "
        "PARENT_BEST_LOSS_STEP % TRAJ_SAVE_EVERY == 0"
    )
    assert "is not a multiple" in text, (
        "launch_experiment.sh must fail with a clear "
        "'not a multiple of TRAJ_SAVE_EVERY' error"
    )


def _run_launcher_stub(
    manifest_body: str,
    tmp_path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Invoke launch_experiment.sh against a stub WT/OUT with the given
    manifest. Sets BB_ONLY=1 and DL_ONLY=1 to skip both work blocks so
    only the manifest gate + STEPS/PARENT_BEST_LOSS_STEP guards are
    exercised. `extra_env` overrides (e.g. STEPS, TRAJ_SAVE_EVERY) let
    tests target the STEPS × TRAJ_SAVE_EVERY invariant without editing
    the manifest.
    Returns the CompletedProcess.
    """
    launcher = SCRIPTS / "launch_experiment.sh"
    wt = tmp_path / "wt"
    exp_scripts = wt / "experiments" / "2026-07-03_b1024_traj_ckpts" / "scripts"
    exp_scripts.mkdir(parents=True)
    (exp_scripts / "train_backbone_b1024.sh").write_text("#!/bin/bash\nexit 0\n")
    (exp_scripts / "downstream_b1024.sh").write_text("#!/bin/bash\nexit 0\n")
    (exp_scripts / "winners.sh.example").write_text("# stub\n")
    out = tmp_path / "out"
    out.mkdir()
    (out / "winners.sh").write_text(manifest_body)
    env = {
        "WT": str(wt),
        "OUT": str(out),
        "BB_ONLY": "1",
        "DL_ONLY": "1",
        "PATH": "/usr/bin:/bin",
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(launcher)],
        env=env, capture_output=True, text=True, check=False,
    )


def _valid_manifest(step: int) -> str:
    return (
        f"LAMBDA_E=1.0\nLAMBDA_H=1.0\nTAU=0.90\n"
        f"PARENT_BEST_LOSS_STEP={step}\n"
        f"WINNERS_VERIFIED_BY=tester\nWINNERS_VERIFIED_AT=2026-07-03\n"
    )


def test_launch_experiment_aborts_on_non_multiple(tmp_path):
    """A non-multiple step must fail fast with the guard message."""
    result = _run_launcher_stub(_valid_manifest(8237), tmp_path)
    assert result.returncode != 0
    assert "is not a multiple" in result.stderr, (
        f"expected multiple-check ABORT, got stderr:\n{result.stderr}"
    )


def test_launch_experiment_accepts_multiple(tmp_path):
    """A valid multiple must pass the guard and proceed to the stubbed
    backbone/downstream (which we've neutered to `true`)."""
    result = _run_launcher_stub(_valid_manifest(8500), tmp_path)
    assert result.returncode == 0, (
        f"expected success for 500-multiple, got rc={result.returncode}, "
        f"stderr:\n{result.stderr}"
    )


def test_launch_experiment_aborts_on_non_integer(tmp_path):
    """A non-integer step must fail fast."""
    result = _run_launcher_stub(_valid_manifest("abc"), tmp_path)
    assert result.returncode != 0
    assert "not a non-negative integer" in result.stderr


def test_launch_experiment_aborts_on_zero(tmp_path):
    """Step 0 is not a valid trajectory checkpoint (loop starts at step 1)."""
    result = _run_launcher_stub(_valid_manifest(0), tmp_path)
    assert result.returncode != 0
    assert "out of range" in result.stderr


# --- STEPS × TRAJ_SAVE_EVERY invariant -----------------------------------
#
# `_step<STEPS>.pth` (the last-locus checkpoint that downstream_b1024.sh
# reads at TOTAL_STEPS) is only emitted when STEPS is a multiple of
# TRAJ_SAVE_EVERY. An operator STEPS override that breaks this invariant
# (e.g. `STEPS=12501` for the 25k-step follow-up) would let backbone
# training finish and then hard-abort the downstream last-cell on a
# missing file. Same guard shape as PARENT_BEST_LOSS_STEP above.


def test_launch_experiment_validates_steps_multiple():
    """Static: launcher must check `STEPS % TRAJ_SAVE_EVERY == 0`."""
    text = read(SCRIPTS / "launch_experiment.sh")
    assert "STEPS % TRAJ_SAVE_EVERY" in text, (
        "launch_experiment.sh does not validate STEPS % TRAJ_SAVE_EVERY == 0"
    )
    # Distinct error message from PARENT_BEST_LOSS_STEP's abort so the
    # operator can tell them apart.
    assert re.search(
        r"STEPS=\$\{?STEPS\}?\s+is not a multiple of TRAJ_SAVE_EVERY",
        text,
    ), "STEPS-multiple abort must name STEPS explicitly"


def test_launch_experiment_aborts_on_non_multiple_steps(tmp_path):
    """STEPS=12501 (non-multiple of 500) must fail fast with the STEPS
    guard, before PARENT_BEST_LOSS_STEP is validated against a broken
    total. Use a manifest with a valid parent step so any failure
    unambiguously points at the STEPS guard."""
    result = _run_launcher_stub(
        _valid_manifest(8500), tmp_path, extra_env={"STEPS": "12501"},
    )
    assert result.returncode != 0
    assert "STEPS=12501 is not a multiple" in result.stderr, (
        f"expected STEPS-multiple ABORT, got stderr:\n{result.stderr}"
    )


def test_launch_experiment_aborts_on_non_integer_steps(tmp_path):
    """A non-integer STEPS override must fail fast."""
    result = _run_launcher_stub(
        _valid_manifest(8500), tmp_path, extra_env={"STEPS": "abc"},
    )
    assert result.returncode != 0
    assert "STEPS='abc' is not a non-negative integer" in result.stderr


def test_launch_experiment_aborts_on_zero_steps(tmp_path):
    """STEPS=0 makes no sense: no training loop, no checkpoints."""
    result = _run_launcher_stub(
        _valid_manifest(8500), tmp_path, extra_env={"STEPS": "0"},
    )
    assert result.returncode != 0
    assert "STEPS=0 must be positive" in result.stderr


def test_launch_experiment_aborts_on_non_positive_traj_save_every(tmp_path):
    """TRAJ_SAVE_EVERY=0 would cause `STEPS % 0` division by zero — must
    be caught before the modulo check runs."""
    result = _run_launcher_stub(
        _valid_manifest(8500), tmp_path, extra_env={"TRAJ_SAVE_EVERY": "0"},
    )
    assert result.returncode != 0
    assert "TRAJ_SAVE_EVERY='0' must be a positive integer" in result.stderr


def test_launch_experiment_accepts_steps_multiple(tmp_path):
    """25000-step follow-up (per #369 §Success criteria) must pass the
    guard when STEPS is set explicitly to a 500-multiple."""
    result = _run_launcher_stub(
        _valid_manifest(8500), tmp_path, extra_env={"STEPS": "25000"},
    )
    assert result.returncode == 0, (
        f"expected success for STEPS=25000, got rc={result.returncode}, "
        f"stderr:\n{result.stderr}"
    )
