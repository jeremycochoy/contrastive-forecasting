"""Consistency tests for the #379 small-model long-training launchers.

The six launcher scripts under
`experiments/2026-07-21_split_pred_rep_small/scripts/` wire the six #374
loss recipes to the small backbone config specified in the issue:

    d_model=128, n_heads=16 (head_dim=8),
    num_encoder_layers=3, num_layers=3,
    T=4096, C=1, batch_size=128, total_steps=200000, save_every=10000,
    extra_save_steps=2500,25000,
    encoder=gru, rev_norm=ewma span=128,
    lr=1e-3, wd=0.1, betas=(0.9, 0.98), seed=20260520.

Each arm additionally carries its own contrastive-loss flags copied verbatim
from the parent #374 launchers. The purpose of these tests is not to
re-verify the loss shapes — those are guarded by `test_loss*` — but to lock
in the small-model config across all six arms so a stray edit to one
launcher does not silently break the sweep.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "2026-07-21_split_pred_rep_small"
SCRIPTS = EXP_DIR / "scripts"

ARM_LAUNCHERS = {
    "arm1": "elisa_arm1_launch.sh",
    "arm3": "elisa_arm3_launch.sh",
    "arm4": "elisa_arm4_launch.sh",
    "arm5": "elisa_arm5_launch.sh",
    "arm6_v2": "elisa_arm6_v2_launch.sh",
    "bimoco": "elisa_bimoco_launch.sh",
}

# Small-model backbone config that every arm must share.
COMMON_BACKBONE = {
    "--t-raw": "4096",
    "--n-channels": "1",
    "--d-model": "128",
    "--n-heads": "16",
    "--num-encoder-layers": "3",
    "--num-layers": "3",
    "--batch-size": "128",
    "--total-steps": "200000",
    "--save-every": "10000",
    "--extra-save-steps": "2500,25000",
    "--lr": "1e-3",
    "--weight-decay": "0.1",
    "--adam-beta1": "0.9",
    "--adam-beta2": "0.98",
    "--seed": "20260520",
    "--rev-norm-kind": "ewma",
    "--rev-norm-span": "128",
    "--encoder-type": "gru",
    "--tau": "0.10",
    "--ema-tau": "0.9",
    "--sigreg-embedding-weight": "1.0",
    "--sigreg-encoding-weight": "1.0",
}

# Per-arm loss-shape / MoCo / alignment flags copied from the parent #374
# launchers. Encoded as (must_contain_all, must_not_contain_any).
ARM_LOSS_FLAGS = {
    "arm1": (
        ["--loss-shape cosine_similarity_batch_split_pred_rep"],
        ["--moco-negatives", "--moco-rep-keys", "--align-loss-weight",
         "--align-moco-loss-weight"],
    ),
    "arm3": (
        ["--loss-shape cosine_similarity_batch_split_pred_rep",
         "--moco-negatives"],
        ["--moco-rep-keys", "--align-loss-weight",
         "--align-moco-loss-weight"],
    ),
    "arm4": (
        ["--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt",
         "--pos-in-denominator", "--subtract-contrastive-floor",
         "--moco-negatives"],
        ["--moco-rep-keys", "--align-loss-weight",
         "--align-moco-loss-weight"],
    ),
    "arm5": (
        ["--loss-shape cosine_similarity_batch_rep_only",
         "--align-loss-weight 1.0"],
        ["--moco-negatives", "--moco-rep-keys",
         "--align-moco-loss-weight"],
    ),
    "arm6_v2": (
        ["--loss-shape cosine_similarity_batch_rep_only",
         "--align-loss-weight 1.0", "--moco-rep-keys"],
        ["--moco-negatives", "--align-moco-loss-weight"],
    ),
    "bimoco": (
        ["--loss-shape cosine_similarity_batch_split_pred_rep",
         "--moco-negatives", "--moco-rep-keys"],
        ["--align-loss-weight", "--align-moco-loss-weight"],
    ),
}


def strip_comments(text: str) -> str:
    """Remove full-line bash comments so token-search sees only code."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def load_bash_vars(text: str) -> dict[str, str]:
    """Extract simple `NAME=value` and `NAME="value"` top-level assignments.

    Handles multiple `;`-separated assignments per line. Just enough to
    resolve `"$VAR"` references in flag values without launching bash.
    Ignores parameter-expansion defaults like ${VAR:-x}.
    """
    out: dict[str, str] = {}
    pat = re.compile(r'([A-Z_][A-Z0-9_]*)=(?:"([^"$]*)"|(\S+))')
    for raw_line in text.splitlines():
        if raw_line.lstrip().startswith("#"):
            continue
        for stmt in raw_line.split(";"):
            m = pat.search(stmt.strip())
            if not m:
                continue
            name, quoted, bare = m.group(1), m.group(2), m.group(3)
            val = quoted if quoted is not None else bare
            if "$" in val:
                continue  # only pin literal values, skip ${...} references
            out.setdefault(name, val)
    return out


def resolve_var(token: str, vars_: dict[str, str]) -> str:
    """`"$NENC"` → vars['NENC'] if present, else the raw token."""
    m = re.fullmatch(r'"?\$([A-Z_][A-Z0-9_]*)"?', token)
    if m and m.group(1) in vars_:
        return vars_[m.group(1)]
    return token


def read_launcher(name: str) -> str:
    path = SCRIPTS / ARM_LAUNCHERS[name]
    assert path.is_file(), f"missing launcher {path}"
    return path.read_text()


def flag_value(script: str, flag: str) -> str | None:
    body = strip_comments(script)
    m = re.search(rf"{re.escape(flag)}\s+(\S+)", body)
    if not m:
        return None
    return resolve_var(m.group(1), load_bash_vars(script))


@pytest.mark.parametrize("arm", sorted(ARM_LAUNCHERS))
def test_backbone_config_shared_across_arms(arm):
    script = read_launcher(arm)
    for flag, expected in COMMON_BACKBONE.items():
        actual = flag_value(script, flag)
        assert actual == expected, (
            f"{arm}: expected `{flag} {expected}`, got `{flag} {actual}`")


@pytest.mark.parametrize("arm", sorted(ARM_LAUNCHERS))
def test_arm_loss_flags(arm):
    script = strip_comments(read_launcher(arm))
    must_have, must_not_have = ARM_LOSS_FLAGS[arm]
    for token in must_have:
        assert token in script, f"{arm}: missing `{token}`"
    for token in must_not_have:
        assert token not in script, f"{arm}: unexpected `{token}`"


@pytest.mark.parametrize("arm", sorted(ARM_LAUNCHERS))
def test_downstream_backbone_step_cells(arm):
    """Downstream evaluates 5 backbone-step cells: 2k, 25k, 50k, 100k, 200k.

    We only require that all five names appear in the launcher so a
    reviewer can grep to confirm; the exact loop structure is not pinned.
    """
    script = read_launcher(arm)
    for step_label in ["2k", "25k", "50k", "100k", "200k"]:
        assert step_label in script, (
            f"{arm}: missing backbone-step cell label `{step_label}`")


@pytest.mark.parametrize("arm", sorted(ARM_LAUNCHERS))
def test_downstream_head_layers_2_and_6(arm):
    script = read_launcher(arm)
    assert " 2 " in script or "HL=2" in script or "head_layers=2" in script \
        or "2L" in script, f"{arm}: no 2L head cell"
    assert " 6 " in script or "HL=6" in script or "head_layers=6" in script \
        or "6L" in script, f"{arm}: no 6L head cell"
