"""Consistency tests for the #379 consolidated backbone-only arm launcher.

`experiments/2026-07-21_split_pred_rep_small/scripts/run_arm.sh` is a
single launcher parameterised by `$ARM ∈ {arm1, arm3, arm4, arm5,
arm6_v2, bimoco}`. Each arm shares the same backbone config (small
model, 200k steps, save-every 25000, extra snapshot at 2500) and only
the per-arm case block picks the loss flags copied verbatim from #374.

These tests do not re-verify the loss shapes — those are guarded by
`test_loss*`. They lock in:

  * every arm's per-arm case block sets NAME, ARM_DESC, and LOSS_ARGS;
  * the small-model backbone config appears exactly once in the
    consolidated body (so a stray edit can't specialise one arm);
  * the launcher is backbone-only — no q-head training, no eval;
  * the WT-under-/tmp guard is present (blocking-2 fix).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "2026-07-21_split_pred_rep_small"
LAUNCHER = EXP_DIR / "scripts" / "run_arm.sh"

ARMS = ("arm1", "arm3", "arm4", "arm5", "arm6_v2", "bimoco")
# #379 tau_rep=1.0 reruns — the 5 arms whose loss has a separate L_rep
# term. arm 4 is xshh_allt (single pooled denom, no split L_rep) and is
# not rerun.
ARMS_TR1 = ("arm1_tr1", "arm3_tr1", "arm5_tr1", "arm6_v2_tr1", "bimoco_tr1")
# #379 no-sigreg-embedding reruns — all 6 base arms with
# --sigreg-embedding-weight 0.0 appended (h_t regulariser kept).
ARMS_NSE = ("arm1_nse", "arm3_nse", "arm4_nse", "arm5_nse", "arm6_v2_nse",
            "bimoco_nse")
# #379 no-CPC reruns — all 6 base arms with --cpc-infonce-weight 0.0
# appended (SIGReg kept as base).
ARMS_NCPC = ("arm1_ncpc", "arm3_ncpc", "arm4_ncpc", "arm5_ncpc",
             "arm6_v2_ncpc", "bimoco_ncpc")

# Small-model backbone config that the shared body must carry verbatim.
BACKBONE_LITERALS = (
    "--batch-size 64",
    "--total-steps \"$STEPS\"",
    "--seed \"$SEED\"",
    "--save-every \"$SAVE_EVERY\"",
    "--extra-save-steps \"$EXTRA_SAVES\"",
    "--t-raw 4096",
    "--n-channels 1",
    "--d-model 64",
    "--n-heads 8",
    "--num-encoder-layers \"$NENC\"",
    "--num-layers \"$NLAY\"",
    "--rev-norm-kind ewma",
    "--rev-norm-span 128",
    "--encoder-type gru",
    "--tau 0.10",
    "--ema-tau 0.9",
    "--sigreg-embedding-weight 1.0",
    "--sigreg-encoding-weight 1.0",
    "--lr 1e-3",
    "--weight-decay 0.1",
    "--adam-beta1 0.9",
    "--adam-beta2 0.98",
    "--depthwise-conv 3",
    "--freq-emb-dim 3",
    "--seasonality-emb-dim 3",
    "--qk-norm",
    "--attn-out-norm",
)

# Per-arm expected NAME suffix + LOSS_ARGS contents.
ARM_EXPECTATIONS = {
    "arm1": dict(
        name_stub="bb_small_arm1_split_pred_rep_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",),
        must_not_have=("--moco-negatives", "--moco-rep-keys",
                       "--align-loss-weight", "--pos-in-denominator"),
    ),
    "arm3": dict(
        name_stub="bb_small_arm3_split_pred_rep_moco_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--moco-negatives"),
        must_not_have=("--moco-rep-keys", "--align-loss-weight",
                       "--pos-in-denominator"),
    ),
    "arm4": dict(
        name_stub="bb_small_arm4_xshh_allt_moco_enc3l3_b64_200k",
        must_have=(
            "--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt",
            "--pos-in-denominator",
            "--subtract-contrastive-floor",
            "--moco-negatives",
        ),
        must_not_have=("--moco-rep-keys", "--align-loss-weight"),
    ),
    "arm5": dict(
        name_stub="bb_small_arm5_lalign_lrep_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_rep_only",
                   "--align-loss-weight 1.0"),
        must_not_have=("--moco-negatives", "--moco-rep-keys",
                       "--pos-in-denominator"),
    ),
    "arm6_v2": dict(
        name_stub="bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_rep_only",
                   "--align-loss-weight 1.0", "--moco-rep-keys"),
        must_not_have=("--moco-negatives", "--pos-in-denominator"),
    ),
    "bimoco": dict(
        name_stub="bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--moco-negatives", "--moco-rep-keys"),
        must_not_have=("--align-loss-weight", "--pos-in-denominator"),
    ),
    # #379 tau_rep=1.0 reruns — same loss-shape flags as the base arm plus
    # `--tau-rep 1.0`. The NAME stub carries the `_tr1_` marker so base and
    # rerun checkpoints never collide on disk.
    "arm1_tr1": dict(
        name_stub="bb_small_arm1_tr1_split_pred_rep_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--tau-rep 1.0"),
        must_not_have=("--moco-negatives", "--moco-rep-keys",
                       "--align-loss-weight", "--pos-in-denominator"),
    ),
    "arm3_tr1": dict(
        name_stub="bb_small_arm3_tr1_split_pred_rep_moco_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--moco-negatives", "--tau-rep 1.0"),
        must_not_have=("--moco-rep-keys", "--align-loss-weight",
                       "--pos-in-denominator"),
    ),
    "arm5_tr1": dict(
        name_stub="bb_small_arm5_tr1_lalign_lrep_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_rep_only",
                   "--align-loss-weight 1.0", "--tau-rep 1.0"),
        must_not_have=("--moco-negatives", "--moco-rep-keys",
                       "--pos-in-denominator"),
    ),
    "arm6_v2_tr1": dict(
        name_stub="bb_small_arm6_v2_tr1_lalign_lrepmoco_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_rep_only",
                   "--align-loss-weight 1.0", "--moco-rep-keys",
                   "--tau-rep 1.0"),
        must_not_have=("--moco-negatives", "--pos-in-denominator"),
    ),
    "bimoco_tr1": dict(
        name_stub="bb_small_bimoco_tr1_split_pred_rep_moco_bothsides_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--moco-negatives", "--moco-rep-keys", "--tau-rep 1.0"),
        must_not_have=("--align-loss-weight", "--pos-in-denominator"),
    ),
    # #379 no-sigreg-embedding (nse) reruns — same loss-shape as the base
    # arm; the `--sigreg-embedding-weight 0.0` override is emitted via
    # EXTRA_ARGS after the trainer's shared `--sigreg-embedding-weight 1.0`
    # (Python argparse's last-value-wins rule zeroes the e_t regulariser).
    # NAME stub carries the `_nse_` marker so base and rerun checkpoints
    # never collide on disk.
    "arm1_nse": dict(
        name_stub="bb_small_arm1_nse_split_pred_rep_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--sigreg-embedding-weight 0.0"),
        must_not_have=("--moco-negatives", "--moco-rep-keys",
                       "--align-loss-weight", "--pos-in-denominator",
                       "--tau-rep", "--cpc-infonce-weight 0.0"),
    ),
    "arm3_nse": dict(
        name_stub="bb_small_arm3_nse_split_pred_rep_moco_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--moco-negatives", "--sigreg-embedding-weight 0.0"),
        must_not_have=("--moco-rep-keys", "--align-loss-weight",
                       "--pos-in-denominator", "--tau-rep",
                       "--cpc-infonce-weight 0.0"),
    ),
    "arm4_nse": dict(
        name_stub="bb_small_arm4_nse_xshh_allt_moco_enc3l3_b64_200k",
        must_have=(
            "--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt",
            "--pos-in-denominator",
            "--subtract-contrastive-floor",
            "--moco-negatives",
            "--sigreg-embedding-weight 0.0",
        ),
        must_not_have=("--moco-rep-keys", "--align-loss-weight",
                       "--tau-rep", "--cpc-infonce-weight 0.0"),
    ),
    "arm5_nse": dict(
        name_stub="bb_small_arm5_nse_lalign_lrep_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_rep_only",
                   "--align-loss-weight 1.0",
                   "--sigreg-embedding-weight 0.0"),
        must_not_have=("--moco-negatives", "--moco-rep-keys",
                       "--pos-in-denominator", "--tau-rep",
                       "--cpc-infonce-weight 0.0"),
    ),
    "arm6_v2_nse": dict(
        name_stub="bb_small_arm6_v2_nse_lalign_lrepmoco_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_rep_only",
                   "--align-loss-weight 1.0", "--moco-rep-keys",
                   "--sigreg-embedding-weight 0.0"),
        must_not_have=("--moco-negatives", "--pos-in-denominator",
                       "--tau-rep", "--cpc-infonce-weight 0.0"),
    ),
    "bimoco_nse": dict(
        name_stub="bb_small_bimoco_nse_split_pred_rep_moco_bothsides_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--moco-negatives", "--moco-rep-keys",
                   "--sigreg-embedding-weight 0.0"),
        must_not_have=("--align-loss-weight", "--pos-in-denominator",
                       "--tau-rep", "--cpc-infonce-weight 0.0"),
    ),
    # #379 no-CPC (ncpc) reruns — same loss-shape as the base arm; the
    # `--cpc-infonce-weight 0.0` override is emitted via EXTRA_ARGS after
    # the trainer's shared `--cpc-infonce-weight 1.0` (last-value-wins
    # disables the CPC auxiliary while keeping SIGReg on).
    "arm1_ncpc": dict(
        name_stub="bb_small_arm1_ncpc_split_pred_rep_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--cpc-infonce-weight 0.0"),
        must_not_have=("--moco-negatives", "--moco-rep-keys",
                       "--align-loss-weight", "--pos-in-denominator",
                       "--tau-rep", "--sigreg-embedding-weight 0.0"),
    ),
    "arm3_ncpc": dict(
        name_stub="bb_small_arm3_ncpc_split_pred_rep_moco_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--moco-negatives", "--cpc-infonce-weight 0.0"),
        must_not_have=("--moco-rep-keys", "--align-loss-weight",
                       "--pos-in-denominator", "--tau-rep",
                       "--sigreg-embedding-weight 0.0"),
    ),
    "arm4_ncpc": dict(
        name_stub="bb_small_arm4_ncpc_xshh_allt_moco_enc3l3_b64_200k",
        must_have=(
            "--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt",
            "--pos-in-denominator",
            "--subtract-contrastive-floor",
            "--moco-negatives",
            "--cpc-infonce-weight 0.0",
        ),
        must_not_have=("--moco-rep-keys", "--align-loss-weight",
                       "--tau-rep", "--sigreg-embedding-weight 0.0"),
    ),
    "arm5_ncpc": dict(
        name_stub="bb_small_arm5_ncpc_lalign_lrep_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_rep_only",
                   "--align-loss-weight 1.0",
                   "--cpc-infonce-weight 0.0"),
        must_not_have=("--moco-negatives", "--moco-rep-keys",
                       "--pos-in-denominator", "--tau-rep",
                       "--sigreg-embedding-weight 0.0"),
    ),
    "arm6_v2_ncpc": dict(
        name_stub="bb_small_arm6_v2_ncpc_lalign_lrepmoco_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_rep_only",
                   "--align-loss-weight 1.0", "--moco-rep-keys",
                   "--cpc-infonce-weight 0.0"),
        must_not_have=("--moco-negatives", "--pos-in-denominator",
                       "--tau-rep", "--sigreg-embedding-weight 0.0"),
    ),
    "bimoco_ncpc": dict(
        name_stub="bb_small_bimoco_ncpc_split_pred_rep_moco_bothsides_enc3l3_b64_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--moco-negatives", "--moco-rep-keys",
                   "--cpc-infonce-weight 0.0"),
        must_not_have=("--align-loss-weight", "--pos-in-denominator",
                       "--tau-rep", "--sigreg-embedding-weight 0.0"),
    ),
}


def strip_comments(text: str) -> str:
    """Remove full-line bash comments so token-search sees only code."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


@pytest.fixture(scope="module")
def launcher_source() -> str:
    return LAUNCHER.read_text()


@pytest.fixture(scope="module")
def launcher_code(launcher_source: str) -> str:
    return strip_comments(launcher_source)


def extract_arm_case_body(code: str, arm: str) -> str:
    """Return the body of the `case "$ARM" in ... <arm>) ... ;;` block."""
    m = re.search(rf'(?m)^\s*{re.escape(arm)}\)\s*\n(.*?)\n\s*;;',
                  code, re.DOTALL)
    assert m is not None, f"no case body for arm {arm!r} in run_arm.sh"
    return m.group(1)


def test_launcher_exists(launcher_source: str):
    assert launcher_source, "run_arm.sh is empty"
    assert 'ARM="${1:?' in launcher_source, (
        "run_arm.sh must take <arm> as its first positional argument.")


@pytest.mark.parametrize("arm", ARMS)
def test_arm_case_block_sets_required_vars(launcher_code: str, arm: str):
    body = extract_arm_case_body(launcher_code, arm)
    for var in ("NAME=", "ARM_DESC=", "LOSS_ARGS="):
        assert var in body, (
            f"arm {arm}: case block must set {var} — got:\n{body}")


@pytest.mark.parametrize("arm", ARMS)
def test_arm_case_block_name_stub(launcher_code: str, arm: str):
    body = extract_arm_case_body(launcher_code, arm)
    stub = ARM_EXPECTATIONS[arm]["name_stub"]
    assert stub in body, (
        f"arm {arm}: NAME must contain {stub!r} — got:\n{body}")


@pytest.mark.parametrize("arm", ARMS)
def test_arm_case_block_loss_args(launcher_code: str, arm: str):
    body = extract_arm_case_body(launcher_code, arm)
    expect = ARM_EXPECTATIONS[arm]
    for token in expect["must_have"]:
        assert token in body, (
            f"arm {arm}: LOSS_ARGS must contain {token!r} — got:\n{body}")
    for token in expect["must_not_have"]:
        assert token not in body, (
            f"arm {arm}: LOSS_ARGS must NOT contain {token!r} — got:\n{body}")


@pytest.mark.parametrize("arm", ARMS_TR1)
def test_tau_rep_arm_case_block_sets_required_vars(launcher_code: str, arm: str):
    body = extract_arm_case_body(launcher_code, arm)
    for var in ("NAME=", "ARM_DESC=", "LOSS_ARGS="):
        assert var in body, (
            f"arm {arm}: case block must set {var} — got:\n{body}")


@pytest.mark.parametrize("arm", ARMS_TR1)
def test_tau_rep_arm_case_block_name_stub(launcher_code: str, arm: str):
    body = extract_arm_case_body(launcher_code, arm)
    stub = ARM_EXPECTATIONS[arm]["name_stub"]
    assert stub in body, (
        f"arm {arm}: NAME must contain {stub!r} — got:\n{body}")


@pytest.mark.parametrize("arm", ARMS_TR1)
def test_tau_rep_arm_case_block_loss_args(launcher_code: str, arm: str):
    body = extract_arm_case_body(launcher_code, arm)
    expect = ARM_EXPECTATIONS[arm]
    for token in expect["must_have"]:
        assert token in body, (
            f"arm {arm}: LOSS_ARGS must contain {token!r} — got:\n{body}")
    for token in expect["must_not_have"]:
        assert token not in body, (
            f"arm {arm}: LOSS_ARGS must NOT contain {token!r} — got:\n{body}")


# ---------------------------------------------------------------------------
# #379 nse / ncpc arm case-block coverage — same shape as ARMS_TR1: each
# arm sets NAME + ARM_DESC + LOSS_ARGS, its NAME carries the `_nse_` /
# `_ncpc_` marker so base and rerun artefacts never collide, and the
# per-arm block emits the override flag via EXTRA_ARGS (must_have below).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS_NSE + ARMS_NCPC)
def test_variant_arm_case_block_sets_required_vars(launcher_code: str, arm: str):
    body = extract_arm_case_body(launcher_code, arm)
    for var in ("NAME=", "ARM_DESC=", "LOSS_ARGS=", "EXTRA_ARGS="):
        assert var in body, (
            f"arm {arm}: case block must set {var} — got:\n{body}")


@pytest.mark.parametrize("arm", ARMS_NSE + ARMS_NCPC)
def test_variant_arm_case_block_name_stub(launcher_code: str, arm: str):
    body = extract_arm_case_body(launcher_code, arm)
    stub = ARM_EXPECTATIONS[arm]["name_stub"]
    assert stub in body, (
        f"arm {arm}: NAME must contain {stub!r} — got:\n{body}")


@pytest.mark.parametrize("arm", ARMS_NSE + ARMS_NCPC)
def test_variant_arm_case_block_flags(launcher_code: str, arm: str):
    body = extract_arm_case_body(launcher_code, arm)
    expect = ARM_EXPECTATIONS[arm]
    for token in expect["must_have"]:
        assert token in body, (
            f"arm {arm}: case block must contain {token!r} — got:\n{body}")
    for token in expect["must_not_have"]:
        assert token not in body, (
            f"arm {arm}: case block must NOT contain {token!r} — got:\n{body}")


def test_extra_args_appended_after_shared_defaults(launcher_code: str):
    """Argparse's last-value-wins rule requires EXTRA_ARGS AFTER defaults.

    `--sigreg-embedding-weight 1.0` and `--cpc-infonce-weight 1.0` are
    part of the shared trainer body. The per-arm EXTRA_ARGS carries
    `--sigreg-embedding-weight 0.0` / `--cpc-infonce-weight 0.0`
    overrides — those MUST land after the shared defaults on the trainer
    command line so Python argparse picks the zero.
    """
    idx_default_sigreg = launcher_code.find("--sigreg-embedding-weight 1.0")
    idx_default_cpc    = launcher_code.find("--cpc-infonce-weight 1.0")
    idx_extra_args     = launcher_code.find('"${EXTRA_ARGS[@]}"')
    assert idx_default_sigreg > 0 and idx_default_cpc > 0, (
        "run_arm.sh must still carry the shared "
        "`--sigreg-embedding-weight 1.0` and `--cpc-infonce-weight 1.0` "
        "defaults in the trainer body.")
    assert idx_extra_args > 0, (
        "run_arm.sh must emit `\"${EXTRA_ARGS[@]}\"` on the trainer "
        "command line (per-arm override channel for nse / ncpc reruns).")
    assert idx_extra_args > idx_default_sigreg, (
        "`\"${EXTRA_ARGS[@]}\"` must appear AFTER "
        "`--sigreg-embedding-weight 1.0` so the nse override wins.")
    assert idx_extra_args > idx_default_cpc, (
        "`\"${EXTRA_ARGS[@]}\"` must appear AFTER "
        "`--cpc-infonce-weight 1.0` so the ncpc override wins.")


def test_extra_args_default_is_empty_array(launcher_code: str):
    """Base + tau_rep arms don't override anything → EXTRA_ARGS must
    default to an empty array so `set -u` + `"${EXTRA_ARGS[@]}"` stays
    well-defined for those arms.
    """
    assert re.search(r'^\s*EXTRA_ARGS=\(\)\s*$', launcher_code, re.MULTILINE), (
        "run_arm.sh must default `EXTRA_ARGS=()` before the case block "
        "so base + tau_rep arms don't hit an unbound-array expansion.")


@pytest.mark.parametrize("literal", BACKBONE_LITERALS)
def test_backbone_config_literal_present(launcher_code: str, literal: str):
    # These are shared across all arms — they must appear in the outer
    # training invocation. One occurrence is fine and expected.
    assert launcher_code.count(literal) >= 1, (
        f"backbone literal {literal!r} missing from run_arm.sh (shared body).")


def test_save_cadence_defaults(launcher_code: str):
    # Pivot: save-every 25000, single extra snapshot at 2500 (labelled `_2k`).
    m_se = re.search(r'SAVE_EVERY="\$\{SAVE_EVERY:-([0-9]+)\}"', launcher_code)
    assert m_se is not None and m_se.group(1) == "25000", (
        f"SAVE_EVERY default must be 25000; got {m_se.group(1) if m_se else 'unset'!r}")
    m_ex = re.search(r'EXTRA_SAVES="\$\{EXTRA_SAVES:-([^}]+)\}"', launcher_code)
    assert m_ex is not None and m_ex.group(1).strip() == "2500", (
        f"EXTRA_SAVES default must be '2500'; got {m_ex.group(1) if m_ex else 'unset'!r}")


def test_launcher_is_backbone_only(launcher_code: str):
    # Pivot: no downstream / q-head / eval sections.
    forbidden = (
        "QTRAIN=", "QEVAL=", "train_head_cell", "eval_cell", "downstream_hl",
        "--head-nhead", "--quantile-head", "--head-arch", "--forecast-len",
        "gift_eval_full_", "BB_STEPS_K", "QEVAL_EXTRA_ARGS", "GPU_2L", "GPU_6L",
        "HEAD_STEPS", "HEAD_WARMUP",
    )
    for token in forbidden:
        assert token not in launcher_code, (
            f"run_arm.sh must be backbone-only — found forbidden token "
            f"{token!r} (issue #379 pivot removed all downstream wiring)")


def test_wt_under_tmp_is_rejected(launcher_code: str):
    # Default WT under $HOME and refuse /tmp.
    assert re.search(r'WT="\$\{WT:-\$HOME/', launcher_code), (
        "run_arm.sh should default WT under $HOME (a persistent checkout), "
        "never /tmp.")
    assert "/tmp/*|/tmp" in launcher_code, (
        "run_arm.sh must reject WT under /tmp with a loud ABORT.")


def test_arm_label_in_complete_log_line(launcher_code: str):
    # The completion log must reference the current arm, not hard-coded "arm 1".
    assert '"$ARM complete:' in launcher_code, (
        'run_arm.sh completion log must read `"$ARM complete: …"`.')


def test_no_dead_bblast_var(launcher_code: str):
    assert "BBLAST=" not in launcher_code, (
        "run_arm.sh must not carry the dead BBLAST assignment.")


def test_orchestrator_sequences_three_phases():
    orch = (EXP_DIR / "scripts" / "orchestrate.sh").read_text()
    body = strip_comments(orch)
    for phase in ("PHASE A", "PHASE B", "PHASE C"):
        assert phase in body, f"orchestrate.sh missing {phase}"
    for arm in ARMS:
        assert f"launch_arm {arm} " in body, (
            f"orchestrate.sh must include `launch_arm {arm} …`")
    # Pivot: no downstream wiring in the orchestrator either.
    for token in ("GPU_2L=", "GPU_6L=", "dl_2L", "dl_6L"):
        assert token not in body, (
            f"orchestrate.sh must not carry downstream token {token!r}")


def test_sync_loop_covers_all_arms_and_classes():
    sync = (EXP_DIR / "sync" / "sync_loop.sh").read_text()
    for arm in ARMS + ARMS_TR1 + ARMS_NSE + ARMS_NCPC:
        assert f"NAME_{arm}=" in sync, f"sync_loop.sh must define NAME_{arm}"
        assert f" {arm} " in sync or f" {arm})" in sync or f"({arm} " in sync \
               or f" {arm}" in sync, (
            f"sync_loop.sh ARMS list must include `{arm}`")
    # b64 name suffix must match the launcher.
    for arm in ARMS:
        assert "_b64_200k_" in sync, "sync_loop.sh names must carry `_b64_200k_` suffix"
        break
    for floor in ("BACKBONE_MIN=", "BACKBONE_OPT_MIN=", "TEXT_MIN="):
        assert floor in sync, f"sync_loop.sh must define per-class floor {floor}"
    # Pivot: no head / eval file classes.
    for token in ("HEAD_MIN=", "HEAD_OPT_MIN=", "gift_eval_full_", "qhead_"):
        assert token not in sync, (
            f"sync_loop.sh must not carry downstream token {token!r}")
    # Backbone-step cadence covers the union of base 6-arm (save-every 25 000
    # + extra {2500}) and #379 tau_rep staged waves (wave 1 save-every 10 000
    # → adds 10/20/30/40). Missing any step here means that arm's
    # `_<N>k.pth` never lands locally.
    for step_k in ("2", "10", "20", "25", "30", "40",
                   "50", "75", "100", "125", "150", "175", "200"):
        assert re.search(rf'BACKBONE_STEPS_K="[^"]*\b{step_k}\b', sync), (
            f"sync_loop.sh BACKBONE_STEPS_K must include step_k={step_k} "
            f"(union of base + tau_rep-wave-1 cadences).")
    assert "/tmp/*|/tmp" in sync, (
        "sync_loop.sh must reject LOCAL_DIR under /tmp.")
    # #379 — every orchestrator log must be pulled so the wave/phase
    # audit trail lands locally alongside the checkpoints. The three
    # new orchestrators (nse, ncpc, base-fresh) each write their own
    # log at the experiment-root results/ dir.
    for log_name in ("orchestrate_tau_rep.log", "orchestrate_no_sigreg_e.log",
                     "orchestrate_no_cpc.log", "orchestrate_base_fresh.log"):
        assert log_name in sync, (
            f"sync_loop.sh must pull `{log_name}` so its #379 rerun-wave "
            "audit trail lands locally.")


# ---------------------------------------------------------------------------
# #379 staged tau_rep=1.0 orchestrator — waves 40 000 → 100 000 → 200 000,
# each wave broken into 3 sub-phases (5 arms across 2 GPUs).
# ---------------------------------------------------------------------------

def _read_orch_tau_rep_code() -> str:
    return strip_comments(
        (EXP_DIR / "scripts" / "orchestrate_tau_rep.sh").read_text())


# Wave index → phase letter (continues A/B/C from the base 6-arm
# orchestrator so a shared log is unambiguous).
WAVE_TO_PHASE = {"1": "D", "2": "E", "3": "F"}


def test_orchestrate_tau_rep_wave_schedule():
    body = _read_orch_tau_rep_code()
    # Wave schedule literals — one entry per wave. Each entry is
    # 'wave|phase_letter|target|save_every|extras'. Locking the
    # target / save_every / letter here is what makes the staged
    # behaviour a contract, not a coincidence.
    for entry in (
        '"1|D|40000|10000|2500,40000"',
        '"2|E|100000|25000|100000"',
        '"3|F|200000|25000|"',
    ):
        assert entry in body, (
            f"orchestrate_tau_rep.sh WAVE_SCHEDULE must contain {entry!r}.")
    # An outer loop that iterates the wave entries — a plain sequence of
    # three inline blocks would let a future edit silently reorder or
    # drop a wave. Enforce the loop form.
    assert re.search(r'for\s+\w+\s+in\s+"\$\{WAVE_SCHEDULE\[@\]\}"', body), (
        "orchestrate_tau_rep.sh must drive waves with `for … in "
        '"${WAVE_SCHEDULE[@]}"` so adding/removing a wave is a single-'
        "line edit.")
    # FINAL_STEPS is the arm's true final and must equal wave-3's target.
    assert re.search(r'FINAL_STEPS=200000\b', body), (
        "orchestrate_tau_rep.sh must set FINAL_STEPS=200000 (the true "
        "final step; run_arm.sh only writes _FINAL.pth when TARGET_STEPS "
        "reaches this).")


def test_orchestrate_tau_rep_wave_to_phase_letter_mapping():
    """wave 1 → PHASE D, wave 2 → PHASE E, wave 3 → PHASE F.

    The letter travels through the orchestrator on three surfaces —
    the schedule entries (asserted in
    ``test_orchestrate_tau_rep_wave_schedule``), the outer loop's
    destructuring, and `run_wave`'s positional signature. If any of
    those drops the letter, log lines silently fall back to bare
    "WAVE 1" style and the D/E/F contract is lost.
    """
    body = _read_orch_tau_rep_code()
    assert re.search(
        r'read\s+-r\s+WAVE\s+LETTER\s+TARGET\s+SAVE_EVERY\s+EXTRAS', body), (
        "orchestrate_tau_rep.sh outer loop must destructure "
        "`WAVE|LETTER|TARGET|SAVE_EVERY|EXTRAS` from each schedule entry.")
    # `run_wave` must accept the letter positionally so it appears in
    # both the phase-start and phase-end logs.
    assert 'local wave="$1" letter="$2"' in body, (
        "run_wave must accept `wave` and `letter` as its first two "
        "positional arguments (via `local wave=\"$1\" letter=\"$2\" …`).")
    # The outer driver line — passes WAVE and LETTER into run_wave in
    # that order, matching run_wave's signature.
    assert re.search(
        r'run_wave\s+"\$WAVE"\s+"\$LETTER"\s+"\$TARGET"\s+"\$SAVE_EVERY"\s+"\$EXTRAS"',
        body), (
        "orchestrate_tau_rep.sh outer loop must call "
        "`run_wave \"$WAVE\" \"$LETTER\" \"$TARGET\" \"$SAVE_EVERY\" "
        "\"$EXTRAS\"` — argument order must match the run_wave signature.")


def test_orchestrate_tau_rep_three_subphases_per_wave():
    body = _read_orch_tau_rep_code()
    # Sub-phase X1 pair, sub-phase X2 pair, sub-phase X3 solo — the shape
    # the base 6-arm orchestrator uses but re-run once per wave. The
    # `run_wave` function is the only place that fans out the arms;
    # asserting the arm-to-GPU assignments there prevents a stray edit
    # from doubling up two arms on GPU 0 (would OOM at bs=64). Sub-phase
    # names use the parent phase letter (`${letter}1`, etc.) so D-wave
    # and E-wave log lines never collide in the shared log.
    assert re.search(r'^run_wave\s*\(\)\s*\{', body, re.MULTILINE), (
        "orchestrate_tau_rep.sh must define a `run_wave()` function so "
        "each wave runs an identical 3-sub-phase pipeline.")
    assert re.search(r'sub-phase \$\{letter\}1.*arm1_tr1.*arm3_tr1', body), (
        "sub-phase ${letter}1 must pair arm1_tr1 (GPU 0) with arm3_tr1 (GPU 1).")
    assert re.search(r'sub-phase \$\{letter\}2.*arm5_tr1.*arm6_v2_tr1', body), (
        "sub-phase ${letter}2 must pair arm5_tr1 (GPU 0) with arm6_v2_tr1 (GPU 1).")
    assert re.search(r'sub-phase \$\{letter\}3.*bimoco_tr1.*GPU 0.*solo', body), (
        "sub-phase ${letter}3 must run bimoco_tr1 alone on GPU 0.")
    # Each arm must appear as a `launch_arm` call so the shape test can
    # grep for it directly (mirrors the base-6 orchestrator's contract).
    for arm in ARMS_TR1:
        assert re.search(rf'\blaunch_arm\s+{re.escape(arm)}\b', body), (
            f"orchestrate_tau_rep.sh must include `launch_arm {arm} …` in "
            f"run_wave (found via regex search on the stripped body).")
    # GPU pairing: three lines that assign `pid_a` / `pid_b` so
    # `wait_pair` gets the right two PIDs. If someone re-numbers the
    # sub-phases, the pair-wait must still barrier correctly.
    assert body.count("pid_a=$!") == 2, (
        "run_wave must background exactly 2 arms per pair (sub-phase ${letter}1 "
        "+ sub-phase ${letter}2).")
    assert body.count("pid_b=$!") == 2, (
        "run_wave must background exactly 2 arms per pair (sub-phase ${letter}1 "
        "+ sub-phase ${letter}2).")


def test_orchestrate_tau_rep_barrier_per_wave():
    body = _read_orch_tau_rep_code()
    # `wait_pair` and the solo direct call inside `run_wave` create the
    # per-wave barrier: `run_wave` cannot return until every arm in the
    # wave finishes. The outer loop then advances to the next wave.
    # `wait_pair` must appear twice inside run_wave (once per pair). The
    # solo bimoco call is synchronous. The outer `for` loop over
    # WAVE_SCHEDULE only calls `run_wave` after the previous returns.
    m = re.search(r'^run_wave\s*\(\)\s*\{(.*?)^\}', body,
                  re.MULTILINE | re.DOTALL)
    assert m is not None, "orchestrate_tau_rep.sh: cannot locate run_wave body"
    inside = m.group(1)
    assert inside.count("wait_pair ") == 2, (
        "run_wave must wait_pair twice (one per 2-arm sub-phase).")
    # Solo call — not backgrounded (no trailing &).
    assert re.search(r'launch_arm\s+bimoco_tr1\b[^&]*$', inside, re.MULTILINE), (
        "run_wave's bimoco_tr1 call must NOT be backgrounded (& absent) "
        "so its return code gates the wave summary.")


def test_orchestrate_tau_rep_wave_end_summary():
    body = _read_orch_tau_rep_code()
    # An end-of-wave summary log makes the resumption / debugging
    # story auditable; without it a partial wave failure is invisible
    # in the orchestrator log.
    assert "count_arms_at_step" in body, (
        "orchestrate_tau_rep.sh must count how many arms reached the "
        "wave target and log the ratio (auditability).")
    assert re.search(r'PHASE\s+\$letter\s+DONE', body), (
        "orchestrate_tau_rep.sh must log `PHASE $letter DONE — wave $wave · "
        "arms at ≥ …` at the end of each wave.")


def test_orchestrate_tau_rep_delegates_target_and_final_to_run_arm():
    body = _read_orch_tau_rep_code()
    # The orchestrator's contract with run_arm.sh: TARGET_STEPS gates
    # this wave's total_steps; FINAL_STEPS gates whether _FINAL.pth is
    # written. Missing either → the intermediate waves would incorrectly
    # write _FINAL.pth (breaking the resume chain).
    assert 'TARGET_STEPS="$target"' in body, (
        "orchestrate_tau_rep.sh must pass TARGET_STEPS to run_arm.sh.")
    assert 'FINAL_STEPS="$FINAL_STEPS"' in body, (
        "orchestrate_tau_rep.sh must pass FINAL_STEPS to run_arm.sh.")
    assert 'SAVE_EVERY="$se"' in body, (
        "orchestrate_tau_rep.sh must pass per-wave SAVE_EVERY to run_arm.sh.")
    assert 'EXTRA_SAVES="$ex"' in body, (
        "orchestrate_tau_rep.sh must pass per-wave EXTRA_SAVES to run_arm.sh.")


def test_orchestrate_tau_rep_target_steps_per_wave():
    """TARGET_STEPS pass-through: wave N's target must be the schedule value.

    Ties the schedule table (targets 40 000 / 100 000 / 200 000) to what
    `launch_arm` actually sends to `run_arm.sh`. Without this, a rewrite
    could keep the schedule literals but silently pass e.g. FINAL_STEPS
    to run_arm.sh (arms would jump straight to 200k on wave 1).
    """
    body = _read_orch_tau_rep_code()
    # launch_arm's TARGET_STEPS assignment must reference the parameter
    # named `target`, which run_wave forwards from its own `$3`.
    assert re.search(
        r'launch_arm\s*\(\).*?local\s+arm="\$1"\s+bb_gpu="\$2"\s+letter="\$3"\s+target="\$4"',
        body, re.DOTALL), (
        "orchestrate_tau_rep.sh launch_arm signature must be "
        "`arm bb_gpu letter target save_every extras` (positions 1..6) — "
        "so run_wave's `$target` reaches run_arm.sh untouched.")
    # And the actual TARGET_STEPS env var passed to the subshell.
    assert re.search(
        r'TARGET_STEPS="\$target"\s+FINAL_STEPS="\$FINAL_STEPS"',
        body), (
        "orchestrate_tau_rep.sh must set both `TARGET_STEPS=$target` and "
        "`FINAL_STEPS=$FINAL_STEPS` on the run_arm.sh sub-shell "
        "(single assignment line).")


# ---------------------------------------------------------------------------
# #379 run_arm.sh wave-support: TARGET_STEPS / FINAL_STEPS handling,
# _FINAL.pth gating, wave idempotency skip.
# ---------------------------------------------------------------------------

def test_run_arm_accepts_target_and_final_steps(launcher_code: str):
    # TARGET_STEPS overrides STEPS; STEPS still recognised as a legacy
    # alias so pre-refactor callers (base 6-arm orchestrator) keep
    # working without edits.
    assert 'TARGET_STEPS="${TARGET_STEPS:-${STEPS:-200000}}"' in launcher_code, (
        "run_arm.sh must derive TARGET_STEPS from TARGET_STEPS ∨ STEPS "
        "∨ default 200000 (in that precedence).")
    assert 'STEPS="$TARGET_STEPS"' in launcher_code, (
        "run_arm.sh must set STEPS=$TARGET_STEPS so the existing "
        "--total-steps \"$STEPS\" line picks up the wave target.")
    assert 'FINAL_STEPS="${FINAL_STEPS:-200000}"' in launcher_code, (
        "run_arm.sh must accept FINAL_STEPS (default 200000).")


def test_run_arm_final_pth_gated_on_target_reaching_final(launcher_code: str):
    # The wave-boundary contract: only write _FINAL.pth (the run_arm.sh
    # skip sentinel) when this launch trained all the way to the arm's
    # true final step. Otherwise the next wave's launcher would
    # short-circuit on the FINAL-exists guard and never resume.
    assert re.search(
        r'if\s+\[\s+"\$TARGET_STEPS"\s+-lt\s+"\$FINAL_STEPS"\s+\]\s*;\s*then\s*\n'
        r'.*?wave complete.*?exit\s+0',
        launcher_code, re.DOTALL), (
        "run_arm.sh must exit 0 without copying _FINAL.pth when "
        "TARGET_STEPS < FINAL_STEPS (intermediate wave).")


def test_run_arm_wave_skip_when_target_already_reached(launcher_code: str):
    # Wave-idempotency: if an existing _<N>k.pth is already at or past
    # TARGET_STEPS on an intermediate wave, re-running the orchestrator
    # should short-circuit rather than pay the trainer startup cost.
    assert "WAVE SKIP" in launcher_code, (
        "run_arm.sh must log a `WAVE SKIP` when a periodic checkpoint "
        "already meets the wave target and this is not the final wave.")
    # The gate must include the < FINAL_STEPS condition (else re-running
    # a completed arm's final wave would skip creating _FINAL.pth).
    assert re.search(
        r'\[\s+"\$TARGET_STEPS"\s+-lt\s+"\$FINAL_STEPS"\s+\]\s*&&\s*'
        r'\[\s+"\$best_ck_k"\s+-ge\s+"\$target_k"\s+\]',
        launcher_code), (
        "run_arm.sh WAVE SKIP guard must combine "
        "TARGET_STEPS < FINAL_STEPS AND best_ck_k ≥ target_k.")


def test_run_arm_wave_endpoint_backfilled_into_extra_saves(launcher_code: str):
    """Intermediate wave: run_arm.sh must ensure TARGET_STEPS is a saved snapshot.

    Rationale: the next wave's launcher resumes from the newest
    `_<N>k.pth`. If TARGET_STEPS isn't a save-every multiple and the
    caller forgets to list it in EXTRA_SAVES, the trainer would stop one
    save-every short and the resume chain breaks. The backfill is
    idempotent (dedup'd by the parse_extra_save_steps 1000-block guard —
    same-value dupes are permitted).
    """
    # The gating condition — only backfill on intermediate waves.
    assert re.search(
        r'if\s+\[\s+"\$TARGET_STEPS"\s+-lt\s+"\$FINAL_STEPS"\s+\]\s*;\s*then'
        r'.*?case\s+",\$EXTRA_SAVES,"',
        launcher_code, re.DOTALL), (
        "run_arm.sh must gate the EXTRA_SAVES wave-endpoint backfill "
        "on `TARGET_STEPS < FINAL_STEPS` (only intermediate waves).")
    # The append (idempotent: `EXTRA_SAVES:+…,` yields "" when empty so
    # we don't accidentally prepend a stray comma).
    assert 'EXTRA_SAVES="${EXTRA_SAVES:+$EXTRA_SAVES,}$TARGET_STEPS"' in launcher_code, (
        "run_arm.sh must append `$TARGET_STEPS` to EXTRA_SAVES with the "
        "`${EXTRA_SAVES:+$EXTRA_SAVES,}` prefix so an empty EXTRA_SAVES "
        "becomes just `$TARGET_STEPS` (no leading comma).")


def test_run_arm_resume_flag_pipes_intermediate_checkpoint(launcher_code: str):
    # Resume path: pick the latest `_<N>k.pth` (excluding optimizer
    # sidecar). The trainer's own resume path takes over from there.
    # `_FINAL.pth` (uppercase L) doesn't match `_*k.pth` — that's the
    # invariant that keeps the sentinel out of the resume candidate list.
    assert re.search(
        r'latest=\$\(ls -t\s+"\$RUNS/\$\{NAME\}"_\*k\.pth\b',
        launcher_code), (
        "run_arm.sh resume-latest logic must ls _<N>k.pth candidates "
        "sorted by mtime.")
    assert 'RESUME="--resume $latest"' in launcher_code, (
        "run_arm.sh must pass `--resume <path>` to the trainer when a "
        "prior checkpoint exists.")


# ---------------------------------------------------------------------------
# #379 nse / ncpc / base-fresh orchestrators — each mirrors the tau_rep
# staged-wave shape with its own phase-letter block (G/H/I, J/K/L,
# M/N/O). MAX_WAVE env var support gates the outer loop so a run can
# stop after a chosen wave (Wave-D-first barrier — all variants hit 40k
# before any advances further).
# ---------------------------------------------------------------------------

# Per-orchestrator spec: file name, arm-suffix, phase letters
# (wave_1 / wave_2 / wave_3), and the sub-phase pairings that the
# corresponding `run_wave` must emit. `subphases` is a list of tuples —
# each tuple is (gpu0_arm, gpu1_arm_or_None). None → solo on GPU 0.
_NEW_ORCHESTRATORS = [
    dict(
        file="orchestrate_no_sigreg_e.sh",
        tag="nse",
        letters=("G", "H", "I"),
        arm_count=6,
        subphases=(
            ("arm1_nse", "arm3_nse"),
            ("arm4_nse", "arm5_nse"),
            ("arm6_v2_nse", "bimoco_nse"),
        ),
    ),
    dict(
        file="orchestrate_no_cpc.sh",
        tag="ncpc",
        letters=("J", "K", "L"),
        arm_count=6,
        subphases=(
            ("arm1_ncpc", "arm3_ncpc"),
            ("arm4_ncpc", "arm5_ncpc"),
            ("arm6_v2_ncpc", "bimoco_ncpc"),
        ),
    ),
    dict(
        file="orchestrate_base_fresh.sh",
        tag="base_fresh",
        letters=("M", "N", "O"),
        arm_count=2,
        subphases=(
            ("arm6_v2", "bimoco"),
        ),
    ),
]


def _read_new_orch(name: str) -> str:
    return strip_comments((EXP_DIR / "scripts" / name).read_text())


@pytest.mark.parametrize("spec", _NEW_ORCHESTRATORS,
                         ids=[s["file"] for s in _NEW_ORCHESTRATORS])
def test_new_orchestrator_wave_schedule(spec):
    body = _read_new_orch(spec["file"])
    wave1, wave2, wave3 = spec["letters"]
    for entry in (
        f'"1|{wave1}|40000|10000|2500,40000"',
        f'"2|{wave2}|100000|25000|100000"',
        f'"3|{wave3}|200000|25000|"',
    ):
        assert entry in body, (
            f"{spec['file']} WAVE_SCHEDULE must contain {entry!r}.")
    assert re.search(r'for\s+\w+\s+in\s+"\$\{WAVE_SCHEDULE\[@\]\}"', body), (
        f"{spec['file']} must drive waves with `for … in "
        '"${WAVE_SCHEDULE[@]}"` so adding/removing a wave is a single '
        "edit.")
    assert re.search(r'FINAL_STEPS=200000\b', body), (
        f"{spec['file']} must set FINAL_STEPS=200000.")


@pytest.mark.parametrize("spec", _NEW_ORCHESTRATORS,
                         ids=[s["file"] for s in _NEW_ORCHESTRATORS])
def test_new_orchestrator_max_wave_support(spec):
    """`MAX_WAVE` env var must, when set, break the outer loop after
    that phase letter — this is the Wave-D-first barrier the researcher
    imposed on the 23-arm sweep. When unset, all waves run as before.
    """
    body = _read_new_orch(spec["file"])
    assert re.search(r'MAX_WAVE="\$\{MAX_WAVE:-\}"', body), (
        f"{spec['file']} must declare `MAX_WAVE=\"${{MAX_WAVE:-}}\"` so "
        "the outer loop treats an unset env var as \"run everything\".")
    assert re.search(
        r'if\s+\[\s+-n\s+"\$MAX_WAVE"\s+\]\s*&&\s*'
        r'\[\s+"\$LETTER"\s*=\s*"\$MAX_WAVE"\s+\]\s*;\s*then'
        r'.*?break',
        body, re.DOTALL), (
        f"{spec['file']} outer loop must `break` after `$LETTER` matches "
        "`$MAX_WAVE` (only when `MAX_WAVE` is non-empty).")


@pytest.mark.parametrize("spec", _NEW_ORCHESTRATORS,
                         ids=[s["file"] for s in _NEW_ORCHESTRATORS])
def test_new_orchestrator_wave_to_phase_letter_mapping(spec):
    body = _read_new_orch(spec["file"])
    assert re.search(
        r'read\s+-r\s+WAVE\s+LETTER\s+TARGET\s+SAVE_EVERY\s+EXTRAS', body), (
        f"{spec['file']} outer loop must destructure "
        "`WAVE|LETTER|TARGET|SAVE_EVERY|EXTRAS` from each schedule entry.")
    assert 'local wave="$1" letter="$2"' in body, (
        f"{spec['file']} run_wave signature must start with "
        "`local wave=\"$1\" letter=\"$2\" …`.")
    assert re.search(
        r'run_wave\s+"\$WAVE"\s+"\$LETTER"\s+"\$TARGET"\s+"\$SAVE_EVERY"\s+"\$EXTRAS"',
        body), (
        f"{spec['file']} outer loop must call `run_wave` with WAVE, "
        "LETTER, TARGET, SAVE_EVERY, EXTRAS in that order.")


@pytest.mark.parametrize("spec", _NEW_ORCHESTRATORS,
                         ids=[s["file"] for s in _NEW_ORCHESTRATORS])
def test_new_orchestrator_subphase_layout(spec):
    """Every arm appears as a `launch_arm` call in the expected sub-phase
    slot. Sub-phase names use the parent phase letter (`${letter}1`,
    etc.) so wave-1 vs wave-2 vs wave-3 log lines never collide.
    """
    body = _read_new_orch(spec["file"])
    assert re.search(r'^run_wave\s*\(\)\s*\{', body, re.MULTILINE), (
        f"{spec['file']} must define a `run_wave()` function so each "
        "wave runs an identical sub-phase pipeline.")
    for idx, pair in enumerate(spec["subphases"], start=1):
        gpu0, gpu1 = pair
        # Sub-phase log line: names both arms + the parent phase letter.
        pattern = rf'sub-phase \$\{{letter\}}{idx}.*{re.escape(gpu0)}'
        if gpu1 is not None:
            pattern += rf'.*{re.escape(gpu1)}'
        assert re.search(pattern, body), (
            f"{spec['file']} sub-phase ${{letter}}{idx} must pair "
            f"{gpu0} (GPU 0) with {gpu1} (GPU 1).")
    # Every arm shows up in an actual launch_arm invocation.
    for pair in spec["subphases"]:
        for arm in pair:
            if arm is None:
                continue
            assert re.search(rf'\blaunch_arm\s+{re.escape(arm)}\b', body), (
                f"{spec['file']} must include `launch_arm {arm} …` in "
                "run_wave.")
    # Pair count = number of sub-phases with a second GPU (2 backgrounded
    # calls per pair → matching pid_a / pid_b counts).
    pair_count = sum(1 for pair in spec["subphases"] if pair[1] is not None)
    assert body.count("pid_a=$!") == pair_count, (
        f"{spec['file']} run_wave must background exactly {pair_count} "
        "pid_a= assignments (one per 2-arm sub-phase).")
    assert body.count("pid_b=$!") == pair_count, (
        f"{spec['file']} run_wave must background exactly {pair_count} "
        "pid_b= assignments (one per 2-arm sub-phase).")


@pytest.mark.parametrize("spec", _NEW_ORCHESTRATORS,
                         ids=[s["file"] for s in _NEW_ORCHESTRATORS])
def test_new_orchestrator_wave_end_summary(spec):
    body = _read_new_orch(spec["file"])
    assert "count_arms_at_step" in body, (
        f"{spec['file']} must count how many arms reached the wave "
        "target and log the ratio.")
    assert re.search(r'PHASE\s+\$letter\s+DONE', body), (
        f"{spec['file']} must log `PHASE $letter DONE — wave $wave · "
        "arms at ≥ …` at the end of each wave.")
    assert f"/ {spec['arm_count']}" in body, (
        f"{spec['file']} phase-summary line must divide by "
        f"{spec['arm_count']} (this orchestrator's arm count).")


@pytest.mark.parametrize("spec", _NEW_ORCHESTRATORS,
                         ids=[s["file"] for s in _NEW_ORCHESTRATORS])
def test_new_orchestrator_delegates_target_and_final(spec):
    body = _read_new_orch(spec["file"])
    assert 'TARGET_STEPS="$target"' in body, (
        f"{spec['file']} must pass TARGET_STEPS to run_arm.sh.")
    assert 'FINAL_STEPS="$FINAL_STEPS"' in body, (
        f"{spec['file']} must pass FINAL_STEPS to run_arm.sh.")
    assert 'SAVE_EVERY="$se"' in body, (
        f"{spec['file']} must pass per-wave SAVE_EVERY to run_arm.sh.")
    assert 'EXTRA_SAVES="$ex"' in body, (
        f"{spec['file']} must pass per-wave EXTRA_SAVES to run_arm.sh.")


def test_smoke_script_is_backbone_only():
    smoke = EXP_DIR / "scripts" / "smoke.sh"
    assert smoke.is_file(), "smoke.sh missing"
    body = smoke.read_text()
    assert "STEPS=200" in body, (
        "smoke.sh should use a very short STEPS so a ~3min run validates "
        "the backbone pipeline.")
    # Pivot: no head / eval steps in the smoke either.
    for token in ("HEAD_STEPS", "QEVAL_EXTRA_ARGS", "qhead_", "gift_eval_full_",
                  "--head-nhead", "--quantile-head"):
        assert token not in body, (
            f"smoke.sh must be backbone-only — found forbidden token {token!r}")
    # Smoke must verify the training-dynamics columns the plots depend on.
    for col in ("ff", "u_batchtime", "u_batchtime_e"):
        assert col in body, (
            f"smoke.sh should verify losses.csv column {col!r} is populated.")
    assert "SMOKE OK" in body, (
        "smoke.sh should print `SMOKE OK` on success.")
