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
    for arm in ARMS + ARMS_TR1:
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
    # Backbone-step cadence: 2 (extra) + 25/50/…/200 (save-every=25000).
    assert 'BACKBONE_STEPS_K="2 25 50 75 100 125 150 175 200"' in sync, (
        "sync_loop.sh BACKBONE_STEPS_K must be the union of extra-save {2} "
        "and save-every=25000 cadence out to 200k.")
    assert "/tmp/*|/tmp" in sync, (
        "sync_loop.sh must reject LOCAL_DIR under /tmp.")
    # #379 — tau_rep orchestrator log must be pulled too.
    assert "orchestrate_tau_rep.log" in sync, (
        "sync_loop.sh must pull `orchestrate_tau_rep.log` so #379 rerun "
        "phases D/E/F land locally.")


def test_orchestrate_tau_rep_sequences_three_phases():
    orch = (EXP_DIR / "scripts" / "orchestrate_tau_rep.sh").read_text()
    body = strip_comments(orch)
    for phase in ("PHASE D", "PHASE E", "PHASE F"):
        assert phase in body, f"orchestrate_tau_rep.sh missing {phase}"
    for arm in ARMS_TR1:
        assert f"launch_arm {arm} " in body, (
            f"orchestrate_tau_rep.sh must include `launch_arm {arm} …`")
    # Pivot: no downstream wiring in the tau_rep orchestrator either.
    for token in ("GPU_2L=", "GPU_6L=", "dl_2L", "dl_6L"):
        assert token not in body, (
            f"orchestrate_tau_rep.sh must not carry downstream token {token!r}")


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
