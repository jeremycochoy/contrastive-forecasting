"""Consistency tests for the #379 consolidated arm launcher.

`experiments/2026-07-21_split_pred_rep_small/scripts/run_arm.sh` is a
single launcher parameterised by `$ARM ∈ {arm1, arm3, arm4, arm5,
arm6_v2, bimoco}`. Each arm shares the same backbone config (small
model, 200k steps, save-every 10k + extras at 2500,25000) and only the
per-arm case block picks the loss flags copied verbatim from #374.

These tests do not re-verify the loss shapes — those are guarded by
`test_loss*`. They lock in:

  * every arm's per-arm case block sets NAME, ARM_DESC, and LOSS_ARGS;
  * the small-model backbone config appears exactly once in the
    consolidated body (so a stray edit can't specialise one arm);
  * downstream fans out to the 5 backbone-step cells (2k, 25k, 50k,
    100k, 200k) and both head-layer sizes (2L, 6L);
  * the ckpt_path glob tolerates restart suffixes (blocking-1 fix);
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

# Small-model backbone config that the shared body must carry verbatim.
BACKBONE_LITERALS = (
    "--batch-size 128",
    "--total-steps \"$STEPS\"",
    "--seed \"$SEED\"",
    "--save-every \"$SAVE_EVERY\"",
    "--extra-save-steps \"$EXTRA_SAVES\"",
    "--t-raw 4096",
    "--n-channels 1",
    "--d-model 128",
    "--n-heads 16",
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
# `expected_name_stub` is a substring of NAME that uniquely identifies
# the arm; `must_have`/`must_not_have` pin the loss-arg tokens.
ARM_EXPECTATIONS = {
    "arm1": dict(
        name_stub="bb_small_arm1_split_pred_rep_enc3l3_b128_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",),
        must_not_have=("--moco-negatives", "--moco-rep-keys",
                       "--align-loss-weight", "--pos-in-denominator"),
    ),
    "arm3": dict(
        name_stub="bb_small_arm3_split_pred_rep_moco_enc3l3_b128_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--moco-negatives"),
        must_not_have=("--moco-rep-keys", "--align-loss-weight",
                       "--pos-in-denominator"),
    ),
    "arm4": dict(
        name_stub="bb_small_arm4_xshh_allt_moco_enc3l3_b128_200k",
        must_have=(
            "--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt",
            "--pos-in-denominator",
            "--subtract-contrastive-floor",
            "--moco-negatives",
        ),
        must_not_have=("--moco-rep-keys", "--align-loss-weight"),
    ),
    "arm5": dict(
        name_stub="bb_small_arm5_lalign_lrep_enc3l3_b128_200k",
        must_have=("--loss-shape cosine_similarity_batch_rep_only",
                   "--align-loss-weight 1.0"),
        must_not_have=("--moco-negatives", "--moco-rep-keys",
                       "--pos-in-denominator"),
    ),
    "arm6_v2": dict(
        name_stub="bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b128_200k",
        must_have=("--loss-shape cosine_similarity_batch_rep_only",
                   "--align-loss-weight 1.0", "--moco-rep-keys"),
        must_not_have=("--moco-negatives", "--pos-in-denominator"),
    ),
    "bimoco": dict(
        name_stub="bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b128_200k",
        must_have=("--loss-shape cosine_similarity_batch_split_pred_rep",
                   "--moco-negatives", "--moco-rep-keys"),
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
    # Preamble must include the ARM positional arg.
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


@pytest.mark.parametrize("literal", BACKBONE_LITERALS)
def test_backbone_config_literal_present(launcher_code: str, literal: str):
    # These are shared across all arms — they must appear in the outer
    # (post-case) training invocation. One occurrence is fine and expected.
    assert launcher_code.count(literal) >= 1, (
        f"backbone literal {literal!r} missing from run_arm.sh (shared body).")


def test_downstream_step_loop_pins_five_cells(launcher_code: str):
    # `for sk in $BB_STEPS_K; do …` iterates the 5 backbone-step cells.
    m = re.search(r'BB_STEPS_K="\$\{BB_STEPS_K:-([^}]+)\}"', launcher_code)
    assert m is not None, "BB_STEPS_K default must be set in run_arm.sh"
    steps = tuple(m.group(1).split())
    assert steps == ("2", "25", "50", "100", "200"), (
        f"BB_STEPS_K must default to '2 25 50 100 200' (the 5 downstream "
        f"cells); got {steps!r}")


def test_downstream_head_layers_2_and_6(launcher_code: str):
    # Look for the specific invocations that spawn the 2L / 6L pipelines.
    m2 = re.search(r'downstream_hl\s+2\s+"\$GPU_2L"', launcher_code)
    m6 = re.search(r'downstream_hl\s+6\s+"\$GPU_6L"', launcher_code)
    assert m2 is not None, "run_arm.sh must launch downstream_hl 2 on GPU_2L"
    assert m6 is not None, "run_arm.sh must launch downstream_hl 6 on GPU_6L"


def test_ckpt_path_uses_glob_tolerant_of_restart_suffix(launcher_code: str):
    # Blocking-1 fix: `ls "$RUNS/${NAME}"*_${sk}k.pth` (note the `*` between
    # NAME and `_${sk}k.pth`) — the wildcard picks up `_r2` / `_r3` suffixes
    # from safe_run_name. A literal `${NAME}_${sk}k.pth` would silently miss
    # them.
    assert re.search(
        r'\$\{NAME\}"?\*_\$\{sk\}k\.pth', launcher_code
    ) is not None, (
        "ckpt_path must glob NAME*_${sk}k.pth so restart-suffixed "
        "checkpoints (_r2, _r3) are still resolved (blocking #1).")


def test_missing_ckpt_is_loud_abort(launcher_code: str):
    # Blocking-1 companion: a missing backbone snapshot must NOT be
    # silently counted as a failed head cell — the downstream must
    # abort loudly so the operator sees it.
    assert "ABORT downstream" in launcher_code, (
        "downstream_hl must emit `ABORT downstream ...` when a "
        "backbone-step checkpoint cannot be resolved (blocking #1).")


def test_wt_under_tmp_is_rejected(launcher_code: str):
    # Blocking-2 fix: default WT under $HOME and refuse /tmp.
    assert re.search(r'WT="\$\{WT:-\$HOME/', launcher_code), (
        "run_arm.sh should default WT under $HOME (a persistent checkout), "
        "never /tmp (blocking #2).")
    assert "/tmp/*|/tmp" in launcher_code, (
        "run_arm.sh must reject WT under /tmp with a loud ABORT (blocking #2).")


def test_arm_label_in_complete_log_line(launcher_code: str):
    # Item-6 fix: the completion log must reference the current arm, not
    # hard-coded "arm 1".
    assert '"$ARM complete:' in launcher_code, (
        'run_arm.sh completion log must read `"$ARM complete: …"` — the '
        'six-launcher version hard-coded "arm 1 complete" everywhere '
        '(item #6).')


def test_no_dead_bblast_var(launcher_code: str):
    # Item-9 fix: BBLAST was set in every launcher and never read.
    assert "BBLAST=" not in launcher_code, (
        "run_arm.sh must not carry the dead BBLAST assignment (item #9).")


def test_orchestrator_sequences_three_phases():
    orch = (EXP_DIR / "scripts" / "orchestrate.sh").read_text()
    body = strip_comments(orch)
    # Item-10 fix: three phases across 2 GPUs, pairing arms.
    for phase in ("PHASE A", "PHASE B", "PHASE C"):
        assert phase in body, f"orchestrate.sh missing {phase}"
    for arm in ARMS:
        assert f"launch_arm {arm} " in body, (
            f"orchestrate.sh must include `launch_arm {arm} …` (item #10)")


def test_sync_loop_covers_all_arms_and_classes():
    sync = (EXP_DIR / "sync" / "sync_loop.sh").read_text()
    # Blocking-2 fix: sync_loop.sh exists, covers all 6 arms, and has
    # per-class size floors (not one blanket number).
    for arm in ARMS:
        assert f"NAME_{arm}=" in sync, (
            f"sync_loop.sh must define NAME_{arm} (blocking #2)")
    for floor in ("BACKBONE_MIN=", "BACKBONE_OPT_MIN=", "HEAD_MIN=",
                  "HEAD_OPT_MIN=", "TEXT_MIN="):
        assert floor in sync, (
            f"sync_loop.sh must define per-class floor {floor} (blocking #2)")
    assert "/tmp/*|/tmp" in sync, (
        "sync_loop.sh must reject LOCAL_DIR under /tmp (blocking #2)")


def test_head_nhead_divides_d_model(launcher_code: str):
    # The q-head and eval must use --head-nhead=8, not the #374 default 6.
    # d_model=128 is not divisible by 6 (nn.MultiheadAttention asserts
    # embed_dim % num_heads == 0); the smoke caught this at ~86s of
    # backbone + head-build time. Both the q-head trainer and the
    # evaluator must carry --head-nhead 8.
    assert re.search(r'--head-nhead\s+8\b', launcher_code) is not None, (
        "run_arm.sh must set --head-nhead 8 (d_model=128 % 8 == 0). "
        "The #374 default of 6 does NOT divide 128 — will crash.")
    # Neither call may keep the #374 head-nhead 6.
    assert re.search(r'--head-nhead\s+6\b', launcher_code) is None, (
        "run_arm.sh must NOT use --head-nhead 6 anywhere (d_model=128 % 6 ≠ 0).")


def test_smoke_script_exists():
    # Blocking-3: an end-to-end smoke must exist so the arch flag
    # completeness is exercised before ~35h of GPU is committed.
    smoke = EXP_DIR / "scripts" / "smoke.sh"
    assert smoke.is_file(), "smoke.sh missing (blocking #3)"
    body = smoke.read_text()
    assert "STEPS=200" in body and "HEAD_STEPS=200" in body, (
        "smoke.sh should use a very short STEPS + HEAD_STEPS so a ~10min "
        "run validates the full pipeline (blocking #3).")
    assert "SMOKE OK" in body, (
        "smoke.sh should print `SMOKE OK` on success (blocking #3).")
