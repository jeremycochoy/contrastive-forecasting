"""Consistency tests for the #390 arm launcher.

`experiments/2026-08-01_lalign_teacher/scripts/run_arm.sh` retrains the 10
#379 cells that carry an L_align term, with the term pointed at the EMA
teacher. The issue's contract is exact: *each cell keeps its #379 command
line and adds `--align-target teacher`. Nothing else changes.*

So these tests compare the two launchers directly rather than re-listing
flags:

  * per arm, the new case block's flags are the #379 case block's flags
    plus `--align-target teacher`, and nothing else;
  * the shared trainer invocation (batch size, architecture, schedule,
    dataset, dtypes) is character-identical to #379's;
  * only the 10 L_align cells are present — the 20 copied cells are not
    reachable from this launcher;
  * checkpoint names cannot collide with #379's.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
OLD_LAUNCHER = (REPO_ROOT / "experiments" / "2026-07-21_split_pred_rep_small"
                / "scripts" / "run_arm.sh")
NEW_LAUNCHER = (REPO_ROOT / "experiments" / "2026-08-01_lalign_teacher"
                / "scripts" / "run_arm.sh")

# The 10 retrained cells: the two L_align arms × the five #379 settings.
ARMS = ("arm5", "arm5_tr1", "arm5_nse", "arm5_ncpc", "arm5_combab",
        "arm6_v2", "arm6_v2_tr1", "arm6_v2_nse", "arm6_v2_ncpc",
        "arm6_v2_combab")

# Cells whose loss has no L_align term — #379's numbers stand, so this
# launcher must not be able to retrain them.
COPIED_ARMS = ("arm1", "arm3", "arm4", "bimoco",
               "arm1_tr1", "arm3_tr1", "arm4_tr1", "bimoco_tr1",
               "arm1_nse", "arm3_nse", "arm4_nse", "bimoco_nse",
               "arm1_ncpc", "arm3_ncpc", "arm4_ncpc", "bimoco_ncpc",
               "arm1_combab", "arm3_combab", "arm4_combab", "bimoco_combab")

ADDED_FLAG = ["--align-target", "teacher"]


def strip_comments(text: str) -> str:
    """Remove full-line bash comments so token-search sees only code."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


@pytest.fixture(scope="module")
def new_source() -> str:
    return NEW_LAUNCHER.read_text()


@pytest.fixture(scope="module")
def new_code(new_source: str) -> str:
    return strip_comments(new_source)


@pytest.fixture(scope="module")
def old_code() -> str:
    return strip_comments(OLD_LAUNCHER.read_text())


def extract_arm_case_body(code: str, arm: str) -> str:
    """Return the body of the `case "$ARM" in ... <arm>) ... ;;` block."""
    m = re.search(rf'(?m)^\s*{re.escape(arm)}\)\s*\n(.*?)\n\s*;;', code,
                  re.DOTALL)
    assert m is not None, f"no case body for arm {arm!r}"
    return m.group(1)


def array_tokens(body: str, name: str) -> list[str]:
    """Flatten a `NAME=(…)` bash array into its whitespace-separated tokens.

    Line continuations are joined first, so a multi-line array reads the
    same as a single-line one.
    """
    m = re.search(rf'{name}=\((.*?)\)', body.replace("\\\n", " "), re.DOTALL)
    return m.group(1).split() if m else []


def flags_of(code: str, arm: str) -> list[str]:
    body = extract_arm_case_body(code, arm)
    return array_tokens(body, "LOSS_ARGS") + array_tokens(body, "EXTRA_ARGS")


def trainer_block(code: str) -> str:
    m = re.search(r'(CUDA_VISIBLE_DEVICES=.*?>>"\$tlog" 2>&1)', code,
                  re.DOTALL)
    assert m is not None, "no trainer invocation found in launcher"
    return m.group(1)


def test_launcher_exists(new_source: str):
    assert new_source, "run_arm.sh is empty"
    assert 'ARM="${1:?' in new_source, (
        "run_arm.sh must take <arm> as its first positional argument.")


@pytest.mark.parametrize("arm", ARMS)
def test_arm_case_block_sets_required_vars(new_code: str, arm: str):
    body = extract_arm_case_body(new_code, arm)
    for var in ("NAME=", "ARM_DESC=", "LOSS_ARGS="):
        assert var in body, (
            f"arm {arm}: case block must set {var} — got:\n{body}")


@pytest.mark.parametrize("arm", ARMS)
def test_flags_are_the_379_command_line_plus_the_new_target(
        new_code: str, old_code: str, arm: str):
    """The issue's contract, checked mechanically against #379's launcher."""
    want = sorted(flags_of(old_code, arm) + ADDED_FLAG)
    got = sorted(flags_of(new_code, arm))
    assert got == want, (
        f"arm {arm}: flags must be #379's plus {' '.join(ADDED_FLAG)}.\n"
        f"  #379: {' '.join(sorted(flags_of(old_code, arm)))}\n"
        f"  #390: {' '.join(got)}")


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_targets_the_teacher(new_code: str, arm: str):
    body = extract_arm_case_body(new_code, arm)
    assert "--align-target teacher" in body.replace("\\\n", " "), (
        f"arm {arm}: the whole point of #390 is `--align-target teacher`.")


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_keeps_the_align_term(new_code: str, arm: str):
    """`--align-target` only means something with L_align switched on —
    train.py rejects the pair, and these are the cells that carry it."""
    assert "--align-loss-weight 1.0" in extract_arm_case_body(new_code, arm)


@pytest.mark.parametrize("arm", ARMS)
def test_checkpoint_names_cannot_collide_with_379(new_code: str,
                                                  old_code: str, arm: str):
    """Same arm, two experiments, one shared `runs/` layout — the NAME must
    differ or a rerun would overwrite #379's artefacts."""
    def name_of(code):
        m = re.search(r'NAME="([^"]+)"',
                      extract_arm_case_body(code, arm))
        assert m is not None, f"arm {arm}: no NAME= in case block"
        return m.group(1)
    assert name_of(new_code) != name_of(old_code)
    assert name_of(new_code).endswith("_alignteacher"), (
        f"arm {arm}: NAME must carry the `_alignteacher` marker.")


@pytest.mark.parametrize("arm", COPIED_ARMS)
def test_cells_without_an_align_term_are_not_reachable(new_code: str,
                                                       arm: str):
    """#379's numbers stand for those 20 cells; retraining them would spend
    compute on an unchanged objective."""
    assert not re.search(rf'(?m)^\s*{re.escape(arm)}\)\s*$', new_code), (
        f"{arm} has no L_align term and must not be a #390 arm.")


def test_trainer_invocation_is_identical_to_379(new_code: str, old_code: str):
    """Everything outside the per-arm case block — architecture, batch size,
    seed, save cadence, dataset, dtypes — must be #379's, verbatim."""
    assert trainer_block(new_code) == trainer_block(old_code)


def test_runs_in_its_own_experiment_directory(new_code: str):
    assert 'OUT="$WT/experiments/2026-08-01_lalign_teacher"' in new_code, (
        "run_arm.sh must write its runs/ and results/ under this "
        "experiment's own directory, never #379's.")


def test_launcher_is_backbone_only(new_code: str):
    forbidden = (
        "QTRAIN=", "QEVAL=", "train_head_cell", "eval_cell", "downstream_hl",
        "--head-nhead", "--quantile-head", "--head-arch", "--forecast-len",
        "gift_eval_full_", "BB_STEPS_K", "QEVAL_EXTRA_ARGS", "GPU_2L",
        "GPU_6L", "HEAD_STEPS", "HEAD_WARMUP",
    )
    for token in forbidden:
        assert token not in new_code, (
            f"run_arm.sh must be backbone-only — found {token!r}")


def test_wave_support_matches_379(new_code: str):
    """Stage 1 → 2 → 3 of the issue is #379's staged-wave protocol: a wave
    trains to TARGET_STEPS, only the final wave writes `_FINAL.pth`, and the
    next wave resumes from the newest `_<N>k.pth`."""
    assert 'TARGET_STEPS="${TARGET_STEPS:-${STEPS:-200000}}"' in new_code
    assert 'FINAL_STEPS="${FINAL_STEPS:-200000}"' in new_code
    assert 'STEPS="$TARGET_STEPS"' in new_code
    assert "WAVE SKIP" in new_code
    assert 'RESUME="--resume $latest"' in new_code


def test_wt_under_tmp_is_rejected(new_code: str):
    assert re.search(r'WT="\$\{WT:-\$HOME/', new_code), (
        "run_arm.sh should default WT under $HOME (a persistent checkout), "
        "never /tmp.")
    assert "/tmp/*|/tmp" in new_code, (
        "run_arm.sh must reject WT under /tmp with a loud ABORT.")
