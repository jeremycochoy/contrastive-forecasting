"""The student control's launcher, pinned against #379's and #390's.

Review item 3: every teacher-vs-student delta in the report crosses a code
boundary, because the teacher numbers are from this branch and the student
numbers are #379's, measured on older code and never re-run. The control
re-runs one arm here with L_align pointed back at the student.

That only settles anything if the control's command line is #379's arm5
line and nothing else. `run_arm_student.sh` does not restate it — it derives
the launcher from `run_arm.sh` by three substitutions and execs the result.
These tests run the same transformation and check the product against both
launchers, so "one flag apart" is a checked claim rather than a comment.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests.test_390_launcher_shape import (
    array_tokens, extract_arm_case_body, flags_of, strip_comments,
    trainer_block,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_390 = REPO_ROOT / "experiments" / "2026-08-01_lalign_teacher"
STUDENT_SH = EXP_390 / "scripts" / "run_arm_student.sh"
CONTROL_SH = EXP_390 / "scripts" / "run_student_control.sh"
LAUNCHER_390 = EXP_390 / "scripts" / "run_arm.sh"
LAUNCHER_379 = (REPO_ROOT / "experiments" / "2026-07-21_split_pred_rep_small"
                / "scripts" / "run_arm.sh")
ARM = "arm5"


def generate(tmp_path: Path) -> str:
    """Run run_arm_student.sh's own sed, and return what it would exec."""
    gen = tmp_path / "generated.sh"
    subprocess.run(
        ["bash", "-c",
         "sed -e 's/align-target teacher/align-target student/g' "
         "    -e 's/alignteacher/alignstudent/g' "
         r"    -e 's/dl_\${ARM}\.log/dl_\${ARM}_student.log/g' "
         f'"{LAUNCHER_390}" > "{gen}"'],
        check=True)
    return gen.read_text()


@pytest.fixture(scope="module")
def gen_code(tmp_path_factory) -> str:
    return strip_comments(generate(tmp_path_factory.mktemp("gen")))


@pytest.fixture(scope="module")
def old_code() -> str:
    return strip_comments(LAUNCHER_379.read_text())


def test_substitutions_in_the_script_match_the_ones_tested():
    """These tests re-run the transformation, so they are only evidence if
    it is the same one. Compare the sed expressions character for
    character."""
    src = STUDENT_SH.read_text()
    for expr in ("s/align-target teacher/align-target student/g",
                 "s/alignteacher/alignstudent/g",
                 r"s/dl_\${ARM}\.log/dl_\${ARM}_student.log/g"):
        assert expr in src, f"run_arm_student.sh no longer applies {expr}"


def test_flags_are_the_379_command_line_plus_the_student_target(gen_code,
                                                                old_code):
    """#379's arm5 ran with no --align-target flag at all, which resolves to
    the student. The control names it explicitly; that is the only token it
    may add."""
    want = sorted(flags_of(old_code, ARM) + ["--align-target", "student"])
    assert sorted(flags_of(gen_code, ARM)) == want


def test_no_teacher_token_survives(gen_code):
    body = extract_arm_case_body(gen_code, ARM).replace("\\\n", " ")
    assert "--align-target student" in body
    assert "teacher" not in body, (
        "a teacher token left in the control's case block means it is not "
        "the control it claims to be")


def test_name_cannot_collide_with_either_experiment(gen_code, old_code):
    """Three runs of arm5 now share one runs/ layout: #379's, #390's
    teacher, and this control. A shared NAME would overwrite artefacts."""
    def name_of(code):
        m = re.search(r'NAME="([^"]+)"', extract_arm_case_body(code, ARM))
        assert m is not None
        return m.group(1)
    new = strip_comments(LAUNCHER_390.read_text())
    names = {name_of(gen_code), name_of(old_code), name_of(new)}
    assert len(names) == 3, f"run names collide: {names}"
    assert name_of(gen_code).endswith("_alignstudent")


def test_trainer_invocation_is_untouched(gen_code):
    """Architecture, batch size, seed, save cadence, dataset, dtypes — the
    control differs from the teacher arms inside the case block only."""
    assert trainer_block(gen_code) == trainer_block(
        strip_comments(LAUNCHER_390.read_text()))


def test_backbone_seed_is_the_same_as_every_other_arm(gen_code):
    assert "SEED=20260520" in gen_code


@pytest.mark.parametrize("arm", ["arm5_nse", "arm6_v2", "arm6_v2_combab"])
def test_every_arm_transforms_the_same_way(gen_code, old_code, arm):
    """The control runs arm5, but the transformation must not be arm5-
    shaped — a later control on another arm has to be one env var away."""
    want = sorted(flags_of(old_code, arm) + ["--align-target", "student"])
    assert sorted(flags_of(gen_code, arm)) == want


def test_script_aborts_when_a_substitution_matches_nothing(tmp_path):
    """A silently-empty substitution would train the teacher arm under a
    student name — the worst outcome available, since nothing downstream
    could tell."""
    fake_scripts = tmp_path / "scripts"
    fake_scripts.mkdir()
    (fake_scripts / "run_arm.sh").write_text(
        '#!/bin/bash\nNAME="bb_small_arm5_nothing_to_substitute"\n')
    (fake_scripts / "run_arm_student.sh").write_text(STUDENT_SH.read_text())
    res = subprocess.run(
        ["bash", str(fake_scripts / "run_arm_student.sh"), ARM],
        capture_output=True, text=True, timeout=60)
    assert res.returncode != 0, (
        "the generator accepted a launcher it changed nothing in")
    assert "ABORT" in res.stderr


def test_control_driver_runs_backbone_then_eval_with_the_student_names():
    """The two stages have to agree on which checkpoints they mean: the
    backbone writes `_alignstudent`, so the eval must resolve the same
    suffix and write to its own cell directory."""
    code = strip_comments(CONTROL_SH.read_text())
    assert "run_arm_student.sh" in code
    assert "eval_arm.sh" in code
    assert "CF390_NAME_SUFFIX=alignstudent" in code
    assert "CELL_TAG=_alignstudent" in code
    assert "TARGET_STEPS=$(( BB_STEP_K * 1000 ))" in code
    assert "SAVE_EVERY=10000" in code, (
        "wave 1's save cadence — the control has to match the wave it is "
        "the control for")


def test_control_uses_the_wave_1_head_budget():
    code = strip_comments(CONTROL_SH.read_text())
    assert 'HEAD_STEPS="${HEAD_STEPS:-15000}"' in code
    assert 'BB_STEP_K="${BB_STEP_K:-40}"' in code


def test_arm_names_default_suffix_is_unchanged():
    """Every existing caller of bb_name must still resolve the teacher
    names; the control only overrides it in its own environment."""
    got = subprocess.run(
        ["bash", "-c",
         f'source "{EXP_390 / "scripts" / "arm_names.sh"}"; bb_name arm5'],
        capture_output=True, text=True, check=True).stdout.strip()
    assert got.endswith("_alignteacher")
    got_student = subprocess.run(
        ["bash", "-c",
         f'CF390_NAME_SUFFIX=alignstudent; '
         f'source "{EXP_390 / "scripts" / "arm_names.sh"}"; bb_name arm5'],
        capture_output=True, text=True, check=True).stdout.strip()
    assert got_student.endswith("_alignstudent")
    assert got[: -len("alignteacher")] == got_student[: -len("alignstudent")]
