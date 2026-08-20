"""#407 — A4's continuation to one full pass over ``small_v1``.

#373 stopped A4 at 200,000 steps, which is 30% of the training data. The
card gives the same run more steps — 300,000, 450,000 and 665,000 — with a
quantile head and a GIFT-Eval at each stop. Nothing else changes.

"Nothing else changes" is the whole contract, and a driver can break it
without an error. So the tests here cover four risks, and each one is a
failure that still produces a plausible curve.

  * The continuation must resume THIS checkpoint. The card pins two md5
    sums. A driver that resumes an earlier leg, or starts at step 0, gives
    a number for every stop.
  * The recipe must live in one place. Every training flag comes from
    #373's ``run_leg_k.sh``. A copy of the flags in this study's driver is
    a second place for them to drift.
  * 665,000 is not a multiple of ``--save-every``, so the last stop needs
    ``--extra-save-steps``. Without it the leg trains for 18 hours and
    writes no checkpoint at the stop.
  * The figure's grey rule is #373's 1.0660. Typed a second time, it is a
    number that no longer tracks the file it came from.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDY = REPO_ROOT / "reports" / "2026-08-20_a4_full_pass"
SCRIPTS = STUDY / "scripts"
FULL_PASS_PY = SCRIPTS / "full_pass.py"
RUN_PASS_SH = SCRIPTS / "run_pass.sh"
COLLECT_SH = SCRIPTS / "collect.sh"
PLOT_PY = SCRIPTS / "plot_full_pass.py"

PARENT = REPO_ROOT / "reports" / "2026-08-08_rollout_depth"
RUN_LEG_K = PARENT / "scripts" / "run_leg_k.sh"
STOP_K = PARENT / "scripts" / "stop_k.sh"
EVAL_LOCAL = PARENT / "scripts" / "eval_local.sh"
PARENT_RESULTS = PARENT / "results"

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")

# Confirmed from small_v1/manifest.json on the HF repo (4274 shards). The
# same constant guards #393's ladder in tests/test_393_ladder.py.
SMALL_V1_ROWS = 42_571_692


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def strip_comments(text: str) -> str:
    """Remove full-line bash comments so a token search sees only code."""
    return "\n".join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith("#"))


@pytest.fixture(scope="module")
def fp():
    return load(FULL_PASS_PY, "cf407_full_pass")


@pytest.fixture(scope="module")
def driver_code() -> str:
    return strip_comments(RUN_PASS_SH.read_text())


# --- 1. the card's stops ---------------------------------------------------


class TestStops:

    def test_the_three_stops(self, fp):
        assert fp.STOPS == [300_000, 450_000, 665_000]

    def test_the_run_continues_and_does_not_restart(self, fp):
        """Every stop is above the checkpoint the card names."""
        assert fp.RESUME_STEP == 200_000
        assert min(fp.STOPS) > fp.RESUME_STEP

    def test_the_parent_stops_are_373s(self, fp):
        assert fp.PARENT_STOPS == [40_000, 100_000, 200_000]

    def test_the_last_stop_is_one_pass(self, fp):
        """665,000 x 64 covers small_v1 to better than 0.1%."""
        rows = fp.STOPS[-1] * fp.BATCH_SIZE
        assert 0.999 <= rows / SMALL_V1_ROWS <= 1.0

    def test_one_pass_arithmetic(self, fp):
        assert fp.steps_for_one_pass(SMALL_V1_ROWS, 64) == 665_182
        assert fp.steps_for_one_pass(640, 64) == 10


# --- 2. the checkpoint the continuation resumes ----------------------------


class TestResumeCheckpoint:

    def test_the_card_pins_both_files(self, fp):
        assert fp.RESUME_MD5 == {
            "cf393_arm6_v2_combab_alignS_cf373k3_r2_200k.pth":
                "f477c03525bf5e169704715511f1c6d7",
            "cf393_arm6_v2_combab_alignS_cf373k3_r2_200k_optimizer.pth":
                "740891276637ff7bce744b1d9109d57a",
        }

    def test_the_optimizer_state_comes_too(self, fp):
        """A resume without it loses the step counter and AdamW momentum."""
        backbone = [n for n in fp.RESUME_MD5
                    if not n.endswith("_optimizer.pth")]
        assert len(backbone) == 1
        sidecar = backbone[0].replace(".pth", "_optimizer.pth")
        assert sidecar in fp.RESUME_MD5

    def test_check_resume_accepts_a_matching_pair(self, fp, tmp_path):
        leg = tmp_path / fp.CELL / f"leg_{fp.RESUME_STEP // 1000}k"
        leg.mkdir(parents=True)
        digests = {}
        for i, name in enumerate(sorted(fp.RESUME_MD5)):
            body = f"payload {i}".encode()
            (leg / name).write_bytes(body)
            digests[name] = hashlib.md5(body).hexdigest()
        assert fp.check_resume(tmp_path, expect=digests) == []

    def test_check_resume_reports_a_wrong_digest(self, fp, tmp_path):
        leg = tmp_path / fp.CELL / f"leg_{fp.RESUME_STEP // 1000}k"
        leg.mkdir(parents=True)
        for name in fp.RESUME_MD5:
            (leg / name).write_bytes(b"wrong")
        problems = fp.check_resume(tmp_path)
        assert len(problems) == 2
        assert all("md5" in p for p in problems)

    def test_check_resume_reports_a_missing_file(self, fp, tmp_path):
        problems = fp.check_resume(tmp_path)
        assert len(problems) == 2
        assert all("missing" in p for p in problems)

    def test_the_driver_runs_the_check_before_it_trains(self, driver_code):
        assert "--check-resume" in driver_code

    @pytest.mark.skipif(not Path("/home/jupyter/cf373_r3/sync").is_dir(),
                        reason="the round-3 root is on elisa only")
    def test_the_checkpoint_on_this_machine_matches_the_card(self, fp):
        assert fp.check_resume("/home/jupyter/cf373_r3/sync") == []


# --- 3. the recipe stays in #373's launcher --------------------------------


# Flags that decide what the backbone learns. Every one of them belongs to
# `run_leg_k.sh`. This study's driver must name none of them.
RECIPE_FLAGS = ("--loss-shape", "--align-loss-weight", "--align-target",
                "--moco-rep-keys", "--tau-rep", "--cpc-infonce-weight",
                "--train-rollout-depth", "--ema-tau", "--batch-size",
                "--lr", "--seed", "--hf-path", "--total-steps")


class TestOneRecipe:

    def test_the_driver_calls_373s_launcher(self, driver_code):
        assert "run_leg_k.sh" in driver_code

    def test_the_driver_calls_373s_stop_script(self, driver_code):
        assert "stop_k.sh" in driver_code

    @pytest.mark.parametrize("flag", RECIPE_FLAGS)
    def test_the_driver_repeats_no_training_flag(self, driver_code, flag):
        assert flag not in driver_code

    def test_the_driver_names_the_cell_the_card_names(self, driver_code):
        assert "arm6_v2_combab_alignS" in driver_code

    def test_the_a4_case_block_is_unchanged(self):
        """Non-regression: A4's four loss flags, in #373's launcher."""
        code = strip_comments(RUN_LEG_K.read_text())
        m = re.search(r'(?m)^\s*arm6_v2_combab_alignS\)\s*\n(.*?)\n\s*;;',
                      code, re.DOTALL)
        assert m, "no arm6_v2_combab_alignS case block in run_leg_k.sh"
        body = " ".join(m.group(1).split())
        assert "--loss-shape cosine_similarity_batch_rep_only" in body
        assert "--align-loss-weight 1.0" in body
        assert "--moco-rep-keys" in body
        assert "--tau-rep 1.0" in body
        assert "--cpc-infonce-weight 0.0" in body
        assert "--align-target student" in body

    def test_the_shared_block_still_carries_k_and_the_seed(self):
        code = strip_comments(RUN_LEG_K.read_text())
        assert '--train-rollout-depth "$K"' in code
        assert 'K="${K:-3}"' in code
        assert 'SEED="${SEED:-20260520}"' in code
        assert "--batch-size 64" in code

    def test_the_driver_holds_k_at_three(self, fp):
        assert fp.K == 3

    def test_the_driver_and_the_module_agree(self, fp, driver_code):
        """Bash cannot import the module, so the two are checked instead."""
        assert f'CELL="{fp.CELL}"' in driver_code
        assert f'CELL_ID="{fp.CELL_ID}"' in driver_code
        assert f"DEPTH={fp.K}" in driver_code
        assert f"HEAD_STEPS={fp.HEAD_STEPS}" in driver_code
        assert f"HEAD_SEED={fp.HEAD_SEED}" in driver_code
        default = " ".join(str(s) for s in fp.STOPS)
        assert f"STOPS=({default})" in driver_code


# --- 4. 665,000 is off the save cadence ------------------------------------


@pytest.fixture(scope="module")
def train_mod():
    """train.py's own save-cadence functions, without running its main()."""
    spec = importlib.util.spec_from_file_location("cf407_train", TRAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


class TestSaveCadence:

    def test_the_launcher_defaults_extra_saves_to_the_target(self):
        code = strip_comments(RUN_LEG_K.read_text())
        assert 'EXTRA_SAVES="${EXTRA_SAVES:-$TARGET_STEPS}"' in code
        assert 'SAVE_EVERY="${SAVE_EVERY:-20000}"' in code

    def test_the_driver_does_not_override_extra_saves(self, driver_code):
        assert "EXTRA_SAVES" not in driver_code

    @pytest.mark.parametrize("stop", [300_000, 450_000, 665_000])
    def test_every_stop_writes_its_checkpoint(self, train_mod, fp, stop):
        extras = train_mod.parse_extra_save_steps(str(stop))
        assert train_mod.should_snapshot(stop, 20_000, extras)

    def test_665k_would_be_lost_without_the_extra_save(self, train_mod):
        """The reason the default matters: 665,000 is not on the grid."""
        assert not train_mod.should_snapshot(665_000, 20_000, set())

    @pytest.mark.parametrize("stop", [300_000, 450_000, 665_000])
    def test_the_checkpoint_name_the_stop_produces(self, fp, stop):
        assert fp.ckpt_name(stop).endswith(f"_{stop // 1000}k.pth")


# --- 5. the head and the eval protocol -------------------------------------


class TestHeadProtocol:

    def test_the_card_s_head_budget_and_seed(self, fp):
        assert fp.HEAD_STEPS == 30_000
        assert fp.HEAD_SEED == 20260722

    def test_both_encoders(self, fp):
        assert fp.HEADS == ["student", "teacher"]

    def test_the_driver_pins_the_budget_and_the_seed(self, driver_code):
        assert "HEAD_STEPS=30000" in driver_code
        assert "HEAD_SEED=20260722" in driver_code

    def test_the_driver_trains_both_heads(self, driver_code):
        assert "student" in driver_code and "teacher" in driver_code

    def test_the_stop_script_still_clips_the_head(self):
        """The card asks for --grad-clip 1.0 on the head, and only there."""
        code = strip_comments(STOP_K.read_text())
        assert "--quantile-head --grad-clip 1.0" in code

    def test_the_backbone_is_never_clipped(self):
        """CLAUDE.md: never grad-clip this project's backbone."""
        assert "--grad-clip" not in strip_comments(RUN_LEG_K.read_text())


class TestEvalProtocol:

    def test_the_official_b4_strategy_at_horizon_16(self):
        code = strip_comments(EVAL_LOCAL.read_text())
        assert "--strategy B4 --forecast-len 16" in code

    def test_the_merge_insists_on_97_configs(self):
        code = strip_comments(EVAL_LOCAL.read_text())
        assert '"$n_rows" -ne 97' in code and '"$n_uniq" -ne 97' in code

    def test_collect_keeps_only_complete_evals(self):
        code = strip_comments(COLLECT_SH.read_text())
        assert "97" in code


# --- 6. the deliverable figure ---------------------------------------------


class TestFigureData:

    def test_the_grey_rule_is_373s_committed_score(self, fp):
        """1.0660 is a file, not a number someone typed twice."""
        published = float(
            (PARENT_RESULTS / "score_A4_k3_bb200k_student.txt").read_text())
        assert fp.BEST_BEFORE == published

    def test_tags_match_the_score_files_373_wrote(self, fp):
        assert fp.tag(200_000, "student") == "A4_k3_bb200k_student"
        assert fp.tag(665_000, "teacher") == "A4_k3_bb665k_teacher"

    def test_the_parent_points_come_off_disk(self, fp):
        got = fp.curve("student", results=None, parent=PARENT_RESULTS)
        assert got[40_000] == 1.0862
        assert got[100_000] == 1.0801
        assert got[200_000] == 1.0660

    def test_both_heads_have_all_three_parent_points(self, fp):
        for head in fp.HEADS:
            got = fp.curve(head, results=None, parent=PARENT_RESULTS)
            assert sorted(got) == fp.PARENT_STOPS

    def test_this_study_extends_the_same_curve(self, fp, tmp_path):
        (tmp_path / "score_A4_k3_bb300k_student.txt").write_text("1.0500\n")
        got = fp.curve("student", results=tmp_path, parent=PARENT_RESULTS)
        assert sorted(got) == [40_000, 100_000, 200_000, 300_000]
        assert got[300_000] == 1.05

    def test_a_stop_with_no_score_is_absent_not_zero(self, fp, tmp_path):
        got = fp.curve("teacher", results=tmp_path, parent=PARENT_RESULTS)
        assert 300_000 not in got

    def test_an_unreadable_score_file_is_absent(self, fp, tmp_path):
        (tmp_path / "score_A4_k3_bb300k_student.txt").write_text("")
        got = fp.curve("student", results=tmp_path, parent=PARENT_RESULTS)
        assert 300_000 not in got


# --- 7. the driver, end to end, with the two child scripts stubbed --------


LEG_STUB = """#!/bin/bash
echo "leg cell=$1 target=$2 RUNS=${RUNS:-} BB_GPU=${BB_GPU:-}" >>"$CF407_LOG"
mkdir -p "$RUNS/$1/leg_$(( $2 / 1000 ))k"
"""

STOP_STUB = """#!/bin/bash
echo "stop cell=$1 k=$2 target=$3 head=$4 ROOT=${CF373_ROOT:-}\
 HEAD_STEPS=${HEAD_STEPS:-} HEAD_SEED=${HEAD_SEED:-} GPU=${BB_GPU:-}" \\
  >>"$CF407_LOG"
n=$(grep -c "head=$4 " "$CF407_LOG")
[ -n "${CF407_FAIL_ONCE:-}" ] && [ "$n" -eq 1 ] && exit 9
exit 0
"""

REAL_ROOT = Path("/home/jupyter/cf373_r3/sync")
needs_checkpoint = pytest.mark.skipif(
    not (REAL_ROOT / "arm6_v2_combab_alignS" / "leg_200k").is_dir(),
    reason="the checkpoint the card pins is on elisa only")


def stub_checkout(tmp_path: Path, leg_body: str = LEG_STUB) -> Path:
    """A checkout whose #373 launcher and stop script only take notes."""
    scripts = tmp_path / "wt" / "reports" / "2026-08-08_rollout_depth" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name, body in (("run_leg_k.sh", leg_body), ("stop_k.sh", STOP_STUB)):
        (scripts / name).write_text(body)
        (scripts / name).chmod(0o755)
    return tmp_path / "wt"


def linked_root(tmp_path: Path) -> Path:
    """The card's checkpoint pair, linked rather than copied (5 MB each)."""
    leg = tmp_path / "runs" / "arm6_v2_combab_alignS" / "leg_200k"
    leg.mkdir(parents=True)
    src = REAL_ROOT / "arm6_v2_combab_alignS" / "leg_200k"
    for name in ("cf393_arm6_v2_combab_alignS_cf373k3_r2_200k.pth",
                 "cf393_arm6_v2_combab_alignS_cf373k3_r2_200k_optimizer.pth"):
        (leg / name).symlink_to(src / name)
    return tmp_path / "runs"


def drive(tmp_path, runs, extra_env=None, stops=(), leg_body=LEG_STUB):
    env = os.environ.copy()
    env.update({
        "WT": str(stub_checkout(tmp_path, leg_body)),
        "RUNS": str(runs),
        "CF407_RESULTS": str(tmp_path / "res"),
        "CF407_LOG": str(tmp_path / "calls.log"),
        # The gate would otherwise wait a day for a card another session
        # filled, which is what it is for outside a test.
        "BB_VRAM_MIB": "0",
        "BB_GPU": "1",
    })
    env.update(extra_env or {})
    proc = subprocess.run(["bash", str(RUN_PASS_SH), *[str(s) for s in stops]],
                          capture_output=True, text=True, env=env, timeout=300)
    log = tmp_path / "calls.log"
    return proc, (log.read_text().splitlines() if log.exists() else [])


class TestDriver:

    def test_it_refuses_a_root_that_is_not_the_cards(self, tmp_path):
        """No checkpoint, no training. The gate runs before the first leg."""
        proc, calls = drive(tmp_path, tmp_path / "empty")
        assert proc.returncode == 3
        assert calls == []

    @needs_checkpoint
    def test_it_walks_the_three_stops_in_order(self, tmp_path):
        proc, calls = drive(tmp_path, linked_root(tmp_path))
        assert proc.returncode == 0, proc.stderr[-2000:]
        legs = [ln for ln in calls if ln.startswith("leg ")]
        assert [ln.split("target=")[1].split()[0] for ln in legs] == \
            ["300000", "450000", "665000"]

    @needs_checkpoint
    def test_each_leg_is_followed_by_both_heads(self, tmp_path):
        _, calls = drive(tmp_path, linked_root(tmp_path))
        shape = [ln.split()[0] + ":" + ln.split("head=")[1].split()[0]
                 if ln.startswith("stop ") else "leg" for ln in calls]
        assert shape == ["leg", "stop:student", "stop:teacher"] * 3

    @needs_checkpoint
    def test_it_hands_the_launcher_the_cell_and_the_target(self, tmp_path):
        _, calls = drive(tmp_path, linked_root(tmp_path), stops=(300_000,))
        leg = [ln for ln in calls if ln.startswith("leg ")][0]
        assert "cell=arm6_v2_combab_alignS" in leg
        assert "target=300000" in leg

    @needs_checkpoint
    def test_the_two_scripts_read_one_root(self, tmp_path):
        """`run_leg_k.sh` takes RUNS, `stop_k.sh` takes CF373_ROOT."""
        runs = linked_root(tmp_path)
        _, calls = drive(tmp_path, runs, stops=(300_000,))
        assert f"RUNS={runs}" in calls[0]
        assert f"ROOT={runs}" in calls[1]

    @needs_checkpoint
    def test_it_pins_the_cards_head_protocol(self, tmp_path):
        _, calls = drive(tmp_path, linked_root(tmp_path), stops=(450_000,))
        for line in [ln for ln in calls if ln.startswith("stop ")]:
            assert "cell=A4 k=3 target=450000" in line
            assert "HEAD_STEPS=30000" in line
            assert "HEAD_SEED=20260722" in line

    @needs_checkpoint
    def test_a_failed_head_is_retried_once(self, tmp_path):
        """A transient must not cost 30,000 GPU steps that already ran."""
        _, calls = drive(tmp_path, linked_root(tmp_path), stops=(300_000,),
                         extra_env={"CF407_FAIL_ONCE": "1"})
        heads = [ln for ln in calls if ln.startswith("stop ")]
        assert len(heads) == 4          # two heads, each attempted twice

    @needs_checkpoint
    def test_a_failed_leg_stops_the_driver(self, tmp_path):
        """The 450k leg resumes the 300k checkpoint. Never skip a leg."""
        proc, calls = drive(
            tmp_path, linked_root(tmp_path),
            leg_body='#!/bin/bash\necho "leg $2" >>"$CF407_LOG"\nexit 1\n')
        assert proc.returncode == 1
        assert len([ln for ln in calls if ln.startswith("leg ")]) == 1


class TestPlot:

    def test_it_writes_a_figure(self, tmp_path):
        out = tmp_path / "full_pass.png"
        (tmp_path / "score_A4_k3_bb300k_student.txt").write_text("1.0500\n")
        (tmp_path / "score_A4_k3_bb300k_teacher.txt").write_text("1.0700\n")
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        proc = subprocess.run(
            [sys.executable, str(PLOT_PY), "--results", str(tmp_path),
             "--parent", str(PARENT_RESULTS), "--out", str(out)],
            capture_output=True, text=True, env=env, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert out.is_file() and out.stat().st_size > 10_000

    def test_it_draws_the_parent_points_alone(self, tmp_path):
        """Before the first new stop lands, the figure is still #373's."""
        out = tmp_path / "empty.png"
        proc = subprocess.run(
            [sys.executable, str(PLOT_PY), "--results", str(tmp_path),
             "--parent", str(PARENT_RESULTS), "--out", str(out)],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert out.is_file()

    def test_one_point_on_the_rule_is_not_a_zero_height_axis(self, tmp_path):
        """The degenerate case: the only value is the rule itself."""
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "score_A4_k3_bb200k_student.txt").write_text("1.0660\n")
        out = tmp_path / "one.png"
        proc = subprocess.run(
            [sys.executable, str(PLOT_PY), "--results", str(tmp_path),
             "--parent", str(parent), "--out", str(out)],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert out.is_file()
