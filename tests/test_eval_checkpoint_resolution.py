"""Eval-time checkpoint selection must never resolve an ambiguity silently.

A resumed run is renamed `<name>_r<N>` by `safe_run_name`, so one (run name,
step) pair can leave several backbones on disk: `NAME_40k.pth` from the first
attempt and `NAME_r3_40k.pth` from the resume. The eval scripts used to pick
between them with `ls -t … | head -1`, i.e. by modification time, and recorded
nothing about the choice. Two different backbones could then be evaluated
under the same cell name, and no output said which file produced the number.

`scripts/resolve_eval_checkpoint.sh` replaces that pick. It holds two
properties:

  1. more than one candidate aborts, listing every candidate, unless the
     caller names the file explicitly;
  2. the resolved path is always printed, so a published number can be traced
     back to the file it came from.

Selecting the *newest* checkpoint of a still-running job is a different job
(resume) and is not what this resolver is for.

Three call sites carry the pick. #379's `eval_2L_gm_mase.sh` is checked from
its source text below; #390's `eval_arm.sh` (eval selection) and `pipeline.sh`
(a presence check, which decides whether an arm reaches the eval stage at all)
are checked by running them.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESOLVER = REPO_ROOT / "scripts" / "resolve_eval_checkpoint.sh"
EVAL_SH = (REPO_ROOT / "experiments" / "2026-07-21_split_pred_rep_small" /
           "scripts" / "eval_2L_gm_mase.sh")
EXP_390 = REPO_ROOT / "experiments" / "2026-08-01_lalign_teacher"
EVAL_390 = EXP_390 / "scripts" / "eval_arm.sh"
PIPELINE_390 = EXP_390 / "scripts" / "pipeline.sh"

NAME = "bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"

# The resolver's exit codes. Tests assert the exact code so a missing or
# unrunnable script (127) can never pass as a deliberate abort.
EXIT_NO_CANDIDATE = 3
EXIT_BAD_OVERRIDE = 4
EXIT_AMBIGUOUS = 5


def touch(runs: Path, filename: str, mtime: int) -> Path:
    """Create an empty checkpoint file with an explicit modification time."""
    path = runs / filename
    path.write_bytes(b"")
    os.utime(path, (mtime, mtime))
    return path


def resolve(runs: Path, name: str, step_k: str, explicit: str | None = None):
    argv = ["bash", str(RESOLVER), str(runs), name, step_k]
    if explicit is not None:
        argv.append(explicit)
    return subprocess.run(argv, capture_output=True, text=True)


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture
def eval_code() -> str:
    return EVAL_SH.read_text()


@pytest.fixture
def eval_code_joined(eval_code: str) -> str:
    """`eval_code` with line continuations folded, so a wrapped command
    matches the same pattern as a one-liner."""
    return re.sub(r"\\\n\s*", " ", eval_code)


# --- 1. one candidate: unchanged behaviour --------------------------------

def test_single_base_candidate_resolves(runs: Path):
    """The common case — one run, never resumed — keeps working."""
    ckpt = touch(runs, f"{NAME}_40k.pth", 1_000)
    r = resolve(runs, NAME, "40")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(ckpt)


def test_single_resumed_candidate_resolves(runs: Path):
    """#379/#390 shape: several cells only ever had the `_r<N>` file.

    Those runs crashed and resumed before their first snapshot, so the base
    name never appears at that step. One candidate, no ambiguity — the
    resolver must return it exactly as the old `ls -t` line did.
    """
    ckpt = touch(runs, f"{NAME}_r3_40k.pth", 1_000)
    r = resolve(runs, NAME, "40")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(ckpt)


def test_optimizer_sidecar_is_not_a_candidate(runs: Path):
    """`_40k_optimizer.pth` sits beside every snapshot; it is not a backbone."""
    ckpt = touch(runs, f"{NAME}_40k.pth", 1_000)
    touch(runs, f"{NAME}_40k_optimizer.pth", 2_000)
    r = resolve(runs, NAME, "40")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(ckpt)


def test_other_steps_are_not_candidates(runs: Path):
    """Step 40 must not match step 400 or step 4."""
    touch(runs, f"{NAME}_400k.pth", 1_000)
    touch(runs, f"{NAME}_4k.pth", 1_000)
    r = resolve(runs, NAME, "40")
    assert r.returncode == EXIT_NO_CANDIDATE
    assert r.stdout.strip() == ""


def test_no_candidate_aborts(runs: Path):
    r = resolve(runs, NAME, "40")
    assert r.returncode == EXIT_NO_CANDIDATE
    assert r.stdout.strip() == ""
    assert "40" in r.stderr and NAME in r.stderr


# --- 2. two candidates: abort, and name them ------------------------------

def test_two_candidates_abort(runs: Path):
    """The mtime pick is gone: a base run and its resume no longer resolve."""
    touch(runs, f"{NAME}_40k.pth", 1_000)
    touch(runs, f"{NAME}_r3_40k.pth", 9_000)  # newest — the old `ls -t` winner
    r = resolve(runs, NAME, "40")
    assert r.returncode == EXIT_AMBIGUOUS, (
        "two backbones match one (name, step) pair; resolving that by "
        "modification time is what produced the #390 arm5 discrepancy")
    assert r.stdout.strip() == "", (
        "nothing may reach stdout on an abort — the caller captures stdout "
        "as the checkpoint path")


def test_abort_message_names_every_candidate(runs: Path):
    """The message has to be actionable: every file that matched, by path."""
    paths = [touch(runs, f"{NAME}_40k.pth", 1_000),
             touch(runs, f"{NAME}_r2_40k.pth", 5_000),
             touch(runs, f"{NAME}_r3_40k.pth", 9_000)]
    r = resolve(runs, NAME, "40")
    assert r.returncode == EXIT_AMBIGUOUS
    for p in paths:
        assert str(p) in r.stderr, f"abort message does not name {p.name}"


def test_abort_message_points_at_the_override(runs: Path):
    """A human reading the abort must learn how to name the file they mean."""
    touch(runs, f"{NAME}_40k.pth", 1_000)
    touch(runs, f"{NAME}_r3_40k.pth", 9_000)
    r = resolve(runs, NAME, "40")
    assert r.returncode == EXIT_AMBIGUOUS
    assert re.search(r"resolve_eval_checkpoint\.sh .*<path>|<explicit-path>",
                     r.stderr), (
        "the ambiguity abort must tell the caller how to name one candidate")


# --- 3. explicit override -------------------------------------------------

def test_explicit_override_selects_the_named_file(runs: Path):
    """The caller names the replicate; ambiguity stops being an error."""
    touch(runs, f"{NAME}_40k.pth", 9_000)
    wanted = touch(runs, f"{NAME}_r3_40k.pth", 1_000)  # older, not the mtime pick
    r = resolve(runs, NAME, "40", str(wanted))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(wanted)


def test_empty_override_is_no_override(runs: Path):
    """Call sites pass `$BB_CHECKPOINT` unquoted-empty when unset."""
    ckpt = touch(runs, f"{NAME}_40k.pth", 1_000)
    r = resolve(runs, NAME, "40", "")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(ckpt)


def test_missing_override_aborts(runs: Path):
    touch(runs, f"{NAME}_40k.pth", 1_000)
    r = resolve(runs, NAME, "40", str(runs / "typo.pth"))
    assert r.returncode == EXIT_BAD_OVERRIDE
    assert r.stdout.strip() == ""
    assert "typo.pth" in r.stderr


# --- 4. the resolved path is always recorded ------------------------------

def test_resolved_path_is_reported(runs: Path):
    ckpt = touch(runs, f"{NAME}_r3_40k.pth", 1_000)
    r = resolve(runs, NAME, "40")
    assert r.returncode == 0, r.stderr
    assert f"-> {ckpt}" in r.stderr, (
        "a successful resolution must report the file it chose, on stderr, so "
        "the choice lands in the caller's log next to the number it produced")


def test_resolved_path_is_reported_under_override(runs: Path):
    touch(runs, f"{NAME}_40k.pth", 9_000)
    wanted = touch(runs, f"{NAME}_r3_40k.pth", 1_000)
    r = resolve(runs, NAME, "40", str(wanted))
    assert r.returncode == 0, r.stderr
    assert f"-> {wanted}" in r.stderr
    assert "override" in r.stderr.lower(), (
        "an overridden pick must say so — it is not the default resolution")


# --- 5. the #379 eval script uses the resolver ----------------------------

def test_eval_script_has_no_mtime_pick(eval_code: str):
    assert not re.search(r"ls -t.*\.pth", eval_code), (
        "eval_2L_gm_mase.sh must not choose a backbone by modification time")


def test_eval_script_calls_the_resolver(eval_code_joined: str):
    assert re.search(
        r'"\$CKPT_RESOLVER"\s+"\$RUNS"\s+"\$NAME"\s+'
        r'"\$BB_STEP_K"\s+"\$BB_CHECKPOINT"', eval_code_joined), (
        "eval_2L_gm_mase.sh must resolve its backbone through "
        "scripts/resolve_eval_checkpoint.sh, passing the BB_CHECKPOINT "
        "override through")


def test_eval_script_takes_the_resolver_from_its_own_checkout(
        eval_code_joined: str):
    """Not from `$WT`.

    `$WT` is a training worktree that can sit on any commit. Loading the
    resolver from there would let a stale checkout fall back to the mtime
    pick, silently, which is the bug this file is about.
    """
    assert re.search(
        r'CKPT_RESOLVER="\$\(cd "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)'
        r'/\.\./\.\./\.\." && pwd\)/scripts/resolve_eval_checkpoint\.sh"',
        eval_code_joined), (
        "eval_2L_gm_mase.sh must load the resolver relative to its own path, "
        "not from $WT")


def test_eval_script_guards_a_missing_resolver(eval_code_joined: str):
    assert re.search(r'\[ -f "\$CKPT_RESOLVER" \]\s*\|\|.*exit',
                     eval_code_joined), (
        "eval_2L_gm_mase.sh must abort with a named path when the resolver is "
        "missing, not die on a bare `No such file or directory`")


def test_eval_script_exposes_the_override(eval_code: str):
    assert 'BB_CHECKPOINT="${BB_CHECKPOINT:-}"' in eval_code, (
        "eval_2L_gm_mase.sh must accept BB_CHECKPOINT=<path> so a human can "
        "name the replicate to evaluate")
    assert "BB_CHECKPOINT" in eval_code.split("Usage:")[1].split("\n#\n")[0], (
        "the BB_CHECKPOINT override must appear in the usage header")


def test_eval_script_aborts_when_resolution_fails(eval_code_joined: str):
    assert re.search(r'BB=\$\([^)]*\$CKPT_RESOLVER[^)]*\)\s*\|\|\s*exit',
                     eval_code_joined), (
        "eval_2L_gm_mase.sh must exit when the resolver aborts, not carry on "
        "with an empty backbone path")


def test_eval_script_logs_the_resolved_path(eval_code: str):
    assert "backbone=$BB)" in eval_code, (
        "eval_2L_gm_mase.sh must log the resolved checkpoint path (not just "
        "its basename) so a published cell traces back to one file")


# --- 6. #390's eval_arm.sh: the same site, the same resolver --------------
# `eval_arm.sh` is #379's eval script rebuilt for #390's ten arms. It carried
# the same `ls -t … | head -1` line and needs the same two properties.

@pytest.fixture
def eval_390_code() -> str:
    return EVAL_390.read_text()


@pytest.fixture
def eval_390_code_joined(eval_390_code: str) -> str:
    return re.sub(r"\\\n\s*", " ", eval_390_code)


def test_390_eval_arm_has_no_mtime_pick(eval_390_code: str):
    assert not re.search(r"ls -t.*\.pth", eval_390_code), (
        "eval_arm.sh must not choose a backbone by modification time")


def test_390_eval_arm_calls_the_resolver(eval_390_code_joined: str):
    assert re.search(
        r'"\$CKPT_RESOLVER"\s+"\$RUNS"\s+"\$NAME"\s+'
        r'"\$BB_STEP_K"\s+"\$BB_CHECKPOINT"', eval_390_code_joined), (
        "eval_arm.sh must resolve its backbone through "
        "scripts/resolve_eval_checkpoint.sh, passing the BB_CHECKPOINT "
        "override through")


def test_390_eval_arm_takes_the_resolver_from_its_own_checkout(
        eval_390_code: str):
    """From `$HERE`, the script's own directory, never from `$WT`.

    `$WT` is a training worktree that can sit on any commit. Loading the
    resolver from there would let a stale checkout fall back to the mtime
    pick, silently, which is the bug this file is about.
    """
    m = re.search(r"(?m)^CKPT_RESOLVER=.*$", eval_390_code)
    assert m, "eval_arm.sh does not define CKPT_RESOLVER"
    assert "$HERE" in m.group(0), (
        "the resolver path must be derived from $HERE (the script's own "
        f"directory); got: {m.group(0)}")
    assert "$WT" not in m.group(0), (
        f"the resolver must not be loaded from $WT; got: {m.group(0)}")


def test_390_eval_arm_guards_a_missing_resolver(eval_390_code_joined: str):
    assert re.search(r'\[ -f "\$CKPT_RESOLVER" \]\s*\|\|.*exit',
                     eval_390_code_joined), (
        "eval_arm.sh must abort with a named path when the resolver is "
        "missing, not die on a bare `No such file or directory`")


def test_390_eval_arm_aborts_when_resolution_fails(eval_390_code_joined: str):
    assert re.search(r'BB=\$\([^)]*\$CKPT_RESOLVER[^)]*\)\s*\|\|\s*exit',
                     eval_390_code_joined), (
        "eval_arm.sh must exit when the resolver aborts, not carry on with "
        "an empty backbone path")


def test_390_eval_arm_exposes_the_override(eval_390_code: str):
    assert 'BB_CHECKPOINT="${BB_CHECKPOINT:-}"' in eval_390_code, (
        "eval_arm.sh must accept BB_CHECKPOINT=<path> so a human can name "
        "the replicate to evaluate")
    assert "BB_CHECKPOINT" in eval_390_code.split("Usage:")[1].split("\n#\n")[0], (
        "the BB_CHECKPOINT override must appear in the usage header")


def test_390_eval_arm_logs_the_resolved_path(eval_390_code: str):
    assert "backbone=$BB)" in eval_390_code, (
        "eval_arm.sh must log the resolved checkpoint path (not just its "
        "basename) so a published cell traces back to one file")


# --- 7. #390's eval_arm.sh, run ------------------------------------------
# Source text says what the script contains; running it says what it does.
# Each case builds a stand-in `$WT` holding only what the script stats.

ARM = "arm5"
STEP_K = "40"
CELL_DIR = f"{ARM}_bb{STEP_K}k_hd15000s"

# A `python3` that records nothing and succeeds. Resolution happens before the
# first call, so every assertion below is reachable without a real trainer.
PY_NOOP = "#!/bin/bash\nexit 0\n"


@pytest.fixture
def scratch() -> Path:
    """A scratch root outside /tmp — eval_arm.sh refuses a WT under it."""
    cache = Path(os.path.expanduser("~/.cache"))
    cache.mkdir(parents=True, exist_ok=True)
    root = tempfile.mkdtemp(prefix="cf390-ckpt-", dir=cache)
    yield Path(root)
    shutil.rmtree(root, ignore_errors=True)


def bb_name_390(arm: str) -> str:
    """The arm's backbone run name, from the script the launchers source."""
    return subprocess.run(
        ["bash", "-c", f'source "{EXP_390 / "scripts" / "arm_names.sh"}"; '
                       f'bb_name {arm}'],
        capture_output=True, text=True, check=True).stdout.strip()


def cf390_arms() -> list[str]:
    """The experiment's ten arms, from the same source."""
    return subprocess.run(
        ["bash", "-c", f'source "{EXP_390 / "scripts" / "arm_names.sh"}"; '
                       f'echo "${{CF390_ARMS[*]}}"'],
        capture_output=True, text=True, check=True).stdout.split()


def make_wt(scratch: Path) -> Path:
    """A stand-in checkout: the four paths eval_arm.sh stats, and a runs dir.

    It deliberately holds no `scripts/resolve_eval_checkpoint.sh` — a $WT
    predating the resolver is exactly the case that must still resolve.
    """
    wt = scratch / "wt"
    gift = wt / "experiments" / "2026-04-13_gift-eval" / "scripts"
    gift.mkdir(parents=True)
    (gift / "train_forecasting_head.py").write_text("")
    (gift / "eval_gift_eval_official.py").write_text("")
    (wt / "experiments" / "hf_token.txt").write_text("hf_stub_token\n")
    (wt / "experiments" / "2026-08-01_lalign_teacher" / "runs").mkdir(
        parents=True)
    return wt


def stub_python3(scratch: Path) -> Path:
    d = scratch / "bin"
    d.mkdir(parents=True, exist_ok=True)
    stub = d / "python3"
    stub.write_text(PY_NOOP)
    stub.chmod(0o755)
    return d


def run_eval_arm(wt: Path, scratch: Path, bb_checkpoint: str | None = None):
    env = {**os.environ,
           "PATH": f"{stub_python3(scratch)}:{os.environ['PATH']}",
           "WT": str(wt), "ARM": ARM, "BB_GPU": "0",
           "BB_STEP_K": STEP_K, "HEAD_STEPS": "15000"}
    if bb_checkpoint is not None:
        env["BB_CHECKPOINT"] = bb_checkpoint
    return subprocess.run(["bash", str(EVAL_390)], env=env,
                          capture_output=True, text=True, timeout=180)


def eval_log(wt: Path) -> str:
    log = (wt / "experiments" / "2026-08-01_lalign_teacher" / "eval_gm_mase" /
           CELL_DIR / "eval.log")
    return log.read_text() if log.is_file() else ""


def test_390_eval_arm_aborts_on_two_candidates(scratch: Path):
    """A base run and its resume at the same step no longer resolve.

    The old line took the newest, so the cell name said `arm5` while the
    number came from whichever replicate happened to land last.
    """
    wt = make_wt(scratch)
    runs = wt / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    name = bb_name_390(ARM)
    base = touch(runs, f"{name}_{STEP_K}k.pth", 1_000)
    resumed = touch(runs, f"{name}_r3_{STEP_K}k.pth", 9_000)  # the `ls -t` pick

    r = run_eval_arm(wt, scratch)
    assert r.returncode == EXIT_AMBIGUOUS, (
        f"expected the resolver's ambiguity exit; got {r.returncode}\n"
        f"{r.stdout}\n{r.stderr}")
    assert str(base) in r.stderr and str(resumed) in r.stderr, (
        "the abort must name every candidate")
    assert not (wt / "experiments" / "2026-08-01_lalign_teacher" /
                "eval_gm_mase").exists(), (
        "an unresolved backbone must not open a cell directory")


def test_390_eval_arm_records_the_resolved_path_in_its_log(scratch: Path):
    """The unambiguous case: one candidate, and the log names the file."""
    wt = make_wt(scratch)
    runs = wt / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    ckpt = touch(runs, f"{bb_name_390(ARM)}_r3_{STEP_K}k.pth", 1_000)

    r = run_eval_arm(wt, scratch)
    assert r.returncode != EXIT_AMBIGUOUS, r.stderr
    assert f"backbone={ckpt}" in eval_log(wt), (
        "the cell's own log must carry the full path of the checkpoint that "
        f"produced its number; log was:\n{eval_log(wt)}")


def test_390_eval_arm_override_selects_the_named_replicate(scratch: Path):
    """BB_CHECKPOINT names the replicate, and the log says which one."""
    wt = make_wt(scratch)
    runs = wt / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    name = bb_name_390(ARM)
    touch(runs, f"{name}_{STEP_K}k.pth", 9_000)          # the mtime winner
    wanted = touch(runs, f"{name}_r3_{STEP_K}k.pth", 1_000)

    r = run_eval_arm(wt, scratch, bb_checkpoint=str(wanted))
    assert r.returncode != EXIT_AMBIGUOUS, r.stderr
    assert f"backbone={wanted}" in eval_log(wt), (
        "the named replicate must be the one evaluated, and be logged; log "
        f"was:\n{eval_log(wt)}")


def test_390_eval_arm_resolves_with_a_wt_that_has_no_resolver(scratch: Path):
    """The resolver is loaded from the script's checkout, not from `$WT`."""
    wt = make_wt(scratch)
    assert not (wt / "scripts" / "resolve_eval_checkpoint.sh").exists()
    runs = wt / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    ckpt = touch(runs, f"{bb_name_390(ARM)}_{STEP_K}k.pth", 1_000)

    r = run_eval_arm(wt, scratch)
    assert "no checkpoint resolver at" not in r.stderr, (
        f"eval_arm.sh looked for the resolver under $WT:\n{r.stderr}")
    assert f"backbone={ckpt}" in eval_log(wt), r.stderr


# --- 8. #390's pipeline.sh: the presence check ---------------------------
# `arms_at_step()` decides which arms reach the eval stage. It is not an eval
# selection, so it takes no override — but it must not pick between two
# candidates either, and it must record what it found.

@pytest.fixture
def pipeline_code() -> str:
    return PIPELINE_390.read_text()


@pytest.fixture
def pipeline_code_joined(pipeline_code: str) -> str:
    return re.sub(r"\\\n\s*", " ", pipeline_code)


def test_390_pipeline_has_no_mtime_pick(pipeline_code: str):
    assert not re.search(r"ls -t.*\.pth", pipeline_code), (
        "pipeline.sh must not decide an arm's readiness by modification time")


def test_390_pipeline_calls_the_resolver(pipeline_code_joined: str):
    assert re.search(r'"\$CKPT_RESOLVER"\s+"\$RUNS"\s+"\$name"\s+"\$k"',
                     pipeline_code_joined), (
        "pipeline.sh's presence check must go through "
        "scripts/resolve_eval_checkpoint.sh")


def test_390_pipeline_takes_the_resolver_from_its_own_checkout(
        pipeline_code: str):
    m = re.search(r"(?m)^CKPT_RESOLVER=.*$", pipeline_code)
    assert m, "pipeline.sh does not define CKPT_RESOLVER"
    assert "$HERE" in m.group(0), (
        f"the resolver path must be derived from $HERE; got: {m.group(0)}")
    assert "$WT" not in m.group(0), (
        f"the resolver must not be loaded from $WT; got: {m.group(0)}")


def test_390_pipeline_guards_a_missing_resolver(pipeline_code_joined: str):
    assert re.search(r'\[ -f "\$CKPT_RESOLVER" \]\s*\|\|.*exit',
                     pipeline_code_joined), (
        "pipeline.sh must abort with a named path when the resolver is "
        "missing, not treat every arm as absent")


# --- 9. #390's pipeline.sh, run ------------------------------------------
# One run of the real pipeline over a stand-in checkout: three arms with a
# 40k and a 100k snapshot each (one base-only, one resume-only, one of each,
# i.e. ambiguous), the other seven absent. The two stages the pipeline shells
# out to are stubbed, so the run exercises the presence check and nothing
# else.

READY_ARMS = ["arm5", "arm5_tr1"]
AMBIGUOUS_ARM = "arm5_nse"

STUB_STAGE = """#!/bin/bash
printf '%s\\n' "$ARMS" >> "$STUB_OUT"
exit 0
"""
STUB_GATE = "#!/usr/bin/env python3\n"  # prints nothing: wave 3 is empty


def make_pipeline_wt(scratch: Path) -> tuple[Path, Path]:
    wt = scratch / "wt"
    exp = wt / "experiments" / "2026-08-01_lalign_teacher"
    runs = exp / "runs"
    runs.mkdir(parents=True)
    (exp / "results").mkdir()
    scripts = exp / "scripts"
    scripts.mkdir()

    stub_out = scratch / "stage_arms.txt"
    for stage in ("orchestrate_pool.sh", "eval_wave.sh"):
        (scripts / stage).write_text(STUB_STAGE)
    (scripts / "select_wave3.py").write_text(STUB_GATE)

    for k in ("40", "100"):
        for arm in READY_ARMS + [AMBIGUOUS_ARM]:
            name = bb_name_390(arm)
            if arm == "arm5_tr1":
                touch(runs, f"{name}_r2_{k}k.pth", 1_000)   # resume only
            else:
                touch(runs, f"{name}_{k}k.pth", 1_000)
            if arm == AMBIGUOUS_ARM:
                touch(runs, f"{name}_r4_{k}k.pth", 9_000)   # the mtime winner
    return wt, stub_out


@pytest.fixture
def pipeline_run(scratch: Path):
    wt, stub_out = make_pipeline_wt(scratch)
    env = {**os.environ, "WT": str(wt), "STUB_OUT": str(stub_out)}
    res = subprocess.run(["bash", str(PIPELINE_390)], env=env,
                         capture_output=True, text=True, timeout=180)
    log = (wt / "experiments" / "2026-08-01_lalign_teacher" / "results" /
           "pipeline.log")
    handed = stub_out.read_text() if stub_out.is_file() else ""
    return res, log.read_text() if log.is_file() else "", handed, wt


def test_390_pipeline_records_the_resolved_path(pipeline_run):
    """An arm the pipeline calls ready must have its file named in the log."""
    _, log, _, wt = pipeline_run
    runs = wt / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    for arm, fname in ((READY_ARMS[0], f"{bb_name_390(READY_ARMS[0])}_40k.pth"),
                       (READY_ARMS[1],
                        f"{bb_name_390(READY_ARMS[1])}_r2_40k.pth")):
        assert str(runs / fname) in log, (
            f"pipeline.log does not record which file made {arm} ready:\n{log}")


def test_390_pipeline_drops_an_ambiguous_arm_naming_both_candidates(
        pipeline_run):
    """Two candidates: no pick, and both paths in the log."""
    _, log, _, wt = pipeline_run
    runs = wt / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    name = bb_name_390(AMBIGUOUS_ARM)
    for fname in (f"{name}_40k.pth", f"{name}_r4_40k.pth"):
        assert str(runs / fname) in log, (
            f"the ambiguity must name {fname} in pipeline.log:\n{log}")


def test_390_pipeline_hands_only_unambiguous_arms_downstream(pipeline_run):
    """The eval stage never receives an arm the pipeline could not resolve.

    The backbone stage gets every live arm; the eval stage gets what
    `arms_at_step` filtered. Those filtered lists are the ones under test.
    """
    _, _, handed, _ = pipeline_run
    calls = [line.split() for line in handed.splitlines() if line.strip()]
    assert calls, "no stage was reached; the pipeline stub run did nothing"
    filtered = [c for c in calls if len(c) < len(cf390_arms())]
    assert filtered, f"no stage was handed a filtered arm list; got {calls}"
    for call in filtered:
        assert sorted(call) == sorted(READY_ARMS), (
            f"the eval stage was handed {call}, expected {READY_ARMS}; "
            f"{AMBIGUOUS_ARM} has two 40k backbones and cannot be resolved")


def test_390_pipeline_arm_list_stays_on_stdout(pipeline_run):
    """`arms_at_step` returns its arm list on stdout; its log lines must not
    join it. A missing `>&2` makes the wave's arm list a log line."""
    _, _, handed, _ = pipeline_run
    for line in handed.splitlines():
        assert "pipeline-390" not in line, (
            f"a log line leaked into the arm list handed downstream: {line!r}")
        for token in line.split():
            assert not token.startswith("/"), (
                f"a path leaked into the arm list handed downstream: {line!r}")
