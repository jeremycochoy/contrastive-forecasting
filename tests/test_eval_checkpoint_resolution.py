"""Eval-time checkpoint selection must never resolve an ambiguity silently.

A resumed run is renamed `<name>_r<N>` by `safe_run_name`, so one (run name,
step) pair can leave several backbones on disk: `NAME_40k.pth` from the first
attempt and `NAME_r3_40k.pth` from the resume. The eval scripts used to pick
between them with `ls -t … | head -1`, i.e. by modification time, and recorded
nothing about the choice. Two different backbones could then be evaluated
under the same cell name, and no output said which file produced the number.

`scripts/resolve_eval_checkpoint.sh` replaces that pick. It holds three
properties:

  1. more than one candidate aborts, listing every candidate, unless the
     caller names the file explicitly;
  2. a caller who names a file must name one that belongs to the (run name,
     step) pair being resolved — the override is a way to pick a replicate,
     not a way to file any checkpoint under any cell name;
  3. the resolved path is always printed, so a published number can be traced
     back to the file it came from.

Selecting the *newest* checkpoint of a still-running job is a different job
(resume) and is not what this resolver is for.

Four call sites carry the pick. #379's `eval_2L_gm_mase.sh` is checked from
its source text below; #390's `eval_arm.sh` (eval selection), `pipeline.sh`
(a presence check, which decides whether an arm reaches the eval stage at
all) and `run_arm.sh` (the `_FINAL.pth` fallback, which names an artefact
after a replicate) are checked by running them.
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
RUN_ARM_390 = EXP_390 / "scripts" / "run_arm.sh"
PARITY_390 = EXP_390 / "scripts" / "verify_ckpt_resolution_parity.sh"

NAME = "bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"

# The resolver's exit codes. Tests assert the exact code so a missing or
# unrunnable script (127) can never pass as a deliberate abort.
EXIT_NO_CANDIDATE = 3
EXIT_BAD_OVERRIDE = 4
EXIT_AMBIGUOUS = 5
EXIT_OVERRIDE_MISMATCH = 6


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


def make_stub_resolver_wt(scratch: Path, rc: int, message: str) -> Path:
    """A checkout holding the real pipeline.sh and a resolver that fails.

    `pipeline.sh` loads the resolver from its own checkout, so the only way
    to make it meet an exit code the real resolver never returns is to give
    it a checkout whose resolver is a stub.
    """
    wt = scratch / "wt-stub-resolver"
    exp = wt / "experiments" / "2026-08-01_lalign_teacher"
    (exp / "runs").mkdir(parents=True)
    (exp / "results").mkdir()
    scripts = exp / "scripts"
    scripts.mkdir()
    for name in ("pipeline.sh", "arm_names.sh"):
        shutil.copy(EXP_390 / "scripts" / name, scripts / name)
    for stage in ("orchestrate_pool.sh", "eval_wave.sh"):
        (scripts / stage).write_text(STUB_STAGE)
    (scripts / "select_wave3.py").write_text(STUB_GATE)
    (wt / "scripts").mkdir()
    (wt / "scripts" / "resolve_eval_checkpoint.sh").write_text(
        f'#!/bin/bash\necho "{message}" >&2\nexit {rc}\n')
    return wt


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
    """Two candidates: no pick, both paths in the log, called what it is."""
    _, log, _, wt = pipeline_run
    runs = wt / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    name = bb_name_390(AMBIGUOUS_ARM)
    for fname in (f"{name}_40k.pth", f"{name}_r4_40k.pth"):
        assert str(runs / fname) in log, (
            f"the ambiguity must name {fname} in pipeline.log:\n{log}")
    assert re.search(rf"DROP {AMBIGUOUS_ARM} — 40k backbone is ambiguous", log), (
        "an ambiguity must be logged as one, not as a generic resolver "
        f"failure:\n{log}")


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


# --- 10. the override must name a file of this (run name, step) pair ------
# The override exists so an operator can say which replicate they mean. It
# must not become a way to file any checkpoint under any cell name: the
# callers name their output cell from (arm, step) whatever the override says,
# so an override from another step or another run reproduces the exact bug
# this resolver removes — a number published under a name it does not belong
# to. It is reached precisely when the operator is already unsure which
# replicate is which, so it checks rather than trusts.

def test_override_from_another_step_aborts(runs: Path):
    """`BB_CHECKPOINT=…_100k.pth` with `BB_STEP_K=40` is the failure case:
    a 100k backbone's numbers in a cell labelled `bb40k`."""
    touch(runs, f"{NAME}_40k.pth", 1_000)
    wrong = touch(runs, f"{NAME}_100k.pth", 1_000)
    r = resolve(runs, NAME, "40", str(wrong))
    assert r.returncode == EXIT_OVERRIDE_MISMATCH, (
        "a checkpoint from another step must not be accepted for this step; "
        f"got rc={r.returncode}\n{r.stderr}")
    assert r.stdout.strip() == "", (
        "nothing may reach stdout on an abort — the caller captures stdout "
        "as the checkpoint path")
    assert str(wrong) in r.stderr and "40" in r.stderr


def test_override_from_another_run_aborts(runs: Path):
    """Same step, different arm — the other run's cell name is not this one."""
    other = "bb_small_arm6_v2_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon"
    wrong = touch(runs, f"{other}_40k.pth", 1_000)
    r = resolve(runs, NAME, "40", str(wrong))
    assert r.returncode == EXIT_OVERRIDE_MISMATCH, (
        f"another run's checkpoint was accepted for '{NAME}'\n{r.stderr}")
    assert r.stdout.strip() == ""


def test_override_abort_names_the_shape_it_wanted(runs: Path):
    """The abort has to say what a valid override looks like."""
    wrong = touch(runs, f"{NAME}_100k.pth", 1_000)
    r = resolve(runs, NAME, "40", str(wrong))
    assert r.returncode == EXIT_OVERRIDE_MISMATCH
    assert f"{NAME}_40k.pth" in r.stderr, (
        "the mismatch abort must name the expected basename")
    assert "_r" in r.stderr, (
        "the mismatch abort must say the `_r<N>` replicate form is accepted "
        f"too; stderr was:\n{r.stderr}")


def test_override_accepts_the_base_name(runs: Path):
    """The two accepted shapes still resolve. Base run, no resume."""
    wanted = touch(runs, f"{NAME}_40k.pth", 1_000)
    r = resolve(runs, NAME, "40", str(wanted))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(wanted)


def test_override_accepts_a_two_digit_replicate(runs: Path):
    """`_r<N>` is any number of digits, not one."""
    touch(runs, f"{NAME}_40k.pth", 9_000)
    wanted = touch(runs, f"{NAME}_r12_40k.pth", 1_000)
    r = resolve(runs, NAME, "40", str(wanted))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(wanted)


def test_override_rejects_the_optimizer_sidecar(runs: Path):
    """`_40k_optimizer.pth` exists for every snapshot and is not a backbone."""
    wrong = touch(runs, f"{NAME}_40k_optimizer.pth", 1_000)
    r = resolve(runs, NAME, "40", str(wrong))
    assert r.returncode == EXIT_OVERRIDE_MISMATCH, (
        f"the optimizer sidecar was accepted as a backbone\n{r.stderr}")


def test_override_rejects_a_non_numeric_replicate_suffix(runs: Path):
    """`_revin_` is a recipe suffix, not a resume — `_r<N>` means digits."""
    wrong = touch(runs, f"{NAME}_revin_40k.pth", 1_000)
    r = resolve(runs, NAME, "40", str(wrong))
    assert r.returncode == EXIT_OVERRIDE_MISMATCH, (
        f"'_revin_' was read as a replicate suffix\n{r.stderr}")


def test_override_may_point_outside_the_runs_dir(runs: Path, tmp_path: Path):
    """The check is on the name, not the directory.

    A replicate archived elsewhere still says which (run, step) it is, and
    naming it is the documented way to evaluate it.
    """
    elsewhere = tmp_path / "archive"
    elsewhere.mkdir()
    wanted = touch(elsewhere, f"{NAME}_r3_40k.pth", 1_000)
    r = resolve(runs, NAME, "40", str(wanted))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(wanted)


def test_390_eval_arm_rejects_an_override_from_another_step(scratch: Path):
    """End to end at the call site the escape hatch was opened for.

    `eval_arm.sh` names its output cell `${ARM}_bb${BB_STEP_K}k_…` whatever
    BB_CHECKPOINT says, so a 100k file under `BB_STEP_K=40` would publish a
    100k backbone's numbers as `bb40k`. It must not start.
    """
    wt = make_wt(scratch)
    runs = wt / "experiments" / "2026-08-01_lalign_teacher" / "runs"
    name = bb_name_390(ARM)
    touch(runs, f"{name}_{STEP_K}k.pth", 1_000)
    wrong = touch(runs, f"{name}_100k.pth", 9_000)

    r = run_eval_arm(wt, scratch, bb_checkpoint=str(wrong))
    assert r.returncode == EXIT_OVERRIDE_MISMATCH, (
        f"expected the override-mismatch exit; got {r.returncode}\n"
        f"{r.stdout}\n{r.stderr}")
    assert not (wt / "experiments" / "2026-08-01_lalign_teacher" /
                "eval_gm_mase").exists(), (
        "a rejected override must not open a cell directory")


# --- 11. `_r<N>` means digits, in the candidate glob too ------------------
# `_r*_` matches any suffix starting with `r`. No arm slug starts with one
# today, so it only ever cost a spurious abort — but the glob should say what
# it means.

def test_a_non_replicate_r_suffix_is_not_a_candidate(runs: Path):
    """`NAME_revin_40k.pth` is a different run, not `NAME` resumed."""
    touch(runs, f"{NAME}_revin_40k.pth", 1_000)
    r = resolve(runs, NAME, "40")
    assert r.returncode == EXIT_NO_CANDIDATE, (
        "'_revin_' was matched as a `_r<N>` resume of this run; got "
        f"rc={r.returncode}, stdout={r.stdout.strip()!r}")


def test_a_non_replicate_r_suffix_does_not_create_an_ambiguity(runs: Path):
    """And it must not block the run that does have one candidate."""
    ckpt = touch(runs, f"{NAME}_40k.pth", 1_000)
    touch(runs, f"{NAME}_revin_40k.pth", 9_000)
    r = resolve(runs, NAME, "40")
    assert r.returncode == 0, (
        "a sibling run's checkpoint made this pair look ambiguous; "
        f"rc={r.returncode}\n{r.stderr}")
    assert r.stdout.strip() == str(ckpt)


# --- 12. pipeline.sh classifies the resolver's exit codes ----------------
# `arms_at_step` maps an exit code to a log line an operator acts on. Calling
# every non-0/non-3 code "ambiguous" hides a broken resolver behind the one
# message that reads as normal.

def test_390_pipeline_matches_the_ambiguous_code_explicitly(pipeline_code:
                                                            str):
    body = pipeline_code.split("arms_at_step()", 1)[1]
    assert re.search(r"(?m)^\s*5\)", body), (
        "arms_at_step must match exit 5 (ambiguous) explicitly, not through "
        "a catch-all that also swallows every other failure")


def test_390_pipeline_reports_an_unexpected_resolver_failure(scratch: Path):
    """A resolver that fails for any other reason must look like a failure.

    Run against a resolver stub that exits 7, i.e. a code `arms_at_step` was
    never taught. The arm is still dropped — no arm is measured on a guess —
    but the log must say the resolver failed and give the code, not report a
    checkpoint ambiguity that does not exist.
    """
    wt = make_stub_resolver_wt(scratch, rc=7,
                               message="resolver stub: something broke")
    res = subprocess.run(
        ["bash", str(wt / "experiments" / "2026-08-01_lalign_teacher" /
                     "scripts" / "pipeline.sh")],
        env={**os.environ, "WT": str(wt), "STUB_OUT": str(scratch / "out.txt")},
        capture_output=True, text=True, timeout=180)
    log = (wt / "experiments" / "2026-08-01_lalign_teacher" / "results" /
           "pipeline.log").read_text()
    assert "rc=7" in log, (
        f"the unexpected resolver exit code is not in the log:\n{log}")
    assert not re.search(r"DROP \S+ — .*ambiguous", log), (
        f"a resolver failure was reported as a checkpoint ambiguity:\n{log}")
    assert res.returncode != 0, (
        "no arm resolved, so the wave cannot continue")


# --- 13. run_arm.sh's `_FINAL.pth` fallback ------------------------------
# Not an eval selection: it names the artefact a later stage would evaluate.
# `cp -f "$(ls -t … | head -1)"` is the same mtime pick across the same two
# replicates, so `_FINAL.pth` could carry either one and nothing would say
# which. Nothing in #390's waves reads `_FINAL.pth`, so this was latent.

@pytest.fixture
def run_arm_code() -> str:
    return RUN_ARM_390.read_text()


def test_run_arm_final_fallback_has_no_mtime_pick(run_arm_code: str):
    assert not re.search(r"cp -f \"\$\(ls -t", run_arm_code), (
        "run_arm.sh must not name _FINAL.pth after a checkpoint chosen by "
        "modification time")


def test_run_arm_takes_the_resolver_from_its_own_checkout(run_arm_code: str):
    m = re.search(r"(?m)^CKPT_RESOLVER=.*$", run_arm_code)
    assert m, "run_arm.sh does not define CKPT_RESOLVER"
    assert "$WT" not in m.group(0), (
        f"the resolver must not be loaded from $WT; got: {m.group(0)}")


def test_run_arm_keeps_the_mtime_pick_for_resume(run_arm_code: str):
    """Non-regression. Resuming a live run *wants* the newest checkpoint;
    the resolver's own header says so. Only the selection at a named step
    changes."""
    assert re.search(r"RESUME_FROM:-\$\(ls -t", run_arm_code), (
        "the --resume selection is a different question and must keep taking "
        "the newest checkpoint")


def run_run_arm(wt: Path, scratch: Path, steps: str = "2000"):
    env = {**os.environ,
           "PATH": f"{stub_python3(scratch)}:{os.environ['PATH']}",
           "WT": str(wt), "BB_GPU": "0",
           "TARGET_STEPS": steps, "FINAL_STEPS": steps}
    return subprocess.run(["bash", str(RUN_ARM_390), ARM], env=env,
                          capture_output=True, text=True, timeout=180)


def make_run_arm_wt(scratch: Path) -> tuple[Path, Path]:
    """A stand-in checkout run_arm.sh accepts, with the trainer stubbed."""
    wt = scratch / "wt"
    exp = wt / "experiments" / "2026-08-01_lalign_teacher"
    runs = exp / "runs"
    runs.mkdir(parents=True)
    (wt / "experiments" / "hf_token.txt").write_text("hf_stub_token\n")
    trainer = wt / "experiments" / "2026-04-27_freq-embedding" / "scripts"
    trainer.mkdir(parents=True)
    (trainer / "train.py").write_text("")
    return wt, runs


def test_run_arm_final_fallback_aborts_on_two_candidates(scratch: Path):
    """A run and its resume both reached the target: no `_FINAL.pth`."""
    wt, runs = make_run_arm_wt(scratch)
    name = bb_name_390(ARM)
    base = touch(runs, f"{name}_2k.pth", 1_000)
    resumed = touch(runs, f"{name}_r3_2k.pth", 9_000)  # the old `ls -t` pick

    r = run_run_arm(wt, scratch)
    assert r.returncode != 0, (
        f"an ambiguous snapshot produced a _FINAL.pth\n{r.stdout}\n{r.stderr}")
    assert not (runs / f"{name}_FINAL.pth").exists(), (
        "_FINAL.pth must not be written from a checkpoint nothing chose")
    out = r.stdout + r.stderr
    assert str(base) in out and str(resumed) in out, (
        f"the abort must name every candidate; got:\n{out}")


def test_run_arm_finds_the_resolver_through_a_symlinked_scripts_dir(
        scratch: Path):
    """The orchestrators reach run_arm.sh through `$WT/…/scripts` symlinked
    at the real checkout. Resolving that path logically walks back up to $WT,
    where the resolver may not exist at all — the stale-checkout case."""
    wt, runs = make_run_arm_wt(scratch)
    exp = wt / "experiments" / "2026-08-01_lalign_teacher"
    (exp / "scripts").symlink_to(EXP_390 / "scripts")
    touch(runs, f"{bb_name_390(ARM)}_2k.pth", 1_000)

    r = subprocess.run(
        ["bash", str(exp / "scripts" / "run_arm.sh"), ARM],
        env={**os.environ,
             "PATH": f"{stub_python3(scratch)}:{os.environ['PATH']}",
             "WT": str(wt), "BB_GPU": "0",
             "TARGET_STEPS": "2000", "FINAL_STEPS": "2000"},
        capture_output=True, text=True, timeout=180)
    assert "no checkpoint resolver" not in r.stdout + r.stderr, (
        f"run_arm.sh looked for the resolver under $WT:\n{r.stdout}{r.stderr}")
    assert r.returncode == 0, r.stdout + r.stderr


def test_run_arm_final_fallback_copies_the_one_snapshot(scratch: Path):
    """The unambiguous case keeps working, byte for byte."""
    wt, runs = make_run_arm_wt(scratch)
    name = bb_name_390(ARM)
    ckpt = runs / f"{name}_r3_2k.pth"
    ckpt.write_bytes(b"backbone-r3")

    r = run_run_arm(wt, scratch)
    final = runs / f"{name}_FINAL.pth"
    assert final.is_file(), (
        f"_FINAL.pth was not written\n{r.stdout}\n{r.stderr}")
    assert final.read_bytes() == b"backbone-r3"
    assert r.returncode == 0, r.stdout + r.stderr


# --- 14. the parity run says what it proves ------------------------------
# `verify_ckpt_resolution_parity.sh` compares the old pick with the resolver
# over the pairs on disk. That is the right check for "the committed numbers
# do not move" and it is the only thing it shows: every pair in #390's runs/
# has one candidate, so the ambiguity path is never taken. Detection is
# covered by the tests above, and the output has to say so rather than let
# `aborted : 0` read as "the resolver never aborts".

def run_parity(runs: Path):
    return subprocess.run(["bash", str(PARITY_390), str(runs)],
                          capture_output=True, text=True, timeout=180)


def test_parity_counts_the_pairs_that_could_be_ambiguous(runs: Path):
    """The limitation becomes a measured number, not a claim."""
    touch(runs, f"{NAME}_40k.pth", 1_000)
    touch(runs, f"{NAME}_100k.pth", 1_000)
    r = run_parity(runs)
    assert re.search(r"multi-candidate pairs\s*:\s*0", r.stdout), (
        f"the parity run must report how many pairs have more than one "
        f"candidate; got:\n{r.stdout}")


def test_parity_counts_a_real_ambiguity(runs: Path):
    """And the count is real: a second replicate at one step moves it."""
    touch(runs, f"{NAME}_40k.pth", 1_000)
    touch(runs, f"{NAME}_r3_40k.pth", 9_000)
    r = run_parity(runs)
    assert re.search(r"multi-candidate pairs\s*:\s*1", r.stdout), (
        f"a pair with two candidates was not counted:\n{r.stdout}")
    assert r.returncode != 0, "an unresolvable pair is not parity"


def test_parity_states_what_it_does_not_show(runs: Path):
    """`aborted : 0` must not read as evidence about detection."""
    touch(runs, f"{NAME}_40k.pth", 1_000)
    r = run_parity(runs)
    assert "test_two_candidates_abort" in r.stdout, (
        "the parity output must point at the tests that do cover detection; "
        f"got:\n{r.stdout}")
    flowed = " ".join(r.stdout.split())  # the note wraps; the claim is one
    assert re.search(r"does not exercise|not exercised", flowed), (
        f"the parity output must say the ambiguity path was not exercised by "
        f"this run; got:\n{r.stdout}")
