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
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESOLVER = REPO_ROOT / "scripts" / "resolve_eval_checkpoint.sh"
EVAL_SH = (REPO_ROOT / "experiments" / "2026-07-21_split_pred_rep_small" /
           "scripts" / "eval_2L_gm_mase.sh")

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
