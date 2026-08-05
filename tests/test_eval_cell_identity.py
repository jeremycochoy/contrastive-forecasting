"""The grammar of a snapshot filename, and the cell name that must cite it.

`scripts/resolve_eval_checkpoint.sh` decides *which* backbone a cell is
measured on. It cannot decide what the cell is *called*, and that is where
the same defect came back: the resolver hands back `<name>_r3_40k.pth`, the
caller names its output directory after (arm, step) alone, and the replicate
lands in the base run's directory — reusing its head and lifting its
aggregate, then publishing that number as the replicate's.

`scripts/eval_cell_identity.sh` holds the two names a checkpoint decides:

  * `replicate_tag`   — `""` for the base run, `_r<N>` for a resume. It is
    the same grammar the resolver validates an override against, which is
    why the resolver sources this file rather than spelling the pattern
    twice.
  * `eval_cell_name`  — the output cell, with the replicate token beside the
    backbone step it qualifies.
  * `eval_cell_summaries` — the summaries of a (slug, step, head steps)
    triple, whichever replicate wrote them. Callers that look a cell up by
    name need it, or a replicate-backed cell reads as missing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LIB = REPO_ROOT / "scripts" / "eval_cell_identity.sh"
RESOLVER = REPO_ROOT / "scripts" / "resolve_eval_checkpoint.sh"

NAME = "bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"


def call(func: str, *args: str) -> subprocess.CompletedProcess:
    argv = 'source "%s"; %s' % (LIB, " ".join([func] + [f'"{a}"'
                                                        for a in args]))
    return subprocess.run(["bash", "-c", argv], capture_output=True, text=True)


# --- 1. the replicate token ----------------------------------------------

def test_base_run_has_no_replicate_token():
    """The committed cells are named from base-run checkpoints and from
    replicates alike; an empty token for the base keeps their names."""
    r = call("replicate_tag", NAME, "40", f"/runs/{NAME}_40k.pth")
    assert r.returncode == 0, r.stderr
    assert r.stdout == ""


def test_a_resume_carries_its_replicate_token():
    r = call("replicate_tag", NAME, "40", f"/runs/{NAME}_r3_40k.pth")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "_r3"


def test_a_two_digit_replicate_keeps_both_digits():
    r = call("replicate_tag", NAME, "40", f"/runs/{NAME}_r12_40k.pth")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "_r12"


def test_two_replicates_do_not_share_a_token():
    """The whole point: the token separates the models the resolver refuses
    to choose between."""
    tags = {call("replicate_tag", NAME, "40", p).stdout
            for p in (f"/runs/{NAME}_40k.pth", f"/runs/{NAME}_r2_40k.pth",
                      f"/runs/{NAME}_r3_40k.pth")}
    assert len(tags) == 3, f"two replicates map to one token: {tags}"


@pytest.mark.parametrize("filename,why", [
    (f"{NAME}_100k.pth", "another step"),
    ("bb_small_arm6_v2_lalign_lrepmoco_40k.pth", "another run"),
    (f"{NAME}_40k_optimizer.pth", "the optimizer sidecar"),
    (f"{NAME}_revin_40k.pth", "a recipe suffix, not a resume"),
    (f"{NAME}_EMERGENCY_40000.pth", "a NaN dump"),
])
def test_a_path_of_another_pair_has_no_token(filename: str, why: str):
    """A caller that cannot name the replicate must not get an empty token
    and file the result under the base run: `""` means the base run."""
    r = call("replicate_tag", NAME, "40", f"/runs/{filename}")
    assert r.returncode != 0, (
        f"{why} was accepted as a replicate of ({NAME}, 40k)")
    assert r.stdout == ""


# --- 2. the cell name -----------------------------------------------------

def test_cell_name_without_a_replicate_is_the_committed_shape():
    """Every cell in the report is this string. It must not move."""
    r = call("eval_cell_name", "arm5", "40", "", "15000")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "arm5_bb40k_hd15000s"


def test_cell_name_carries_the_replicate_beside_its_step():
    r = call("eval_cell_name", "arm5_nse", "200", "_r3", "30000")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "arm5_nse_bb200k_r3_hd30000s"


def test_a_replicate_cell_is_not_the_base_cell():
    base = call("eval_cell_name", "arm5", "40", "", "15000").stdout
    repl = call("eval_cell_name", "arm5", "40", "_r3", "15000").stdout
    assert base != repl, (
        "base and replicate share one cell name, so they share one output "
        "directory, one head and one aggregate")


# --- 3. finding a cell whichever replicate wrote it -----------------------

def summaries(root: Path, *args: str) -> list[str]:
    r = call("eval_cell_summaries", str(root), *args)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_a_base_cell_summary_is_found(tmp_path: Path):
    f = tmp_path / "arm5_bb40k_hd15000s_summary.txt"
    f.write_text("x\n")
    assert summaries(tmp_path, "arm5", "40", "15000") == [str(f)]


def test_a_replicate_cell_summary_is_found(tmp_path: Path):
    """Without this, the wave-3 gate reads a measured replicate cell as
    missing and stops the arm."""
    f = tmp_path / "arm5_bb100k_r2_hd30000s_summary.txt"
    f.write_text("x\n")
    assert summaries(tmp_path, "arm5", "100", "30000") == [str(f)]


def test_both_replicates_are_returned_not_one(tmp_path: Path):
    """Two measured replicates of one cell is an ambiguity for the caller to
    refuse, not something to resolve here by picking."""
    base = tmp_path / "arm5_bb40k_hd15000s_summary.txt"
    repl = tmp_path / "arm5_bb40k_r3_hd15000s_summary.txt"
    for f in (base, repl):
        f.write_text("x\n")
    assert sorted(summaries(tmp_path, "arm5", "40", "15000")) == \
        sorted([str(base), str(repl)])


def test_no_summary_is_no_output(tmp_path: Path):
    assert summaries(tmp_path, "arm5", "40", "15000") == []


def test_another_step_is_not_this_cells_summary(tmp_path: Path):
    (tmp_path / "arm5_bb100k_hd30000s_summary.txt").write_text("x\n")
    assert summaries(tmp_path, "arm5", "40", "15000") == []


def test_a_non_replicate_suffix_is_not_a_replicate_cell(tmp_path: Path):
    """`_r*_` also matches `_revin_`; the resolver's glob says `_r[0-9]*_`
    and this one has to say the same."""
    (tmp_path / "arm5_bb40k_revin_hd15000s_summary.txt").write_text("x\n")
    assert summaries(tmp_path, "arm5", "40", "15000") == []


# --- 4. the resolver reads its grammar from here --------------------------

def test_the_resolver_sources_this_library():
    """One grammar. A second copy of `(_r[0-9]+)?` drifts from this one, and
    a drifted copy is how the override check stops matching what the cell
    name is built from."""
    src = RESOLVER.read_text()
    assert "eval_cell_identity.sh" in src, (
        "resolve_eval_checkpoint.sh must take the snapshot-name grammar from "
        "scripts/eval_cell_identity.sh, not spell it out a second time")
    assert "ckpt_is_run_step" in src, (
        "the override check must use the library's predicate")
