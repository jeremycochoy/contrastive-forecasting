"""The grammar of a snapshot filename, and the cell name that must cite it.

`scripts/resolve_eval_checkpoint.sh` decides *which* backbone a cell is
measured on. It cannot decide what the cell is *called*, and that is where
the same defect came back: the resolver hands back `<name>_r3_40k.pth`, the
caller names its output directory after (arm, step) alone, and the replicate
lands in the base run's directory — reusing its head and lifting its
aggregate, then publishing that number as the replicate's.

The head seed is the same story with a different token. `HEAD_SEED` decides
the number — it is the head trainer's `--seed` — so a cell measured under
another seed is another measurement. Before it was part of the identity,
`HEAD_SEED=20260723` for an already-measured cell resolved to the base
cell's directory, skipped head-train on the other seed's head, found 97 rows,
skipped GIFT-Eval and rewrote the summary with the other seed's aggregate,
exit 0. `CELL_TAG=_s<seed>` avoided that by convention, and a convention is
not an identity.

`scripts/eval_cell_identity.sh` holds the names a checkpoint and a seed
decide:

  * `replicate_tag`   — `""` for the base run, `_r<N>` for a resume. It is
    the same grammar the resolver validates an override against, which is
    why the resolver sources this file rather than spelling the pattern
    twice.
  * `head_seed_tag`   — `""` for the seed every wave ran, `_s<seed>` for any
    other. Empty for the default for the same reason the base run's
    replicate token is: the committed cell names must not move.
  * `eval_cell_name`  — the output cell, with each token beside the thing it
    qualifies.
  * `eval_cell_summaries` — the summaries of a (slug, step, head steps, head
    seed) coordinate, whichever replicate wrote them. Callers that look a
    cell up by name need it, or a replicate-backed cell reads as missing.

`scripts/eval_cell_identity.py` is the same grammar for the Python readers —
the wave-3 gate, the table builder, the bootstrap, the figures. Two bindings,
not two grammars: the parity section below generates every name both ways and
requires them to agree character for character.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LIB = REPO_ROOT / "scripts" / "eval_cell_identity.sh"
PYLIB = REPO_ROOT / "scripts" / "eval_cell_identity.py"
RESOLVER = REPO_ROOT / "scripts" / "resolve_eval_checkpoint.sh"
EXP_390 = REPO_ROOT / "experiments" / "2026-08-01_lalign_teacher"
REPORT_390 = REPO_ROOT / "reports" / "2026-08-04_lalign_teacher"

NAME = "bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"

# The seed every wave, and so every committed cell, was measured under.
WAVE_SEED = "20260722"
OTHER_SEED = "20260723"


def call(func: str, *args: str) -> subprocess.CompletedProcess:
    argv = 'source "%s"; %s' % (LIB, " ".join([func] + [f'"{a}"'
                                                        for a in args]))
    return subprocess.run(["bash", "-c", argv], capture_output=True, text=True)


def shell_var(name: str) -> str:
    """The value the library gives a constant, as a caller sourcing it sees."""
    r = subprocess.run(["bash", "-c", f'source "{LIB}"; printf "%s" "${name}"'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cid():
    return load(PYLIB, "cf_eval_cell_identity")


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


# --- 2. the head-seed token ----------------------------------------------

def test_the_default_head_seed_is_the_seed_every_wave_ran():
    """Every committed cell was measured at this seed and carries no token
    for it. Moving the default renames all 238 of them."""
    assert shell_var("EVAL_DEFAULT_HEAD_SEED") == WAVE_SEED


def test_the_wave_seed_has_no_token():
    r = call("head_seed_tag", WAVE_SEED)
    assert r.returncode == 0, r.stderr
    assert r.stdout == ""


def test_another_head_seed_carries_its_own_token():
    r = call("head_seed_tag", OTHER_SEED)
    assert r.returncode == 0, r.stderr
    assert r.stdout == f"_s{OTHER_SEED}"


# --- 2b. what a seed may be, in both bindings ----------------------------
# The token has to be readable back, or the name says one thing and every
# reader of it says another. Bash printed `_s` in front of whatever it was
# handed while Python parsed `_s<six or more digits>`, so `HEAD_SEED=7` wrote
# `arm5_s7_bb40k_hd15000s` and the readers called that the *wave* seed's cell
# of an arm named `arm5_s7`: the wrong seed in the table, and a collision with
# the real wave cell in `controlled_delta.py`. A seed is a run of digits on
# both sides, and anything else is refused rather than named.

BAD_SEEDS = ["", "abc", "20260722x", "2026 0722", "-1", "20260722.0"]


@pytest.mark.parametrize("seed", BAD_SEEDS, ids=repr)
def test_bash_refuses_a_seed_it_cannot_write_a_readable_token_for(seed: str):
    r = call("head_seed_tag", seed)
    assert r.returncode != 0, (
        f"the token {r.stdout!r} was written for seed {seed!r}, and no reader "
        "can parse that back into a seed")
    assert r.stdout == ""


@pytest.mark.parametrize("seed", BAD_SEEDS, ids=repr)
def test_a_seed_bash_cannot_name_does_not_become_the_default_cell(seed: str):
    """The refusal has to travel out of the `$( )` the token is built in. A
    caller that gets an empty token gets the *default* cell's name back with
    exit 0, which is the wave's own cell under another seed's number."""
    r = call("eval_cell_name", "arm5", "40", "", "15000", seed)
    assert r.returncode != 0, (
        f"seed {seed!r} was answered with the cell name {r.stdout!r}")
    assert r.stdout == ""


@pytest.mark.parametrize("seed", BAD_SEEDS, ids=repr)
def test_python_refuses_the_same_seeds_bash_refuses(cid, seed: str):
    with pytest.raises(ValueError):
        cid.head_seed_tag(seed)


@pytest.mark.parametrize("seed", ["7", "42", "1234567", WAVE_SEED, OTHER_SEED])
def test_a_seed_of_any_length_reads_back_as_itself(cid, seed: str):
    """Both bindings write the token, and the Python one parses it back to
    the seed it was written for — whatever its length."""
    name = call("eval_cell_name", "arm5", "40", "", "15000", seed).stdout
    assert cid.cell_name("arm5", "40", "", "15000", seed) == name
    parsed = cid.parse_cell(name)
    assert parsed is not None, name
    assert cid.split_head_seed(parsed.slug) == ("arm5", seed)


# --- 3. the cell name -----------------------------------------------------

def test_cell_name_without_a_replicate_is_the_committed_shape():
    """Every cell in the report is this string. It must not move."""
    r = call("eval_cell_name", "arm5", "40", "", "15000", WAVE_SEED)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "arm5_bb40k_hd15000s"


def test_cell_name_carries_the_replicate_beside_its_step():
    r = call("eval_cell_name", "arm5_nse", "200", "_r3", "30000", WAVE_SEED)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "arm5_nse_bb200k_r3_hd30000s"


def test_a_replicate_cell_is_not_the_base_cell():
    base = call("eval_cell_name", "arm5", "40", "", "15000", WAVE_SEED).stdout
    repl = call("eval_cell_name", "arm5", "40", "_r3", "15000", WAVE_SEED).stdout
    assert base != repl, (
        "base and replicate share one cell name, so they share one output "
        "directory, one head and one aggregate")


def test_cell_name_carries_the_head_seed_beside_the_slug():
    """The shape the committed replicate-seed cells already have, now
    produced by the identity rather than by a caller's CELL_TAG."""
    r = call("eval_cell_name", "arm5_nse", "200", "", "30000", OTHER_SEED)
    assert r.returncode == 0, r.stderr
    assert r.stdout == f"arm5_nse_s{OTHER_SEED}_bb200k_hd30000s"


def test_a_second_head_seed_is_not_the_first_seeds_cell():
    """The gap this closes. One directory for two seeds means head-train
    SKIPs on the first seed's head, the 97-row check skips GIFT-Eval and
    lifts the first seed's aggregate, and the second seed publishes a number
    it never measured — exit 0."""
    first = call("eval_cell_name", "arm5", "40", "", "15000", WAVE_SEED).stdout
    second = call("eval_cell_name", "arm5", "40", "", "15000", OTHER_SEED).stdout
    assert first != second, (
        "two head seeds share one cell name, so they share one output "
        "directory, one head and one aggregate")


def test_the_head_seed_is_not_optional():
    """A caller that forgets the seed must fail loudly, not silently name
    the default cell — that is the difference between an identity and a
    convention. Every caller runs under `set -u`."""
    r = subprocess.run(
        ["bash", "-c",
         f'set -u; source "{LIB}"; eval_cell_name arm5 40 "" 15000'],
        capture_output=True, text=True)
    assert r.returncode != 0, (
        f"a four-argument call was answered with {r.stdout!r} instead of "
        "refusing")


# --- 4. finding a cell whichever replicate wrote it -----------------------

def summaries(root: Path, *args: str) -> list[str]:
    r = call("eval_cell_summaries", str(root), *args)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_a_base_cell_summary_is_found(tmp_path: Path):
    f = tmp_path / "arm5_bb40k_hd15000s_summary.txt"
    f.write_text("x\n")
    assert summaries(tmp_path, "arm5", "40", "15000", WAVE_SEED) == [str(f)]


def test_a_replicate_cell_summary_is_found(tmp_path: Path):
    """Without this, the wave-3 gate reads a measured replicate cell as
    missing and stops the arm."""
    f = tmp_path / "arm5_bb100k_r2_hd30000s_summary.txt"
    f.write_text("x\n")
    assert summaries(tmp_path, "arm5", "100", "30000", WAVE_SEED) == [str(f)]


def test_both_replicates_are_returned_not_one(tmp_path: Path):
    """Two measured replicates of one cell is an ambiguity for the caller to
    refuse, not something to resolve here by picking."""
    base = tmp_path / "arm5_bb40k_hd15000s_summary.txt"
    repl = tmp_path / "arm5_bb40k_r3_hd15000s_summary.txt"
    for f in (base, repl):
        f.write_text("x\n")
    assert sorted(summaries(tmp_path, "arm5", "40", "15000", WAVE_SEED)) == \
        sorted([str(base), str(repl)])


def test_no_summary_is_no_output(tmp_path: Path):
    assert summaries(tmp_path, "arm5", "40", "15000", WAVE_SEED) == []


def test_another_step_is_not_this_cells_summary(tmp_path: Path):
    (tmp_path / "arm5_bb100k_hd30000s_summary.txt").write_text("x\n")
    assert summaries(tmp_path, "arm5", "40", "15000", WAVE_SEED) == []


def test_another_head_seed_is_not_this_cells_summary(tmp_path: Path):
    """A seed's cell must not be answered with another seed's file, or the
    lookup hands back a number measured under a seed it was not asked for."""
    (tmp_path / f"arm5_s{OTHER_SEED}_bb40k_hd15000s_summary.txt").write_text("x")
    assert summaries(tmp_path, "arm5", "40", "15000", WAVE_SEED) == []


def test_a_head_seed_cell_is_found_by_its_own_seed(tmp_path: Path):
    f = tmp_path / f"arm5_s{OTHER_SEED}_bb40k_hd15000s_summary.txt"
    f.write_text("x\n")
    assert summaries(tmp_path, "arm5", "40", "15000", OTHER_SEED) == [str(f)]


def test_a_replicate_cell_of_another_seed_is_not_this_cells_summary(
        tmp_path: Path):
    """The replicate half of the lookup carries the seed too. Without it, a
    lookup for one seed is answered by another seed's `_r<N>` cell — the
    same wrong number, one glob further along."""
    (tmp_path / "arm5_bb40k_r2_hd15000s_summary.txt").write_text("x\n")
    assert summaries(tmp_path, "arm5", "40", "15000", OTHER_SEED) == []


def test_a_head_seed_cell_measured_on_a_replicate_is_found(tmp_path: Path):
    """Both tokens at once: another seed's head on a resumed backbone."""
    f = tmp_path / f"arm5_s{OTHER_SEED}_bb40k_r2_hd15000s_summary.txt"
    f.write_text("x\n")
    assert summaries(tmp_path, "arm5", "40", "15000", OTHER_SEED) == [str(f)]


def test_a_non_replicate_suffix_is_not_a_replicate_cell(tmp_path: Path):
    """`_r*_` also matches `_revin_`; the resolver's glob says `_r[0-9]*_`
    and this one has to say the same."""
    (tmp_path / "arm5_bb40k_revin_hd15000s_summary.txt").write_text("x\n")
    assert summaries(tmp_path, "arm5", "40", "15000", WAVE_SEED) == []


def test_a_replicate_like_suffix_is_not_a_replicate_cell(tmp_path: Path):
    """`_r3x` sits between the glob that gathers and the grammar that
    decides: `_r[0-9]*` matches it, `_r[0-9]+` does not. Handed back it is a
    cell name `parse_cell` refuses, so the two bindings disagree about what
    this directory holds — the glob narrows, the grammar decides."""
    (tmp_path / "arm5_bb40k_r3x_hd15000s_summary.txt").write_text("x\n")
    assert summaries(tmp_path, "arm5", "40", "15000", WAVE_SEED) == []


# --- 5. the resolver reads its grammar from here --------------------------

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


# --- 6. the Python half says the same thing -------------------------------
# The readers are Python: the wave-3 gate, the table builder, the bootstrap,
# the figures. They had the grammar written out again, once each, without the
# replicate — so a replicate-backed cell read as missing everywhere at once.
# One binding per language is fine; two grammars is not, and this is what
# stops the second one from appearing.

# (slug, backbone step k, replicate token, head steps, head seed). Covers the
# base run and a resume, the wave seed and another, and both at once.
NAME_MATRIX = [
    ("arm5", "40", "", "15000", WAVE_SEED),
    ("arm5", "40", "_r3", "15000", WAVE_SEED),
    ("arm5_nse", "200", "", "30000", OTHER_SEED),
    ("arm5_nse", "200", "_r12", "30000", OTHER_SEED),
    ("arm5_combab_alignstudent", "100", "", "30000", WAVE_SEED),
    ("arm6_v2_ncpc", "100", "_r2", "30000", "20260724"),
    # Not every seed is an eight-digit date. A matrix that only ever asks for
    # one shape cannot see the two bindings disagree about the others.
    ("arm5", "40", "", "15000", "7"),
    ("arm5_combab", "100", "_r2", "30000", "42"),
    ("arm6_v2_nse_alignstudent", "200", "", "30000", "1234567"),
]


@pytest.mark.parametrize("slug,bb,repl,hd,seed", NAME_MATRIX)
def test_python_and_bash_build_the_same_cell_name(cid, slug, bb, repl, hd,
                                                  seed):
    got_sh = call("eval_cell_name", slug, bb, repl, hd, seed).stdout
    got_py = cid.cell_name(slug, bb, repl, hd, seed)
    assert got_py == got_sh, (
        f"the two bindings disagree: bash says {got_sh!r}, Python says "
        f"{got_py!r}")


def test_python_and_bash_agree_on_the_default_head_seed(cid):
    assert cid.DEFAULT_HEAD_SEED == shell_var("EVAL_DEFAULT_HEAD_SEED")


@pytest.mark.parametrize("seed,tag", [(WAVE_SEED, ""),
                                      (OTHER_SEED, f"_s{OTHER_SEED}")])
def test_python_and_bash_agree_on_the_head_seed_token(cid, seed, tag):
    assert cid.head_seed_tag(seed) == tag == call("head_seed_tag", seed).stdout


@pytest.mark.parametrize("seed,want", [
    (WAVE_SEED, ["arm5_bb40k_hd15000s_summary.txt",
                 "arm5_bb40k_r3_hd15000s_summary.txt"]),
    (OTHER_SEED, [f"arm5_s{OTHER_SEED}_bb40k_hd15000s_summary.txt",
                  f"arm5_s{OTHER_SEED}_bb40k_r3_hd15000s_summary.txt"]),
])
def test_python_and_bash_find_the_same_summaries(cid, tmp_path: Path, seed,
                                                 want):
    """The lookup, not just the name. Both must return this seed's base cell
    and its replicate, and none of: the other seed's, another step's, a
    `_revin_` sibling, or `_r3x` — which the bash glob matches and the Python
    pattern does not."""
    for n in ("arm5_bb40k_hd15000s", "arm5_bb40k_r3_hd15000s",
              "arm5_bb40k_revin_hd15000s", "arm5_bb40k_r3x_hd15000s",
              f"arm5_s{OTHER_SEED}_bb40k_hd15000s",
              f"arm5_s{OTHER_SEED}_bb40k_r3_hd15000s", "arm5_bb100k_hd30000s"):
        (tmp_path / f"{n}_summary.txt").write_text("x\n")
    got_py = [str(p) for p in cid.cell_paths(tmp_path, "arm5", 40, 15000, seed,
                                             suffix="_summary.txt")]
    assert sorted(got_py) == sorted(
        summaries(tmp_path, "arm5", "40", "15000", seed))
    assert [Path(p).name for p in got_py] == want


def test_python_parses_back_what_it_builds(cid):
    for slug, bb, repl, hd, seed in NAME_MATRIX:
        got = cid.parse_cell(cid.cell_name(slug, bb, repl, hd, seed))
        assert got is not None
        assert (got.bb_k, got.replicate, got.head_steps) == (int(bb), repl,
                                                             int(hd))
        assert cid.split_head_seed(got.slug) == (slug, seed)


def test_python_refuses_a_name_that_is_not_a_cell(cid):
    for bad in ("arm5_bb40k", "arm5_hd15000s", "arm5_bb40k_revin_hd15000s"):
        assert cid.parse_cell(bad) is None, bad


# --- 7. every reader takes its grammar from here --------------------------
# A reader that spells the name itself reads a replicate-backed or
# seed-tagged cell as missing: dropped from the CI table, dropped from the
# figures, and re-run from scratch by the batch script.

PY_READERS = [
    EXP_390 / "scripts" / "select_wave3.py",
    EXP_390 / "scripts" / "make_gm_table.py",
    EXP_390 / "scripts" / "eval_bootstrap.py",
    EXP_390 / "scripts" / "controlled_delta.py",
    EXP_390 / "scripts" / "snapshot_reproduction.py",
    EXP_390 / "scripts" / "verify_head_seeds_40k.py",
    REPORT_390 / "plots" / "_cells.py",
]

SH_READERS = [
    EXP_390 / "scripts" / "eval_arm.sh",
    EXP_390 / "scripts" / "eval_wave.sh",
    EXP_390 / "scripts" / "run_student_control.sh",
    EXP_390 / "scripts" / "run_student_control_batch.sh",
    REPO_ROOT / "experiments" / "2026-07-21_split_pred_rep_small" / "scripts"
    / "eval_2L_gm_mase.sh",
]


@pytest.mark.parametrize("path", PY_READERS, ids=lambda p: p.name)
def test_a_python_reader_imports_the_shared_grammar(path: Path):
    src = path.read_text()
    assert "eval_cell_identity" in src, (
        f"{path.name} names cells without the shared grammar; a replicate- or "
        "seed-tagged cell reads as missing there")
    assert "_bb{" not in src and '_bb%d' not in src, (
        f"{path.name} still builds a cell name by hand")


@pytest.mark.parametrize("path", SH_READERS, ids=lambda p: p.name)
def test_a_shell_reader_sources_the_shared_grammar(path: Path):
    src = path.read_text()
    assert "eval_cell_identity.sh" in src, (
        f"{path.name} names cells without the shared grammar")


# --- 8. one exit code, one operator action --------------------------------

def test_the_bad_tag_code_is_the_librarys_and_both_evals_use_it():
    """A value the cell cannot be named from is one operator action — name it
    in the grammar the cells use. It exited 25 in #390's eval and 20 in
    #379's, so the same failure read as two different things depending on
    which script ran.

    Both refusals, in both scripts: the replicate the resolver handed back,
    and the head seed the cell name is built from. A second number for the
    second one is the same split again, one refusal further along.
    """
    assert shell_var("E_BAD_TAG") == "25"
    for path in (EXP_390 / "scripts" / "eval_arm.sh",
                 REPO_ROOT / "experiments" / "2026-07-21_split_pred_rep_small"
                 / "scripts" / "eval_2L_gm_mase.sh"):
        src = path.read_text()
        assert src.count("exit $E_BAD_TAG") == 2, (
            f"{path.name} spells the library's bad-tag exit code "
            f"{src.count('exit $E_BAD_TAG')} times; both the replicate-tag "
            "refusal and the cell-name refusal use it")


# --- 9. the readers, run ---------------------------------------------------
# Source text says a reader imports the grammar. These say it resolves the
# cell. Each one is the same fixture: one measured cell, named the way a
# resumed backbone names it, and a reader asked for it by its untagged
# coordinate.

AGG = "Aggregate GM-Relative MASE (97 configs): 1.2345\n"


def test_the_figures_find_a_replicate_backed_cell(tmp_path, monkeypatch):
    """`_cells.read_aggregate` feeds every figure in the report. A cell it
    reads as missing is a cell dropped from the progression plot, the
    per-panel plot, the rankings and the radars at once."""
    cells = load(REPORT_390 / "plots" / "_cells.py", "cf390_cells")
    rep = tmp_path / "rep"
    rep.mkdir()
    (rep / "arm5_bb100k_r2_hd30000s_summary.txt").write_text(AGG)
    monkeypatch.setattr(cells, "REP", rep)
    monkeypatch.setattr(cells, "EXP", tmp_path / "absent")
    assert cells.read_aggregate("arm5", 100, 30000) == 1.2345


def test_the_figures_refuse_two_replicates_of_one_cell(tmp_path, monkeypatch):
    """The choice the resolver refuses to make. A figure must not make it
    silently by taking whichever it globbed first."""
    cells = load(REPORT_390 / "plots" / "_cells.py", "cf390_cells")
    rep = tmp_path / "rep"
    rep.mkdir()
    for tag in ("", "_r2"):
        (rep / f"arm5_bb100k{tag}_hd30000s_summary.txt").write_text(AGG)
    monkeypatch.setattr(cells, "REP", rep)
    monkeypatch.setattr(cells, "EXP", tmp_path / "absent")
    with pytest.raises(SystemExit):
        cells.read_aggregate("arm5", 100, 30000)


def test_the_bootstrap_finds_a_replicate_backed_cell(tmp_path):
    """A cell the bootstrap reads as missing is a row missing from the CI
    table, and the report judges every claimed gap against that table."""
    boot = load(EXP_390 / "scripts" / "eval_bootstrap.py", "cf390_boot")
    cell = tmp_path / "eval_gm_mase" / "arm5_bb40k_r2_hd15000s"
    cell.mkdir(parents=True)
    (cell / "all_results.csv").write_text("dataset\n")
    assert boot.cell_results(tmp_path, "arm5", 40, 15000) == \
        cell / "all_results.csv"


def test_the_bootstrap_refuses_two_replicates_of_one_cell(tmp_path):
    boot = load(EXP_390 / "scripts" / "eval_bootstrap.py", "cf390_boot")
    for tag in ("", "_r2"):
        d = tmp_path / "eval_gm_mase" / f"arm5_bb40k{tag}_hd15000s"
        d.mkdir(parents=True)
        (d / "all_results.csv").write_text("dataset\n")
    with pytest.raises(SystemExit):
        boot.cell_results(tmp_path, "arm5", 40, 15000)


@pytest.fixture
def scratch():
    """A scratch root outside /tmp — the batch script refuses a $WT under it."""
    cache = Path(os.path.expanduser("~/.cache"))
    cache.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="cf390-cellid-", dir=cache))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def batch_sandbox(root: Path) -> Path:
    """The batch script, its pool helper and the identity library, copied at
    the same relative depth, with the per-cell driver replaced by a recorder.

    A copy rather than the checkout: the real `run_student_control.sh`
    generates a launcher into the repo's own `scripts/` and starts a 40 000-
    step backbone run. Only the skip check is under test here.
    """
    repo = root / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "eval_cell_identity.sh",
                repo / "scripts")
    scripts = repo / "experiments" / "2026-08-01_lalign_teacher" / "scripts"
    scripts.mkdir(parents=True)
    for name in ("run_student_control_batch.sh", "gpu_pool.sh"):
        shutil.copy(EXP_390 / "scripts" / name, scripts / name)
    (scripts / "run_student_control.sh").write_text(
        '#!/bin/bash\necho "DRIVER RAN $ARM"\n')
    return scripts / "run_student_control_batch.sh"


def run_student_batch(root: Path, *, measured: str | None,
                      env_extra: dict[str, str] | None = None):
    """Run the batch over arm5 with `measured` (a cell name) already on disk."""
    script = batch_sandbox(root)
    wt = root / "wt"
    evals = wt / "experiments" / "2026-08-01_lalign_teacher" / "eval_gm_mase"
    evals.mkdir(parents=True)
    if measured:
        (evals / f"{measured}_summary.txt").write_text(AGG)
    return subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "WT": str(wt), "ARMS": "arm5", "SLOTS_PER_GPU": "1",
             **(env_extra or {})},
        capture_output=True, text=True, timeout=180)


def test_the_student_batch_skips_a_replicate_backed_cell(scratch: Path):
    """The batch script's own skip check. Reading a measured cell as missing
    re-runs a 40 000-step backbone and a 15 000-step head from scratch, and
    then measures the same thing twice under two names."""
    r = run_student_batch(
        scratch, measured="arm5_alignstudent_bb40k_r2_hd15000s")
    assert "SKIP arm5" in r.stdout, (
        "the batch script read a measured replicate-backed cell as missing "
        f"and would re-run it:\n{r.stdout}\n{r.stderr}")
    assert "DRIVER RAN" not in r.stdout, r.stdout


def test_the_student_batch_does_not_skip_an_unmeasured_cell(scratch: Path):
    """The other half: the skip fires on a measured cell and only on one, or
    the batch reports arms it never ran."""
    r = run_student_batch(scratch, measured=None)
    assert "SKIP arm5" not in r.stdout, r.stdout
    assert "DRIVER RAN arm5" in r.stdout, r.stdout


# --- 10. a wave counts the cells it wrote, not another seed's -------------
# `HEAD_SEED` reaches `eval_arm.sh` from the environment, so an exported one
# renames every cell a wave writes. These scripts then looked the results up
# at the library's default: a wave that measured all ten arms perfectly
# reported ten missing cells, and the batch script's skip check answered a
# seed's question with another seed's cell. The seed a wave runs and the seed
# it counts are one variable, bound once and passed on.

EVAL_ARM_RECORDER = """#!/bin/bash
# Stand-in for eval_arm.sh: no head, no GIFT-Eval, just the cell those two
# would have written — named by the shared identity out of the environment
# this was handed, the way the real one names it.
set -uo pipefail
ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ROOT/scripts/eval_cell_identity.sh"
HEAD_SEED="${HEAD_SEED:-$EVAL_DEFAULT_HEAD_SEED}"
CELL_TAG="${CELL_TAG:-}"
CELL="$(eval_cell_name "${ARM}${CELL_TAG}" "$BB_STEP_K" "" "$HEAD_STEPS" \
                       "$HEAD_SEED")"
OUT="$WT/experiments/2026-08-01_lalign_teacher/eval_gm_mase"
mkdir -p "$OUT"
printf '%s\n' "__AGG__" > "$OUT/${CELL}_summary.txt"
echo "EVAL RAN arm=$ARM seed=$HEAD_SEED cell=$CELL"
"""


def wave_sandbox(root: Path) -> Path:
    """The wave and student-control scripts over a recorder, at the depth the
    real tree has. Returns the sandbox repo, which is also its own `$WT`.

    A copy rather than the checkout, for the same reason as `batch_sandbox`:
    the real stages train a backbone and a head and run a 97-config eval.
    What is under test is which cell each script writes and which cell it
    then counts.
    """
    repo = root / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "eval_cell_identity.sh",
                repo / "scripts")
    scripts = repo / "experiments" / "2026-08-01_lalign_teacher" / "scripts"
    scripts.mkdir(parents=True)
    for name in ("eval_wave.sh", "run_student_control.sh", "arm_names.sh",
                 "gpu_pool.sh"):
        shutil.copy(EXP_390 / "scripts" / name, scripts / name)
    (scripts / "eval_arm.sh").write_text(
        EVAL_ARM_RECORDER.replace("__AGG__", AGG.strip()))
    (scripts / "run_arm_student.sh").write_text(
        '#!/bin/bash\necho "BACKBONE RAN $1"\n')
    return repo


def run_wave_script(repo: Path, name: str, **env_extra):
    scripts = repo / "experiments" / "2026-08-01_lalign_teacher" / "scripts"
    return subprocess.run(
        ["bash", str(scripts / name)],
        env={**os.environ, "WT": str(repo), "WAVE": "1", "ARMS": "arm5",
             "ARM": "arm5", "BB_GPU": "0", "SLOTS_PER_GPU": "1", **env_extra},
        capture_output=True, text=True, timeout=300)


def test_a_wave_counts_the_cells_it_measured(scratch: Path):
    """Non-regression at the wave's own seed: one arm measured, one counted."""
    r = run_wave_script(wave_sandbox(scratch), "eval_wave.sh")
    assert "arm5_bb40k_hd15000s: Aggregate" in r.stdout, r.stdout
    assert "measurement: 1 / 1" in r.stdout, r.stdout


def test_a_wave_under_another_head_seed_counts_its_own_cells(scratch: Path):
    """The desync. Every cell the wave writes carries the seed; the tally
    asked for the default, so a wave where nothing failed logged MISSING for
    all ten arms and the operator re-ran a finished wave."""
    repo = wave_sandbox(scratch)
    r = run_wave_script(repo, "eval_wave.sh", HEAD_SEED=OTHER_SEED)
    # The stage's own output goes to the wave log; the tally goes to stdout.
    stage = (repo / "experiments" / "2026-08-01_lalign_teacher" / "results"
             / "eval_wave1.log").read_text()
    assert f"cell=arm5_s{OTHER_SEED}_bb40k_hd15000s" in stage, (
        f"the eval stage did not run under the wave's seed:\n{stage}")
    assert "MISSING or partial" not in r.stdout, (
        "the wave measured its cell and then counted at another seed:\n"
        f"{r.stdout}")
    assert "measurement: 1 / 1" in r.stdout, r.stdout


def test_the_student_control_reports_the_cell_of_its_own_seed(scratch: Path):
    """Same two halves in the single-cell driver: it hands `eval_arm.sh` a
    seed and then reads the summary back by name."""
    r = run_wave_script(wave_sandbox(scratch), "run_student_control.sh",
                        HEAD_SEED=OTHER_SEED)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert f"cell=arm5_alignstudent_s{OTHER_SEED}_bb40k_hd15000s" in r.stdout, (
        r.stdout)
    assert "DONE — Aggregate" in r.stdout, (
        f"the control read its own measurement as missing:\n{r.stdout}")


def test_the_student_batch_skip_check_asks_for_the_seed_it_will_measure(
        scratch: Path):
    """The batch's skip check under an ambient seed. The default seed's cell
    is on disk and the batch is about to measure another seed's, so this arm
    has not been measured — skipping it drops a cell from the sweep and
    reports the run as complete."""
    r = run_student_batch(scratch, measured="arm5_alignstudent_bb40k_hd15000s",
                          env_extra={"HEAD_SEED": OTHER_SEED})
    assert "SKIP arm5" not in r.stdout, (
        "another seed's cell answered this seed's skip check:\n"
        f"{r.stdout}\n{r.stderr}")
    assert "DRIVER RAN arm5" in r.stdout, r.stdout


def test_the_student_batch_skips_the_cell_of_its_own_seed(scratch: Path):
    """And the other half, or the skip check never fires under a seed and the
    batch re-measures everything it already has."""
    r = run_student_batch(
        scratch, measured=f"arm5_alignstudent_s{OTHER_SEED}_bb40k_hd15000s",
        env_extra={"HEAD_SEED": OTHER_SEED})
    assert "SKIP arm5" in r.stdout, (
        f"the batch re-runs a cell it has already measured:\n{r.stdout}")
    assert "DRIVER RAN" not in r.stdout, r.stdout
