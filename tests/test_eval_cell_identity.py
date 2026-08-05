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
    and its replicate, and neither the other seed's, another step's, nor a
    `_revin_` sibling."""
    for n in ("arm5_bb40k_hd15000s", "arm5_bb40k_r3_hd15000s",
              "arm5_bb40k_revin_hd15000s", f"arm5_s{OTHER_SEED}_bb40k_hd15000s",
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
    """`replicate_tag` refusing is one operator action — name the replicate
    you mean. It exited 25 in #390's eval and 20 in #379's, so the same
    failure read as two different things depending on which script ran."""
    assert shell_var("E_BAD_TAG") == "25"
    for path in (EXP_390 / "scripts" / "eval_arm.sh",
                 REPO_ROOT / "experiments" / "2026-07-21_split_pred_rep_small"
                 / "scripts" / "eval_2L_gm_mase.sh"):
        src = path.read_text()
        assert "exit $E_BAD_TAG" in src, (
            f"{path.name} does not use the library's bad-tag exit code")


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


def run_student_batch(root: Path, *, measured: str | None):
    """Run the batch over arm5 with `measured` (a cell name) already on disk."""
    script = batch_sandbox(root)
    wt = root / "wt"
    evals = wt / "experiments" / "2026-08-01_lalign_teacher" / "eval_gm_mase"
    evals.mkdir(parents=True)
    if measured:
        (evals / f"{measured}_summary.txt").write_text(AGG)
    return subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "WT": str(wt), "ARMS": "arm5", "SLOTS_PER_GPU": "1"},
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
