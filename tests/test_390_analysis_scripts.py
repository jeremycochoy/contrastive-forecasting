"""The analysis scripts the report's numbers come out of.

`make_gm_table.py` builds the one file the report's central comparison is
rebuilt from. A cell tag parsed wrong silently files a student measurement
as a teacher one, which is the exact confusion review item 6 was about.

`eval_bootstrap.py` reports the interval every claimed gap is judged
against. Its one substantive choice is resampling whole datasets rather
than configs; if the clustering were wrong the intervals would come out
too narrow and every gap would look real.

`controlled_delta.py` is the headline table. Its whole reason to exist is
that a teacher number from this branch minus a student number from #379's
sweep is not a measurement of `--align-target`: arm5's two sides sit 0.0977
apart under the identical command line, because #379 published that row off a
resumed backbone replicate (`replicate_provenance.py`). So the tests here
insist that the controlled delta's two sides both come from this branch,
that the snapshot shift is carried in its own column rather than folded
away, and that a partial table cannot be written by accident.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "experiments" / "2026-08-01_lalign_teacher" / "scripts"
GM_TABLE = SCRIPTS / "make_gm_table.py"
BOOTSTRAP = SCRIPTS / "eval_bootstrap.py"
CONTROLLED = SCRIPTS / "controlled_delta.py"
PROVENANCE = SCRIPTS / "replicate_provenance.py"
SPREAD = SCRIPTS / "seed_spread.py"
RESULTS = REPO_ROOT / "reports" / "2026-08-04_lalign_teacher" / "results"
RESULTS_379 = (REPO_ROOT / "reports" / "2026-07-21_split_pred_rep_small"
               / "results")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gm():
    return load(GM_TABLE, "cf390_make_gm_table")


@pytest.fixture(scope="module")
def boot():
    return load(BOOTSTRAP, "cf390_eval_bootstrap")


@pytest.fixture(scope="module")
def ctl():
    return load(CONTROLLED, "cf390_controlled_delta")


# --- cell-tag parsing ----------------------------------------------------

@pytest.mark.parametrize("arm,bare,target,copied,seed", [
    ("arm6_v2_nse", "arm6_v2_nse", None, False, "20260722"),
    # The control: student target, measured on this branch.
    ("arm5_alignstudent", "arm5", "student", False, "20260722"),
    # #379's copy of the same arm and target. These two must not collide.
    ("arm5_alignstudent379", "arm5", "student", True, "20260722"),
    ("arm5_alignteacher", "arm5", "teacher", False, "20260722"),
    ("arm5_nse_s20260723", "arm5_nse", None, False, "20260723"),
    ("arm5_nse_alignstudent379_s20260724", "arm5_nse", "student", True,
     "20260724"),
    # A variant token that merely looks like a tag must survive.
    ("arm1_combab", "arm1_combab", None, False, "20260722"),
])
def test_strip_tags(gm, arm, bare, target, copied, seed):
    assert gm.strip_tags(arm, "20260722") == (bare, target, copied, seed)


def test_the_control_and_the_379_copy_do_not_collide(gm):
    """Same arm, same target, different code. If the tags collapsed, the
    control's number would overwrite the number it is compared against."""
    ctl = gm.strip_tags("arm5_alignstudent", "20260722")
    cp = gm.strip_tags("arm5_alignstudent379", "20260722")
    assert ctl[:2] == cp[:2]
    assert ctl[2] is False and cp[2] is True


def test_seeds_are_read_from_the_scripts_not_typed(gm):
    """A constant typed into the table can drift from what ran. These come
    off the launchers, and the reader fails loudly if they disagree."""
    bb_seed, head_seed = gm.read_seeds()
    assert bb_seed == "20260520"
    assert head_seed == "20260722"
    assert f"SEED={bb_seed}" in (
        REPO_ROOT / "experiments/2026-07-21_split_pred_rep_small/scripts"
        / "run_arm.sh").read_text()
    assert f"SEED={bb_seed}" in (
        REPO_ROOT / "experiments/2026-08-01_lalign_teacher/scripts"
        / "run_arm.sh").read_text()


# --- the committed table -------------------------------------------------

@pytest.fixture(scope="module")
def table_rows():
    path = RESULTS / "gm_relative_mase.csv"
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_table_has_the_review_columns(table_rows):
    for col in ("align_target", "bb_seed", "head_seed"):
        assert col in table_rows[0], f"gm_relative_mase.csv has no {col}"


def test_every_teacher_cell_has_a_student_counterpart(table_rows):
    """The report's central comparison. Without the pair the delta cannot
    be rebuilt from this file, which is what review item 6 asked for."""
    teacher = {(r["arm_slug"], r["bb_steps"]) for r in table_rows
               if r["align_target"] == "teacher"
               and r["head_seed"] == "20260722"}
    student = {(r["arm_slug"], r["bb_steps"]) for r in table_rows
               if r["align_target"] == "student"}
    assert teacher, "no teacher cells in the table"
    assert not (teacher - student), (
        f"teacher cells with no student counterpart: {sorted(teacher - student)}")


def test_arms_without_an_align_term_are_marked_none(table_rows):
    """arm1 / arm3 / arm4 / bimoco carry no L_align term, so naming a
    target for them would invent a distinction that does not exist."""
    for r in table_rows:
        base = r["arm_slug"].split("_")[0]
        if base in ("arm1", "arm3", "arm4", "bimoco"):
            assert r["align_target"] == "none", r
        else:
            assert r["align_target"] in ("teacher", "student"), r


def test_every_cell_is_over_97_configs(table_rows):
    assert {r["n_configs"] for r in table_rows} == {"97"}


def test_table_reproduces_from_the_committed_results(tmp_path, table_rows):
    """Rebuild it and diff. A table that cannot be regenerated from the
    committed CSVs is not evidence."""
    out = tmp_path / "rebuilt.csv"
    res = subprocess.run([sys.executable, str(GM_TABLE), str(RESULTS),
                          str(out)], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert out.read_text() == (RESULTS / "gm_relative_mase.csv").read_text()


# --- the bootstrap -------------------------------------------------------

def test_cluster_bootstrap_resamples_whole_groups(boot):
    """Two datasets, four configs each, opposite signs. Resampling whole
    datasets can only ever produce means of -1, 0 or +1; a config-level
    resample produces the values in between. That difference is the whole
    point of clustering."""
    d = np.array([-1.0] * 4 + [1.0] * 4)
    groups = np.array(["a"] * 4 + ["b"] * 4)
    rng = np.random.default_rng(0)
    ratios = boot.cluster_bootstrap(d, groups, 500, rng)
    means = np.log(ratios)
    assert set(np.round(means, 9)) <= {-1.0, 0.0, 1.0}, (
        "cluster bootstrap produced a mean no whole-dataset resample can give")
    simple = np.log(boot.simple_bootstrap(d, 500, rng))
    assert not set(np.round(simple, 9)) <= {-1.0, 0.0, 1.0}, (
        "the config-level bootstrap should mix within a dataset")


def test_dataset_key_is_the_first_path_component(boot):
    assert boot.dataset_of("electricity/15T/short") == "electricity"
    assert boot.dataset_of("bizitobs_l2c/5T/long") == "bizitobs_l2c"


def test_ratio_equals_the_quotient_of_the_two_aggregates(boot):
    """exp(mean of paired per-config log differences) is GM_t / GM_s
    exactly, because the seasonal-naive denominator is shared per config.
    That identity is what lets the CI be read as an interval on the ratio."""
    configs = [f"ds{i}/1H/short" for i in range(20)]
    rng = np.random.default_rng(1)
    t = {c: float(v) for c, v in zip(configs, rng.uniform(0.5, 3, 20))}
    s = {c: float(v) for c, v in zip(configs, rng.uniform(0.5, 3, 20))}
    n = {c: float(v) for c, v in zip(configs, rng.uniform(0.5, 3, 20))}
    d = np.array([math.log(t[c]) - math.log(s[c]) for c in configs])
    gm_t = boot.gm_relative(t, n, configs)
    gm_s = boot.gm_relative(s, n, configs)
    assert math.exp(d.mean()) == pytest.approx(gm_t / gm_s, rel=1e-12)


@pytest.fixture(scope="module")
def ci_rows():
    with open(RESULTS / "eval_bootstrap_ci.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_committed_cis_are_dataset_clustered(ci_rows):
    """28 base datasets behind 97 configs. If n_datasets came out as 97 the
    clustering silently did nothing."""
    assert ci_rows, "no CI rows committed"
    for r in ci_rows:
        assert r["n_configs"] == "97"
        assert r["n_datasets"] == "28"


def test_dataset_level_intervals_are_the_wider_ones(ci_rows):
    """The reason for clustering, stated as a check: treating correlated
    configs as independent draws cannot give a wider interval."""
    wider = sum(float(r["ci95_width_dataset_boot"])
                >= float(r["ci95_width_config_boot"]) for r in ci_rows)
    assert wider >= 0.8 * len(ci_rows), (
        f"only {wider}/{len(ci_rows)} dataset-level intervals are the wider "
        "one — the clustering is probably not doing what it claims")


def test_paired_tests_cover_both_measured_waves():
    with open(RESULTS / "eval_paired_tests.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["bb_steps"] for r in rows] == ["40000", "100000"], (
        "the paired test must run at both backbone steps where all 10 arms "
        "have a teacher and a student cell — and not at 200k, where the two "
        "sides are different selections.")
    for r in rows:
        assert r["n_arms"] == "10"


# --- the code snapshot ---------------------------------------------------

def test_table_names_the_code_snapshot(table_rows):
    """Without this column a reader cannot tell a controlled delta from a
    cross-experiment one, and the two differ by 0.0977 on arm5 alone."""
    assert "code_snapshot" in table_rows[0]
    assert {r["code_snapshot"] for r in table_rows} == {"#379-sweep",
                                                        "#390-branch"}


def test_code_snapshot_agrees_with_source(table_rows):
    """A copied cell was measured by #379's code and never re-run; a cell
    measured here ran on this branch. The two columns say the same thing
    from different sides, so a disagreement means one of them is wrong."""
    for r in table_rows:
        expect = "#379-sweep" if r["source"] == "#379" else "#390-branch"
        assert r["code_snapshot"] == expect, r


def test_both_student_columns_survive_for_the_controlled_cells(table_rows):
    """The issue asks for #379's number; only the same-branch number
    supports a statement about the flag. Reporting one in place of the
    other is the failure this guards: at every arm/step that has a
    same-branch student control, #379's row must still be present."""
    student = {}
    for r in table_rows:
        if r["align_target"] != "student":
            continue
        student.setdefault((r["arm_slug"], r["bb_steps"]), set()).add(
            r["code_snapshot"])
    controlled = [k for k, v in student.items() if "#390-branch" in v]
    assert controlled, "no same-branch student control in the table at all"
    for key in controlled:
        assert "#379-sweep" in student[key], (
            f"{key}: the same-branch student control replaced #379's number "
            "instead of being added beside it")


def test_the_control_is_not_filed_as_the_teacher(table_rows):
    """`_alignstudent` and a bare cell name differ by one tag. If the tag
    were dropped the control would be counted as a teacher measurement and
    the delta would come out zero."""
    for r in table_rows:
        if "_alignstudent" in r["cell"]:
            assert r["align_target"] == "student", r


# --- the controlled delta ------------------------------------------------

def test_paired_boot_resamples_whole_datasets(ctl):
    """Same argument as the cluster-bootstrap test above, on the paired
    (delta, ratio) draw: two datasets, so only three distinct resamples of
    whole datasets exist and the ratio can only take three values."""
    configs = [f"a/1H/c{i}" for i in range(4)] + [f"b/1H/c{i}" for i in range(4)]
    naive = {c: 1.0 for c in configs}
    t = {c: (math.e if c.startswith("a") else 1.0) for c in configs}
    s = {c: (1.0 if c.startswith("a") else math.e) for c in configs}
    rng = np.random.default_rng(0)
    delta, ratio = ctl.paired_cluster_boot(t, s, naive, configs, 400, rng)
    assert set(np.round(np.log(ratio), 9)) <= {-1.0, 0.0, 1.0}, (
        "a draw that mixes configs within a dataset got through")
    # Delta and ratio must describe the SAME draw, or the CSV's two
    # intervals would be about different resamples.
    assert np.all((delta < 0) == (ratio < 1.0))


def test_partial_table_is_refused_without_the_flag(tmp_path):
    """The headline table is all ten cells. A run that quietly wrote three
    would read as the whole comparison."""
    res = subprocess.run(
        [sys.executable, str(CONTROLLED),
         "--results", str(RESULTS),
         "--student-379-results", str(RESULTS_379),
         "--out-delta", str(tmp_path / "d.csv"),
         "--out-tests", str(tmp_path / "t.csv"),
         "--n-boot", "50"],
        capture_output=True, text=True)
    if res.returncode == 0:
        # All ten controls have landed; then it must have written ten rows.
        with open(tmp_path / "d.csv", newline="") as fh:
            assert len(list(csv.DictReader(fh))) == 10
    else:
        assert res.returncode == 2, res.stdout + res.stderr
        assert "REFUSING to write" in res.stdout
        assert not (tmp_path / "d.csv").exists()


def test_only_40k_can_be_controlled(tmp_path):
    """Nothing was re-run past step 40 000 with the student target, so a
    100k controlled delta cannot exist. The flag refuses rather than
    silently falling back to #379's student cell."""
    res = subprocess.run(
        [sys.executable, str(CONTROLLED),
         "--results", str(RESULTS),
         "--student-379-results", str(RESULTS_379),
         "--bb-steps-k", "100", "--head-steps", "30000",
         "--out-delta", str(tmp_path / "d.csv"),
         "--out-tests", str(tmp_path / "t.csv")],
        capture_output=True, text=True)
    assert res.returncode != 0
    assert "invalid choice" in res.stderr


# --- which replicate #379 published -------------------------------------

@pytest.fixture(scope="module")
def prov():
    return load(PROVENANCE, "cf390_replicate_provenance")


@pytest.fixture(scope="module")
def prov_rows():
    path = RESULTS / "replicate_provenance_40k.csv"
    if not path.is_file():
        pytest.skip("replicate_provenance_40k.csv not built yet")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("filename,tag", [
    ("bb_small_arm5_lalign_lrep_x_r3_40k.pth", "r3"),
    ("bb_small_arm5_lalign_lrep_x_r12_40k.pth", "r12"),
    ("bb_small_arm5_lalign_lrep_x_40k.pth", "r1"),
    # `_r2` must be the run-name suffix, not any `r<N>` inside the arm token.
    ("bb_small_arm5_r2rep_lalign_lrep_x_40k.pth", "r1"),
])
def test_replicate_tag(prov, filename, tag):
    assert prov.replicate_tag(filename) == tag


def test_agreement_counts_only_exact_steps(prov):
    a = {s: {"loss": "1.0", "ff": "0.5"} for s in (1, 2, 3)}
    b = {s: {"loss": "1.0", "ff": "0.5"} for s in (1, 2, 3)}
    assert prov.agreement(a, b) == (3, 3, 0.0)
    b[2] = {"loss": "1.25", "ff": "0.5"}
    ident, shared, worst = prov.agreement(a, b)
    assert (ident, shared) == (2, 3) and worst == pytest.approx(0.25)


def test_agreement_ignores_steps_past_the_snapshot(prov):
    """A replicate that ran past 40 000 must not be scored on steps the 40k
    checkpoint never saw."""
    a = {1: {"loss": "1.0"}, prov.BB_STEPS + 1: {"loss": "1.0"}}
    b = {1: {"loss": "1.0"}, prov.BB_STEPS + 1: {"loss": "9.0"}}
    assert prov.agreement(a, b) == (1, 1, 0.0)


def test_every_arm_repeats_a_379_replicate(prov_rows):
    """The whole provenance reading rests on this: the branch re-run is a
    step-for-step repeat of one of #379's own runs. An arm matching none
    would mean the training path itself moved."""
    assert len(prov_rows) == 10
    for r in prov_rows:
        assert r["branch_matches_replicate"] != "none", r


def test_the_one_arm_that_misses_is_the_one_published_off_a_resume(prov_rows):
    """arm5's 0.0977 is a comparison of two different backbones, not a
    re-run of one. If a future pass makes some other arm miss, or makes
    arm5 miss while its published row came from the base run, this reading
    is wrong and the report must not keep asserting it."""
    missed = {r["arm_slug"] for r in prov_rows if r["reproduces"] == "no"}
    resumed = {r["arm_slug"] for r in prov_rows
               if r["replicate_used_379"] != "r1"}
    assert missed == resumed, (
        f"arms that miss #379's number: {missed}; arms published off a "
        f"resumed replicate: {resumed}")


def test_the_resumed_replicate_is_a_genuinely_different_run(prov_rows):
    """Same flags and same backbone seed, but the branch repeats r1 exactly
    while the published row used another replicate — so the two numbers
    differ because they are two backbones, not because one path is noisy."""
    for r in prov_rows:
        if r["replicate_used_379"] == "r1":
            continue
        assert r["branch_matches_replicate"] == "r1", r
        assert float(r["snapshot_diff"]) != 0.0, r


# --- the head-seed bar ---------------------------------------------------

SPREAD_COLS = ("arm_slug,variant,align_target,code_snapshot,bb_steps,"
               "head_steps,bb_seed,head_seed,cell,gm_rel_mase,n_configs,"
               "source")


def _spread_table(path, cells):
    """cells: (arm, target, bb_steps, head_seed, value)"""
    with open(path, "w") as fh:
        fh.write(SPREAD_COLS + "\n")
        for arm, target, bb, seed, val in cells:
            fh.write(f"{arm},base,{target},#390-branch,{bb},15000,20260520,"
                     f"{seed},{arm}_c{seed},{val},97,#390\n")


def _run_spread(tmp_path, cells, extra=()):
    table = tmp_path / "gm.csv"
    out = tmp_path / "spread.csv"
    _spread_table(table, cells)
    res = subprocess.run(
        [sys.executable, str(SPREAD), "--table", str(table),
         "--out", str(out), *extra],
        capture_output=True, text=True)
    return res, out


def test_seed_spread_keys_on_the_align_target(tmp_path):
    """At backbone 40k the teacher and the student side of one arm are two
    different backbones. Grouped on (arm, step) alone they would merge into
    one row and report a spread that is really a teacher-student gap."""
    res, out = _run_spread(tmp_path, [
        ("arm5", "teacher", 40000, "20260722", "1.3515"),
        ("arm5", "teacher", 40000, "20260723", "1.3550"),
        ("arm5", "student", 40000, "20260722", "1.4501"),
        ("arm5", "student", 40000, "20260723", "1.4530"),
    ])
    assert res.returncode == 0, res.stdout + res.stderr
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2, rows
    assert {r["align_target"] for r in rows} == {"teacher", "student"}
    for r in rows:
        assert float(r["range"]) < 0.01, r


def test_the_bar_is_read_off_40k_cells(tmp_path):
    """The whole point of measuring at 40k: a 200k cell's range must not
    become the bar for a 40k delta."""
    res, _ = _run_spread(tmp_path, [
        ("arm5", "teacher", 40000, "20260722", "1.3515"),
        ("arm5", "teacher", 40000, "20260723", "1.3560"),
        ("arm5_nse", "teacher", 200000, "20260722", "1.7979"),
        ("arm5_nse", "teacher", 200000, "20260723", "1.8887"),
    ])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "largest 40k head-seed range: 0.0045" in res.stdout, res.stdout
    assert "0.0908" not in res.stdout.split("backbone 40000")[1]


def test_missing_40k_replicate_is_said_out_loud(tmp_path):
    """A bar carried across from 100k/200k is the failure this guards. With
    no 40k replicate the script must refuse to imply one exists."""
    res, _ = _run_spread(tmp_path, [
        ("arm5_nse", "teacher", 200000, "20260722", "1.7979"),
        ("arm5_nse", "teacher", 200000, "20260723", "1.8887"),
    ])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "NO 40k CELL CARRIES A REPLICATE HEAD SEED" in res.stdout


def test_committed_spread_measures_both_sides_at_40k():
    """The controlled comparison is teacher against student at backbone 40k,
    so the bar has to exist on both sides of it."""
    path = RESULTS / "seed_spread.csv"
    if not path.is_file():
        pytest.skip("seed_spread.csv not built yet")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    at40 = [r for r in rows if int(r["bb_steps"]) == 40000]
    if not at40:
        pytest.skip("the 40k head-seed replicates have not landed yet")
    assert {r["align_target"] for r in at40} == {"teacher", "student"}, at40
    for r in at40:
        assert int(r["n_seeds"]) >= 3, r
