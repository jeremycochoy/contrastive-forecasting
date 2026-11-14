"""#462: the scores of the width-64 GM-Relative MASE graph
(reports/2026-11-14_moirai_width64/scripts/plot_width64_gm_mase.py).

``panel_scores`` gives the GM of one per-config table over the 97 configs and
over each horizon. ``table_scores`` reads the table of each MPE stop that has
one. ``hp64_scores`` reads the table of each stop of a width-64 run that has a
score file, and compares the GM with that file. The tests make the tables and
the score files in temp folders, in the layout of ~/cf_runs/hp64/ff.
"""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (REPO_ROOT / "reports" / "2026-11-14_moirai_width64" / "scripts"
          / "plot_width64_gm_mase.py")
# The 97 GIFT-Eval configs: 55 short, 21 medium and 21 long.
SIZES = {"short": 55, "medium": 21, "long": 21}
CONFIGS = [f"d{i}/H/{h}" for h, n in SIZES.items() for i in range(n)]
NAIVE = {c: 1.0 + i % 7 for i, c in enumerate(CONFIGS)}
# The MASE of the model divided by the MASE of Seasonal Naive, by horizon.
EARLY = {"short": 0.5, "medium": 2.0, "long": 0.8}
LATE = {"short": 0.25, "medium": 1.5, "long": 0.6}


def load_script():
    spec = importlib.util.spec_from_file_location("plot_width64_gm_mase", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot = load_script()


def mase_of(ratios):
    """``{config: MASE}``, with the ratio ``ratios[h]`` for each config of
    the horizon ``h``."""
    return {c: ratios[c.rsplit("/", 1)[1]] * NAIVE[c] for c in CONFIGS}


def expected(ratios):
    """The panel scores of ``mase_of(ratios)``."""
    logs = sum(n * math.log(ratios[h]) for h, n in SIZES.items())
    return {None: math.exp(logs / len(CONFIGS)), **ratios}


def write_table(path, mase):
    """A per-config table in the layout of all_results.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model", "eval_metrics/MASE[0.5]", "domain"])
        for config, value in mase.items():
            writer.writerow([config, "m", value, "Energy"])


def hp64_stop(ff, run, stop, ratios, score):
    """The files of one stop of a width-64 run: the table of its eval and its
    score file. ``run`` is the part of the name after ``_sphere``. ``score``
    None: no score file."""
    name = (f"g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere{run}"
            f"_bb{stop}k_h30k_control")
    write_table(ff / "heads" / "eval" / name / "gift" / "all_results.csv",
                mase_of(ratios))
    if score is not None:
        (ff / "results").mkdir(parents=True, exist_ok=True)
        (ff / "results" / f"score_{name}.txt").write_text(score)


def score_text(ratios):
    """A score file as hp64 writes it: the GM to 4 decimals."""
    return f"{expected(ratios)[None]:.4f}\n"


def test_panel_scores_gives_the_gm_over_all_configs_and_each_horizon():
    assert plot.panel_scores(mase_of(EARLY), NAIVE) == pytest.approx(
        expected(EARLY))


def test_a_table_without_the_97_configs_is_an_error():
    mase = mase_of(EARLY)
    del mase[CONFIGS[0]]
    with pytest.raises(ValueError, match="96 configs"):
        plot.panel_scores(mase, NAIVE)


def test_table_scores_reads_the_table_of_each_stop_that_has_one(tmp_path):
    pattern = str(tmp_path / "mpe_native{}k.csv")
    write_table(Path(pattern.format(10)), mase_of(EARLY))
    write_table(Path(pattern.format(166)), mase_of(LATE))
    scores = plot.table_scores(pattern, NAIVE)
    assert set(scores) == {10, 166}
    assert scores[10] == pytest.approx(expected(EARLY))
    assert scores[166] == pytest.approx(expected(LATE))


def test_table_scores_gives_no_stop_before_the_first_table(tmp_path):
    assert plot.table_scores(str(tmp_path / "new_native{}k.csv"), NAIVE) == {}


def test_hp64_scores_reads_each_stop_that_has_a_score(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "FF", tmp_path)
    hp64_stop(tmp_path, "_sigf_rollfix", 0, EARLY, score_text(EARLY))
    hp64_stop(tmp_path, "_sigf_rollfix", 10, LATE, score_text(LATE))
    hp64_stop(tmp_path, "_sigf_rollfix", 25, LATE, "")
    hp64_stop(tmp_path, "_sigf_rollfix", 50, LATE, None)
    scores = plot.hp64_scores("SIGReg-F", "", [0, 10, 25, 50], NAIVE)
    assert set(scores) == {0, 10}
    assert scores[0] == pytest.approx(expected(EARLY))
    assert scores[10] == pytest.approx(expected(LATE))


def test_hp64_scores_reads_the_files_of_the_cooldown(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "FF", tmp_path)
    hp64_stop(tmp_path, "_align10_rollfix", 150, EARLY, score_text(EARLY))
    hp64_stop(tmp_path, "_align10_rollfix_cdsqrt135", 150, LATE,
              score_text(LATE))
    scores = plot.hp64_scores("10-L-ALIGN", "_cdsqrt135", [150, 166], NAIVE)
    assert set(scores) == {150}
    assert scores[150] == pytest.approx(expected(LATE))


def test_a_score_file_that_does_not_agree_with_its_table_is_an_error(
        tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "FF", tmp_path)
    hp64_stop(tmp_path, "_rollfix", 10, EARLY,
              f"{expected(EARLY)[None] + 2e-4:.4f}\n")
    with pytest.raises(AssertionError, match="sphere_rollfix_bb10k"):
        plot.hp64_scores("base", "", [10], NAIVE)
