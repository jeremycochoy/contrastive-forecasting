"""#482: the figures of the paper ablations
(reports/2026-11-27_paper_ablations/scripts/plot_ablations.py).

The script reads one score file per run and stop, in the layout of the
follower (``<ff>/results/score_<run>_bb<k>k_h30k_control.txt``), for the
reference and for the 15 rows. The tests write such files in temp folders.
"""
import csv
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "reports" / "2026-11-27_paper_ablations" / "scripts"
REFERENCE = {10: 0.9863, 25: 0.9510, 50: 0.9325, 75: 0.9336, 100: 0.9345,
             125: 0.9412, 150: 0.9260, 166: 0.9303}


def load():
    spec = importlib.util.spec_from_file_location("plot_ablations_482",
                                                  SCRIPTS / "plot_ablations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot = load()
RUNS = plot.load_rows()


def write_score(ff, run, k, value):
    path = ff / "results" / f"score_{run}_bb{k}k_h30k_control.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value:.4f}\n")


@pytest.fixture
def folders(tmp_path):
    """The reference at each stop, and row r at 0.93 + r / 1000 at each
    stop. Row 4 has scores up to 100k only."""
    ref, abl = tmp_path / "ref", tmp_path / "abl"
    for k, value in REFERENCE.items():
        write_score(ref, RUNS.REF, k, value)
    for row, short, *_ in RUNS.ROWS:
        for k in plot.STOPS:
            if row != 4 or k <= 100:
                write_score(abl, RUNS.run_name(row, short), k, 0.93 + row / 1000)
    return ref, abl


def test_the_table_has_each_row_at_each_stop_with_the_reference(folders):
    lines = plot.table(*folders)
    assert len(lines) == 15 * len(plot.STOPS)
    for line in lines:
        assert line["reference"] == REFERENCE[line["stop_k"]]
        want = (None if line["row"] == 4 and line["stop_k"] > 100
                else 0.93 + line["row"] / 1000)
        assert line["gm_mase"] == (pytest.approx(want) if want else None)


def test_a_missing_score_file_reads_as_none(tmp_path):
    assert plot.score(tmp_path, "nothing", 166) is None


def test_main_writes_the_table_and_the_two_figures(folders, tmp_path):
    out = tmp_path / "report"
    plot.main(["--ref-ff", str(folders[0]), "--ablation-ff", str(folders[1]),
               "--out", str(out)])
    with open(out / "results" / "ablation_gm_mase.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 120
    final = [r for r in rows if r["stop_k"] == "166"]
    assert [r["reference"] for r in final] == ["0.9303"] * 15
    assert final[3]["gm_mase"] == ""  # row 4 has no 166k score
    for name in ("ablation_gm_mase_166k.png", "ablation_gm_mase_steps.png"):
        assert (out / "plots" / name).stat().st_size > 10_000
