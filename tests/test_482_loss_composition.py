"""#482: the loss composition figures of the paper ablations
(reports/2026-11-27_paper_ablations/scripts/plot_loss_composition.py).

The trainer logs one row per step in ``<run>_losses.csv``, and each resume
writes a new leg (``<run>_r2_losses.csv``, ...). The tests write small CSVs
with the columns of the trainer in temp folders.
"""
import csv
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (REPO_ROOT / "reports" / "2026-11-27_paper_ablations" / "scripts"
          / "plot_loss_composition.py")
HEADER = ["step", "loss", "sigreg_e", "sigreg_h", "rep_w", "l_pred", "l_rep",
          "l_align", "cos_err_d0", "cos_err_d1"]
# The weight of each logged column in the loss, and its value in the CSVs.
WEIGHTS = {"l_rep": 0.1, "l_align": 10.0, "sigreg_e": 1.0, "sigreg_h": 1.0}
VALUES = {"l_rep": 14.0, "l_align": 0.3, "sigreg_e": 0.2, "sigreg_h": 0.1}
FAR = 0.5  # the weighted L_align at distances 2 to 4: no column


def load():
    spec = importlib.util.spec_from_file_location("plot_loss_composition_482",
                                                  SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot = load()


def write_leg(path, steps, scale=1.0, drop=(), rollout=True):
    """A losses CSV with the VALUES times ``scale`` at each step. Each column
    in ``drop`` is empty, as the trainer logs a part that a row removes.
    With ``rollout`` False, the CSV has no cos_err_d<j> column and no FAR
    term, as a run at rollout depth 0."""
    values = {c: v * scale for c, v in VALUES.items() if c not in drop}
    loss = sum(WEIGHTS[c] * v for c, v in values.items())
    row = {"loss": loss + (FAR * scale if rollout else 0.0),
           "rep_w": 0.0 if "l_rep" in drop else 0.1, "l_pred": "",
           "cos_err_d0": 0.4, "cos_err_d1": 0.5,
           **{c: values.get(c, "") for c in VALUES}}
    header = HEADER if rollout else [c for c in HEADER if "cos_err" not in c]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for step in steps:
            writer.writerow([step if c == "step" else row[c] for c in header])
    return path


def test_the_parts_add_up_to_the_total(tmp_path):
    steps, parts = plot.read_parts([write_leg(tmp_path / "a_losses.csv",
                                              range(1, 3001))])
    want = {"align_dist1": 3.0, "align_dist2to4": FAR, "sigreg_e": 0.2,
            "sigreg_h": 0.1, "rep": 1.4, "total": 5.2}
    assert set(parts) == set(want)
    for name, value in want.items():
        np.testing.assert_allclose(parts[name], value)
    centres, means = plot.binned(steps, parts)
    np.testing.assert_allclose(sum(means[p] for p in plot.PARTS),
                               means["total"])


def test_a_run_at_rollout_depth_0_logs_each_part(tmp_path):
    # Row 15: no L_align at distances 2 to 4, so no part is a remainder.
    path = write_leg(tmp_path / "a_losses.csv", range(1, 3001), rollout=False)
    _, parts = plot.read_parts([path])
    assert "align_dist2to4" not in parts
    np.testing.assert_allclose(sum(parts[p] for p in plot.PARTS if p in parts),
                               parts["total"])


def test_the_join_of_two_legs_drops_the_earlier_rows_from_the_later_start(
        tmp_path):
    # The first leg logs to step 80. The resume logs again from its save at 50.
    write_leg(tmp_path / "a_losses.csv", range(1, 81))
    write_leg(tmp_path / "a_r2_losses.csv", range(51, 101), scale=10.0)
    steps, parts = plot.read_parts(plot.loss_legs(tmp_path, "a"))
    assert steps.tolist() == list(range(1, 101))
    np.testing.assert_allclose(parts["align_dist1"][steps <= 50], 3.0)
    np.testing.assert_allclose(parts["align_dist1"][steps > 50], 30.0)


def test_the_legs_go_in_the_order_of_their_first_steps(tmp_path):
    # The reference: r3 from 40, r4 from 45, then a box move named its new
    # leg r2 again (from 50). A different run in the folder is not a leg.
    for leg, first, last in [("a", 1, 36), ("a_r3", 41, 46), ("a_r4", 46, 52),
                             ("a_r2", 51, 167), ("a_cd135", 1, 9)]:
        write_leg(tmp_path / f"{leg}_losses.csv", range(first, last))
    legs = plot.loss_legs(tmp_path, "a")
    assert [p.name for p in legs] == ["a_losses.csv", "a_r3_losses.csv",
                                      "a_r4_losses.csv", "a_r2_losses.csv"]
    steps, _ = plot.read_parts(legs)
    assert steps.tolist() == list(range(1, 36)) + list(range(41, 167))


@pytest.mark.parametrize("drop, rollout, part", [
    (("sigreg_e",), True, "sigreg_e"),  # row 12
    (("sigreg_h",), True, "sigreg_h"),  # row 13
    (("l_rep",), True, "rep"),  # row 14
    ((), False, "align_dist2to4"),  # row 15
])
def test_a_row_with_no_part_of_a_kind_draws_no_area_for_it(
        tmp_path, drop, rollout, part):
    path = write_leg(tmp_path / "a_losses.csv", range(1000, 5001), drop=drop,
                     rollout=rollout)
    centres, means = plot.binned(*plot.read_parts([path]))
    assert part not in means
    fig, ax = plt.subplots()
    plot.draw_stack(ax, centres, means, top=10.0)
    drawn = {c.get_gid() for c in ax.collections}
    plt.close(fig)
    assert drawn == set(plot.PARTS) - {part}
    np.testing.assert_allclose(sum(means[p] for p in drawn), means["total"])


def test_a_negative_part_goes_below_zero():
    centres = np.array([2e3, 2e4, 1e5])
    means = {"total": np.full(3, 3.0), "align_dist1": np.full(3, 4.0),
             "rep": np.full(3, -1.0)}
    layers = {part: (low, high) for part, low, high in plot.stack_layers(means)}
    np.testing.assert_allclose(layers["align_dist1"], [np.zeros(3), np.full(3, 4.0)])
    np.testing.assert_allclose(layers["rep"], [np.full(3, -1.0), np.zeros(3)])
    low, high = plot.y_range({"a": (centres, means)})
    assert low == pytest.approx(-1.05) and high == pytest.approx(4.2)


def test_the_y_range_ignores_the_warmup_and_a_clipped_panel_gives_the_peak():
    centres = np.array([2e3, 2e4, 1e5])
    tall = {"total": np.array([40.0, 20.0, 18.0])}
    low_run = {"total": np.array([10.0, 12.0, 11.0])}
    for means in (tall, low_run):
        means["align_dist1"] = means["total"]
    low, high = plot.y_range({"a": (centres, tall), "b": (centres, low_run)})
    assert low == 0.0 and high == pytest.approx(21.0)
    notes = []
    for means in (tall, low_run):
        fig, ax = plt.subplots()
        plot.draw_stack(ax, centres, means, high)
        notes.append([t.get_text() for t in ax.texts])
        plt.close(fig)
    assert notes == [["peak 40.0"], []]


def test_main_draws_partial_runs_and_writes_the_table(tmp_path):
    ref, root, out = tmp_path / "ref", tmp_path / "runs", tmp_path / "out"
    ref.mkdir()
    write_leg(ref / f"{plot.REF}_losses.csv", range(1, 166_001, 50))
    (root / plot.REF2 / "ckpt").mkdir(parents=True)
    write_leg(root / plot.REF2 / "ckpt" / f"{plot.REF2}_losses.csv", range(1, 30_001, 50))
    removed = {12: (("sigreg_e",), True), 13: (("sigreg_h",), True),
               14: (("l_rep",), True), 15: ((), False)}
    for row, short, *_ in plot.ROWS:
        name = plot.run_name(row, short)
        (root / name / "ckpt").mkdir(parents=True)
        drop, rollout = removed.get(row, ((), True))
        write_leg(root / name / "ckpt" / f"{name}_losses.csv",
                  range(1, 60_001, 50), drop=drop, rollout=rollout)
    plot.main(["--ref-ckpt", str(ref), "--runs-root", str(root),
               "--out", str(out)])
    for png in ("loss_composition.png", "loss_small_terms.png"):
        assert (out / "plots" / png).stat().st_size > 10_000
    with open(out / "results" / "loss_parts_binned.tsv") as f:
        table = list(csv.DictReader(f, delimiter="\t"))
    names = [plot.REF, plot.REF2] + [plot.run_name(r, s) for r, s, *_ in plot.ROWS]
    assert list(dict.fromkeys(line["run"] for line in table)) == names
    row1 = [line for line in table if line["run"] == names[2]]
    edges = np.geomspace(plot.FIRST, plot.LAST, plot.BINS + 1)
    low = edges[edges <= 59_951].max()  # the last bin holds the last steps
    assert float(row1[-1]["bin_centre_step"]) == pytest.approx(
        np.sqrt(low * 59_951), abs=0.5)
    for row, part in [(12, "sigreg_e"), (13, "sigreg_h"), (14, "rep"),
                      (15, "align_dist2to4")]:
        lines = [line for line in table if line["run"] == names[row + 1]]
        assert lines and all(line[part] == "" for line in lines)
        assert all(line["align_dist1"] != "" for line in lines)
