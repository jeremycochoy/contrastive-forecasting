"""Write the closing report (owner, 10-06): the figures, then one table, and no
other text. The best score and its step come from results/gm_trajectories.tsv
(scripts/gm_trajectories.py writes it), so a new score updates the table.

Usage: python3 scripts/make_report.py   (after scripts/make_plots.sh)
"""
from collections import defaultdict
from pathlib import Path

from run_style import CODE, P

STUDY = Path(__file__).resolve().parent.parent
OUT = STUDY / "moirai_small_size.md"
TITLE = "# Our contrastive model and our copy of Moirai on GIFT-Eval"
FIGURES = ["gm_mase_rates", "gm_mase_rates_all", "gm_mase_rates_ours_one_patch_size",
           "gm_mase_rates_ours_patch_sizes", "gm_mase_rates_moirai", "gm_mase_radar_moirai_top2",
           "gm_mase_radar_ours_top", "lr_schedules", "loss_terms_412om_vs_cyan"]

# The recipes of the lr figure.
MOIRAI = "lr 1e-3, warmup over 10k, cosine to 0 at 166k"
CYAN = "lr cosine 5.6e-5 to 1e-6 by 200k, then 1e-6"
WARMUP = "lr 0 to 1e-3 over 10k, straight line to 5.6e-5 at 20k, then 5.6e-5"
# Ours: the contrastive objective (L_rep with MoCo keys, L_align on the teacher,
# SIGReg, rollout depth 3). The L_rep weight goes from 1 to 0 by 10k unless the
# row says otherwise. Moirai: our copy, trained on the values.
# run: (configuration, batch size, training recipe)
RUNS = {
    P + "_lr10x": ("Ours, one patch size, EWMA, old data", 64, "lr 5.6e-5"),
    P + "_lr10xb": ("As ABC, second seed", 64, "lr 5.6e-5"),
    P + "_lr30x": ("Ours, one patch size, EWMA, old data", 64, "lr 1.8e-5"),
    P + "_lr100x": ("Ours, one patch size, EWMA, old data", 64, "lr 5.6e-6"),
    P + "_cos665k": ("Ours, one patch size, EWMA, old data", 64, "lr cosine 6e-5 to 1e-6 over 665k"),
    P + "_cos200k": ("Ours, one patch size, EWMA, old data", 64, "lr cosine 5e-5 to 1e-6 by 200k, then 1e-6"),
    "cf419_cos200k": ("Ours, one patch size, EWMA, new data", 64, CYAN),
    "cf419ms": ("As BLK, with mean/std in place of EWMA", 64, CYAN),
    "cf412om": ("Ours, patch sizes 8 to 128, mean/std, new data, loss bug", 256, MOIRAI),
    "cf412oc": ("Ours, patch sizes 8 to 128, mean/std, new data, loss bug", 64, CYAN),
    "cf412oc2": ("Ours, patch sizes 8 to 128, mean/std, new data", 64, CYAN),
    "cf412oe2": ("Ours, patch sizes 8 to 128, EWMA, new data", 64, CYAN),
    "cf412om2": ("Ours, patch sizes 8 to 128, mean/std, new data", 256, MOIRAI),
    "cf412oa2": ("Ours, patch sizes 8 to 128, mean/std, new data", 64, "lr 5.6e-5"),
    "cf412ow2": ("Ours, patch sizes 8 to 128, mean/std, new data", 64, WARMUP),
    "cf412or2": ("As OWF, with the L_rep weight at 1 for the whole run", 64, WARMUP),
    "cf412ol2": ("As OWR", 64, "As OWF, with the straight line to 5.6e-5 at 40k"),
    "cf412bm": ("As OMF, with L_pred and L_rep (MoCo, tau 1, no L_align), "
                "the L_rep weight at 1", 256, MOIRAI),
    "cf412bw": ("As OBM, with the L_rep weight from 1 to 0 by 10k", 256, WARMUP),
    "cf412al": ("As OAF", 256, "lr 5.6e-5"),
    "cf419_moirai_native": ("Moirai, its own head, EWMA, new data", 256, MOIRAI),
    "cf415_moirai_native": ("Moirai, its own head, EWMA, old data", 256, MOIRAI),
    "cf421f_moirai_native": ("Moirai, patch heads, mean/std, new data", 256, MOIRAI),
    "cf421ew_moirai_native": ("Moirai, patch heads, EWMA, new data", 256, MOIRAI),
    "cf421n_moirai_native": ("As MPM, with a term on the RMS of each patch", 256, MOIRAI),
}


def best_scores():
    """{run: (best GM-Relative MASE, its step in thousands)}."""
    points = defaultdict(list)
    for line in open(STUDY / "results" / "gm_trajectories.tsv"):
        run, stop_k, score = line.split()
        points[run].append((float(score), int(stop_k)))
    return {run: min(pts) for run, pts in points.items()}


def table():
    best = best_scores()
    rows = ["| Code | Configuration | Batch | Training recipe | Best GM-Relative MASE | Step of the best |",
            "|---|---|---|---|---|---|"]
    for run, (config, batch, recipe) in RUNS.items():
        if run in best:
            score, step = best[run]
            rows.append(f"| {CODE[run]} | {config} | {batch} | {recipe} | {score:.4f} | {step:,}k |")
    return "\n".join(rows)


def main():
    figures = "\n\n".join(f"![{name}](plots/{name}.png)" for name in FIGURES)
    OUT.write_text(f"{TITLE}\n\n{figures}\n\n{table()}\n")
    print(f"wrote {OUT}")


main()
