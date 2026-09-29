"""#419 and #421: each run on all of GiftEvalPretrain at its best checkpoint,
against the best run on the old data.

A stop enters once its score file is in results/ and its per-config CSV is in
results/per_config/radar. The colours are those of the curves figure.
"""
import re
from radar_lib import RADAR, STUDY, draw
from run_style import P, colour, line

# Run name in results/, CSV prefix in the radar folder, legend label. The
# colour and the line style come from run_style.py.
# The stopped runs cf419_moirai_native and cf421fb_moirai_native keep their
# CSVs in the radar folder; the owner took them off this figure.
NEW_DATA = [
    ("cf419_cos200k", "cos419_", "Ours"),
    ("cf421f_moirai_native", "moirai421f_native", "Moirai + patch heads + mean/std"),
    ("cf421n_moirai_native", "moirai421n_native", "Moirai + patch heads + mean/std + RMS term"),
]
OLD_BEST = ("cyan665", "Ours, old data, 665k   1.1369",
            colour(P + "_cos200k"), line(P + "_cos200k"))
TITLE = ("Relative MASE per GIFT-Eval dataset (geometric mean over its configs), each run at its best checkpoint\n"
         "Solid lines, Ours: our contrastive model. Dashed lines, Moirai: our copy of Moirai.\n"
         "New data: all of GiftEvalPretrain. Old data: its series of 4,096 points or more.\n"
         "Green ring: seasonal naive (1.0). Inside is better. The hardest datasets are at the top.")


def scored_stops(arm, csv_prefix):
    """{stop in k steps: GM-Relative MASE} for the stops with a score and a radar CSV."""
    out = {}
    for f in (STUDY / "results").glob(f"score_{arm}_bb*k_h30k_student.txt"):
        k = int(re.search(r"_bb(\d+)k_", f.name).group(1))
        if (RADAR / f"{csv_prefix}{k}k.csv").exists():
            out[k] = float(f.read_text().split()[0])
    return out


def new_data_arms():
    """The best scored stop of each new-data run, when it has one."""
    arms = []
    for arm, prefix, label in NEW_DATA:
        stops = scored_stops(arm, prefix)
        if stops:
            k = min(stops, key=stops.get)
            arms.append((f"{prefix}{k}k", f"{label}, new data, {k}k   {stops[k]:.4f}",
                         colour(arm), line(arm)))
    return arms


def print_table(arms, per_ds):
    """One row per dataset with the value of each arm. The first new-data arm's
    largest gains over the old best come first."""
    names = [a[0] for a in arms]
    old, new = per_ds[names[0]], per_ds[names[1]]
    print("dataset\t" + "\t".join(names))
    for d in sorted(new, key=lambda d: new[d] / old.get(d, new[d])):
        print(d + "\t" + "\t".join(f"{per_ds[n].get(d, float('nan')):.3f}" for n in names))


def main():
    new = new_data_arms()
    if not new:
        return print("radar 419: no new-data score yet")
    arms = [OLD_BEST] + new
    per_ds = draw(arms, TITLE, STUDY / "plots/gm_mase_radar_419.png")
    print_table(arms, per_ds)


if __name__ == "__main__":
    main()
