"""#419: the runs on all of GiftEvalPretrain against the best run on the old data.

The contrastive arm shows its newest scored stop. The value-space arm shows its
best native stop. A stop enters once its score file is in results/ and its
per-config CSV is in results/per_config/radar.
"""
import re
from radar_lib import RADAR, STUDY, draw

OLD_BEST = ("cyan665", "contrastive, cosine to 1e-6 by 200k, 665k, old data   1.1369", "#17becf")
TITLE = ("Issue #419: the runs on the new data against the best run on the old data\n"
         "New data: all of GiftEvalPretrain. Old data: only its series of 4,096 points or more.\n"
         "Relative MASE per GIFT-Eval dataset, geometric mean over its configs.\n"
         "The green ring is 1.0, the seasonal-naive level. Inside it is better.")


def scored_stops(arm, csv_prefix):
    """{stop in k steps: GM-Relative MASE} for the stops with a score and a radar CSV."""
    out = {}
    for f in (STUDY / "results").glob(f"score_{arm}_bb*k_h30k_student.txt"):
        k = int(re.search(r"_bb(\d+)k_", f.name).group(1))
        if (RADAR / f"{csv_prefix}{k}k.csv").exists():
            out[k] = float(f.read_text().split()[0])
    return out


def new_data_arms():
    """The newest contrastive stop and the best value-space stop, when they exist."""
    arms, cos = [], scored_stops("cf419_cos200k", "cos419_")
    moirai = scored_stops("cf419_moirai_native", "moirai419_native")
    if cos:
        k = max(cos)
        arms.append((f"cos419_{k}k", f"contrastive, cosine to 1e-6 by 200k, {k}k, new data   {cos[k]:.4f}", "#1f77b4"))
    if moirai:
        k = min(moirai, key=moirai.get)
        arms.append((f"moirai419_native{k}k", f"value space, Moirai schedule, {k}k, its own head, new data   {moirai[k]:.4f}", "#8c564b"))
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
