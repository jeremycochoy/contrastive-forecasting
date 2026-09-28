"""#414 and #415: the value-space reference against the best contrastive model."""
from radar_lib import STUDY, draw

# The best contrastive run against the value-space reference with the Moirai
# schedule. The flat-rate run trained without that schedule, so it stays out.
ARMS = [("cyan665",    "Ours, 665k   1.1369", "#17becf"),
        ("moirai_native166k", "Moirai, own head, 166k   1.2072", "#8c564b")]
TITLE = ("Relative MASE per GIFT-Eval dataset (geometric mean over its configs), old data\n"
         "Ours: our contrastive model. Moirai: our copy of Moirai.\n"
         "Green ring: seasonal naive (1.0). Inside is better. The hardest datasets are at the top.")

draw(ARMS, TITLE, STUDY / "plots/gm_mase_radar_value.png")
print("radar written")
