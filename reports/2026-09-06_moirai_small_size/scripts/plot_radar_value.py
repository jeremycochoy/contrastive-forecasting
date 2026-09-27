"""#414 and #415: the value-space reference against the best contrastive model."""
from radar_lib import STUDY, draw

# The best contrastive run against the value-space reference with the Moirai
# schedule. The flat-rate run trained without that schedule, so it stays out.
ARMS = [("cyan665",    "contrastive, cosine to 1e-6 by 200k, 665k   1.1369", "#17becf"),
        ("moirai_native166k", "value space, Moirai schedule, 166k, its own head   1.2072", "#8c564b")]
TITLE = ("Issues #414 and #415: the value-space reference against the best contrastive model\n"
         "Relative MASE per GIFT-Eval dataset, geometric mean over its configs.\n"
         "The green ring is 1.0, the seasonal-naive level. Inside it is better.")

draw(ARMS, TITLE, STUDY / "plots/gm_mase_radar_value.png")
print("radar written")
