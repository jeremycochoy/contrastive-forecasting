"""#414 and #415: the value-space reference against the best contrastive model."""
from radar_lib import STUDY, draw
from run_style import MOIRAI, P, colour, line

# The best contrastive run against the value-space reference with the Moirai
# schedule. The flat-rate run trained without that schedule, so it stays out.
ARMS = [("cyan665", "Ours, 665k   1.1369", colour(P + "_cos200k"), line(P + "_cos200k")),
        ("moirai_native166k", "Moirai, own head, old data, 166k   1.2072",
         colour("cf415_moirai_native"), MOIRAI)]
TITLE = ("Relative MASE per GIFT-Eval dataset (geometric mean over its configs), old data\n"
         "Solid line, Ours: our contrastive model. Dashed line, Moirai: our copy of Moirai.\n"
         "Green ring: seasonal naive (1.0). Inside is better. The hardest datasets are at the top.")

draw(ARMS, TITLE, STUDY / "plots/gm_mase_radar_value.png")
print("radar written")
