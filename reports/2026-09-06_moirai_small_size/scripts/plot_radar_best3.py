"""The best two Moirai runs against the best of ours (owner, 10-02): three
curves, each run at its best checkpoint. The best of ours is CYN at 665k,
1.1369, on the old data; the two Moirai runs train on the new data."""
from radar_lib import STUDY, draw
from run_style import MOIRAI, P, colour, line, tagged

CYN = P + "_cos200k"
ARMS = [("cyan665", tagged(CYN, "Ours, lr cosine by 200k, old data, 665k   1.1369"),
         colour(CYN), line(CYN)),
        ("moirai421f_native166k",
         tagged("cf421f_moirai_native", "Moirai + patch heads + mean/std, new data, 166k   0.9250"),
         colour("cf421f_moirai_native"), MOIRAI),
        ("moirai421ew_native75k",
         tagged("cf421ew_moirai_native", "Moirai + patch heads + EWMA, new data, 75k   0.9625"),
         colour("cf421ew_moirai_native"), MOIRAI)]
TITLE = ("Relative MASE per GIFT-Eval dataset (geometric mean over its configs): "
         "the best of ours and the best two Moirai runs\n"
         "Solid line, Ours: our contrastive model. Dashed lines, Moirai: our copy of Moirai.\n"
         "New data: all of GiftEvalPretrain. Old data: its series of 4,096 points or more.\n"
         "Green ring: seasonal naive (1.0). Inside is better. The hardest datasets are at the top.")

draw(ARMS, TITLE, STUDY / "plots/gm_mase_radar_best3.png")
print("radar written")
