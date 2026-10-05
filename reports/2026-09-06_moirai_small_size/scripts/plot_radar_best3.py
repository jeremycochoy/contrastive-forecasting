"""The best two Moirai runs against the best of ours (owner, 10-02): three
curves, each run at its best checkpoint. The best of ours is BLK at 200k,
1.1262. All three runs train on the new data."""
from radar_lib import STUDY, draw
from run_style import MOIRAI, colour, line, tagged

BLK = "cf419_cos200k"
ARMS = [("cos419_200k", tagged(BLK, "Ours, lr cosine by 200k, new data, 200k   1.1262"),
         colour(BLK), line(BLK)),
        ("moirai421f_native166k",
         tagged("cf421f_moirai_native", "Moirai + patch heads + mean/std, new data, 166k   0.9250"),
         colour("cf421f_moirai_native"), MOIRAI),
        ("moirai421ew_native166k",
         tagged("cf421ew_moirai_native", "Moirai + patch heads + EWMA, new data, 166k   0.9362"),
         colour("cf421ew_moirai_native"), MOIRAI)]
TITLE = ("Relative MASE per GIFT-Eval dataset (geometric mean over its configs): "
         "the best of ours and the best two Moirai runs\n"
         "Solid line, Ours: our contrastive model. Dashed lines, Moirai: our copy of Moirai.\n"
         "New data: all of GiftEvalPretrain.\n"
         "Green ring: seasonal naive (1.0). Inside is better. The hardest datasets are at the top.")

draw(ARMS, TITLE, STUDY / "plots/gm_mase_radar_best3.png")
print("radar written")
