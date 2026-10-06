"""The two radars of the closing report (owner, 10-06): the two best Moirai
runs, and the four best runs of ours against the best Moirai run (MPM, red
dashed). Each run at its best checkpoint."""
from radar_lib import STUDY, draw
from run_style import MOIRAI, P, colour, line, tagged

MPM, MPE = "cf421f_moirai_native", "cf421ew_moirai_native"
MPM_ARM = ("moirai421f_native166k", tagged(MPM, "Moirai + patch heads + mean/std, new data, 166k   0.9250"),
           colour(MPM), MOIRAI)
MPE_ARM = ("moirai421ew_native166k", tagged(MPE, "Moirai + patch heads + EWMA, new data, 166k   0.9362"),
           colour(MPE), MOIRAI)
# The four best runs of ours. The per-config files of ABC and MIN are in traj/.
OURS = [("cos419_200k", "cf419_cos200k", "Ours, lr cosine by 200k, new data, 200k   1.1262"),
        ("cyan665", P + "_cos200k", "Ours, lr cosine by 200k, old data, 665k   1.1369"),
        ("../traj/blue_240", P + "_lr10x", "Ours, lr 5.6e-5, old data, 240k   1.1403"),
        ("../traj/yellow_1000", P + "_lr100x", "Ours, lr 5.6e-6, old data, 1,000k   1.1435")]
NOTE = ("Geometric mean over the configs of each dataset. Green ring: seasonal naive (1.0). "
        "Inside is better.\nSolid lines, Ours: our contrastive model. Dashed lines, Moirai: "
        "our copy of Moirai.")

draw([MPM_ARM, MPE_ARM],
     "Relative MASE per GIFT-Eval dataset: the two best Moirai runs\n" + NOTE.split("\n")[0],
     STUDY / "plots/gm_mase_radar_moirai_top2.png")
draw([(name, tagged(run, label), colour(run), line(run)) for name, run, label in OURS] + [MPM_ARM],
     "Relative MASE per GIFT-Eval dataset: the four best runs of ours, and MPM\n" + NOTE,
     STUDY / "plots/gm_mase_radar_ours_top.png")
print("radars written")
