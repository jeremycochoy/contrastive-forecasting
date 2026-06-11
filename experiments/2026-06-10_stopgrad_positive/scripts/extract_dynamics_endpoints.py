#!/usr/bin/env python3
"""Late-training endpoint values quoted in the report's dynamics section.

For each arm, the mean over the final 120 logged steps (the right edge of the
plot's 120-step MA) of the metrics quoted in stopgrad_positive.md →
results/dynamics_endpoints.csv. Same source CSVs as plot_training_metrics.py.
"""
import csv

EXP = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-10_stopgrad_positive"
ARMS = [
    ("reference", f"{EXP}/results/reference/bb_allt08_xftrip_nobn_enc3_qk_aon_b1024_losses.csv"),
    ("stop_grad", f"{EXP}/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_losses.csv"),
]
METRICS = ["loss", "ff", "fp", "cross_batch", "r2_naive", "r2_random",
           "u_batch", "u_temporal", "auc", "top1"]
WINDOW = 120
OUT = "/tmp/cf-sgpos/experiments/2026-06-10_stopgrad_positive/results/dynamics_endpoints.csv"

vals = {}
for arm, path in ARMS:
    rows = list(csv.DictReader(open(path)))[-WINDOW:]
    vals[arm] = {m: sum(float(r[m]) for r in rows) / len(rows) for m in METRICS}
    vals[arm]["last_step"] = float(rows[-1]["step"])

with open(OUT, "w") as f:
    w = csv.writer(f)
    f.write(f"# mean over the final {WINDOW} logged steps (right edge of the "
            "plot's 120-step MA); loss is floor-subtracted as logged\n")
    w.writerow(["metric"] + [a for a, _ in ARMS])
    for m in METRICS + ["last_step"]:
        w.writerow([m] + [f"{vals[a][m]:.4f}" for a, _ in ARMS])
print(open(OUT).read())
