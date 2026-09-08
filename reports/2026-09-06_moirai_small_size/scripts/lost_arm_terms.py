"""The loss terms of the two lost arms, from their losses CSVs.

For each lost arm it prints, per column:
  - the 200-row rolling mean at step 2,000 and at the last logged step
  - the AUC peak (raw) and the first raw reading under 0.55 after warm-up
  - the step where the guard's 500-row median first went under 0.55

Writes results/lost_arm_terms.txt.
"""

from pathlib import Path

import pandas as pd

ROOT = Path("/home/jupyter/checkpoints_backup/cf-412")
ARMS = {
    "k32_r200_08": ROOT
    / "k32_r200_08/arm6_v2_combab_alignT/leg_40k"
    / "cf393_arm6_v2_combab_alignT_cf373k32_cf412_k32_r200_08_losses.csv",
    "k32_r100_09_dec": ROOT
    / "k32_r100_09_dec/arm6_v2_combab_alignT/leg_40k"
    / "cf393_arm6_v2_combab_alignT_cf373k32_cf412_k32_r100_09_dec_losses.csv",
}
COLUMNS = ["u_temporal", "u_batch", "l_align"]
WARMUP, THRESHOLD, WINDOW = 1000, 0.55, 500

out = []
for arm, path in ARMS.items():
    df = pd.read_csv(path)
    out.append(f"{arm}  ({len(df)} rows, last step {df.step.iloc[-1]})")
    for col in COLUMNS:
        smooth = df[col].rolling(200, min_periods=1).mean()
        start = smooth[df.step <= 2000].iloc[-1]
        out.append(f"  {col:11s} step 2,000: {start:.4f}   last: {smooth.iloc[-1]:.4f}")
    peak = df.auc.idxmax()
    out.append(f"  auc peak (raw)   {df.auc.max():.4f} at step {df.step[peak]}")
    warm = df[df.step >= WARMUP]
    raw = warm[warm.auc < THRESHOLD]
    out.append(f"  auc raw < {THRESHOLD}   first at step {raw.step.iloc[0]}")
    med = df.assign(med=df.auc.rolling(WINDOW).median())
    med = med[(med.step >= WARMUP) & med.med.notna() & (med.med < THRESHOLD)]
    out.append(f"  auc median < {THRESHOLD} (guard) first at step {med.step.iloc[0]}")

text = "\n".join(out) + "\n"
print(text, end="")
Path(__file__).resolve().parent.parent.joinpath(
    "results", "lost_arm_terms.txt"
).write_text(text)
