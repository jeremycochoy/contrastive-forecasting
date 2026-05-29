#!/usr/bin/env python3
"""#322 — visualise the batch-1024 collapses at BOTH LR 1e-3 and 5e-4 (5e-4 only delays it).

Three β·0.8% trajectories on shared axes:
  - #320 batch 256, LR 1e-3  (stable reference)
  - #322 batch 1024, LR 1e-3 (diverges ~step 4500: loss ↑, gap → 0)
  - #322 batch 1024, LR 5e-4 (the fix: gap stays high, no collapse)

Left panel: contrastive loss vs step, LOG-LOG. Right: the gap (forecast vs
future-step margin) vs step, log-x — the clearest view of the representational
collapse. Data: the b1024 runs are parsed from the (appended) training log
[1e-3 section then 5e-4 section]; b256 from #320's losses CSV.
"""
import csv
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
LOG = f"{ROOT}/2026-05-29_forked_6Lf_b1024/results/run_bb_beta_forked2_6Lf_b1024.log"
B256_CSV = f"{ROOT}/2026-05-26_forked_6Lf/runs/bb_beta_forked2_6Lf_50k_losses.csv"
OUT = ("/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/"
       "forked-6Lf-b1024/experiments/2026-05-29_forked_6Lf_b1024/plots/collapse_handling.png")

STEP_RE = re.compile(r"^\[\s*(\d+)\]\s+loss=([-\d.]+).*?gap=([-\d.]+)")
LR_RE = re.compile(r"lr=([\d.eE+-]+)")


def parse_log_runs(path):
    """Return list of (lr_str, [(step, loss, gap), ...]) split on 'Training for' lines."""
    runs, cur = [], None
    with open(path) as f:
        for line in f:
            if "Training for" in line:
                m = LR_RE.search(line)
                cur = (m.group(1) if m else "?", [])
                runs.append(cur)
                continue
            if cur is None:
                continue
            m = STEP_RE.match(line)
            if m:
                cur[1].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return runs


def read_csv_curve(path):
    steps, loss, gap = [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(float(row["step"])))
            loss.append(float(row["loss"]))
            gap.append(float(row["gap"]))
    return steps, loss, gap


runs = parse_log_runs(LOG)
# map by lr (first run with lr~0.001 = diverged, lr~0.0005 = fix)
by_lr = {}
for lr, pts in runs:
    if pts:
        by_lr.setdefault(lr, pts)
b256_s, b256_l, b256_g = read_csv_curve(B256_CSV)

fig, (axL, axG) = plt.subplots(1, 2, figsize=(14, 5.5))

series = [
    ("b256, lr 1e-3 (#320, stable)", b256_s, b256_l, b256_g, "#2ca02c", "-"),
]
# b1024 runs from log
diverged = by_lr.get("0.001") or by_lr.get("1e-3")
fix = by_lr.get("0.0005") or by_lr.get("5e-4")
if diverged:
    s = [p[0] for p in diverged]; l = [p[1] for p in diverged]; g = [p[2] for p in diverged]
    series.append(("b1024 lr 1e-3 (collapses ~step 4500)", s, l, g, "#d62728", "-"))
if fix:
    s = [p[0] for p in fix]; l = [p[1] for p in fix]; g = [p[2] for p in fix]
    series.append(("b1024 lr 5e-4 (collapses ~step 6000)", s, l, g, "#1f77b4", "-"))
# τ=0.20 probe (still running): read from its smoke losses CSV
TAU_CSV = f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/smoke_beta_b1024_losses.csv"
if os.path.exists(TAU_CSV):
    ts, tl, tg = read_csv_curve(TAU_CSV)
    if ts:
        series.append((f"b1024 lr 5e-4 τ=0.20 (probe, running, @step {ts[-1]})",
                       ts, tl, tg, "#9467bd", "-"))

for label, s, l, g, c, ls in series:
    # drop step 0 for log axes
    pts = [(x, y, gg) for x, y, gg in zip(s, l, g) if x and x > 0]
    xs = [p[0] for p in pts]
    axL.plot(xs, [p[1] for p in pts], ls, color=c, label=label, lw=1.8)
    axG.plot(xs, [p[2] for p in pts], ls, color=c, label=label, lw=1.8)

axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlabel("training step (log)"); axL.set_ylabel("contrastive loss (log)")
axL.set_title("Training loss — log-log")
axL.grid(True, which="both", alpha=0.3); axL.legend(fontsize=9)

axG.set_xscale("log")
axG.axhline(0, color="k", lw=0.8, ls=":")
axG.set_xlabel("training step (log)"); axG.set_ylabel("gap (forecast↔future margin)")
axG.set_title("Representation health (gap) — collapse vs handled")
axG.grid(True, which="both", alpha=0.3); axG.legend(fontsize=9)

fig.suptitle("#322 β·0.8% backbone: batch-1024 collapses at BOTH LR 1e-3 and 5e-4 (5e-4 only delays it)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")
# quick textual summary
for label, s, l, g, c, ls in series:
    if s:
        print(f"  {label}: steps {s[0]}..{s[-1]}, loss {l[0]:.2f}->{l[-1]:.2f}, gap {g[0]:.2f}->{g[-1]:.2f}")
