#!/usr/bin/env python3
"""#322 — batch-1024 collapse at LR 1e-3 and 5e-4 (5e-4 only delays it).

Left: contrastive loss MINUS the InfoNCE floor log(1+N·e^(−1/τ)) — rebased so ~0 = the
uniformity floor, and comparable across batch sizes / τ (which set different floors).
Right: gap (forecast↔future margin). b1024 runs parsed from the (appended) training log
[1e-3 then 5e-4 sections]; b256 from #320's CSV; τ=0.20 probe from its smoke CSV.
"""
import csv
import math
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WT = "/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024"
sys.path.insert(0, WT)
try:
    from src.loss import infonce_floor, _effective_negative_count

    def floor_of(B, tau):
        N = _effective_negative_count("cosine_similarity_batch_full_hh_negs", B, 64, 1)
        if N is None:
            N = B * (B + 63)
        return infonce_floor(tau, N)
except Exception:
    def floor_of(B, tau):
        N = B * (B + 63)
        return math.log1p(N * math.exp(-1.0 / tau))

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
LOG = f"{ROOT}/2026-05-29_forked_6Lf_b1024/results/run_bb_beta_forked2_6Lf_b1024.log"
B256_CSV = f"{ROOT}/2026-05-26_forked_6Lf/runs/bb_beta_forked2_6Lf_50k_losses.csv"
TAU_CSV = f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/smoke_beta_b1024_losses.csv"
OUT = f"{WT}/experiments/2026-05-29_forked_6Lf_b1024/plots/collapse_handling.png"

STEP_RE = re.compile(r"^\[\s*(\d+)\]\s+loss=([-\d.]+).*?gap=([-\d.]+)")
LR_RE = re.compile(r"lr=([\d.eE+-]+)")


def parse_log_runs(path):
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
    s, l, g = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            s.append(int(float(row["step"]))); l.append(float(row["loss"])); g.append(float(row["gap"]))
    return s, l, g


runs = {lr: pts for lr, pts in parse_log_runs(LOG) if pts}
b256 = read_csv_curve(B256_CSV)
# series: (label, steps, loss, gap, color, B, tau, already_floor_subtracted)
series = [("b256, lr 1e-3 (#320, stable)", *b256, "#2ca02c", 256, 0.10, False)]
div = runs.get("0.001") or runs.get("1e-3")
fix = runs.get("0.0005") or runs.get("5e-4")
if div:
    series.append(("b1024 lr 1e-3 (collapses ~4500)", [p[0] for p in div], [p[1] for p in div],
                   [p[2] for p in div], "#d62728", 1024, 0.10, False))
if fix:
    series.append(("b1024 lr 5e-4 (collapses ~6000)", [p[0] for p in fix], [p[1] for p in fix],
                   [p[2] for p in fix], "#1f77b4", 1024, 0.10, False))
if os.path.exists(TAU_CSV):
    ts, tl, tg = read_csv_curve(TAU_CSV)
    if ts:
        series.append((f"b1024 lr5e-4 τ=0.20 (collapses ~5700)", ts, tl, tg,
                       "#9467bd", 1024, 0.20, False))
# QK-norm red line (lr 1e-3 + --qk-norm). Loss is ALREADY floor-subtracted (run uses
# --subtract-contrastive-floor) → already=True.
QK_CSV = f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/bb_beta_forked2_qknorm_6Lf_b1024_losses.csv"
if os.path.exists(QK_CSV):
    qs, ql, qg = read_csv_curve(QK_CSV)
    if qs:
        series.append((f"b1024 lr1e-3 +QK-norm (collapses)", qs, ql, qg,
                       "#ff7f0e", 1024, 0.10, True))
AON_CSV = f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/bb_beta_forked2_aon_6Lf_b1024_losses.csv"
if os.path.exists(AON_CSV):
    a_s, a_l, a_g = read_csv_curve(AON_CSV)
    if a_s:
        series.append((f"b1024 lr1e-3 +attn-out-only (logits→NaN)", a_s, a_l, a_g,
                       "#17becf", 1024, 0.10, True))
QKAON_CSV = f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/bb_beta_forked2_qk_aon_6Lf_b1024_losses.csv"
if os.path.exists(QKAON_CSV):
    z_s, z_l, z_g = read_csv_curve(QKAON_CSV)
    if z_s:
        series.append((f"b1024 lr1e-3 +QK-norm+attn-out (@step {z_s[-1]})", z_s, z_l, z_g,
                       "#000000", 1024, 0.10, True))

fig, (axL, axG) = plt.subplots(1, 2, figsize=(14, 5.5))
for label, s, l, g, c, B, tau, already in series:
    fl = 0.0 if already else floor_of(B, tau)
    pts = [(x, y - fl, gg) for x, y, gg in zip(s, l, g) if x and x > 0 and (y - fl) > 0]
    xs = [p[0] for p in pts]
    axL.plot(xs, [p[1] for p in pts], "-", color=c, label=label, lw=1.8)
    axG.plot([p[0] for p in zip(s, l, g) if p[0] > 0],
             [gg for x, _, gg in zip(s, l, g) if x > 0], "-", color=c, label=label, lw=1.8)

axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlabel("training step (log)"); axL.set_ylabel("contrastive loss − InfoNCE floor (log)")
axL.set_title("Loss above the floor — log-log")
axL.grid(True, which="both", alpha=0.3); axL.legend(fontsize=8)
axG.set_xscale("log"); axG.axhline(0, color="k", lw=0.8, ls=":")
axG.set_xlabel("training step (log)"); axG.set_ylabel("gap (forecast↔future margin)")
axG.set_title("Representation health (gap)")
axG.grid(True, which="both", alpha=0.3); axG.legend(fontsize=8)
fig.suptitle("#322 β·0.8% b1024: LR 1e-3 / 5e-4 / τ=0.20 / QK-norm all collapse — attn-out-RMSNorm (cyan) holds so far (early; loss rebased to floor)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")
for label, s, l, g, c, B, tau, already in series:
    print(f"  floor(B={B},τ={tau}) = {floor_of(B, tau):.3f}  | {label}: loss {l[0]:.2f}->{l[-1]:.2f}")
