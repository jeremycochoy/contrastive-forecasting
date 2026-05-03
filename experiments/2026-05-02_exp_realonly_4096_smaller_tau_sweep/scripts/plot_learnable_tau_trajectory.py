"""Trajectory of the learnable τ (CLIP-style log_inv_tau) across training.

Parses lines like `[ 17200] loss=... τ=0.0587` from the local run.log of the
#32 learnable τ run and plots τ vs step. Adds horizontal reference lines at
the three fixed-τ values (0.05, 0.07, 0.20) used in #27 so the trajectory is
contextualized against the sweep range.

Caveat: the local run.log only retains τ samples from step ~17,100 onward
(the original run was on a vast.ai instance that was destroyed pre-resume).
We mark the missing range explicitly with a dashed line back to the init
value τ=0.07 at step 0.
"""
import os
import re
import matplotlib.pyplot as plt

ROOT = "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(os.path.dirname(HERE), "plots", "learnable_tau_trajectory.png")
LOG = f"{ROOT}/sync_realonly_4096_smaller_learnable_tau/learnable/run.log"

LINE_RE = re.compile(r"^\[\s*(\d+)\]\s+loss=.*τ=([0-9.]+)")


def parse_log(path):
    pts = []
    with open(path, "r") as f:
        for ln in f:
            m = LINE_RE.search(ln)
            if m:
                pts.append((int(m.group(1)), float(m.group(2))))
    pts.sort()
    return pts


pts = parse_log(LOG)
assert pts, "no τ samples found"
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]
first_step = xs[0]
last_step = xs[-1]

fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))

# Pre-data dashed segment from init (step 0, τ=0.07) → first observed (step ~17.1k).
ax.plot([0, first_step], [0.07, ys[0]], color="tab:red", linestyle=":", linewidth=1.0, alpha=0.6,
        label=f"unobserved (step 0..{first_step}, init τ=0.07)")
ax.plot(xs, ys, color="tab:red", linewidth=1.6, label=f"learnable τ (n={len(pts)} samples)")

# Reference lines for fixed τ used in the #27 sweep.
for fixed_tau, color, label in [(0.05, "tab:blue", "fixed τ=0.05"),
                                  (0.07, "tab:orange", "fixed τ=0.07 (init for learnable)"),
                                  (0.20, "tab:green", "fixed τ=0.20")]:
    ax.axhline(fixed_tau, color=color, linestyle="--", linewidth=1.0, alpha=0.7, label=label)

ax.scatter([0], [0.07], color="tab:red", marker="^", s=60, zorder=5, label="init point")
ax.scatter([last_step], [ys[-1]], color="tab:red", marker="o", s=60, zorder=5,
           label=f"end (step {last_step}, τ={ys[-1]:.4f})")

ax.set_xlabel("step")
ax.set_ylabel("τ")
ax.set_title("Learnable τ trajectory — #32 (init τ=0.07, log_inv_tau, clamped [0.01, 1.0])")
ax.set_xlim(left=-500)
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"Wrote {OUT_PNG}")
print(f"  range: step {first_step}..{last_step}, τ {ys[0]:.4f} → {ys[-1]:.4f} (Δ={ys[0]-ys[-1]:+.4f})")
