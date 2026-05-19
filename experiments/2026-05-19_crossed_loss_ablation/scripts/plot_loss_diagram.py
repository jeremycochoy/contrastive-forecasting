"""Contrastive loss structure on a (time × batch) ladder.

Two batch lines (b, b′); each line carries an f (forecaster) and an h
(encoder) node at every time t-2..t+2. Horizontal = time, vertical =
batch. From a single anchor f_{b,t} we draw, deliberately sparse (cf.
the square-loss diagram — not every link):
  • positive            f_{b,t} → h_{b,t+1}
  • (A) f↔h  l≠t+1       the loss-of-record crossed negative
  • (B) h↔h  l≠t         encoder–encoder sibling
  • (C) f↔f  l≠t         forecaster–forecaster sibling
  • cross-batch b≠b′     f_{b,t} ↔ h_{b′,t+1}  (the batch axis)
Single channel — same-time cross-channel negatives are not drawn.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(11, 7.2))
ax.set_xlim(0.0, 6.4)
ax.set_ylim(0.0, 9.6)
ax.axis("off")

TI = ["t-2", "t-1", "t", "t+1", "t+2"]
X = {tt: 1.4 + 1.05 * i for i, tt in enumerate(TI)}
# four rows: batch b group (f over h, top), batch b' group (bottom)
Y = {"fb": 8.2, "hb": 6.6, "fB": 3.0, "hB": 1.4}
ANCH, POS = "t", "t+1"

C_POS = "#2a9d2a"   # green  positive
C_FH  = "#c22020"   # red    (A) f↔h
C_HH  = "#1a6ec2"   # blue   (B) h↔h
C_FF  = "#d07000"   # orange (C) f↔f
C_XB  = "#7a3fb3"   # purple cross-batch
F_FACE, H_FACE = "#d0e8ff", "#ffd0d0"

N = {}
for tt in TI:
    for row in Y:
        N[(row, tt)] = (X[tt], Y[row])

def link(a, b, color, lw, rad, head=False, alpha=0.9, ls="-"):
    x0, y0 = N[a]; x1, y1 = N[b]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>" if head else "-",
                                color=color, lw=lw, alpha=alpha,
                                linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}"))

# ── (C) f↔f  within the f_b row, l≠t (arc above) ──
for tt, r in (("t-1", 0.32), ("t+2", -0.32)):
    link(("fb", ANCH), ("fb", tt), C_FF, 1.9, r)
# ── (B) h↔h  within the h_b row, l≠t (arc below) ──
for tt, r in (("t-1", -0.32), ("t+2", 0.32)):
    link(("hb", ANCH), ("hb", tt), C_HH, 1.9, r)
# ── (A) f↔h  f_{b,t} → h_{b,l}, l≠t+1 ──
for tt, r in (("t-1", 0.10), ("t+2", -0.12)):
    link(("fb", ANCH), ("hb", tt), C_FH, 1.9, r)
# ── cross-batch (batch axis): f_{b,t} ↔ h_{b′,t+1}, b≠b′ ──
link(("fb", ANCH), ("hB", POS), C_XB, 1.9, -0.06, ls=(0, (5, 3)))
# ── positive: f_{b,t} → h_{b,t+1} ──
link(("fb", ANCH), ("hb", POS), C_POS, 3.0, 0.0, head=True)
# faint: batch b′ shares the identical structure (its own positive)
link(("fB", ANCH), ("hB", POS), C_POS, 2.0, 0.0, head=True, alpha=0.28)

# ── nodes ──
for (row, tt), (x, y) in N.items():
    face = F_FACE if row[0] == "f" else H_FACE
    edge, lw = "#333", 1.5
    if row == "fb" and tt == ANCH:
        edge, lw = "#d07000", 3.0          # anchor f_{b,t}
    if row == "hb" and tt == POS:
        edge, lw = C_POS, 3.0              # positive target h_{b,t+1}
    ax.scatter(x, y, s=520, zorder=5, color=face,
               edgecolors=edge, linewidths=lw)

# node labels (math), batch subscript
SUB = {"fb": "f_{b,%s}", "hb": "h_{b,%s}",
       "fB": "f_{b',%s}", "hB": "h_{b',%s}"}
for (row, tt), (x, y) in N.items():
    dy = 0.42 if row[0] == "f" else -0.42
    va = "bottom" if row[0] == "f" else "top"
    ax.text(x, y + dy, rf"${SUB[row] % tt}$", ha="center", va=va,
            fontsize=10, fontweight="bold")

# row / axis guides
ax.text(0.15, (Y["fb"] + Y["hb"]) / 2, "batch  b", rotation=90,
        ha="center", va="center", fontsize=10, color="#555",
        fontweight="bold")
ax.text(0.15, (Y["fB"] + Y["hB"]) / 2, "batch  b'", rotation=90,
        ha="center", va="center", fontsize=10, color="#999",
        fontweight="bold")
ax.annotate("", xy=(0.62, 1.0), xytext=(0.62, 8.6),
            arrowprops=dict(arrowstyle="<->", color="#ccc", lw=1.3))
ax.text(0.92, 4.9, "batch axis", rotation=90, ha="center", va="center",
        fontsize=8.5, color="#bbb", style="italic")
ax.annotate("", xy=(6.05, 0.55), xytext=(1.1, 0.55),
            arrowprops=dict(arrowstyle="->", color="#ccc", lw=1.2))
ax.text(6.0, 0.78, "time", ha="right", fontsize=8.5, color="#bbb",
        style="italic")

ax.set_title(
    "Contrastive loss — positive + crossed negatives over time & batch\n"
    "(A) fₜ↔hₗ  ·  (B) hₜ↔hₗ  ·  (C) fₜ↔fₗ   (single channel; "
    "same-time cross-channel negatives omitted)",
    fontsize=11, pad=8)

leg = [
    mpatches.Patch(color=C_POS, label=r"positive  $f_{b,t}\!\to\!h_{b,t+1}$"),
    mpatches.Patch(color=C_FH,  label=r"(A) $f_t\!\leftrightarrow\!h_\ell$,"
                                      r" $\ell\neq t{+}1$"),
    mpatches.Patch(color=C_HH,  label=r"(B) $h_t\!\leftrightarrow\!h_\ell$,"
                                      r" $\ell\neq t$"),
    mpatches.Patch(color=C_FF,  label=r"(C) $f_t\!\leftrightarrow\!f_\ell$,"
                                      r" $\ell\neq t$"),
    mpatches.Patch(color=C_XB,  label=r"cross-batch  $f_{b,t}\!\leftrightarrow\!"
                                      r"h_{b',t+1}$, $b\neq b'$"),
]
ax.legend(handles=leg, loc="lower center", ncol=3, fontsize=8.6,
          framealpha=0.95, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.13))
ax.text(0.5, -0.205,
        "Links drawn from one anchor $f_{b,t}$ for clarity (the loss sums "
        "them over all anchors). A = loss-of-record; B, C = the siblings; "
        "cross-batch is shared by every arm.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.2,
        color="#777")

out = ("/home/jupyter/cf-wt-crossed-loss/experiments/"
       "2026-05-19_crossed_loss_ablation/plots/loss_diagram.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved → {out}")
