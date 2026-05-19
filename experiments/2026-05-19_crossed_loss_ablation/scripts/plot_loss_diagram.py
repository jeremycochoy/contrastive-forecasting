"""Contrastive loss structure on a (time × batch) ladder.

Two batch lines b (top) / b′ (bottom); each carries an f (forecaster)
and an h (encoder) node at every time t-2..t+2. Horizontal = time,
vertical = batch.

Arrow convention: positives are drawn ▶◀ (heads inward — *attract*);
every negative is drawn ◀▶ (heads at the extremities — *repel*).

  • positive            f_{·,τ} → h_{·,τ+1}   — ALL, both ladders
  • (A) f↔h  l≠t+1       full fan from one anchor f_{b,t}  (top ladder)
  • (B) h↔h  l≠t         full fan from one anchor h_{b′,t} (bottom)
  • (C) f↔f  l≠t         full fan from one anchor f_{b′,t} (bottom)
  • cross-batch b≠b′     a few links between the two ladders
Single channel — same-time cross-channel negatives are not drawn.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(15, 9.6))
ax.set_xlim(0.0, 7.0)
ax.set_ylim(0.0, 10.2)
ax.axis("off")

TI = ["t-2", "t-1", "t", "t+1", "t+2"]
IX = {tt: i for i, tt in enumerate(TI)}
X = {tt: 1.6 + 1.15 * i for i, tt in enumerate(TI)}
Y = {"fb": 8.7, "hb": 7.0, "fB": 3.1, "hB": 1.4}
ANCH, POS = "t", "t+1"

C_POS = "#2a9d2a"   # green  positive (inward)
C_FH  = "#c22020"   # red    (A) f↔h
C_HH  = "#1a6ec2"   # blue   (B) h↔h
C_FF  = "#d07000"   # orange (C) f↔f
C_XB  = "#7a3fb3"   # purple cross-batch
F_FACE, H_FACE = "#d0e8ff", "#ffd0d0"

N = {(r, tt): (X[tt], Y[r]) for tt in TI for r in Y}
SHR = 14  # keep arrowheads off the node discs


def neg(a, b, color, rad=0.0, lw=1.7):
    """◀▶ — heads at both extremities (repulsion)."""
    ax.annotate("", xy=N[b], xytext=N[a], zorder=2,
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=lw,
                                alpha=.82, shrinkA=SHR, shrinkB=SHR,
                                connectionstyle=f"arc3,rad={rad}",
                                mutation_scale=17))


def pos(a, b):
    """▶◀ — two heads meeting at the midpoint (attraction)."""
    (xa, ya), (xb, yb) = N[a], N[b]
    mid = ((xa + xb) / 2.0, (ya + yb) / 2.0)
    for end in (a, b):
        ax.annotate("", xy=mid, xytext=N[end], zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=2.4,
                                    alpha=.95, shrinkA=SHR, shrinkB=1,
                                    mutation_scale=18))


# ── ALL positives, both ladders: f_{·,τ} → h_{·,τ+1} ──
for fr, hr in (("fb", "hb"), ("fB", "hB")):
    for i in range(len(TI) - 1):
        pos((fr, TI[i]), (hr, TI[i + 1]))

# ── (A) full fan from f_{b,t}: f_{b,t} ↔ h_{b,l}, l ≠ t+1 (top) ──
for tt in TI:
    if tt == POS:
        continue
    rad = 0.16 * (IX[tt] - IX[ANCH])
    neg(("fb", ANCH), ("hb", tt), C_FH, rad=rad)

# ── (B) full fan from h_{b′,t}: h_{b′,t} ↔ h_{b′,l}, l ≠ t (bottom) ──
for tt in TI:
    if tt == ANCH:
        continue
    rad = -0.34 if IX[tt] < IX[ANCH] else 0.34
    neg(("hB", ANCH), ("hB", tt), C_HH, rad=rad)

# ── (C) full fan from f_{b′,t}: f_{b′,t} ↔ f_{b′,l}, l ≠ t (bottom) ──
for tt in TI:
    if tt == ANCH:
        continue
    rad = 0.34 if IX[tt] < IX[ANCH] else -0.34
    neg(("fB", ANCH), ("fB", tt), C_FF, rad=rad)

# ── a few cross-batch (b ≠ b′) ──
for a, b, r in (
    (("fb", "t"), ("hB", "t+1"), -0.05),
    (("hb", "t+1"), ("hB", "t+1"), 0.05),
    (("fb", "t"), ("fB", "t"), 0.10),
    (("hb", "t-1"), ("hB", "t-1"), -0.05),
):
    neg(a, b, C_XB, rad=r, lw=1.4)

# ── nodes ──
ANCHORS = {("fb", ANCH): C_FH, ("hB", ANCH): C_HH, ("fB", ANCH): C_FF}
for (r, tt), (x, y) in N.items():
    face = F_FACE if r[0] == "f" else H_FACE
    edge, lw = "#333", 1.4
    if (r, tt) in ANCHORS:
        edge, lw = ANCHORS[(r, tt)], 3.0
    ax.scatter(x, y, s=620, zorder=5, color=face, edgecolors=edge,
               linewidths=lw)

SUB = {"fb": "f_{b,%s}", "hb": "h_{b,%s}",
       "fB": "f_{b',%s}", "hB": "h_{b',%s}"}
for (r, tt), (x, y) in N.items():
    dy = 0.40 if r[0] == "f" else -0.40
    va = "bottom" if r[0] == "f" else "top"
    ax.text(x, y + dy, rf"${SUB[r] % tt}$", ha="center", va=va,
            fontsize=10, fontweight="bold")

ax.text(0.18, (Y["fb"] + Y["hb"]) / 2, "batch  b", rotation=90,
        ha="center", va="center", fontsize=10, color="#555",
        fontweight="bold")
ax.text(0.18, (Y["fB"] + Y["hB"]) / 2, "batch  b'", rotation=90,
        ha="center", va="center", fontsize=10, color="#999",
        fontweight="bold")
ax.annotate("", xy=(0.62, 1.0), xytext=(0.62, 9.1),
            arrowprops=dict(arrowstyle="<->", color="#ccc", lw=1.3))
ax.text(0.92, 5.05, "batch axis", rotation=90, ha="center", va="center",
        fontsize=8.5, color="#bbb", style="italic")
ax.annotate("", xy=(6.7, 0.55), xytext=(1.2, 0.55),
            arrowprops=dict(arrowstyle="->", color="#ccc", lw=1.2))
ax.text(6.65, 0.78, "time", ha="right", fontsize=8.5, color="#bbb",
        style="italic")

ax.set_title(
    "Contrastive loss — positive + crossed negatives over time & batch\n"
    "(A) fₜ↔hₗ  ·  (B) hₜ↔hₗ  ·  (C) fₜ↔fₗ   "
    "(single channel; same-time cross-channel negatives omitted)",
    fontsize=11, pad=8)

leg = [
    mpatches.Patch(color=C_POS, label=r"positive ▶◀ (attract)  "
                                      r"$f_{\cdot,\tau}\!\to\!h_{\cdot,\tau+1}$"),
    mpatches.Patch(color=C_FH,  label=r"(A) ◀▶  $f_t\!\leftrightarrow\!h_\ell$,"
                                      r" $\ell\neq t{+}1$  (top)"),
    mpatches.Patch(color=C_HH,  label=r"(B) ◀▶  $h_t\!\leftrightarrow\!h_\ell$,"
                                      r" $\ell\neq t$  (bottom)"),
    mpatches.Patch(color=C_FF,  label=r"(C) ◀▶  $f_t\!\leftrightarrow\!f_\ell$,"
                                      r" $\ell\neq t$  (bottom)"),
    mpatches.Patch(color=C_XB,  label=r"cross-batch ◀▶  $b\neq b'$"),
]
ax.legend(handles=leg, loc="lower center", ncol=3, fontsize=8.6,
          framealpha=0.95, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.12))
ax.text(0.5, -0.185,
        "Negatives ◀▶ push apart; the positive ▶◀ pulls together. Each "
        "crossed family is fanned from one anchor for clarity (the loss "
        "sums over all anchors); A is the loss-of-record, B/C the siblings.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.2,
        color="#777")

out = ("/home/jupyter/cf-wt-crossed-loss/experiments/"
       "2026-05-19_crossed_loss_ablation/plots/loss_diagram.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved → {out}")
