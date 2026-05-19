"""#307 loss diagram — the 3 NEW cross-branch ablation arms on a
(time × batch) ladder, side-by-side. Same primitives as #303 (A/B/C +
cross-batch) so the new arms are read as colored *combinations*.

Three panels:
  1. (B)+(C)    full_hh_ff_negs      — blue h↔h + orange f↔f, full purple xb
  2. (A)+(B)+(C) full_fh_hh_ff_negs  — red + blue + orange, full purple xb
  3. (B) xbfree full_hh_negs_xbf     — blue h↔h, purple xb keeps h↔h+f↔f,
     **drops f↔h cross-batch** (the structural twist).

Same arrow convention as #303: positives ▶◀ inward (attract), negatives
◀▶ outward (repel).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

C_POS = "#2a9d2a"
C_FH  = "#c22020"
C_HH  = "#1a6ec2"
C_FF  = "#d07000"
C_XB  = "#7a3fb3"   # cross-batch (kept arms)
C_XBD = "#cccccc"   # cross-batch DROPPED (xbfree only)
F_FACE, H_FACE = "#d0e8ff", "#ffd0d0"

TI = ["t-2", "t-1", "t", "t+1", "t+2"]
IX = {tt: i for i, tt in enumerate(TI)}
X = {tt: 1.4 + 1.05 * i for i, tt in enumerate(TI)}
Y = {"fb": 8.5, "hb": 6.9, "fB": 3.1, "hB": 1.5}
ANCH, POS = "t", "t+1"
SHR = 13

fig, axes = plt.subplots(1, 3, figsize=(22.5, 8.2))


def draw(ax, *, use_fh, use_hh, use_ff, xb_dropped_fh, title, subtitle):
    ax.set_xlim(0.0, 7.0); ax.set_ylim(0.0, 10.0); ax.axis("off")
    N = {(r, tt): (X[tt], Y[r]) for tt in TI for r in Y}

    def neg(a, b, color, rad=0.0, lw=1.6, alpha=.82, style="<|-|>"):
        ax.annotate("", xy=N[b], xytext=N[a], zorder=2,
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    alpha=alpha, shrinkA=SHR, shrinkB=SHR,
                                    connectionstyle=f"arc3,rad={rad}",
                                    mutation_scale=15))

    def pos(a, b):
        (xa, ya), (xb, yb) = N[a], N[b]
        mid = ((xa + xb) / 2, (ya + yb) / 2)
        for end in (a, b):
            ax.annotate("", xy=mid, xytext=N[end], zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=C_POS,
                                        lw=2.2, alpha=.95, shrinkA=SHR,
                                        shrinkB=1, mutation_scale=16))

    # All positives, both ladders
    for fr, hr in (("fb", "hb"), ("fB", "hB")):
        for i in range(len(TI) - 1):
            pos((fr, TI[i]), (hr, TI[i + 1]))

    anchors = {}
    if use_fh:  # (A) f↔h, l≠t+1, top
        for tt in TI:
            if tt == POS:
                continue
            neg(("fb", ANCH), ("hb", tt), C_FH,
                rad=0.15 * (IX[tt] - IX[ANCH]))
        anchors[("fb", ANCH)] = C_FH
    if use_hh:  # (B) h↔h, l≠t, bottom
        for tt in TI:
            if tt == ANCH:
                continue
            rad = -0.34 if IX[tt] < IX[ANCH] else 0.34
            neg(("hB", ANCH), ("hB", tt), C_HH, rad=rad)
        anchors[("hB", ANCH)] = C_HH
    if use_ff:  # (C) f↔f, l≠t, bottom
        for tt in TI:
            if tt == ANCH:
                continue
            rad = 0.34 if IX[tt] < IX[ANCH] else -0.34
            neg(("fB", ANCH), ("fB", tt), C_FF, rad=rad)
        anchors[("fB", ANCH)] = C_FF

    # Cross-batch (b≠b′) — sample, between ladders
    xb_links = [
        # (a, b, rad, kind)
        (("fb", "t"),   ("hB", "t+1"), -0.05, "fh"),  # f↔h cross-batch
        (("hb", "t+1"), ("fB", "t"),    0.05, "fh"),  # f↔h cross-batch
        (("hb", "t-1"), ("hB", "t-1"), -0.05, "hh"),  # h↔h cross-batch
        (("fb", "t"),   ("fB", "t"),    0.10, "ff"),  # f↔f cross-batch
    ]
    for a, b, r, kind in xb_links:
        if xb_dropped_fh and kind == "fh":
            neg(a, b, C_XBD, rad=r, lw=1.0, alpha=.45, style="-")
        else:
            neg(a, b, C_XB, rad=r, lw=1.3)

    # Nodes
    for (r, tt), (x, y) in N.items():
        face = F_FACE if r[0] == "f" else H_FACE
        edge, lw = "#333", 1.3
        if (r, tt) in anchors:
            edge, lw = anchors[(r, tt)], 3.0
        ax.scatter(x, y, s=540, zorder=5, color=face, edgecolors=edge,
                   linewidths=lw)
    SUB = {"fb": "f_{b,%s}", "hb": "h_{b,%s}",
           "fB": "f_{b',%s}", "hB": "h_{b',%s}"}
    for (r, tt), (x, y) in N.items():
        dy = 0.36 if r[0] == "f" else -0.36
        va = "bottom" if r[0] == "f" else "top"
        ax.text(x, y + dy, rf"${SUB[r] % tt}$", ha="center", va=va,
                fontsize=8.6, fontweight="bold")
    ax.text(0.18, (Y["fb"] + Y["hb"]) / 2, "b", rotation=90,
            ha="center", va="center", fontsize=10, color="#555",
            fontweight="bold")
    ax.text(0.18, (Y["fB"] + Y["hB"]) / 2, "b'", rotation=90,
            ha="center", va="center", fontsize=10, color="#999",
            fontweight="bold")
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=6)


draw(axes[0], use_fh=False, use_hh=True, use_ff=True,
     xb_dropped_fh=False,
     title="(B)+(C)  full_hh_ff_negs",
     subtitle="all-time h↔h + f↔f within each branch · full cross-batch")
draw(axes[1], use_fh=True, use_hh=True, use_ff=True,
     xb_dropped_fh=False,
     title="(A)+(B)+(C)  full_fh_hh_ff_negs",
     subtitle="all three within-batch fans · full cross-batch")
draw(axes[2], use_fh=False, use_hh=True, use_ff=False,
     xb_dropped_fh=True,
     title="(B) cross-branch-free  full_hh_negs_xbf",
     subtitle="all-time h↔h ·  NO f↔h anywhere (positive retained)")

leg = [
    mpatches.Patch(color=C_POS, label=r"positive ▶◀  $f_{\cdot,\tau}\to h_{\cdot,\tau+1}$"),
    mpatches.Patch(color=C_FH,  label=r"(A) $f_t\!\leftrightarrow\!h_\ell$, $\ell\neq t{+}1$"),
    mpatches.Patch(color=C_HH,  label=r"(B) $h_t\!\leftrightarrow\!h_\ell$, $\ell\neq t$"),
    mpatches.Patch(color=C_FF,  label=r"(C) $f_t\!\leftrightarrow\!f_\ell$, $\ell\neq t$"),
    mpatches.Patch(color=C_XB,  label=r"cross-batch ◀▶  $b\neq b'$ (kept)"),
    mpatches.Patch(color=C_XBD, label=r"cross-batch  $f\!\leftrightarrow\!h$  DROPPED (xbfree only)"),
]
fig.legend(handles=leg, loc="lower center", ncol=6, fontsize=9,
           framealpha=0.95, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.01))
fig.suptitle("#307 cross-branch ablation — three new arms on a "
             "(time × batch) ladder.  Same anchor convention as #303: each "
             "crossed family is fanned from one anchor for clarity (the "
             "loss sums over all anchors).", fontsize=12, y=0.98)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
out = ("/home/jupyter/cf-wt-crossed-loss/experiments/"
       "2026-05-19_crossed_loss_xbranch_ablation/plots/loss_diagram.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved → {out}")
