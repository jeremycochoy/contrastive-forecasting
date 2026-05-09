"""Generate the (batch × time) square diagram — each vertex is an (f, h) pair."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# ── Each square corner holds TWO nodes: f (left) and h (right), offset by ±D ──
# Corner centres in the (batch, time) grid:
#   TL=(b, t-1)  TR=(b, t)
#   BL=(b',t-1)  BR=(b',t)
CX = {  "TL": 2.5, "TR": 7.5, "BL": 2.5, "BR": 7.5  }
CY = {  "TL": 7.5, "TR": 7.5, "BL": 2.5, "BR": 2.5  }
D = 0.52          # half-separation between f and h within a corner
R = 0.22          # node radius for scatter

# Node positions: each corner → (f_x, f_y), (h_x, h_y)
def fn(corner): return (CX[corner]-D, CY[corner])   # f node
def hn(corner): return (CX[corner]+D, CY[corner])   # h node

NODES = {
    "f_TL": fn("TL"), "h_TL": hn("TL"),
    "f_TR": fn("TR"), "h_TR": hn("TR"),
    "f_BL": fn("BL"), "h_BL": hn("BL"),
    "f_BR": fn("BR"), "h_BR": hn("BR"),
}

# ── colours ────────────────────────────────────────────────────────────────────
C_POS   = "#2a9d2a"   # green   — positive
C_LEFT  = "#1a6ec2"   # blue    — left/right edges NEW (f<>f, h<>h same time)
C_TOPB  = "#888888"   # grey    — top/bottom edges (temporal, already covered)
C_RIGHT = "#c22020"   # red     — h-side batch edge (NEW)
C_DIAG  = "#d07000"   # orange  — diagonal (existing, kept)

def arrow(src, dst, color, lw, arrowhead=True, ls="-", rad=0.0):
    x0, y0 = NODES[src]; x1, y1 = NODES[dst]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->" if arrowhead else "-",
                    color=color, lw=lw, linestyle=ls,
                    connectionstyle=f"arc3,rad={rad}"),
                zorder=3)

# ── POSITIVE pairs: f→h within each corner (×4) ───────────────────────────────
for c in ("TL", "TR", "BL", "BR"):
    arrow(f"f_{c}", f"h_{c}", C_POS, 2.4)

# ── TOP edge: TL <> TR (temporal, same batch b) — grey, 2 arrows ──────────────
# f_TL <> f_TR
arrow("f_TL", "f_TR", C_TOPB, 1.6, rad=-0.15)   # f side top
# h_TL <> h_TR
arrow("h_TL", "h_TR", C_TOPB, 1.6, rad=-0.15)   # h side top

# ── BOTTOM edge: BL <> BR (temporal, same batch b') — grey, 2 arrows ──────────
arrow("f_BL", "f_BR", C_TOPB, 1.6, rad=0.15)
arrow("h_BL", "h_BR", C_TOPB, 1.6, rad=0.15)

# ── LEFT edge: TL <> BL (batch axis at time t-1) — blue f-side, red h-side ────
arrow("f_TL", "f_BL", C_LEFT,  2.0, arrowhead=False, rad=0.15)
arrow("h_TL", "h_BL", C_RIGHT, 2.0, arrowhead=False, rad=0.15)

# ── RIGHT edge: TR <> BR (batch axis at time t) — blue f-side, red h-side ─────
arrow("f_TR", "f_BR", C_LEFT,  2.0, arrowhead=False, rad=-0.15)
arrow("h_TR", "h_BR", C_RIGHT, 2.0, arrowhead=False, rad=-0.15)

# ── DIAGONAL: f_TR → h_BR  (existing neg_cross_batch_forecast_embedding) ───────
arrow("f_TR", "h_BR", C_DIAG, 1.8, arrowhead=True, ls=(0,(6,3)), rad=0.08)

# ── Draw nodes on top ──────────────────────────────────────────────────────────
F_COLOR = "#d0e8ff"   # light blue for f
H_COLOR = "#ffd0d0"   # light red  for h
for k, (x, y) in NODES.items():
    c = F_COLOR if k.startswith("f") else H_COLOR
    ax.scatter(x, y, s=420, zorder=5, color=c, edgecolors="#333", linewidths=1.6)

# node labels
NL = {
    "f_TL": r"$f_{b,t-1}$",  "h_TL": r"$h_{b,t}$",
    "f_TR": r"$f_{b,t}$",    "h_TR": r"$h_{b,t+1}$",
    "f_BL": r"$f_{b',t-1}$", "h_BL": r"$h_{b',t}$",
    "f_BR": r"$f_{b',t}$",   "h_BR": r"$h_{b',t+1}$",
}
for k, (x, y) in NODES.items():
    corner = k[2:]
    dy = 0.45 if CY[corner] > 5 else -0.45
    va = "bottom" if CY[corner] > 5 else "top"
    ax.text(x, y+dy, NL[k], ha="center", va=va, fontsize=10, fontweight="bold")

# ── corner labels (prediction pair names) ─────────────────────────────────────
for corner, label in [("TL", r"$(b,\,t-1)$"), ("TR", r"$(b,\,t)$"),
                       ("BL", r"$(b',t-1)$"), ("BR", r"$(b',\,t)$")]:
    dx = -1.2 if corner in ("TL","BL") else 1.2
    ax.text(CX[corner]+dx, CY[corner],
            label, ha="center", va="center", fontsize=9, color="#888", style="italic")

# ── axis labels ────────────────────────────────────────────────────────────────
ax.annotate("", xy=(1.0, CY["BL"]-0.15), xytext=(1.0, CY["TL"]+0.15),
            arrowprops=dict(arrowstyle="<->", color="#bbb", lw=1.3))
ax.text(0.6, (CY["TL"]+CY["BL"])/2, "batch\naxis", ha="center", va="center",
        fontsize=8.5, color="#bbb", style="italic", rotation=90)
ax.text(CX["TL"], CY["TL"]+1.5, r"time $t-1$", ha="center", fontsize=10,
        color="#666", style="italic")
ax.text(CX["TR"], CY["TR"]+1.5, r"time $t$", ha="center", fontsize=10,
        color="#666", style="italic")

# ── diagonal label ─────────────────────────────────────────────────────────────
ax.text(5.5, 4.8,
        "neg_cross_batch\n_forecast_embedding\n"+r"$f_{b,t}\leftrightarrow h_{b',t+1}$"+"\n(existing, kept)",
        ha="center", va="center", fontsize=8.5, color=C_DIAG,
        bbox=dict(fc="white", ec="#e8c880", boxstyle="round,pad=0.35", alpha=0.95))

# ── edge labels ────────────────────────────────────────────────────────────────
# top grey
ax.text((CX["TL"]+CX["TR"])/2, CY["TL"]+0.85,
        r"$neg\_zy$: $f\leftrightarrow f$ / $h\leftrightarrow h$ adjacent $t$  (existing)",
        ha="center", fontsize=8.5, color=C_TOPB)
# bottom grey
ax.text((CX["BL"]+CX["BR"])/2, CY["BL"]-0.85,
        r"$neg\_zy$: same (existing)",
        ha="center", fontsize=8.5, color=C_TOPB)
# left blue
ax.text(CX["TL"]-1.65, (CY["TL"]+CY["BL"])/2,
        "neg_cross_batch\n_forecast  (NEW)\n"+r"$f_{b,t}\leftrightarrow f_{b',t}$",
        ha="center", va="center", fontsize=8.5, color=C_LEFT)
# right red
ax.text(CX["TR"]+1.65, (CY["TR"]+CY["BR"])/2,
        "neg_cross_batch\n_embedding  (NEW)\n"+r"$h_{b,t+1}\leftrightarrow h_{b',t+1}$",
        ha="center", va="center", fontsize=8.5, color=C_RIGHT)

# ── title ──────────────────────────────────────────────────────────────────────
ax.set_title("Square loss — (batch × time) negative structure\n"
             r"each vertex = prediction pair $(f_t \rightarrow h_{t+1})$",
             fontsize=11, pad=8)

# ── legend ─────────────────────────────────────────────────────────────────────
items = [
    mpatches.Patch(color=C_POS,   label="positive  f→h within vertex (existing)"),
    mpatches.Patch(color=C_TOPB,  label="temporal edges  f↔f, h↔h adj. t  (neg_zy, existing)"),
    mpatches.Patch(color=C_DIAG,  label="diagonal  f_{b,t}↔h_{b',t+1}  (existing, kept)"),
    mpatches.Patch(color=C_LEFT,  label="f↔f cross-batch  neg_cross_batch_forecast  (NEW)"),
    mpatches.Patch(color=C_RIGHT, label="h↔h cross-batch  neg_cross_batch_embedding  (NEW)"),
]
ax.legend(handles=items, loc="lower center", fontsize=8.2,
          framealpha=0.95, edgecolor="#ccc", ncol=2,
          bbox_to_anchor=(0.5, -0.01))

out = "experiments/2026-05-09_exp_loss_extensions/plots/square_diagram.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved → {out}")
