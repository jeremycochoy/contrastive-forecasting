"""Additional negative terms — 2×3 (batch × time) diagram.

Each vertex = prediction pair (f_{b,τ-1} → h_{b,τ}).
All NEW uncovered pairs shown as dashed coloured lines (referenced at tensor
index s = t-1, i.e. the middle-→-right sub-square).
Existing covered edges shown muted for context.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(-1.0, 16.0)
ax.set_ylim(0.5, 12.5)
ax.axis("off")

# ── Grid layout ────────────────────────────────────────────────────────────────
CX_L, CX_M, CX_R = 3.0, 7.5, 12.0   # column x positions
CY_T, CY_B       = 8.8, 3.8          # row y positions (top=b, bottom=b')
D = 0.52                              # f/h half-separation within a vertex

CX = {"TL": CX_L, "TM": CX_M, "TR": CX_R,
      "BL": CX_L, "BM": CX_M, "BR": CX_R}
CY = {"TL": CY_T, "TM": CY_T, "TR": CY_T,
      "BL": CY_B, "BM": CY_B, "BR": CY_B}

def fn(v): return (CX[v] - D, CY[v])
def hn(v): return (CX[v] + D, CY[v])

NODES = {}
for v in ("TL", "TM", "TR", "BL", "BM", "BR"):
    NODES[f"f_{v}"] = fn(v)
    NODES[f"h_{v}"] = hn(v)

# ── Arrow helper ───────────────────────────────────────────────────────────────
def arrow(src, dst, color, lw, arrowhead=False, ls="-", rad=0.0, alpha=1.0, zorder=3):
    x0, y0 = NODES[src]; x1, y1 = NODES[dst]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->" if arrowhead else "-",
                    color=color, lw=lw, linestyle=ls, alpha=alpha,
                    connectionstyle=f"arc3,rad={rad}"),
                zorder=zorder)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_POS   = "#2a9d2a"   # green   — positive
C_TOPB  = "#bbbbbb"   # grey    — neg_zy temporal
C_DIAG  = "#c8a030"   # orange  — neg_cross_batch_forecast_embedding
C_FSIDE = "#7aabdc"   # blue    — neg_cross_batch_forecast (new square)
C_HSIDE = "#dc7a7a"   # red     — neg_cross_batch_embedding (new square)
ALPHA_E = 0.40        # opacity for existing edges

# New uncovered
C_WB1 = "#9b59b6"   # purple  — within-batch 1-miss
C_WB2 = "#c0186a"   # magenta — within-batch 2-hop
C_CB0 = "#17a58d"   # teal    — cross-batch 0-hop
C_CB1 = "#e67e22"   # orange  — cross-batch 1-miss
C_CB2 = "#c0392b"   # dark red — cross-batch 2-hop
C_CTH = "#5a7a00"   # olive   — cross-time h
C_CTF = "#7b3f00"   # brown   — cross-time f

# ── EXISTING: positives ────────────────────────────────────────────────────────
for v in ("TL", "TM", "TR", "BL", "BM", "BR"):
    arrow(f"f_{v}", f"h_{v}", C_POS, 2.4, arrowhead=True, zorder=5)

# ── EXISTING: neg_zy (temporal, adj. columns, same row) ───────────────────────
for a, b, r in (
    ("f_TL", "f_TM", -0.18), ("f_TM", "f_TR", -0.18),
    ("h_TL", "h_TM", -0.18), ("h_TM", "h_TR", -0.18),
    ("f_BL", "f_BM",  0.18), ("f_BM", "f_BR",  0.18),
    ("h_BL", "h_BM",  0.18), ("h_BM", "h_BR",  0.18),
):
    arrow(a, b, C_TOPB, 1.3, rad=r, alpha=ALPHA_E)

# ── EXISTING: neg_cross_batch_forecast_embedding (f→h same col, cross-batch) ──
for col in ("L", "M", "R"):
    arrow(f"f_T{col}", f"h_B{col}", C_DIAG, 1.3,
          ls=(0, (5, 3)), rad=0.06, alpha=ALPHA_E, arrowhead=True)

# ── EXISTING: neg_cross_batch_forecast (f↔f same col, cross-batch) ────────────
for col in ("L", "M", "R"):
    arrow(f"f_T{col}", f"f_B{col}", C_FSIDE, 1.3,
          rad=0.22, alpha=ALPHA_E)

# ── EXISTING: neg_cross_batch_embedding (h↔h same col, cross-batch) ──────────
for col in ("L", "M", "R"):
    arrow(f"h_T{col}", f"h_B{col}", C_HSIDE, 1.3,
          rad=-0.22, alpha=ALPHA_E)

# ── NEW UNCOVERED (all at tensor index s = t-1) ────────────────────────────────
# Mapping:  f_s=f_TM  f_{s+1}=f_TR  h_s=h_TL  h_{s+1}=h_TM
#           f_s'=f_BM f_{s+1}'=f_BR h_s'=h_BL h_{s+1}'=h_BM

# 1. within-b 1-miss: f_{s+1}^b  ↔ h_{s+1}^b  → f_TR ↔ h_TM
arrow("f_TR", "h_TM", C_WB1, 2.1, ls=(0, (4, 3)), rad=-0.40, zorder=4)

# 2. within-b 2-hop:  f_{s+1}^b  ↔ h_s^b      → f_TR ↔ h_TL
arrow("f_TR", "h_TL", C_WB2, 2.1, ls=(0, (4, 3)), rad=-0.28, zorder=4)

# 3. cross-b  0-hop:  h_s^b      ↔ f_s^{b'}   → h_TL ↔ f_BM
arrow("h_TL", "f_BM", C_CB0, 2.1, ls=(0, (4, 3)), rad= 0.22, zorder=4)

# 4. cross-b  1-miss: f_{s+1}^b  ↔ h_{s+1}^{b'} → f_TR ↔ h_BM
arrow("f_TR", "h_BM", C_CB1, 2.1, ls=(0, (4, 3)), rad= 0.18, zorder=4)

# 5. cross-b  2-hop:  f_{s+1}^b  ↔ h_s^{b'}   → f_TR ↔ h_BL
arrow("f_TR", "h_BL", C_CB2, 2.1, ls=(0, (4, 3)), rad= 0.10, zorder=4)

# 6. cross-time h:    h_s^b      ↔ h_{s+1}^{b'} → h_TL ↔ h_BM
arrow("h_TL", "h_BM", C_CTH, 2.1, ls=(0, (4, 3)), rad=-0.18, zorder=4)

# 7. cross-time f:    f_s^b      ↔ f_{s+1}^{b'} → f_TM ↔ f_BR
arrow("f_TM", "f_BR", C_CTF, 2.1, ls=(0, (4, 3)), rad= 0.30, zorder=4)

# ── AUTO-SHIFTED ECHOES ────────────────────────────────────────────────────────
# Each loss term above applies at every anchor s and every (b, b') with b ≠ b'.
# The 7 primaries above are drawn at the centred anchor s = t-1 on row b.
# Below: every additional copy that lands inside the 6-vertex window — i.e.
# anchor s shifted to s±1, and/or the batch label-swap b ↔ b' for cross-b
# terms. Drawn in the SAME colour as the source so the legend is unchanged.

# WB1 (purple) — within-b (f_τ, h_τ); τ ∈ {t-1, t}, both rows
arrow("f_TM", "h_TL", C_WB1, 2.1, ls=(0, (4, 3)), rad=-0.40, zorder=4)  # b, τ=t-1
arrow("f_BM", "h_BL", C_WB1, 2.1, ls=(0, (4, 3)), rad= 0.40, zorder=4)  # b', τ=t-1
arrow("f_BR", "h_BM", C_WB1, 2.1, ls=(0, (4, 3)), rad= 0.40, zorder=4)  # b', τ=t

# WB2 (magenta) — within-b (f_τ, h_{τ-1}); only τ=t fits (τ=t-1 needs h_{t-2})
arrow("f_BR", "h_BL", C_WB2, 2.1, ls=(0, (4, 3)), rad= 0.28, zorder=4)  # b', τ=t

# CB0 (teal) — cross-b (h_b, f_{b'}) same τ; τ ∈ {t-1, t}
arrow("h_BL", "f_TM", C_CB0, 2.1, ls=(0, (4, 3)), rad=-0.22, zorder=4)  # τ=t-1, b↔b'
arrow("h_TM", "f_BR", C_CB0, 2.1, ls=(0, (4, 3)), rad= 0.22, zorder=4)  # τ=t
arrow("h_BM", "f_TR", C_CB0, 2.1, ls=(0, (4, 3)), rad=-0.22, zorder=4)  # τ=t, b↔b'

# CB1 (orange) — cross-b (f_b, h_{b'}) same τ (same set as CB0 under b↔b')
arrow("f_TM", "h_BL", C_CB1, 2.1, ls=(0, (4, 3)), rad= 0.18, zorder=4)  # τ=t-1
arrow("f_BM", "h_TL", C_CB1, 2.1, ls=(0, (4, 3)), rad=-0.18, zorder=4)  # τ=t-1, b↔b'
arrow("f_BR", "h_TM", C_CB1, 2.1, ls=(0, (4, 3)), rad=-0.18, zorder=4)  # τ=t, b↔b'

# CB2 (dark red) — cross-b (f_τ, h_{τ-1}); only τ=t fits
arrow("f_BR", "h_TL", C_CB2, 2.1, ls=(0, (4, 3)), rad=-0.10, zorder=4)  # τ=t, b↔b'

# CTH (olive) — cross-b (h_{τ-1}, h_τ); τ ∈ {t, t+1}
arrow("h_BL", "h_TM", C_CTH, 2.1, ls=(0, (4, 3)), rad= 0.18, zorder=4)  # τ=t, b↔b'
arrow("h_TM", "h_BR", C_CTH, 2.1, ls=(0, (4, 3)), rad=-0.18, zorder=4)  # τ=t+1
arrow("h_BM", "h_TR", C_CTH, 2.1, ls=(0, (4, 3)), rad= 0.18, zorder=4)  # τ=t+1, b↔b'

# CTF (brown) — cross-b (f_τ, f_{τ+1}); τ ∈ {t-2, t-1}
arrow("f_TL", "f_BM", C_CTF, 2.1, ls=(0, (4, 3)), rad= 0.30, zorder=4)  # τ=t-2
arrow("f_BL", "f_TM", C_CTF, 2.1, ls=(0, (4, 3)), rad=-0.30, zorder=4)  # τ=t-2, b↔b'
arrow("f_BM", "f_TR", C_CTF, 2.1, ls=(0, (4, 3)), rad=-0.30, zorder=4)  # τ=t-1, b↔b'

# ── Nodes (drawn on top) ───────────────────────────────────────────────────────
F_COLOR = "#d0e8ff"
H_COLOR = "#ffd0d0"
for k, (x, y) in NODES.items():
    c = F_COLOR if k.startswith("f") else H_COLOR
    ax.scatter(x, y, s=440, zorder=6, color=c, edgecolors="#333", linewidths=1.6)

# ── Node labels ────────────────────────────────────────────────────────────────
NL = {
    "f_TL": r"$f_{b,t\!-\!2}$",  "h_TL": r"$h_{b,t\!-\!1}$",
    "f_TM": r"$f_{b,t\!-\!1}$",  "h_TM": r"$h_{b,t}$",
    "f_TR": r"$f_{b,t}$",         "h_TR": r"$h_{b,t\!+\!1}$",
    "f_BL": r"$f_{b',t\!-\!2}$", "h_BL": r"$h_{b',t\!-\!1}$",
    "f_BM": r"$f_{b',t\!-\!1}$", "h_BM": r"$h_{b',t}$",
    "f_BR": r"$f_{b',t}$",        "h_BR": r"$h_{b',t\!+\!1}$",
}
for k, (x, y) in NODES.items():
    v = k[2:]
    dy = 0.48 if CY[v] > 6 else -0.48
    va = "bottom" if CY[v] > 6 else "top"
    ax.text(x, y + dy, NL[k], ha="center", va=va, fontsize=8.5, fontweight="bold")

# ── Column / row headers ────────────────────────────────────────────────────────
for cx, lbl in ((CX_L, r"time $t\!-\!1$"), (CX_M, r"time $t$"), (CX_R, r"time $t\!+\!1$")):
    ax.text(cx, CY_T + 2.0, lbl, ha="center", fontsize=10.5, color="#555", style="italic")

ax.annotate("", xy=(0.3, CY_B - 0.5), xytext=(0.3, CY_T + 0.5),
            arrowprops=dict(arrowstyle="<->", color="#bbb", lw=1.3))
ax.text(-0.1, (CY_T + CY_B) / 2, "batch\naxis",
        ha="center", va="center", fontsize=8.5, color="#aaa",
        style="italic", rotation=90)
ax.text(1.2, CY_T, "batch $b$",  ha="left", va="center", fontsize=9, color="#555")
ax.text(1.2, CY_B, "batch $b'$", ha="left", va="center", fontsize=9, color="#555")

# ── Title ──────────────────────────────────────────────────────────────────────
ax.set_title(
    "Uncovered negatives — (batch × time) extended view\n"
    r"Vertex = $(f_{b,\tau-1} \to h_{b,\tau})$.  "
    r"Dashed = new terms.  Faded = existing covered.  "
    r"Each colour drawn at every anchor $s$ and $(b,b')$ swap that fits the window.",
    fontsize=10.5, pad=10)

# ── Legend ─────────────────────────────────────────────────────────────────────
exist_items = [
    mpatches.Patch(color=C_POS,              label=r"positive $f\!\to\!h$ within vertex"),
    mpatches.Patch(color=C_TOPB, alpha=0.7,  label=r"$neg\_zy$: $f\!\leftrightarrow\!f$, $h\!\leftrightarrow\!h$ adj. $t$ (existing)"),
    mpatches.Patch(color=C_DIAG, alpha=0.7,  label=r"diag: $f_t^b\!\leftrightarrow\!h_{t+1}^{b'}$ (existing)"),
    mpatches.Patch(color=C_FSIDE,alpha=0.7,  label=r"$f_t^b\!\leftrightarrow\!f_t^{b'}$ (new square)"),
    mpatches.Patch(color=C_HSIDE,alpha=0.7,  label=r"$h_{t+1}^b\!\leftrightarrow\!h_{t+1}^{b'}$ (new square)"),
]
new_items = [
    Line2D([0],[0], color=C_WB1, lw=2.0, ls=(0,(4,2)),
           label=r"within-b 1-miss  $f_{t+1}^b\!\leftrightarrow\!h_{t+1}^b$"),
    Line2D([0],[0], color=C_WB2, lw=2.0, ls=(0,(4,2)),
           label=r"within-b 2-hop   $f_{t+1}^b\!\leftrightarrow\!h_t^b$"),
    Line2D([0],[0], color=C_CB0, lw=2.0, ls=(0,(4,2)),
           label=r"cross-b  0-hop   $h_t^b\!\leftrightarrow\!f_t^{b'}$"),
    Line2D([0],[0], color=C_CB1, lw=2.0, ls=(0,(4,2)),
           label=r"cross-b  1-miss  $f_{t+1}^b\!\leftrightarrow\!h_{t+1}^{b'}$"),
    Line2D([0],[0], color=C_CB2, lw=2.0, ls=(0,(4,2)),
           label=r"cross-b  2-hop   $f_{t+1}^b\!\leftrightarrow\!h_t^{b'}$"),
    Line2D([0],[0], color=C_CTH, lw=2.0, ls=(0,(4,2)),
           label=r"cross-time $h$   $h_t^b\!\leftrightarrow\!h_{t+1}^{b'}$"),
    Line2D([0],[0], color=C_CTF, lw=2.0, ls=(0,(4,2)),
           label=r"cross-time $f$   $f_t^b\!\leftrightarrow\!f_{t+1}^{b'}$"),
]
ax.legend(handles=exist_items + new_items,
          loc="lower center", fontsize=8.2,
          framealpha=0.95, edgecolor="#ccc", ncol=3,
          bbox_to_anchor=(0.5, -0.02))

out = "experiments/2026-05-09_exp_additional_negatives/plots/additional_negatives_diagram.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved → {out}")
