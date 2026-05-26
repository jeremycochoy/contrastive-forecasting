"""Explanatory schematics for #318 — Deny the positional shortcut.

Two clean, neutral (design-only, no results) figures for the report:

  fork_schematic.png  Data-side mechanism: one forked-ARIMA pair (x^A, x^B)
                      sharing an exact past then diverging at the fork point.
  design.png          Boxes-and-arrows schema of *what was tested* — the two
                      ways the positional shortcut is denied (loss-side h<->h
                      repulsion; data-side forked injection), the injection
                      fractions per loss, and the 2L/6L q-head eval footnote.

Run:  PYTHONPATH=<worktree> MPLBACKEND=Agg python experiments/.../scripts/schematics.py

No results are depicted — design / mechanism only.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.synthetic_forked_arma import generate_forked_arma_batch

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.normpath(os.path.join(HERE, "..", "plots"))
os.makedirs(PLOTS, exist_ok=True)

# Report arm colours (from scripts/plots.py).
C_XSHH, C_ALLT = "#1f77b4", "#2ca02c"            # loss-side: same-step / all-time
C_FORK, C_FORK2, C_BFORK = "#ff7f0e", "#8c564b", "#e377c2"   # allt 50% / allt 0.8% / beta 0.8%
C_FORK10, C_BFORK10 = "#bcbd22", "#17becf"       # forked 10%: all-time / beta
C_BETA = "#d62728"                                # beta baseline

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12,
    "savefig.dpi": 130,
    "figure.dpi": 130,
})


# --------------------------------------------------------------------------- #
# IMAGE 1 — fork_schematic.png
# --------------------------------------------------------------------------- #
def fork_schematic(path: str) -> int:
    """Plot one forked pair; return the fork index it shows."""
    X = generate_forked_arma_batch(
        2, T_raw=1024, C=1, rng=np.random.default_rng(0),
        integrate=True, return_labels=True)[0]
    xa = X[0, :, 0].numpy()
    xb = X[1, :, 0].numpy()
    t = np.arange(len(xa))

    # Fork index = first t where the branches differ (prefix is bit-exact).
    diff = np.abs(xa - xb)
    nz = np.where(diff > 1e-6)[0]
    fork = int(nz[0]) if len(nz) else len(xa)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))

    # Shaded regions: shared past (left) / divergent futures (right).
    ax.axvspan(0, fork, color="0.92", zorder=0)
    ax.axvspan(fork, len(xa) - 1, color="#fff3e6", zorder=0)

    # Shared prefix drawn once (the two series overlap exactly here).
    ax.plot(t[:fork + 1], xa[:fork + 1], color="0.25", lw=2.0,
            zorder=3, label=r"shared past  $x^A_{1:l}\!=\!x^B_{1:l}$",
            solid_capstyle="round")
    # Divergent branches.
    ax.plot(t[fork:], xa[fork:], color=C_FORK, lw=1.8, zorder=3,
            label="future $x^A$", solid_capstyle="round")
    ax.plot(t[fork:], xb[fork:], color=C_XSHH, lw=1.8, zorder=3,
            label="future $x^B$", solid_capstyle="round")

    # Fork line + label (placed just inside the top of the axes so it never
    # collides with the title).
    ax.axvline(fork, color="k", ls="--", lw=1.3, zorder=4)
    ymin, ymax = ax.get_ylim()
    ax.annotate(f"fork  (l = {fork})", xy=(fork, ymax),
                xytext=(fork, ymax - 0.05 * (ymax - ymin)),
                ha="center", va="top", fontsize=10.5, fontweight="bold",
                annotation_clip=False, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    # Region annotations (placed low, inside the axes).
    ylo = ymin + 0.10 * (ymax - ymin)
    ax.text(fork * 0.5, ylo, "shared past", ha="center", va="center",
            fontsize=10, color="0.30", zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    ax.text(fork + (len(xa) - 1 - fork) * 0.5, ylo, "divergent futures",
            ha="center", va="center", fontsize=10, color="#b5651d", zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e0b080", alpha=0.9))

    ax.set_xlim(0, len(xa) - 1)
    ax.set_xlabel("time  $t$")
    ax.set_ylabel("value")
    ax.set_title("Data-side: forked-ARIMA continuation (same past → two futures)")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92, ncol=1)
    ax.margins(x=0)
    fig.tight_layout(pad=0.6)
    fig.savefig(path)
    plt.close(fig)
    return fork


# --------------------------------------------------------------------------- #
# IMAGE 2 — design.png
# --------------------------------------------------------------------------- #
def _box(ax, xy, w, h, text, fc, ec="0.35", fontsize=9.5, fontweight="normal",
         tc="black", lw=1.2):
    """Rounded box centred at xy=(cx, cy); returns (cx, cy, w, h)."""
    cx, cy = xy
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", zorder=3,
            fontsize=fontsize, fontweight=fontweight, color=tc, wrap=True)
    return (cx, cy, w, h)


def _arrow(ax, a, b, text=None, color="0.30", lw=1.5, rad=0.0,
           text_kw=None, shrink=2.0):
    """Arrow from box-edge of a to box-edge of b (a, b are (cx, cy, w, h))."""
    ax_, ay, aw, ah = a
    bx, by, bw, bh = b
    # Connect from right edge of a to left edge of b when b is to the right;
    # otherwise from bottom of a to top of b.
    if bx > ax_ + aw / 2:
        p0 = (ax_ + aw / 2, ay)
        p1 = (bx - bw / 2, by)
    elif by < ay:
        p0 = (ax_, ay - ah / 2)
        p1 = (bx, by + bh / 2)
    else:
        p0 = (ax_ + aw / 2, ay)
        p1 = (bx - bw / 2, by)
    arr = FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=13, lw=lw, color=color,
        shrinkA=shrink, shrinkB=shrink, zorder=1)
    ax.add_patch(arr)
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        kw = dict(ha="center", va="center", fontsize=8.5, color=color,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                            alpha=0.85))
        if text_kw:
            kw.update(text_kw)
        ax.text(mx, my, text, zorder=4, **kw)


def _tint(hex_color: str, amount: float = 0.78) -> tuple:
    """Lighten a hex colour toward white (amount in [0,1], higher = lighter)."""
    c = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(c + (1.0 - c) * amount)


def design(path: str) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def _chip(cx, cy, label, color, w=7.4, h=5.4):
        return _box(ax, (cx, cy), w, h, label, fc=_tint(color, 0.42),
                    ec=color, fontsize=9.0)

    def _fan(parent, chips, color):
        """Short arrows from a parent box's right edge to each chip's left edge."""
        px = parent[0] + parent[2] / 2
        py = parent[1]
        for (cx, cy, cw, ch) in chips:
            arr = FancyArrowPatch(
                (px, py), (cx - cw / 2, cy),
                connectionstyle="arc3,rad=0.0", arrowstyle="-|>",
                mutation_scale=11, lw=1.1, color=color,
                shrinkA=1.5, shrinkB=1.5, zorder=1)
            ax.add_patch(arr)

    # --- Left: beta baseline -------------------------------------------------
    beta = _box(ax, (11, 60), 19, 19,
                "β baseline\n(contrastive loss;\ncross-time negative has a\n"
                "positional-shortcut escape)",
                fc=_tint(C_BETA, 0.84), ec=C_BETA, fontsize=9.5, lw=1.6)

    # --- "Deny the shortcut" hub --------------------------------------------
    hub = _box(ax, (33, 60), 13, 9, "Deny the\nshortcut",
               fc="0.93", ec="0.4", fontsize=10, fontweight="bold")
    _arrow(ax, beta, hub, color="0.35", lw=1.7)

    # --- Loss-side branch (upper) -------------------------------------------
    loss_hdr = _box(ax, (54, 84), 22, 9.5,
                    "Loss-side\ncross-series h↔h repulsion",
                    fc=_tint(C_XSHH, 0.80), ec=C_XSHH, fontsize=9.5,
                    fontweight="bold")
    _arrow(ax, hub, loss_hdr, color=C_XSHH, lw=1.7, rad=-0.18)

    ss = _box(ax, (81, 90), 15, 7.6, "same-step",
              fc=_tint(C_XSHH, 0.62), ec=C_XSHH, fontsize=9.5)
    at = _box(ax, (81, 79), 15, 7.6, "all-time",
              fc=_tint(C_ALLT, 0.55), ec=C_ALLT, fontsize=9.5)
    _arrow(ax, loss_hdr, ss, color=C_XSHH, lw=1.4, rad=0.12)
    _arrow(ax, loss_hdr, at, color=C_ALLT, lw=1.4, rad=-0.12)

    # --- Data-side branch (lower) -------------------------------------------
    data_hdr = _box(ax, (54, 34), 22, 11,
                    "Data-side\nforked-ARIMA injection\n"
                    "(same past → divergent futures)",
                    fc=_tint(C_FORK, 0.78), ec=C_FORK, fontsize=9.5,
                    fontweight="bold")
    _arrow(ax, hub, data_hdr, color=C_FORK, lw=1.7, rad=0.18)

    # Two loss "lanes" the fork is injected on, each with its fraction chips
    # laid out to the right (no overlap).
    on_beta = _box(ax, (77, 46), 16, 8.4, "on the β loss",
                   fc="white", ec=C_BFORK10, fontsize=9.5, lw=1.6)
    on_allt = _box(ax, (77, 22), 16, 8.4, "on the all-time loss",
                   fc="white", ec=C_FORK, fontsize=9.5, lw=1.6)
    _arrow(ax, data_hdr, on_beta, color=C_BFORK10, lw=1.5, rad=0.16)
    _arrow(ax, data_hdr, on_allt, color=C_FORK, lw=1.5, rad=-0.16)

    # β loss: 0.8% (brown=allt-0.8 analogue → use β·0.8 colour C_BFORK), 10% (C_BFORK10)
    b08 = _chip(90.5, 50, "0.8%", C_BFORK)
    b10 = _chip(90.5, 42, "10%", C_BFORK10)
    _fan(on_beta, [b08, b10], C_BFORK10)

    # all-time loss: 0.8% (C_FORK2), 10% (C_FORK10), 50% (C_FORK)
    a08 = _chip(90.5, 28, "0.8%", C_FORK2)
    a10 = _chip(90.5, 21, "10%", C_FORK10)
    a50 = _chip(90.5, 14, "50%", C_FORK)
    _fan(on_allt, [a08, a10, a50], C_FORK)

    # --- Footnote box --------------------------------------------------------
    ax.add_patch(FancyBboxPatch(
        (4, 2.5), 92, 6.8, boxstyle="round,pad=0.2,rounding_size=0.4",
        linewidth=1.0, edgecolor="0.55", facecolor="0.96", zorder=2))
    ax.text(50, 5.9,
            "every backbone evaluated with a 2L and a 6L quantile q-head",
            ha="center", va="center", fontsize=10.5, fontstyle="italic",
            zorder=3)

    ax.set_title("#318 — two ways to deny the positional shortcut",
                 fontsize=12.5, pad=8)
    fig.tight_layout(pad=0.5)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    fork_path = os.path.join(PLOTS, "fork_schematic.png")
    design_path = os.path.join(PLOTS, "design.png")
    fork = fork_schematic(fork_path)
    design(design_path)
    print(f"fork_schematic.png  -> {fork_path}  (fork index = {fork})")
    print(f"design.png          -> {design_path}")


if __name__ == "__main__":
    main()
