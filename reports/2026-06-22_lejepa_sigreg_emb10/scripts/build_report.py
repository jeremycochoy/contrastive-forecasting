#!/usr/bin/env python3
# SIGReg-emb10 report builder: plots + gm_table.csv from training CSV + GIFT-Eval summaries.
# Compares against three references: enc3+CPC, EMA-target enc3+CPC, SIGReg λ_e=λ_h=0.1.
import argparse, csv, math, os, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reference GM values, transcribed from each arm's published gm_table.csv at its own code rev.
REF_GM = {
    ("cpc_enc3", "2L", "best"): 1.1846,
    ("cpc_enc3", "2L", "last"): 1.1531,
    ("cpc_enc3", "6L", "best"): 1.1584,
    ("cpc_enc3", "6L", "last"): 1.1436,
}
EMA_GM = {
    ("ema_enc3", "2L", "best"): 1.1614,
    ("ema_enc3", "2L", "last"): 1.1817,
    ("ema_enc3", "6L", "best"): 1.1576,
    ("ema_enc3", "6L", "last"): 1.1597,
}
SIGREG01_GM = {  # prior SIGReg arm with shared λ=0.1, transcribed from its gm_table.csv.
    ("sigreg01_enc3", "2L", "best"): 1.1610,
    ("sigreg01_enc3", "2L", "last"): 1.1758,
    ("sigreg01_enc3", "6L", "best"): 1.1543,
    ("sigreg01_enc3", "6L", "last"): 1.1556,
}

ARM_LABEL = {
    "cpc_enc3":      "enc3+CPC, B=1024",
    "ema_enc3":      "EMA-target enc3+CPC, B=1024",
    "sigreg01_enc3": "SIGReg + EMA-target, B=512 (λ_e=λ_h=0.1)",
    "sigreg10_enc3": "SIGReg + EMA-target, B=512 (λ_e=1.0, λ_h=0.1)",
}

ARM_COLOR = {
    "cpc_enc3":      "#888888",
    "ema_enc3":      "#1f77b4",
    "sigreg01_enc3": "#d62728",
    "sigreg10_enc3": "#2ca02c",
}


def parse_per_config_rel_mase(summary_path: Path) -> dict[str, float]:
    """Return {config: relative_mase} parsed from a GIFT-Eval summary.txt.
    The summary's per-config table format is `config  MASE  SN_MASE  Relative`
    (whitespace-separated, config containing '/'); the last column is rel-MASE."""
    out: dict[str, float] = {}
    if not summary_path.exists():
        return out
    for line in summary_path.read_text().splitlines():
        p = line.split()
        if len(p) == 4 and "/" in p[0]:
            try:
                out[p[0]] = float(p[3])
            except ValueError:
                pass
    return out


def paired_bootstrap_ci_log(
    a: np.ndarray, b: np.ndarray, B: int = 10_000, ci: float = 0.95, seed: int = 20260622,
) -> tuple[float, float, float, float]:
    """Paired bootstrap on the log-rel-MASE difference for two arrays of per-config
    rel-MASE values. GM-Rel MASE is exp(mean(log(rel))), so the natural paired delta
    is mean(log(a) - log(b)); GM(a)/GM(b) = exp(that). Returns (mean_log_delta,
    log_lo, log_hi, p_below_zero). a and b are aligned by index (same configs)."""
    rng = np.random.default_rng(seed)
    la, lb = np.log(a), np.log(b)
    d = la - lb
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    boot = d[idx].mean(axis=1)
    lo = float(np.quantile(boot, (1 - ci) / 2))
    hi = float(np.quantile(boot, 1 - (1 - ci) / 2))
    return float(d.mean()), lo, hi, float((boot < 0).mean())


def compute_bootstrap_cells(
    sig10_results: Path, sig10_tag: str,
    sig01_results: Path, sig01_tag: str,
    out_csv: Path, B: int = 10_000,
) -> list[dict]:
    """For each of the 4 (head, ckpt) cells, compute the paired bootstrap CI of
    GM(this arm)/GM(prior arm). Saves bootstrap_ci.csv with absolute and
    log-delta CIs."""
    cells = [("2L", "best", ""), ("2L", "last", "_last"),
             ("6L", "best", ""), ("6L", "last", "_last")]
    rows: list[dict] = []
    for head, ckpt, suf in cells:
        sub10 = sig10_results / f"gift_eval_full_{sig10_tag}{suf}_{head}" / "summary.txt"
        sub01 = sig01_results / f"gift_eval_full_{sig01_tag}{suf}_{head}" / "summary.txt"
        r10 = parse_per_config_rel_mase(sub10)
        r01 = parse_per_config_rel_mase(sub01)
        common = sorted(set(r10) & set(r01))
        a = np.array([r10[c] for c in common])
        b = np.array([r01[c] for c in common])
        n = len(common)
        if n == 0:
            rows.append(dict(head=head, ckpt=ckpt, n=0,
                             mean_log_delta=float("nan"),
                             ci_lo_log=float("nan"), ci_hi_log=float("nan"),
                             gm_ratio=float("nan"),
                             gm_ratio_lo=float("nan"), gm_ratio_hi=float("nan"),
                             gm_delta_abs=float("nan"),
                             gm_delta_lo=float("nan"), gm_delta_hi=float("nan"),
                             p_below_zero=float("nan")))
            continue
        m, lo, hi, p_neg = paired_bootstrap_ci_log(a, b, B=B)
        gm10 = float(np.exp(np.log(a).mean()))
        gm01 = float(np.exp(np.log(b).mean()))
        rows.append(dict(
            head=head, ckpt=ckpt, n=n,
            mean_log_delta=m,
            ci_lo_log=lo, ci_hi_log=hi,
            gm_ratio=float(np.exp(m)),
            gm_ratio_lo=float(np.exp(lo)),
            gm_ratio_hi=float(np.exp(hi)),
            gm_delta_abs=gm01 * (float(np.exp(m)) - 1.0),
            gm_delta_lo=gm01 * (float(np.exp(lo)) - 1.0),
            gm_delta_hi=gm01 * (float(np.exp(hi)) - 1.0),
            p_below_zero=p_neg,
        ))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    return rows


def parse_gm(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    txt = summary_path.read_text()
    m = re.search(r"Aggregate GM-Relative MASE.*?:\s*([0-9.]+)", txt)
    return float(m.group(1)) if m else None


def parse_n_configs(summary_path: Path) -> int:
    if not summary_path.exists():
        return 0
    txt = summary_path.read_text()
    m = re.search(r"Aggregate GM-Relative MASE \((\d+) configs\)", txt)
    return int(m.group(1)) if m else 0


def write_gm_table(
    results_dir: Path, out_csv: Path, tag: str,
    ci_rows: list[dict] | None = None,
):
    """Build gm_table.csv. The reference rows for cpc/ema/sigreg01 are
    transcribed; sigreg10 cells are parsed from this experiment's summaries.
    Bootstrap CI columns (mean delta vs the λ_e=0.1 prior arm and its 95% CI)
    are merged in for the sigreg10 cells when ci_rows is provided."""
    rows: list[dict] = []
    for d in (REF_GM, EMA_GM, SIGREG01_GM):
        for (arm, head, ckpt), gm in d.items():
            rows.append(dict(arm=arm, label=ARM_LABEL[arm], head=head, ckpt=ckpt, gm=gm, n=97,
                             gm_delta_vs_prior=float("nan"),
                             gm_delta_lo=float("nan"), gm_delta_hi=float("nan"),
                             p_below_zero=float("nan")))

    sigreg10_map = {
        ("2L", "best"): results_dir / f"gift_eval_full_{tag}_2L" / "summary.txt",
        ("2L", "last"): results_dir / f"gift_eval_full_{tag}_last_2L" / "summary.txt",
        ("6L", "best"): results_dir / f"gift_eval_full_{tag}_6L" / "summary.txt",
        ("6L", "last"): results_dir / f"gift_eval_full_{tag}_last_6L" / "summary.txt",
    }
    ci_by_cell = {(r["head"], r["ckpt"]): r for r in (ci_rows or [])}
    for (head, ckpt), sp in sigreg10_map.items():
        gm = parse_gm(sp)
        n = parse_n_configs(sp)
        if gm is None:
            continue
        ci = ci_by_cell.get((head, ckpt), {})
        rows.append(dict(arm="sigreg10_enc3", label=ARM_LABEL["sigreg10_enc3"],
                         head=head, ckpt=ckpt, gm=gm, n=n,
                         gm_delta_vs_prior=ci.get("gm_delta_abs", float("nan")),
                         gm_delta_lo=ci.get("gm_delta_lo", float("nan")),
                         gm_delta_hi=ci.get("gm_delta_hi", float("nan")),
                         p_below_zero=ci.get("p_below_zero", float("nan"))))

    def _fmt(v: float, prec: int) -> str:
        if isinstance(v, float) and math.isnan(v):
            return ""
        return f"{v:.{prec}f}"

    fieldnames = ["arm", "label", "head", "ckpt", "gm", "n",
                  "gm_delta_vs_prior", "gm_delta_lo", "gm_delta_hi", "p_below_zero"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "arm": r["arm"], "label": r["label"],
                "head": r["head"], "ckpt": r["ckpt"],
                "gm": _fmt(r["gm"], 4),
                "n": r["n"],
                "gm_delta_vs_prior": _fmt(r["gm_delta_vs_prior"], 4),
                "gm_delta_lo":       _fmt(r["gm_delta_lo"], 4),
                "gm_delta_hi":       _fmt(r["gm_delta_hi"], 4),
                "p_below_zero":      _fmt(r["p_below_zero"], 4),
            })
    return rows


def plot_sigreg_inspection(sig10_csv: Path, sig01_csv: Path | None, out: Path):
    """Compare sigreg_e / sigreg_h / u_batch_e / u_temporal_e between the two
    λ_e weights. All four panels use log y-axis so the bottom row's tiny values
    (1/K ≈ 0.0026, u_batch_e ≈ 0.03–0.04) are readable instead of crammed at
    the baseline of a [0,1] linear axis. Shaded bands mark the Early-50
    (steps 1–50) and Tail-50 (last 50) windows used in Annex B; each panel
    prints the per-window means so the early-vs-late tension is visible
    directly on the figure."""
    s10 = pd.read_csv(sig10_csv)
    s01 = pd.read_csv(sig01_csv) if (sig01_csv and sig01_csv.exists()) else None
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    panels = [
        ("sigreg_e",     "L_SIGReg(e_t)",     "{:.3e}"),
        ("sigreg_h",     "L_SIGReg(h_t)",     "{:.3e}"),
        ("u_batch_e",    "u_batch (e_t)",     "{:.4f}"),
        ("u_temporal_e", "u_temporal (e_t)",  "{:.4f}"),
    ]
    K = 384
    n_window = 50
    final_step10 = int(s10["step"].iloc[-1])
    early_lo, early_hi = 1, n_window
    tail_lo, tail_hi = final_step10 - n_window + 1, final_step10
    band_color = "#fde0a8"
    edge_color = "#b8860b"
    for ax, (col, title, fmt) in zip(axes.ravel(), panels):
        ax.axvspan(early_lo, early_hi, color=band_color, alpha=0.9, zorder=0)
        ax.axvspan(tail_lo, tail_hi, color=band_color, alpha=0.9, zorder=0)
        ax.axvline(early_hi, color=edge_color, ls="--", lw=0.9, alpha=0.7, zorder=1)
        ax.axvline(tail_lo,  color=edge_color, ls="--", lw=0.9, alpha=0.7, zorder=1)
        ax.plot(s10["step"], s10[col].rolling(50, min_periods=1).mean(),
                label="λ_e=1.0 (this arm)", color=ARM_COLOR["sigreg10_enc3"], lw=1.6)
        if s01 is not None and col in s01.columns:
            ax.plot(s01["step"], s01[col].rolling(50, min_periods=1).mean(),
                    label="λ_e=0.1 (prior arm)", color=ARM_COLOR["sigreg01_enc3"], lw=1.2)
        if col in ("u_batch_e", "u_temporal_e"):
            ax.axhline(1.0 / K, color="k", ls=":", alpha=0.5,
                       label=f"1/K = 1/{K} ≈ {1/K:.4f}")

        def _window_mean(df: pd.DataFrame, lo: int, hi: int) -> float:
            sub = df[(df["step"] >= lo) & (df["step"] <= hi)]
            return float(sub[col].mean()) if len(sub) and col in sub.columns else float("nan")

        early10 = _window_mean(s10, early_lo, early_hi)
        tail10  = _window_mean(s10, tail_lo,  tail_hi)
        early01 = _window_mean(s01, early_lo, early_hi) if s01 is not None else float("nan")
        tail01  = _window_mean(s01, tail_lo,  tail_hi)  if s01 is not None else float("nan")
        early_box = (
            f"Early-50\nλ1.0 {fmt}\nλ0.1 {fmt}".format(early10, early01)
        )
        tail_box = (
            f"Tail-50\nλ1.0 {fmt}\nλ0.1 {fmt}".format(tail10, tail01)
        )
        ax.text(0.01, 0.02, early_box, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=edge_color, alpha=0.9))
        ax.text(0.99, 0.02, tail_box, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=edge_color, alpha=0.9))
        ax.set_yscale("log")
        ax.set_xlabel("step"); ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3, which="both")
    fig.suptitle(
        "Embedding-side SIGReg trajectory: λ_e=1.0 vs λ_e=0.1 — log y-axis; "
        "amber bands = Early-50 (steps 1–50, left) and Tail-50 (last 50, right)",
        y=0.995,
    )
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_gm_bars(rows: list[dict], ci_rows: list[dict], out: Path):
    """Grouped bars over the 4 (head, ckpt) cells with per-arm GM-Rel MASE.
    For the sigreg10 (λ_e=1.0) arm only, overlay paired-bootstrap 95% CI whiskers
    on absolute-GM scale (re-anchored at the prior λ_e=0.1 arm's GM as reference)
    and draw a horizontal tick at each prior-arm cell to anchor against the
    direct reference."""
    df = pd.DataFrame(rows)
    arms_order = ["cpc_enc3", "ema_enc3", "sigreg01_enc3", "sigreg10_enc3"]
    cells = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
    ci_by_cell = {(r["head"], r["ckpt"]): r for r in ci_rows}
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x = np.arange(len(cells))
    w = 0.2
    # vals_for_lim used to fit y-limits across arms + whiskers.
    vals_for_lim: list[float] = []
    for i, arm in enumerate(arms_order):
        vals = []
        for head, ckpt in cells:
            r = df[(df["arm"] == arm) & (df["head"] == head) & (df["ckpt"] == ckpt)]
            v = float(r["gm"].values[0]) if len(r) else float("nan")
            vals.append(v)
            if not math.isnan(v):
                vals_for_lim.append(v)
        offs = (i - (len(arms_order) - 1) / 2) * w
        ax.bar(x + offs, vals, w, label=ARM_LABEL[arm], color=ARM_COLOR[arm])
        for xi, vi in zip(x + offs, vals):
            if not math.isnan(vi):
                ax.text(xi, vi + 0.004, f"{vi:.4f}", ha="center", va="bottom", fontsize=6.5)
        # Whiskers + prior-arm anchor only on the λ_e=1.0 bars.
        if arm == "sigreg10_enc3":
            for k, (head, ckpt) in enumerate(cells):
                ci = ci_by_cell.get((head, ckpt))
                if ci is None or math.isnan(ci.get("gm_delta_abs", float("nan"))):
                    continue
                gm10 = vals[k]
                gm01_row = df[(df["arm"] == "sigreg01_enc3") &
                              (df["head"] == head) & (df["ckpt"] == ckpt)]
                gm01 = float(gm01_row["gm"].values[0])
                # CI is on (GM10 - GM01); re-centred at GM10.
                lo, hi = ci["gm_delta_lo"], ci["gm_delta_hi"]
                y_lo, y_hi = gm01 + lo, gm01 + hi
                ax.errorbar(x[k] + offs, gm10,
                            yerr=[[max(0.0, gm10 - y_lo)], [max(0.0, y_hi - gm10)]],
                            fmt="none", ecolor="k", elinewidth=1.0, capsize=3, capthick=1.0)
                vals_for_lim.extend([y_lo, y_hi])
    # Horizontal tick for each prior-arm cell — spans the prior + this-arm bar
    # pair so the eye can compare directly.
    offs_01 = (arms_order.index("sigreg01_enc3") - (len(arms_order) - 1) / 2) * w
    offs_10 = (arms_order.index("sigreg10_enc3") - (len(arms_order) - 1) / 2) * w
    for k, (head, ckpt) in enumerate(cells):
        r01 = df[(df["arm"] == "sigreg01_enc3") &
                 (df["head"] == head) & (df["ckpt"] == ckpt)]
        if not len(r01):
            continue
        y = float(r01["gm"].values[0])
        ax.hlines(y, x[k] + offs_01 - w / 2, x[k] + offs_10 + w / 2,
                  color="#d62728", lw=1.2, ls=(0, (3, 2)), alpha=0.8, zorder=4)
    ax.plot([], [], color="#d62728", ls=(0, (3, 2)), lw=1.2,
            label="λ_e=0.1 anchor")
    ax.plot([], [], color="k", marker="|", linestyle="None", markersize=8,
            label="bootstrap 95% CI (λ_e=1.0 − λ_e=0.1)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}/{c}" for h, c in cells])
    ax.set_ylabel("GM-Rel MASE (lower = better)")
    ax.set_title("GIFT-Eval full-97 GM-Rel MASE — head-matched (whiskers = paired bootstrap, B=10k)")
    # axhline(1.0) (seasonal-naive parity) is dropped because the GM range is
    # [1.14, 1.18] — including 1.0 would compress the bar differences; the
    # metric's 1.0 = SN parity is defined in the report's vocabulary section.
    if vals_for_lim:
        lo = min(vals_for_lim) - 0.02
        hi = max(vals_for_lim) + 0.04
        ax.set_ylim(lo, hi)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _dataset_to_domain(all_csv: Path) -> dict[str, str]:
    """Map GIFT-Eval config (e.g. `loop_seattle/5T/short`) to its `domain` field
    (e.g. `Transport`). Read once per (head, ckpt) eval directory."""
    m: dict[str, str] = {}
    if not all_csv.exists():
        return m
    for r in csv.DictReader(open(all_csv)):
        m[r["dataset"]] = r.get("domain", "Other")
    return m


def _gm_by_domain(rels: dict[str, float], dmap: dict[str, str]) -> dict[str, float]:
    """Aggregate per-config rel-MASE into geometric mean per domain."""
    acc: dict[str, list[float]] = {}
    for cfg, rel in rels.items():
        dom = dmap.get(cfg)
        if rel <= 0 or dom is None:
            continue
        acc.setdefault(dom, []).append(math.log(rel))
    return {d: math.exp(sum(v) / len(v)) for d, v in acc.items()}


def plot_perdomain_radar(
    sig10_results: Path, sig10_tag: str,
    sig01_results: Path, sig01_tag: str,
    out: Path,
):
    """2 panels (2L | 6L), 4 curves each (λ_e ∈ {1.0, 0.1} × {best, last}).
    Palette: light-red (λ_e=0.1, best) / dark-red (λ_e=0.1, last) /
    light-green (λ_e=1.0, best) / dark-green (λ_e=1.0, last). Radial log scale;
    ring at 1.0 marks seasonal-naive parity; lower = better."""
    HEADS = ["2L", "6L"]
    LIGHT_RED, DARK_RED   = "#f08183", "#8a1416"
    LIGHT_GREEN, DARK_GREEN = "#6dc56e", "#185f1a"
    CURVES = [  # (root, tag, suf, colour, linestyle, label)
        (sig10_results, sig10_tag, "",      LIGHT_GREEN, "-", "λ_e=1.0 · best"),
        (sig10_results, sig10_tag, "_last", DARK_GREEN,  "-", "λ_e=1.0 · last"),
        (sig01_results, sig01_tag, "",      LIGHT_RED,   "-", "λ_e=0.1 · best"),
        (sig01_results, sig01_tag, "_last", DARK_RED,    "-", "λ_e=0.1 · last"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), subplot_kw=dict(polar=True))
    for ax, head in zip(axes, HEADS):
        cells = []
        for root, tag, suf, col, ls, lab in CURVES:
            sub = root / f"gift_eval_full_{tag}{suf}_{head}"
            rel = parse_per_config_rel_mase(sub / "summary.txt")
            dmap = _dataset_to_domain(sub / "all_results.csv")
            gm = _gm_by_domain(rel, dmap)
            if gm:
                cells.append((lab, gm, col, ls))
        if not cells:
            ax.text(0.5, 0.5, "no eval", transform=ax.transAxes); continue
        domains = sorted(set().union(*(g for _, g, _, _ in cells)))
        theta = np.linspace(0, 2 * np.pi, len(domains), endpoint=False)
        theta_closed = np.concatenate([theta, theta[:1]])
        vals = [v for _, g, _, _ in cells for v in g.values()]
        lo, hi = max(0.5, min(vals) * 0.92), max(vals) * 1.06
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_xticks(theta); ax.set_xticklabels(domains, fontsize=8)
        ax.set_rscale("log"); ax.set_ylim(lo, hi)
        rticks = [t for t in (0.8, 1.0, 1.2, 1.5, 2.0) if lo < t < hi]
        ax.set_yticks(rticks)
        ax.set_yticklabels([f"{t:g}" for t in rticks], fontsize=7, color="0.4")
        ax.set_rlabel_position(90)
        ax.plot(theta_closed, [1.0] * len(theta_closed),
                color="k", ls=(0, (2, 2)), lw=0.8, alpha=0.6, zorder=1)
        for lab, g, col, ls in cells:
            v = np.array([g.get(d, np.nan) for d in domains]
                         + [g.get(domains[0], np.nan)])
            ax.plot(theta_closed, v, color=col, ls=ls, lw=1.6, zorder=3,
                    marker="o", markersize=3, label=lab)
        ax.set_title(f"{head} q-head", fontsize=11, pad=14)
        ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.06),
                  fontsize=8, frameon=False, ncol=1)
    fig.suptitle(
        "Per-domain GM relative MASE on GIFT-Eval full-97 "
        "(radial log scale; ring at 1.0 = seasonal-naive; lower = better; "
        "light shade = best-loss ckpt, dark shade = last)", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def _window_summary(window: pd.DataFrame, lam_e: float) -> dict[str, float]:
    """Window means. `sigreg_e_over_loss_ratio` is the mean of the PER-STEP
    fraction `sigreg_e / loss` (not the ratio of window means) — this matches
    the convention previously used in Annex B and is the well-defined per-step
    contribution of the SIGReg term to total loss. Multiply by λ_e at the
    consumer to get the loss-fraction `λ_e · L_SIGReg(e_t) / loss`."""
    return {
        "u_batch":      float(window["u_batch"].mean()),
        "u_batch_e":    float(window["u_batch_e"].mean()),
        "u_temporal":   float(window["u_temporal"].mean()),
        "u_temporal_e": float(window["u_temporal_e"].mean()),
        "sigreg_e":     float(window["sigreg_e"].mean()),
        "sigreg_h":     float(window["sigreg_h"].mean()),
        "loss":         float(window["loss"].mean()),
        "sigreg_e_over_loss_ratio": float((window["sigreg_e"] / window["loss"]).mean()),
    }


def final_trajectories(sigreg_csv: Path, n_tail: int = 50) -> dict[str, float]:
    s = pd.read_csv(sigreg_csv)
    out = _window_summary(s.tail(n_tail), lam_e=1.0)
    out["final_step"] = int(s["step"].iloc[-1])
    return out


def early_trajectories(sigreg_csv: Path, n_early: int = 50) -> dict[str, float]:
    s = pd.read_csv(sigreg_csv).head(n_early)
    out = _window_summary(s, lam_e=1.0)
    out["early_step_lo"] = int(s["step"].iloc[0])
    out["early_step_hi"] = int(s["step"].iloc[-1])
    return out


def write_trajectories_with_commentary(
    out_path: Path,
    traj10: dict[str, float], traj01: dict[str, float] | None,
    early10: dict[str, float], early01: dict[str, float] | None,
    K: int = 384,
):
    """Verdict block + early-50/tail-50 trajectory comparison between this
    arm (λ_e=1.0) and the prior arm (λ_e=0.1). Plain prose: no PR/issue refs,
    no mechanistic hypothesis, no journey commentary."""
    lines: list[str] = [
        "# VERDICT (single seed 20260520):",
        "#",
        "# Q  Does stronger embedding-side pressure change the time course of",
        "#    u_batch_e / u_temporal_e / L_SIGReg(e_t) / λ_e·L_SIGReg(e_t)/loss,",
        "#    and is that linked to downstream GM-Rel MASE?",
        "#",
        "# Facts (this arm λ_e=1.0 vs prior arm λ_e=0.1, single seed 20260520):",
        "#",
        "#   Early-50 (steps 1–50):",
        "#     u_batch_e, u_temporal_e, L_SIGReg(e_t) shift by <1e-5 between arms",
        "#     (indistinguishable at this resolution). λ_e·L_SIGReg(e_t)/loss",
        "#     shifts by ~+2.0e-4 (explicit 10× λ_e scaling on near-identical",
        "#     L_SIGReg(e_t)).",
        "#",
        "#   Tail-50 (last 50 of 12 500 steps):",
        "#     u_batch_e: −0.0099 (this − prior); u_temporal_e: −0.0077;",
        "#     L_SIGReg(e_t): −1.8e-4; λ_e·L_SIGReg(e_t)/loss: +1.6e-4;",
        "#     loss: +0.30. End-state direction on u_batch_e / u_temporal_e /",
        "#     L_SIGReg(e_t) is opposite the small (≈0) early-window shift —",
        "#     they end LOWER under the 10× weight, not higher.",
        "#",
        "#   Downstream (GIFT-Eval full-97, head-matched, paired bootstrap B=10 000):",
        "#     Point Δ_GM (λ_e=1.0 − λ_e=0.1) is negative in all 4 (head, ckpt)",
        "#     cells, range [−0.014, −0.007]; all 4 paired-bootstrap 95% CIs",
        "#     include zero; P(Δ<0) in [0.83, 0.95].",
        "#",
    ]
    if traj01 and early01:
        u_b_e_10  = traj10["u_batch_e"];    u_b_e_01  = traj01["u_batch_e"]
        u_t_e_10  = traj10["u_temporal_e"]; u_t_e_01  = traj01["u_temporal_e"]
        u_b_10    = traj10["u_batch"];      u_b_01    = traj01["u_batch"]
        u_t_10    = traj10["u_temporal"];   u_t_01    = traj01["u_temporal"]
        sr_e_10   = traj10["sigreg_e"];     sr_e_01   = traj01["sigreg_e"]
        sr_h_10   = traj10["sigreg_h"];     sr_h_01   = traj01["sigreg_h"]
        loss_10   = traj10["loss"];         loss_01   = traj01["loss"]
        eu_b_e_10 = early10["u_batch_e"];    eu_b_e_01 = early01["u_batch_e"]
        eu_t_e_10 = early10["u_temporal_e"]; eu_t_e_01 = early01["u_temporal_e"]
        esr_e_10  = early10["sigreg_e"];     esr_e_01  = early01["sigreg_e"]
        eloss_10  = early10["loss"];         eloss_01  = early01["loss"]
        lam_e_10, lam_e_01 = 1.0, 0.1
        # Mean-of-per-step-ratio (matches Annex B convention).
        ratio_loss_10  = lam_e_10 * traj10["sigreg_e_over_loss_ratio"]
        ratio_loss_01  = lam_e_01 * traj01["sigreg_e_over_loss_ratio"]
        eratio_loss_10 = lam_e_10 * early10["sigreg_e_over_loss_ratio"]
        eratio_loss_01 = lam_e_01 * early01["sigreg_e_over_loss_ratio"]
        lines += [
            "# Early-50 vs Tail-50 trajectory means",
            "#",
            "# u_batch_e (this − prior):",
            f"#   Early-50:  this {eu_b_e_10:.6f} | prior {eu_b_e_01:.6f} | Δ {eu_b_e_10 - eu_b_e_01:+.2e}",
            f"#   Tail-50:   this {u_b_e_10:.6f} | prior {u_b_e_01:.6f} | Δ {u_b_e_10 - u_b_e_01:+.4f}",
            f"#              (in 1/K units: this {u_b_e_10*K:.1f}× | prior {u_b_e_01*K:.1f}×)",
            "#",
            "# u_temporal_e (this − prior):",
            f"#   Early-50:  this {eu_t_e_10:.6f} | prior {eu_t_e_01:.6f} | Δ {eu_t_e_10 - eu_t_e_01:+.2e}",
            f"#   Tail-50:   this {u_t_e_10:.6f} | prior {u_t_e_01:.6f} | Δ {u_t_e_10 - u_t_e_01:+.4f}",
            "#",
            "# L_SIGReg(e_t) (this − prior):",
            f"#   Early-50:  this {esr_e_10:.3e} | prior {esr_e_01:.3e} | Δ {esr_e_10 - esr_e_01:+.2e}",
            f"#   Tail-50:   this {sr_e_10:.3e} | prior {sr_e_01:.3e} | Δ {sr_e_10 - sr_e_01:+.2e}",
            "#",
            "# λ_e · L_SIGReg(e_t) / loss (this − prior):",
            f"#   Early-50:  this {eratio_loss_10:.3e} | prior {eratio_loss_01:.3e} | Δ {eratio_loss_10 - eratio_loss_01:+.2e}",
            f"#   Tail-50:   this {ratio_loss_10:.3e} | prior {ratio_loss_01:.3e} | Δ {ratio_loss_10 - ratio_loss_01:+.2e}",
            f"#              (Tail-50 ratio of ratios this/prior = {ratio_loss_10/ratio_loss_01:.2f}×;",
            f"#               under a 10× λ_e bump because L_SIGReg(e_t) self-suppressed)",
            "#",
            "# Other tail-50 quantities (this − prior):",
            f"#   u_batch (h_t):    this {u_b_10:.4f}  prior {u_b_01:.4f}  Δ {u_b_10 - u_b_01:+.4f}",
            f"#   u_temporal (h_t): this {u_t_10:.4f}  prior {u_t_01:.4f}  Δ {u_t_10 - u_t_01:+.4f}",
            f"#   L_SIGReg(h_t):    this {sr_h_10:.3e}  prior {sr_h_01:.3e}  Δ {sr_h_10 - sr_h_01:+.2e}",
            f"#   loss:             this {loss_10:.4f}  prior {loss_01:.4f}  Δ {loss_10 - loss_01:+.4f}",
            "",
            "# Raw tail-50 means below (key<TAB>value).",
        ]
    for k, v in traj10.items():
        lines.append(f"{k}\t{v}")
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", required=True, type=Path)
    p.add_argument("--sig10-tag", default="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_emb10")
    p.add_argument("--sig01-csv", type=Path,
        default=Path("/tmp/contrastive-forecasting-359/reports/2026-06-20_lejepa_sigreg/runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_losses.csv"))
    p.add_argument("--sig01-results", type=Path,
        default=Path("/tmp/contrastive-forecasting-359/reports/2026-06-20_lejepa_sigreg/results"))
    p.add_argument("--sig01-tag", default="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc")
    args = p.parse_args()

    report = args.report_dir
    runs = report / "runs"
    results = report / "results"
    plots = report / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    sig10_csv = runs / f"bb_{args.sig10_tag}_losses.csv"
    if not sig10_csv.exists():
        sys.exit(f"missing: {sig10_csv}")

    ci_rows = compute_bootstrap_cells(
        sig10_results=results, sig10_tag=args.sig10_tag,
        sig01_results=args.sig01_results, sig01_tag=args.sig01_tag,
        out_csv=results / "bootstrap_ci.csv",
    )
    rows = write_gm_table(results, results / "gm_table.csv", args.sig10_tag, ci_rows=ci_rows)
    plot_sigreg_inspection(sig10_csv, args.sig01_csv, plots / "sigreg_e_inspection.png")
    plot_gm_bars(rows, ci_rows, plots / "gm_rel_mase.png")
    plot_perdomain_radar(
        sig10_results=results, sig10_tag=args.sig10_tag,
        sig01_results=args.sig01_results, sig01_tag=args.sig01_tag,
        out=plots / "perdomain_radar.png",
    )

    traj_new   = final_trajectories(sig10_csv)
    traj_ref   = final_trajectories(args.sig01_csv) if args.sig01_csv.exists() else None
    early_new  = early_trajectories(sig10_csv)
    early_ref  = early_trajectories(args.sig01_csv) if args.sig01_csv.exists() else None
    write_trajectories_with_commentary(
        results / "final_trajectories.txt",
        traj_new, traj_ref, early_new, early_ref,
    )
    print("FINAL trajectories (last 50 rows) — λ_e=1.0:")
    for k, v in traj_new.items():
        print(f"  {k}: {v}")
    if traj_ref is not None:
        print("\nReference trajectories (last 50 rows) — λ_e=0.1 (prior arm):")
        for k, v in traj_ref.items():
            print(f"  {k}: {v}")
    print("\nEARLY trajectories (first 50 rows) — λ_e=1.0:")
    for k, v in early_new.items():
        print(f"  {k}: {v}")
    if early_ref is not None:
        print("\nReference EARLY trajectories (first 50 rows) — λ_e=0.1 (prior arm):")
        for k, v in early_ref.items():
            print(f"  {k}: {v}")
    print(f"\nrows in gm_table: {len(rows)}")
    for r in rows:
        if r["arm"] == "sigreg10_enc3":
            gm_d = r["gm_delta_vs_prior"]
            lo, hi = r["gm_delta_lo"], r["gm_delta_hi"]
            ci_str = ""
            if not (isinstance(gm_d, float) and math.isnan(gm_d)):
                ci_str = f"  Δ={gm_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
            print(f"  {r['head']}/{r['ckpt']}  GM={r['gm']:.4f}  (n={r['n']}){ci_str}")
    print("\nBootstrap CI cells (λ_e=1.0 − λ_e=0.1):")
    for r in ci_rows:
        print(f"  {r['head']}/{r['ckpt']}  n={r['n']}  "
              f"GM ratio (10/01) = {r['gm_ratio']:.4f}  "
              f"95% CI [{r['gm_ratio_lo']:.4f}, {r['gm_ratio_hi']:.4f}]  "
              f"abs Δ = {r['gm_delta_abs']:+.4f}  "
              f"95% CI [{r['gm_delta_lo']:+.4f}, {r['gm_delta_hi']:+.4f}]  "
              f"P(Δ<0) = {r['p_below_zero']:.4f}")
