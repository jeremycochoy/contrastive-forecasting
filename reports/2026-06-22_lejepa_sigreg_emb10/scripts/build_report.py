#!/usr/bin/env python3
# #359 SIGReg-emb10 report builder: plots + gm_table.csv from training CSV + GIFT-Eval summaries.
# Compares against four references: enc3+CPC (#344), EMA-target (#353), SIGReg λ_e=λ_h=0.1 (#355),
# and the new SIGReg λ_embedding=1.0 / λ_encoding=0.1 arm (#359).
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
SIGREG01_GM = {  # the #355 SIGReg arm with shared λ=0.1 (transcribed from #355 gm_table.csv)
    ("sigreg01_enc3", "2L", "best"): 1.1610,
    ("sigreg01_enc3", "2L", "last"): 1.1758,
    ("sigreg01_enc3", "6L", "best"): 1.1543,
    ("sigreg01_enc3", "6L", "last"): 1.1556,
}
# #357 (SIGReg + ema-tau 0.98) — PR #358 open, eval not yet landed. NaN row keeps the
# slot visible so the report writer can backfill once #357 lands.
SIGREG_TAU098_GM = {
    ("sigreg_tau098", "2L", "best"): float("nan"),
    ("sigreg_tau098", "2L", "last"): float("nan"),
    ("sigreg_tau098", "6L", "best"): float("nan"),
    ("sigreg_tau098", "6L", "last"): float("nan"),
}

ARM_LABEL = {
    "cpc_enc3":      "enc3+CPC, B=1024",
    "ema_enc3":      "EMA-target enc3+CPC, B=1024",
    "sigreg01_enc3": "SIGReg + EMA-target, B=512 (λ_e=λ_h=0.1)",
    "sigreg10_enc3": "SIGReg + EMA-target, B=512 (λ_e=1.0, λ_h=0.1)",
    "sigreg_tau098": "SIGReg + EMA-target, B=512 (λ_e=λ_h=0.1, τ=0.98) [#357 — pending]",
}

ARM_COLOR = {
    "cpc_enc3":      "#888888",
    "ema_enc3":      "#1f77b4",
    "sigreg01_enc3": "#d62728",
    "sigreg10_enc3": "#2ca02c",
    "sigreg_tau098": "#9467bd",
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
    GM(#359)/GM(#355). Saves bootstrap_ci.csv with absolute and log-delta CIs."""
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
    transcribed; sigreg10 cells are parsed from this experiment's summaries;
    a sigreg_tau098 placeholder row is included so the writer can see the
    pending #357 slot. Bootstrap CI columns (mean delta vs #355 and its 95% CI)
    are merged in for the sigreg10 cells when ci_rows is provided."""
    rows: list[dict] = []
    for d in (REF_GM, EMA_GM, SIGREG01_GM):
        for (arm, head, ckpt), gm in d.items():
            rows.append(dict(arm=arm, label=ARM_LABEL[arm], head=head, ckpt=ckpt, gm=gm, n=97,
                             gm_delta_vs_355=float("nan"),
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
                         gm_delta_vs_355=ci.get("gm_delta_abs", float("nan")),
                         gm_delta_lo=ci.get("gm_delta_lo", float("nan")),
                         gm_delta_hi=ci.get("gm_delta_hi", float("nan")),
                         p_below_zero=ci.get("p_below_zero", float("nan"))))

    for (arm, head, ckpt), gm in SIGREG_TAU098_GM.items():
        rows.append(dict(arm=arm, label=ARM_LABEL[arm], head=head, ckpt=ckpt, gm=gm, n=0,
                         gm_delta_vs_355=float("nan"),
                         gm_delta_lo=float("nan"), gm_delta_hi=float("nan"),
                         p_below_zero=float("nan")))

    def _fmt(v: float, prec: int) -> str:
        if isinstance(v, float) and math.isnan(v):
            return ""
        return f"{v:.{prec}f}"

    fieldnames = ["arm", "label", "head", "ckpt", "gm", "n",
                  "gm_delta_vs_355", "gm_delta_lo", "gm_delta_hi", "p_below_zero"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "arm": r["arm"], "label": r["label"],
                "head": r["head"], "ckpt": r["ckpt"],
                "gm": _fmt(r["gm"], 4),
                "n": r["n"],
                "gm_delta_vs_355": _fmt(r["gm_delta_vs_355"], 4),
                "gm_delta_lo":     _fmt(r["gm_delta_lo"], 4),
                "gm_delta_hi":     _fmt(r["gm_delta_hi"], 4),
                "p_below_zero":    _fmt(r["p_below_zero"], 4),
            })
    return rows


def plot_loss_curves(
    sig10_csv: Path, sig01_csv: Path | None, ema_csv: Path | None, cpc_csv: Path | None, out: Path,
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = pd.read_csv(sig10_csv)
    ax.plot(s["step"], s["loss"].rolling(50, min_periods=1).mean(),
            label=ARM_LABEL["sigreg10_enc3"], color=ARM_COLOR["sigreg10_enc3"], lw=1.6)
    for arm, csv_path in (("sigreg01_enc3", sig01_csv), ("ema_enc3", ema_csv), ("cpc_enc3", cpc_csv)):
        if csv_path and csv_path.exists():
            d = pd.read_csv(csv_path)
            ax.plot(d["step"], d["loss"].rolling(50, min_periods=1).mean(),
                    label=ARM_LABEL[arm], color=ARM_COLOR[arm], lw=1.2)
    ax.set_xlabel("step"); ax.set_ylabel("loss (50-step rolling mean)")
    ax.set_title("Training loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_sigreg_inspection(sig10_csv: Path, sig01_csv: Path | None, out: Path):
    """Compare sigreg_e / sigreg_h / u_batch_e / u_temporal_e between the two
    λ_e weights. All four panels use log y-axis so the bottom row's tiny values
    (1/K ≈ 0.0026, u_batch_e ≈ 0.03–0.04) are readable instead of crammed at
    the baseline of a [0,1] linear axis."""
    s10 = pd.read_csv(sig10_csv)
    s01 = pd.read_csv(sig01_csv) if (sig01_csv and sig01_csv.exists()) else None
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        ("sigreg_e",     "L_SIGReg(e_t)"),
        ("sigreg_h",     "L_SIGReg(h_t)"),
        ("u_batch_e",    "u_batch (e_t)"),
        ("u_temporal_e", "u_temporal (e_t)"),
    ]
    K = 384
    for ax, (col, title) in zip(axes.ravel(), panels):
        ax.plot(s10["step"], s10[col].rolling(50, min_periods=1).mean(),
                label="λ_e=1.0 (this arm)", color=ARM_COLOR["sigreg10_enc3"], lw=1.6)
        if s01 is not None and col in s01.columns:
            ax.plot(s01["step"], s01[col].rolling(50, min_periods=1).mean(),
                    label="λ_e=0.1 (prior arm)", color=ARM_COLOR["sigreg01_enc3"], lw=1.2)
        if col in ("u_batch_e", "u_temporal_e"):
            ax.axhline(1.0 / K, color="k", ls=":", alpha=0.5,
                       label=f"1/K = 1/{K} ≈ {1/K:.4f}")
        ax.set_yscale("log")
        ax.set_xlabel("step"); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.suptitle("Embedding-side SIGReg trajectory: λ_e=1.0 vs λ_e=0.1 — log y-axis on all panels")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_uniformity(
    sig10_csv: Path, sig01_csv: Path | None, ema_csv: Path | None, cpc_csv: Path | None, out: Path,
):
    """h_t uniformity only — e_t curves are crushed at the [0, 1] baseline and
    are already shown clearly on the log y-axis of sigreg_e_inspection.png."""
    s10 = pd.read_csv(sig10_csv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, kind in zip(axes, ("batch", "temporal")):
        ax.plot(s10["step"], s10[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                label=f"u_{kind} (h_t) — {ARM_LABEL['sigreg10_enc3']}",
                color=ARM_COLOR["sigreg10_enc3"], lw=1.6)
        if sig01_csv and sig01_csv.exists():
            s01 = pd.read_csv(sig01_csv)
            ax.plot(s01["step"], s01[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                    label=f"u_{kind} (h_t) — {ARM_LABEL['sigreg01_enc3']}",
                    color=ARM_COLOR["sigreg01_enc3"], lw=1.2)
        for arm, csv_path in (("ema_enc3", ema_csv), ("cpc_enc3", cpc_csv)):
            if csv_path and csv_path.exists():
                d = pd.read_csv(csv_path)
                if f"u_{kind}" in d.columns:
                    ax.plot(d["step"], d[f"u_{kind}"].rolling(50, min_periods=1).mean(),
                            label=f"u_{kind} (h_t) — {ARM_LABEL[arm]}",
                            color=ARM_COLOR[arm], lw=0.9)
        ax.set_xlabel("step")
        ax.set_ylabel("effective dimensionality (h_t)")
        ax.set_title(f"u_{kind} ({'cross-batch' if kind=='batch' else 'cross-time'})")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Encoder-output uniformity h_t (cos²-based dim_usage; clipped to [1/K, 1]). "
                 "e_t curves are on sigreg_e_inspection.png.")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def plot_gm_bars(rows: list[dict], ci_rows: list[dict], out: Path):
    """Grouped bars over the 4 (head, ckpt) cells with per-arm GM-Rel MASE.
    For the #359 (sigreg10) arm only, overlay paired-bootstrap 95% CI whiskers
    on absolute-GM scale (re-anchored at #355's GM as reference) and draw a
    horizontal tick at each #355 cell to anchor #359 against the direct reference."""
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
        # Whiskers + #355 anchor only on the #359 bars.
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
    # Horizontal tick for each #355 cell — spans the #355 + #359 bar pair so the
    # eye can compare directly.
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
    m: dict[str, str] = {}
    if not all_csv.exists():
        return m
    for r in csv.DictReader(open(all_csv)):
        m[r["dataset"]] = r.get("domain", "Other")
    return m


def _config_relatives(summary_txt: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not summary_txt.exists():
        return out
    for line in open(summary_txt):
        p = line.split()
        if len(p) == 4 and "/" in p[0]:
            try:
                out[p[0]] = float(p[3])
            except ValueError:
                pass
    return out


def _gm_by_domain(rels: dict[str, float], dmap: dict[str, str]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for cfg, rel in rels.items():
        dom = dmap.get(cfg)
        if rel <= 0 or dom is None:
            continue
        acc.setdefault(dom, []).append(math.log(rel))
    return {d: math.exp(sum(v) / len(v)) for d, v in acc.items()}


def _domain_bootstrap_ci(
    rels: dict[str, float], dmap: dict[str, str], B: int = 10_000, ci: float = 0.95,
    seed: int = 20260622,
) -> dict[str, tuple[float, float, float]]:
    """Per-domain bootstrap CI on GM. Returns {domain: (gm, lo, hi)}."""
    rng = np.random.default_rng(seed)
    acc: dict[str, list[float]] = {}
    for cfg, rel in rels.items():
        dom = dmap.get(cfg)
        if rel <= 0 or dom is None:
            continue
        acc.setdefault(dom, []).append(math.log(rel))
    out: dict[str, tuple[float, float, float]] = {}
    for d, logs in acc.items():
        arr = np.array(logs)
        n = len(arr)
        if n == 0:
            continue
        idx = rng.integers(0, n, size=(B, n))
        boot = np.exp(arr[idx].mean(axis=1))
        out[d] = (
            float(np.exp(arr.mean())),
            float(np.quantile(boot, (1 - ci) / 2)),
            float(np.quantile(boot, 1 - (1 - ci) / 2)),
        )
    return out


def plot_perdomain_radar(
    sig10_results: Path, sig01_results: Path, ema_results: Path, cpc_results: Path,
    sig10_tag: str, sig01_tag: str, ema_tag: str, cpc_tag: str, out: Path,
):
    """2×2 small-multiples (head ∈ {2L,6L} × ckpt ∈ {best,last}), 4 arm curves
    per panel — split out from the previous 8-curve overlay so the green (#359)
    and red (#355) SIGReg curves no longer merge. The #359 curve in each panel
    (using that panel's own ckpt) carries the per-domain bootstrap 95% CI as a
    shaded radial fill."""
    HEADS = ["2L", "6L"]
    CKPTS = [("best", ""), ("last", "_last")]
    ARM_SPECS = [
        ("cpc_enc3",      cpc_results,   cpc_tag,   ARM_COLOR["cpc_enc3"]),
        ("ema_enc3",      ema_results,   ema_tag,   ARM_COLOR["ema_enc3"]),
        ("sigreg01_enc3", sig01_results, sig01_tag, ARM_COLOR["sigreg01_enc3"]),
        ("sigreg10_enc3", sig10_results, sig10_tag, ARM_COLOR["sigreg10_enc3"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), subplot_kw=dict(polar=True))
    for row, head in enumerate(HEADS):
        for colj, (ckpt, suf) in enumerate(CKPTS):
            ax = axes[row, colj]
            cells: list[tuple[str, dict[str, float], str]] = []
            sig10_ci: dict[str, tuple[float, float, float]] | None = None
            for arm, root, tag, colr in ARM_SPECS:
                sub = root / f"gift_eval_full_{tag}{suf}_{head}"
                rel = _config_relatives(sub / "summary.txt")
                dmap = _dataset_to_domain(sub / "all_results.csv")
                gm = _gm_by_domain(rel, dmap)
                if gm:
                    cells.append((ARM_LABEL[arm], gm, colr))
                if arm == "sigreg10_enc3" and rel and dmap:
                    sig10_ci = _domain_bootstrap_ci(rel, dmap)
            if not cells:
                ax.text(0.5, 0.5, "no eval", transform=ax.transAxes)
                continue
            domains = sorted(set().union(*(g for _, g, _ in cells)))
            theta = np.linspace(0, 2 * np.pi, len(domains), endpoint=False)
            theta_closed = np.concatenate([theta, theta[:1]])
            vals = [v for _, g, _ in cells for v in g.values()]
            if sig10_ci:
                vals.extend(v for _, lo, hi in sig10_ci.values() for v in (lo, hi))
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
            if sig10_ci:
                lo_v = np.array([sig10_ci.get(d, (np.nan, np.nan, np.nan))[1] for d in domains])
                hi_v = np.array([sig10_ci.get(d, (np.nan, np.nan, np.nan))[2] for d in domains])
                lo_v = np.concatenate([lo_v, lo_v[:1]])
                hi_v = np.concatenate([hi_v, hi_v[:1]])
                ax.fill_between(theta_closed, lo_v, hi_v,
                                color=ARM_COLOR["sigreg10_enc3"], alpha=0.20, zorder=2,
                                label=f"λ_e=1.0 · {ckpt}  bootstrap 95% CI")
            for lab, g, colr in cells:
                v = np.array([g.get(d, np.nan) for d in domains]
                             + [g.get(domains[0], np.nan)])
                ax.plot(theta_closed, v, color=colr, lw=1.6, zorder=3,
                        marker="o", markersize=3.5, label=lab)
            ax.set_title(f"{head} q-head · {ckpt}", fontsize=11, pad=12)
            ax.legend(loc="upper left", bbox_to_anchor=(-0.10, -0.05),
                      fontsize=7, frameon=False, ncol=1)
    fig.suptitle(
        "Per-domain GM relative MASE on GIFT-Eval full-97 "
        "(radial log scale; ring at 1.0 = seasonal-naive; lower = better; "
        "4-curve small-multiples × {2L,6L} × {best,last}; shaded = λ_e=1.0 per-domain bootstrap 95% CI)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_per_config_delta(
    sig10_results: Path, sig01_results: Path,
    sig10_tag: str, sig01_tag: str,
    ci_rows: list[dict],
    out: Path,
):
    """For each (head, ckpt), show the per-config rel-MASE delta (#359 − #355)
    across the 97 configs as a strip-scatter; overlay the absolute GM delta
    (#359 − #355) ± its paired-bootstrap 95% CI computed on the log-ratio of
    GMs — the same CI methodology used by gm_rel_mase.png / gm_table.csv, so
    a single CI methodology is reported throughout. Negative = #359 better."""
    cells = [("2L", "best", ""), ("2L", "last", "_last"),
             ("6L", "best", ""), ("6L", "last", "_last")]
    ci_by_cell = {(r["head"], r["ckpt"]): r for r in ci_rows}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    rng = np.random.default_rng(20260622)
    for ax, (head, ckpt, suf) in zip(axes, cells):
        sub10 = sig10_results / f"gift_eval_full_{sig10_tag}{suf}_{head}" / "summary.txt"
        sub01 = sig01_results / f"gift_eval_full_{sig01_tag}{suf}_{head}" / "summary.txt"
        r10 = parse_per_config_rel_mase(sub10)
        r01 = parse_per_config_rel_mase(sub01)
        common = sorted(set(r10) & set(r01))
        d = np.array([r10[c] - r01[c] for c in common])
        x_jit = rng.uniform(-0.18, 0.18, size=len(d))
        ax.axhline(0, color="k", lw=0.7, alpha=0.6)
        ax.scatter(x_jit, d, s=9, alpha=0.55, color=ARM_COLOR["sigreg10_enc3"])
        ci = ci_by_cell.get((head, ckpt), {})
        # Use the same CI as gm_rel_mase.png / gm_table.csv: paired bootstrap
        # on the log-ratio of GMs, converted to absolute GM-delta scale via
        # GM(#355) * (exp(log_ratio_quantile) - 1). gm_delta_abs / lo / hi are
        # already in that scale in ci_by_cell.
        gm_delta = ci.get("gm_delta_abs", float("nan"))
        gm_lo    = ci.get("gm_delta_lo",  float("nan"))
        gm_hi    = ci.get("gm_delta_hi",  float("nan"))
        p_neg    = ci.get("p_below_zero", float("nan"))
        if not (isinstance(gm_delta, float) and math.isnan(gm_delta)):
            ax.errorbar([0], [gm_delta],
                        yerr=[[gm_delta - gm_lo], [gm_hi - gm_delta]],
                        fmt="D", color="k", ecolor="k",
                        capsize=5, capthick=1.2, elinewidth=1.2, markersize=6,
                        zorder=5,
                        label=(
                            f"Δ_GM={gm_delta:+.3f}\n"
                            f"95% CI [{gm_lo:+.3f}, {gm_hi:+.3f}]\n"
                            f"P(Δ<0)={p_neg:.2f}"
                        ))
        ax.set_xlim(-0.45, 0.45); ax.set_xticks([])
        ax.set_title(f"{head}/{ckpt}\nn={len(d)}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel("Per-config rel-MASE  Δ = λ_e=1.0 − λ_e=0.1\n(< 0 → λ_e=1.0 better on that config)")
    fig.suptitle(
        "Per-config rel-MASE deltas across the 97 GIFT-Eval configs (scatter); "
        "black diamond = Δ_GM(λ_e=1.0 − λ_e=0.1) with 95% CI from paired bootstrap on "
        "log-ratio of GMs — same statistic as gm_rel_mase.png / gm_table.csv",
        fontsize=10,
    )
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def final_trajectories(sigreg_csv: Path, n_tail: int = 50) -> dict[str, float]:
    s = pd.read_csv(sigreg_csv)
    tail = s.tail(n_tail)
    out = {
        "u_batch":      float(tail["u_batch"].mean()),
        "u_batch_e":    float(tail["u_batch_e"].mean()),
        "u_temporal":   float(tail["u_temporal"].mean()),
        "u_temporal_e": float(tail["u_temporal_e"].mean()),
        "sigreg_e":     float(tail["sigreg_e"].mean()),
        "sigreg_h":     float(tail["sigreg_h"].mean()),
        "loss":         float(tail["loss"].mean()),
        "final_step":   int(s["step"].iloc[-1]),
    }
    return out


def write_trajectories_with_commentary(
    out_path: Path, traj10: dict[str, float], traj01: dict[str, float] | None,
    K: int = 384,
):
    """Replace the legacy bare-key-value trajectories file with a commentary block
    that calls the embedding-side uniformity direction correctly (it FELL under
    a 10x λ_e bump, did not lift off) plus the per-term sigreg → total-loss
    fraction note (λ_e * sigreg_e / loss went ~7.6x while λ_e went 10x because
    sigreg_e itself partially self-suppressed)."""
    lines: list[str] = [
        "# VERDICT (issue #359, single seed 20260520 as specified):",
        '#   * Q1 "Does the bumped weight wake the embedding-side term?"  → NO.',
        "#     u_batch_e moved TOWARD 1/K (16.8× → 13.0×), not away. u_temporal_e,",
        "#     u_batch, u_temporal all also fell. The 10× λ_e bump did not lift e_t",
        "#     uniformity off the 1/K floor.",
        '#   * Q2 "Does it move downstream?"  → NO (not at α=0.05).',
        "#     Point Δ on GM-Rel MASE = #359 − #355 in [−0.014, −0.007] across the",
        "#     4 (head, ckpt) cells; ALL FOUR paired-bootstrap 95% CIs (B=10 000,",
        "#     N=97 per-config rel-MASE deltas) include zero. P(Δ<0) in [0.83, 0.95].",
        "#     Direction is consistent (4/4 cells point-negative) but magnitudes are",
        "#     not separable from single-seed paired-config noise.",
        "#",
    ]
    if traj01:
        u_b_e_10 = traj10["u_batch_e"]; u_b_e_01 = traj01["u_batch_e"]
        u_t_e_10 = traj10["u_temporal_e"]; u_t_e_01 = traj01["u_temporal_e"]
        u_b_10 = traj10["u_batch"]; u_b_01 = traj01["u_batch"]
        u_t_10 = traj10["u_temporal"]; u_t_01 = traj01["u_temporal"]
        sr_e_10 = traj10["sigreg_e"]; sr_e_01 = traj01["sigreg_e"]
        sr_h_10 = traj10["sigreg_h"]; sr_h_01 = traj01["sigreg_h"]
        loss_10 = traj10["loss"]; loss_01 = traj01["loss"]
        lam_e_10, lam_e_01 = 1.0, 0.1
        ratio_loss_10 = lam_e_10 * sr_e_10 / loss_10
        ratio_loss_01 = lam_e_01 * sr_e_01 / loss_01
        lines += [
            "# Tail-50 trajectory facts (#359 SIGReg λ_e=1.0 vs #355 SIGReg λ_e=0.1)",
            "#",
            "# u_batch_e DIRECTION (corrects the earlier PR #360 comment that said",
            "# u_batch_e 'lifts off' under λ_e=1.0 — the data says the opposite).",
            f"#   u_batch_e:     #355 λ_e=0.1 → {u_b_e_01:.4f} ≈ {u_b_e_01 * K:.1f}× 1/K  ",
            f"#                  #359 λ_e=1.0 → {u_b_e_10:.4f} ≈ {u_b_e_10 * K:.1f}× 1/K   "
            "(FELL — moved TOWARD 1/K, not away)",
            f"#   u_temporal_e:  #355 → {u_t_e_01:.4f}  →  #359 → {u_t_e_10:.4f}   (also fell)",
            f"#   u_batch:       #355 → {u_b_01:.4f}  →  #359 → {u_b_10:.4f}     (also fell)",
            f"#   u_temporal:    #355 → {u_t_01:.4f}  →  #359 → {u_t_10:.4f}     (also fell)",
            "#",
            "# SIGReg λ_e · sigreg_e / total-loss fraction:",
            f"#   #355 λ_e=0.1:  λ_e · sigreg_e / loss = {ratio_loss_01:.2e}",
            f"#   #359 λ_e=1.0:  λ_e · sigreg_e / loss = {ratio_loss_10:.2e}   "
            f"(~{ratio_loss_10 / ratio_loss_01:.1f}× under a 10× λ_e bump because",
            f"#                  sigreg_e itself fell {sr_e_01:.2e} → {sr_e_10:.2e} — partial self-suppression).",
            "#",
            f"#   sigreg_h tail-50: #355 → {sr_h_01:.2e}  →  #359 → {sr_h_10:.2e}",
            f"#   loss     tail-50: #355 → {loss_01:.4f}    →  #359 → {loss_10:.4f}",
            "#",
            "# #357 (PR #358, --ema-tau 0.98) — eval has not yet landed; the 4 cells",
            "# carry NaN placeholders in gm_table.csv so the report writer can backfill.",
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
    p.add_argument("--ema-csv", type=Path,
        default=Path("/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-19_ema_target_encoder/runs/bb_allt08_xftrip_nobn_enc3_emateach_qk_aon_b1024_cpc_losses.csv"))
    p.add_argument("--cpc-csv", type=Path,
        default=Path("/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc_losses.csv"))
    p.add_argument("--ema-results", type=Path,
        default=Path("/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-19_ema_target_encoder/results"))
    p.add_argument("--cpc-results", type=Path,
        default=Path("/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux/results"))
    p.add_argument("--ema-tag", default="allt08_xftrip_nobn_enc3_emateach_qk_aon_b1024_cpc")
    p.add_argument("--cpc-tag", default="allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc")
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
    plot_loss_curves(sig10_csv, args.sig01_csv, args.ema_csv, args.cpc_csv, plots / "loss_curve.png")
    plot_sigreg_inspection(sig10_csv, args.sig01_csv, plots / "sigreg_e_inspection.png")
    plot_uniformity(sig10_csv, args.sig01_csv, args.ema_csv, args.cpc_csv, plots / "uniformity.png")
    plot_gm_bars(rows, ci_rows, plots / "gm_rel_mase.png")
    plot_perdomain_radar(
        sig10_results=results, sig01_results=args.sig01_results,
        ema_results=args.ema_results, cpc_results=args.cpc_results,
        sig10_tag=args.sig10_tag, sig01_tag=args.sig01_tag,
        ema_tag=args.ema_tag, cpc_tag=args.cpc_tag,
        out=plots / "perdomain_radar.png",
    )
    plot_per_config_delta(
        sig10_results=results, sig01_results=args.sig01_results,
        sig10_tag=args.sig10_tag, sig01_tag=args.sig01_tag,
        ci_rows=ci_rows, out=plots / "per_config_delta.png",
    )

    traj_new = final_trajectories(sig10_csv)
    traj_ref = final_trajectories(args.sig01_csv) if args.sig01_csv.exists() else None
    write_trajectories_with_commentary(
        results / "final_trajectories.txt", traj_new, traj_ref,
    )
    print("FINAL trajectories (last 50 rows) — λ_e=1.0:")
    for k, v in traj_new.items():
        print(f"  {k}: {v}")
    if traj_ref is not None:
        print("\nReference trajectories (last 50 rows) — λ_e=0.1 (#355):")
        for k, v in traj_ref.items():
            print(f"  {k}: {v}")
    print(f"\nrows in gm_table: {len(rows)}")
    for r in rows:
        if r["arm"] == "sigreg10_enc3":
            gm_d = r["gm_delta_vs_355"]
            lo, hi = r["gm_delta_lo"], r["gm_delta_hi"]
            ci_str = ""
            if not (isinstance(gm_d, float) and math.isnan(gm_d)):
                ci_str = f"  Δ={gm_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
            print(f"  {r['head']}/{r['ckpt']}  GM={r['gm']:.4f}  (n={r['n']}){ci_str}")
    print("\nBootstrap CI cells (#359 − #355):")
    for r in ci_rows:
        print(f"  {r['head']}/{r['ckpt']}  n={r['n']}  "
              f"GM ratio (10/01) = {r['gm_ratio']:.4f}  "
              f"95% CI [{r['gm_ratio_lo']:.4f}, {r['gm_ratio_hi']:.4f}]  "
              f"abs Δ = {r['gm_delta_abs']:+.4f}  "
              f"95% CI [{r['gm_delta_lo']:+.4f}, {r['gm_delta_hi']:+.4f}]  "
              f"P(Δ<0) = {r['p_below_zero']:.4f}")
