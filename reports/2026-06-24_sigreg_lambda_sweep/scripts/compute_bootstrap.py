#!/usr/bin/env python3
# #363 SIGReg λ-sweep — paired bootstrap CI builder.
#
# Statistic: mean(log(rel_A) − log(rel_B)) over the 97 GIFT-Eval configs,
# B=10000, seed 20260624. Aligned per config. Converts the log-delta CI to
# the absolute GM-Rel MASE scale via GM_B * (exp(quantile) − 1).
#
# Anchors: per-cell summary.txt under prior report dirs; sweep arms: per-cell
# summary.txt under experiments/2026-06-24_sigreg_lambda_sweep/results/.
#
# Outputs:
#   results/bootstrap_ci_vs_359.csv  — every sweep arm vs #359 (sigreg10) per cell
#   results/bootstrap_ci_vs_arm1.csv — arms 2/3/5 vs arm 1 (emb100_enc01) per cell
#   results/gm_table.csv             — anchors + sweep arms × 4 cells
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ANCHOR_GM = {
    ("cpc_enc3",      "2L", "best"): 1.1846,
    ("cpc_enc3",      "2L", "last"): 1.1531,
    ("cpc_enc3",      "6L", "best"): 1.1584,
    ("cpc_enc3",      "6L", "last"): 1.1436,
    ("ema_enc3",      "2L", "best"): 1.1614,
    ("ema_enc3",      "2L", "last"): 1.1817,
    ("ema_enc3",      "6L", "best"): 1.1576,
    ("ema_enc3",      "6L", "last"): 1.1597,
    ("sigreg01_enc3", "2L", "best"): 1.1610,
    ("sigreg01_enc3", "2L", "last"): 1.1758,
    ("sigreg01_enc3", "6L", "best"): 1.1543,
    ("sigreg01_enc3", "6L", "last"): 1.1556,
    ("sigreg10_enc3", "2L", "best"): 1.1470,
    ("sigreg10_enc3", "2L", "last"): 1.1681,
    ("sigreg10_enc3", "6L", "best"): 1.1408,
    ("sigreg10_enc3", "6L", "last"): 1.1482,
}

ARM_LABEL = {
    "cpc_enc3":          "enc3+CPC, B=1024",
    "ema_enc3":          "EMA-target enc3+CPC, B=1024",
    "sigreg01_enc3":     "SIGReg λ_e=0.1, λ_h=0.1",
    "sigreg10_enc3":     "SIGReg λ_e=1.0, λ_h=0.1",
    "emb100_enc01":      "SIGReg λ_e=10.0, λ_h=0.1",
    "emb100_enc10":      "SIGReg λ_e=10.0, λ_h=1.0",
    "emb100_enc100":     "SIGReg λ_e=10.0, λ_h=10.0",
    "emb10_enc10":       "SIGReg λ_e=1.0, λ_h=1.0",
    "emb1000_enc01":     "SIGReg λ_e=100.0, λ_h=0.1",
    "emb10000_enc10":    "SIGReg λ_e=1000.0, λ_h=1.0",
}

SWEEP_ARMS = [
    "emb100_enc01", "emb100_enc10", "emb100_enc100",
    "emb10_enc10", "emb1000_enc01", "emb10000_enc10",
]
ANCHOR_ORDER = ["cpc_enc3", "ema_enc3", "sigreg01_enc3", "sigreg10_enc3"]
CELLS = [("2L", "best", ""), ("2L", "last", "_last"),
         ("6L", "best", ""), ("6L", "last", "_last")]

BASE_TAG = "allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc"


def parse_per_config_rel_mase(summary_path: Path) -> dict[str, float]:
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


def parse_gm(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    txt = summary_path.read_text()
    m = re.search(r"Aggregate GM-Relative MASE.*?:\s*([0-9.]+)", txt)
    return float(m.group(1)) if m else None


def paired_bootstrap_ci_log(
    a: np.ndarray, b: np.ndarray, B: int = 10_000, ci: float = 0.95, seed: int = 20260624,
):
    rng = np.random.default_rng(seed)
    la, lb = np.log(a), np.log(b)
    d = la - lb
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    boot = d[idx].mean(axis=1)
    lo = float(np.quantile(boot, (1 - ci) / 2))
    hi = float(np.quantile(boot, 1 - (1 - ci) / 2))
    return float(d.mean()), lo, hi, float((boot < 0).mean())


def sweep_summary(results: Path, suffix: str, head: str, ckpt_suf: str) -> Path:
    return results / f"gift_eval_full_{BASE_TAG}_{suffix}{ckpt_suf}_{head}" / "summary.txt"


def anchor_summary(anchor_results: Path, anchor_tag: str, head: str, ckpt_suf: str) -> Path:
    return anchor_results / f"gift_eval_full_{anchor_tag}{ckpt_suf}_{head}" / "summary.txt"


def bootstrap_table(arm_results: Path, baseline_results: Path, baseline_tag: str,
                    sweep_arms: list[str]) -> list[dict]:
    rows: list[dict] = []
    for arm in sweep_arms:
        for head, ckpt, suf in CELLS:
            a_path = sweep_summary(arm_results, arm, head, suf)
            b_path = anchor_summary(baseline_results, baseline_tag, head, suf)
            ra = parse_per_config_rel_mase(a_path)
            rb = parse_per_config_rel_mase(b_path)
            common = sorted(set(ra) & set(rb))
            if not common:
                rows.append(dict(arm=arm, head=head, ckpt=ckpt, n=0,
                                 mean_log_delta=float("nan"),
                                 ci_lo_log=float("nan"), ci_hi_log=float("nan"),
                                 gm_a=float("nan"), gm_b=float("nan"),
                                 gm_delta_abs=float("nan"),
                                 gm_delta_lo=float("nan"), gm_delta_hi=float("nan"),
                                 p_below_zero=float("nan")))
                continue
            a = np.array([ra[c] for c in common])
            b = np.array([rb[c] for c in common])
            m, lo, hi, p_neg = paired_bootstrap_ci_log(a, b)
            gm_a = float(np.exp(np.log(a).mean()))
            gm_b = float(np.exp(np.log(b).mean()))
            rows.append(dict(
                arm=arm, head=head, ckpt=ckpt, n=len(common),
                mean_log_delta=m,
                ci_lo_log=lo, ci_hi_log=hi,
                gm_a=gm_a, gm_b=gm_b,
                gm_delta_abs=gm_b * (float(np.exp(m)) - 1.0),
                gm_delta_lo=gm_b * (float(np.exp(lo)) - 1.0),
                gm_delta_hi=gm_b * (float(np.exp(hi)) - 1.0),
                p_below_zero=p_neg,
            ))
    return rows


def write_csv(rows: list[dict], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("")
        return
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            row = {}
            for k, v in r.items():
                if isinstance(v, float):
                    row[k] = f"{v:.6f}" if abs(v) < 1 else f"{v:.4f}"
                else:
                    row[k] = v
            w.writerow(row)


def write_gm_table(arm_results: Path, out_csv: Path) -> list[dict]:
    rows: list[dict] = []
    for (arm, head, ckpt), gm in ANCHOR_GM.items():
        rows.append(dict(arm=arm, label=ARM_LABEL[arm], head=head, ckpt=ckpt, gm=gm, n=97))
    for arm in SWEEP_ARMS:
        for head, ckpt, suf in CELLS:
            gm = parse_gm(sweep_summary(arm_results, arm, head, suf))
            if gm is not None:
                rows.append(dict(arm=arm, label=ARM_LABEL[arm], head=head, ckpt=ckpt, gm=gm, n=97))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "label", "head", "ckpt", "gm", "n"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "gm": f"{r['gm']:.4f}"})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm-results", type=Path, required=True,
                   help="experiments/2026-06-24_sigreg_lambda_sweep/results")
    p.add_argument("--sig10-results", type=Path, required=True,
                   help="reports/2026-06-22_lejepa_sigreg_emb10/results")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    out_results = args.out_dir / "results"
    out_results.mkdir(parents=True, exist_ok=True)

    rows = write_gm_table(args.arm_results, out_results / "gm_table.csv")
    print(f"gm_table.csv rows: {len(rows)}")

    sigreg10_tag = f"{BASE_TAG}_emb10"
    rows_vs_359 = bootstrap_table(args.arm_results, args.sig10_results, sigreg10_tag, SWEEP_ARMS)
    write_csv(rows_vs_359, out_results / "bootstrap_ci_vs_359.csv")
    print(f"bootstrap_ci_vs_359.csv rows: {len(rows_vs_359)}")

    arms_vs_arm1 = [a for a in SWEEP_ARMS if a != "emb100_enc01"]
    arm1_tag = f"{BASE_TAG}_emb100_enc01"
    rows_vs_arm1 = bootstrap_table(args.arm_results, args.arm_results, arm1_tag, arms_vs_arm1)
    write_csv(rows_vs_arm1, out_results / "bootstrap_ci_vs_arm1.csv")
    print(f"bootstrap_ci_vs_arm1.csv rows: {len(rows_vs_arm1)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
