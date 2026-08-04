#!/usr/bin/env python3
"""Training-stability read-out for every backbone in the report.

Three cells sit far off their neighbours: arm6_v2 base jumps 1.4322 -> 1.9057
between 40k and 100k, arm5_nse reads 1.8887 at 200k, and the copied
arm1_combab reads 3.1251 at 40k. A cell that diverged and a cell that trained
cleanly to a bad optimum are different facts and belong in the report as
different sentences, so this measures which each one is instead of guessing.

Every backbone is scored, not just the three, because "unusual" only means
something against the other 30. Per run, over the concatenation of its
`_losses.csv` legs in step order:

  n_nonfinite     NaN or inf in loss / gap. Any is divergence.
  loss_final      mean loss over the last 5% of steps.
  loss_min        minimum over the whole run.
  rise_from_min   loss_final - loss_min. Large means the run walked back up.
  max_jump_iqr    biggest step-to-step loss jump, in units of the run's own
                  inter-quartile range. A spike shows here and nowhere else.
  qk_max          largest attention logit magnitude seen (attn_amplitude).
  resid_max       largest post-FFN residual magnitude.
  qk_trend        last-decile mean / first-decile mean of qk_logit_maxabs.
                  Unbounded growth is the amplitude failure mode this
                  diagnostic exists to catch.

Usage:
    python3 inspect_anomalies.py --results <report results dir> \
        --extra-results <other report results dir> --out anomaly.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict

import numpy as np

LEG_RE = re.compile(r"^(?P<base>.+?)(?:_r(?P<leg>\d+))?_losses\.csv$")
ATTN_RE = re.compile(r"^(?P<base>.+?)(?:_r(?P<leg>\d+))?_attn_amplitude\.csv$")

# The cells the review named, by backbone run-name stem.
FLAGGED = {
    "bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher":
        "arm6_v2 base — 1.4322 at 40k, 1.9057 at 100k",
    "bb_small_arm5_nse_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_alignteacher":
        "arm5_nse — 1.8887 at 200k",
    "bb_small_arm1_combab_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090":
        "arm1_combab (copied from #379) — 3.1251 at 40k",
}


def group_legs(dirpath: str, pattern: re.Pattern) -> dict[str, list[str]]:
    out: dict[str, list[tuple[int, str]]] = defaultdict(list)
    if not os.path.isdir(dirpath):
        return {}
    for name in os.listdir(dirpath):
        m = pattern.match(name)
        if not m:
            continue
        leg = int(m.group("leg") or 1)
        out[m.group("base")].append((leg, os.path.join(dirpath, name)))
    return {b: [p for _, p in sorted(v)] for b, v in out.items()}


def read_col(paths: list[str], col: str) -> tuple[np.ndarray, np.ndarray]:
    """(step, value) over the concatenated legs, ordered by step."""
    steps, vals = [], []
    for p in paths:
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                v = row.get(col, "")
                if v == "":
                    continue
                steps.append(float(row["step"]))
                vals.append(float(v))
    if not steps:
        return np.array([]), np.array([])
    s = np.array(steps)
    v = np.array(vals)
    order = np.argsort(s, kind="stable")
    return s[order], v[order]


def loss_stats(paths: list[str]) -> dict:
    step, loss = read_col(paths, "loss")
    _, gap = read_col(paths, "gap")
    if loss.size == 0:
        return {}
    finite = np.isfinite(loss)
    n_nonfinite = int((~finite).sum() + (~np.isfinite(gap)).sum())
    lf = loss[finite]
    tail = max(1, int(0.05 * lf.size))
    q75, q25 = np.percentile(lf, [75, 25])
    iqr = max(float(q75 - q25), 1e-9)
    jumps = np.abs(np.diff(lf))
    return {
        "n_rows": int(loss.size),
        "step_max": int(step.max()),
        "n_nonfinite": n_nonfinite,
        "loss_final": float(lf[-tail:].mean()),
        "loss_min": float(lf.min()),
        "loss_max": float(lf.max()),
        "rise_from_min": float(lf[-tail:].mean() - lf.min()),
        "max_jump_iqr": float(jumps.max() / iqr) if jumps.size else 0.0,
    }


def attn_stats(paths: list[str]) -> dict:
    step, qk = read_col(paths, "qk_logit_maxabs")
    _, resid = read_col(paths, "resid_post_ffn_maxabs")
    if qk.size == 0:
        return {"qk_max": "", "resid_max": "", "qk_trend": ""}
    d = max(1, qk.size // 10)
    first, last = qk[:d].mean(), qk[-d:].mean()
    return {
        "qk_max": float(np.nanmax(qk)),
        "resid_max": float(np.nanmax(resid)) if resid.size else "",
        "qk_trend": float(last / first) if first > 0 else "",
    }


def pct_rank(values: list[float], x: float) -> float:
    v = np.array([t for t in values if t is not None and np.isfinite(t)])
    return float((v <= x).mean() * 100.0)


# The report measures at wave ends, so a whole-run summary can hide a wave.
# arm6_v2's jump is entirely inside 40k -> 100k.
WINDOWS = ((0, 40000), (40000, 100000), (100000, 200000))


def window_rows(base: str, curve_paths: list[str],
                attn_paths: list[str]) -> list[dict]:
    step, loss = read_col(curve_paths, "loss")
    astep, qk = read_col(attn_paths, "qk_logit_maxabs")
    out = []
    for lo, hi in WINDOWS:
        m = (step > lo) & (step <= hi)
        if not m.any():
            continue
        w = loss[m]
        w = w[np.isfinite(w)]
        if w.size == 0:
            continue
        tail = max(1, int(0.05 * w.size))
        am = (astep > lo) & (astep <= hi)
        qw = qk[am] if qk.size else np.array([])
        row = {
            "run": base, "window": f"{lo//1000}k-{hi//1000}k",
            "n_rows": int(w.size),
            "loss_first": f"{float(w[:tail].mean()):.4f}",
            "loss_last": f"{float(w[-tail:].mean()):.4f}",
            "loss_min": f"{float(w.min()):.4f}",
            "delta": f"{float(w[-tail:].mean() - w[:tail].mean()):+.4f}",
            "qk_max": f"{float(np.nanmax(qw)):.2f}" if qw.size else "",
            "qk_mean": f"{float(np.nanmean(qw)):.2f}" if qw.size else "",
        }
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--extra-results", default=None,
                    help="second report's results dir, for copied cells")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-windows", default=None,
                    help="per-wave-window breakdown for every run")
    args = ap.parse_args()

    curves: dict[str, list[str]] = {}
    attn: dict[str, list[str]] = {}
    for root in filter(None, (args.results, args.extra_results)):
        for base, paths in group_legs(os.path.join(root, "training_curves"),
                                      LEG_RE).items():
            curves.setdefault(base, paths)
        for base, paths in group_legs(os.path.join(root, "attn_amplitude"),
                                      ATTN_RE).items():
            attn.setdefault(base, paths)

    rows = []
    for base in sorted(curves):
        st = loss_stats(curves[base])
        if not st:
            continue
        st.update(attn_stats(attn.get(base, [])))
        st["run"] = base
        st["flagged"] = FLAGGED.get(base, "")
        rows.append(st)

    fields = ["run", "flagged", "n_rows", "step_max", "n_nonfinite",
              "loss_final", "loss_min", "loss_max", "rise_from_min",
              "max_jump_iqr", "qk_max", "resid_max", "qk_trend"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float)
                            else r.get(k, "")) for k in fields})

    if args.out_windows:
        wrows = []
        for base in sorted(curves):
            wrows += window_rows(base, curves[base], attn.get(base, []))
        with open(args.out_windows, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(wrows[0]))
            w.writeheader()
            w.writerows(wrows)

    print(f"{len(rows)} backbones scored\n")
    print(f"any non-finite loss/gap anywhere: "
          f"{sum(r['n_nonfinite'] for r in rows)}")
    for key in ("rise_from_min", "max_jump_iqr", "qk_max", "qk_trend"):
        vals = [r[key] for r in rows
                if isinstance(r.get(key), float) and math.isfinite(r[key])]
        if vals:
            print(f"  {key:14s} median={np.median(vals):8.3f} "
                  f"p90={np.percentile(vals, 90):8.3f} "
                  f"max={max(vals):8.3f}")
    print()
    for r in rows:
        if not r["flagged"]:
            continue
        print(f"--- {r['flagged']}")
        print(f"    run            {r['run']}")
        print(f"    steps          {r['step_max']} over {r['n_rows']} rows")
        print(f"    non-finite     {r['n_nonfinite']}")
        print(f"    loss min/final {r['loss_min']:.4f} / {r['loss_final']:.4f}"
              f"   rise {r['rise_from_min']:+.4f} "
              f"(pctile {pct_rank([x['rise_from_min'] for x in rows], r['rise_from_min']):.0f})")
        print(f"    max jump       {r['max_jump_iqr']:.2f} IQR "
              f"(pctile {pct_rank([x['max_jump_iqr'] for x in rows], r['max_jump_iqr']):.0f})")
        if isinstance(r.get("qk_max"), float):
            print(f"    qk max         {r['qk_max']:.2f} "
                  f"(pctile {pct_rank([x['qk_max'] for x in rows if isinstance(x.get('qk_max'), float)], r['qk_max']):.0f})"
                  f"   trend x{r['qk_trend']:.2f}")
            print(f"    resid max      {r['resid_max']:.2f}")
        for wr in window_rows(r["run"], curves[r["run"]],
                              attn.get(r["run"], [])):
            print(f"    {wr['window']:>10s}  loss {wr['loss_first']} -> "
                  f"{wr['loss_last']} ({wr['delta']})  "
                  f"qk max {wr['qk_max'] or '-'} mean {wr['qk_mean'] or '-'}")
        print()


if __name__ == "__main__":
    main()
