#!/usr/bin/env python3
"""Emit the report's Markdown tables straight from the committed CSVs.

Four tables, so the report never holds a hand-typed number:

  1. GM-Relative MASE, all 30 cells, three backbone horizons.
  2. GM-Relative MASE per dataset domain, the cells evaluated at 200k.
  3. Latent drift per setting against base.
  4. The comparison table: each retrained cell's earlier-sweep value beside
     its teacher-target value at 40k / 100k / 200k.

Drift definition for table 3: the mean `1 - cos` displacement of `h_t` (and
of `e_t`) over every adjacent-checkpoint pair of that arm. A setting counts
as "lower than base" for one loss recipe when its mean is below the base
setting's mean of the same recipe. Six recipes, so the count runs out of 6
and the p-value is a two-sided exact binomial test on it.

    make_report_tables.py <results-dir> <out.md>
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "reports" / "2026-08-04_lalign_teacher" / "plots"))
from _cells import per_domain_relative_mase  # noqa: E402

from scipy import stats  # noqa: E402

ARMS = ["arm1", "arm3", "arm4", "arm5", "arm6_v2", "bimoco"]
VARIANTS = ["", "_tr1", "_nse", "_ncpc", "_combab"]
VAR_SHORT = {"": "base", "_tr1": "tr1", "_nse": "nse",
             "_ncpc": "ncpc", "_combab": "combab"}
RETRAINED = ["arm5", "arm5_tr1", "arm5_nse", "arm5_ncpc", "arm5_combab",
             "arm6_v2", "arm6_v2_tr1", "arm6_v2_nse", "arm6_v2_ncpc",
             "arm6_v2_combab"]
HORIZONS = [(40, 15000), (100, 30000), (200, 30000)]
DOMAIN_ORDER = ["Energy", "Web/CloudOps", "Transport", "Nature",
                "Econ/Fin", "Healthcare", "Sales"]


def pretty(slug: str) -> str:
    for v in ("_tr1", "_nse", "_ncpc", "_combab"):
        if slug.endswith(v):
            return f"{slug[:-len(v)]} {v[1:]}"
    return f"{slug} base"


def main() -> None:
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = list(csv.DictReader(open(res / "gm_relative_mase.csv", newline="")))

    # cell value at the default head seed, teacher side for the retrained arms
    val: dict[tuple[str, int], float] = {}
    student379: dict[tuple[str, int], float] = {}
    for r in rows:
        if r["head_seed"] != "20260722":
            continue
        k = (r["arm_slug"], int(r["bb_steps"]))
        if r["align_target"] in ("none", "teacher"):
            val[k] = float(r["gm_rel_mase"])
        elif r["align_target"] == "student" and r["source"] == "#379":
            student379[k] = float(r["gm_rel_mase"])

    L = []

    # --- table 1: all 30 cells ------------------------------------------
    L.append("| Rank @100k | Cell | bb 40k (head 15k) | bb 100k (head 30k) "
             "| 40k→100k | bb 200k (head 30k) | 100k→200k |")
    L.append("|---|---|---|---|---|---|---|")
    cells = []
    for arm in ARMS:
        for var in VARIANTS:
            slug = f"{arm}{var}"
            v = [val.get((slug, bb * 1000)) for bb, _ in HORIZONS]
            if any(x is not None for x in v):
                cells.append((slug, v))
    cells.sort(key=lambda c: c[1][1] if c[1][1] is not None else 9e9)
    for i, (slug, v) in enumerate(cells, 1):
        d1 = f"{v[1] - v[0]:+.4f}" if v[0] and v[1] else "—"
        d2 = f"{v[2] - v[1]:+.4f}" if v[1] and v[2] else "—"
        f = lambda x: f"{x:.4f}" if x is not None else "—"
        star = " ⟲" if slug in RETRAINED else ""
        L.append(f"| {i} | `{pretty(slug)}`{star} | {f(v[0])} | {f(v[1])} "
                 f"| {d1} | {f(v[2])} | {d2} |")
    L.append("")

    # --- table 2: per domain, cells at 200k -----------------------------
    at200 = [slug for slug, v in cells if v[2] is not None]
    per: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for slug in at200:
        m, c = per_domain_relative_mase(slug, 200, 30000)
        if m:
            per[slug] = m
            counts.update(c)
    doms = [d for d in DOMAIN_ORDER if d in counts]
    doms += [d for d in sorted(counts) if d not in doms]
    hdr = " | ".join(f"{d} ({counts[d]})" for d in doms)
    L.append(f"| Cell (bb 200k) | {hdr} | all 97 |")
    L.append("|---" * (len(doms) + 2) + "|")
    for slug in sorted(per, key=lambda s: val[(s, 200000)]):
        cs = " | ".join(
            (f"**{per[slug][d]:.3f}**" if per[slug][d] < 1.0
             else f"{per[slug][d]:.3f}") if d in per[slug] else "—"
            for d in doms)
        star = " ⟲" if slug in RETRAINED else ""
        L.append(f"| `{pretty(slug)}`{star} | {cs} "
                 f"| {val[(slug, 200000)]:.4f} |")
    below = " | ".join(
        f"{sum(1 for s in per if per[s].get(d, 9) < 1.0)}/{len(per)}"
        for d in doms)
    L.append(f"| *cells below 1.0, of {len(per)}* | {below} | 0/{len(per)} |")
    L.append("")

    # --- table 3: latent drift, setting vs base -------------------------
    drift: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(res / "latent_movement_pairs.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            drift[r["arm_slug"]].append((float(r["drift_h"]),
                                         float(r["drift_e"])))
    mean = {k: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v))
            for k, v in drift.items()}
    L.append("| Setting | `h_t` drift lower than base / 6 | `h_t` p "
             "| `e_t` drift lower than base / 6 | `e_t` p |")
    L.append("|---|---|---|---|---|")
    for var in ("_ncpc", "_nse", "_tr1", "_combab"):
        wins_h, wins_e, names = 0, 0, []
        n = 0
        for arm in ARMS:
            if arm not in mean or f"{arm}{var}" not in mean:
                continue
            n += 1
            if mean[f"{arm}{var}"][0] < mean[arm][0]:
                wins_h += 1
                names.append(arm)
            if mean[f"{arm}{var}"][1] < mean[arm][1]:
                wins_e += 1
        ph = stats.binomtest(wins_h, n, 0.5).pvalue
        pe = stats.binomtest(wins_e, n, 0.5).pvalue
        who = f" ({' / '.join(names)})" if 0 < wins_h < n else ""
        L.append(f"| `{VAR_SHORT[var]}` | {wins_h}/{n}{who} | {ph:.3f} "
                 f"| {wins_e}/{n} | {pe:.3f} |")
    L.append("")

    # --- table 4: comparison, earlier sweep vs teacher target -----------
    L.append("| Cell | 40k earlier | 40k teacher | 100k earlier "
             "| 100k teacher | 200k earlier | 200k teacher |")
    L.append("|---|---|---|---|---|---|---|")
    f = lambda x: f"{x:.4f}" if x is not None else "—"
    for slug in RETRAINED:
        c = []
        for bb, _ in HORIZONS:
            c.append(f(student379.get((slug, bb * 1000))))
            c.append(f(val.get((slug, bb * 1000))))
        L.append(f"| `{pretty(slug)}` | " + " | ".join(c) + " |")
    L.append("")

    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}  ({len(cells)} cells, {len(per)} at 200k)")


if __name__ == "__main__":
    main()
