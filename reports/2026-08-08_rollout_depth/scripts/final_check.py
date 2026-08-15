#!/usr/bin/env python3
"""#373 — the closing check on items 3 and 6, from the per-config CSVs.

Two jobs.

1. Cross-check. This recomputes every GM-Relative MASE and every paired
   interval that items 3 and 6 rest on, with an implementation written
   against the same definition but not the same code. It uses the reference
   seed, so a matching row is a real agreement and not a coincidence.

2. Close one hole. The study reports A3's bb200k head gap as a number
   (student 1.3998, teacher 1.2913) but never put an interval on it, and the
   second student draw never faced the teacher at all. Both contrasts land
   here.

Definitions, as everywhere else in this study:
  Relative_i        = MASE_i / SN_MASE_i          per GIFT-Eval config
  GM-Relative MASE  = exp(mean_i log Relative_i)  over the 97 configs
  delta             = GM(arm B) - GM(arm A) in log space, so negative
                      means arm B scores better.
The resampling unit is the DATASET, not the config: `m_dense/H/short`,
`.../medium` and `.../long` are three configs of one series.

Usage:
  final_check.py [--iters 10000] [--seed 20260809] [--out results/final_check.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RES = STUDY / "results"
EVAL = RES / "eval"
SN_REF = (STUDY.parent / "2026-07-21_split_pred_rep_small" / "results"
          / "seasonal_naive_all_results.csv")
MASE = "eval_metrics/MASE[0.5]"

# The cells items 3 and 6 rest on. Name -> eval directory.
CELLS = {
    "B1 k=0            student": "G6_B1_k0_bb40k_student",
    "B1 k=0            teacher": "G6_B1_k0_bb40k_teacher",
    "B1 k=0 L_align x4 student": "G_B1_k0_aw4_bb40k_student",
    "B1 k=0 L_align x4 teacher": "G_B1_k0_aw4_bb40k_teacher",
    "B1 k=3            student": "B1_k3_bb40k_student",
    "B1 k=3            teacher": "B1_k3_bb40k_teacher",
    "A3 bb200k draw1   student": "A3_k3_bb200k_student",
    "A3 bb200k draw2   student": "A3_k3_bb200k_student_s20260723",
    "A3 bb200k         teacher": "A3_k3_bb200k_teacher",
}

# label -> (arm A, arm B). Negative delta means B is better than A.
CONTRASTS = [
    # item 3 — does the re-weight alone carry B1's k=3 win?
    ("item3 alignx4_vs_k0   student", "G6_B1_k0_bb40k_student",
     "G_B1_k0_aw4_bb40k_student"),
    ("item3 alignx4_vs_k0   teacher", "G6_B1_k0_bb40k_teacher",
     "G_B1_k0_aw4_bb40k_teacher"),
    ("item3 k3_vs_k0        student", "G6_B1_k0_bb40k_student",
     "B1_k3_bb40k_student"),
    ("item3 k3_vs_k0        teacher", "G6_B1_k0_bb40k_teacher",
     "B1_k3_bb40k_teacher"),
    ("item3 k3_vs_alignx4   student", "G_B1_k0_aw4_bb40k_student",
     "B1_k3_bb40k_student"),
    ("item3 k3_vs_alignx4   teacher", "G_B1_k0_aw4_bb40k_teacher",
     "B1_k3_bb40k_teacher"),
    # item 6 — the head-seed redraw, and both draws against the teacher.
    ("item6 draw2_vs_draw1  student", "A3_k3_bb200k_student",
     "A3_k3_bb200k_student_s20260723"),
    ("item6 teacher_vs_draw1        ", "A3_k3_bb200k_student",
     "A3_k3_bb200k_teacher"),
    ("item6 teacher_vs_draw2        ", "A3_k3_bb200k_student_s20260723",
     "A3_k3_bb200k_teacher"),
]


def read_mase(path: Path) -> dict:
    """{config: MASE}, dropping non-positive and non-finite entries."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                v = float(r[MASE])
            except (KeyError, TypeError, ValueError):
                continue
            if v > 0 and math.isfinite(v):
                out[r["dataset"]] = v
    return out


def csv_for(tag: str) -> Path:
    return EVAL / tag / "all_results.csv"


def gm(vals) -> float:
    return math.exp(sum(vals) / len(vals)) if vals else float("nan")


def log_rel(arm: dict, sn: dict, keys) -> dict:
    return {d: math.log(arm[d] / sn[d]) for d in keys}


def split(datasets):
    """The three config groups this study reports on."""
    out = {"all": list(datasets), "short": [], "medium_long": []}
    for ds in datasets:
        term = ds.rsplit("/", 1)[-1]
        if term == "short":
            out["short"].append(ds)
        elif term in ("medium", "long"):
            out["medium_long"].append(ds)
    return out


def bootstrap(la, lb, members, keys, clusters, iters, rng):
    """Observed delta plus the 95% dataset-cluster percentile interval."""
    member = set(members)
    obs = gm([lb[d] for d in members]) - gm([la[d] for d in members])
    draws = []
    for _ in range(iters):
        pick = [clusters[keys[rng.randrange(len(keys))]]
                for _ in range(len(keys))]
        sel = [d for grp in pick for d in grp if d in member]
        if not sel:
            continue
        draws.append(gm([lb[d] for d in sel]) - gm([la[d] for d in sel]))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    share = sum(1 for d in draws if d < 0) / len(draws)
    return obs, lo, hi, share


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--out", default=str(RES / "final_check.csv"))
    args = ap.parse_args(argv)

    sn = read_mase(SN_REF)
    print(f"seasonal-naive reference: {SN_REF.name}, {len(sn)} configs\n")

    # ---- 1. the scores, recomputed from the per-config rows -------------
    print("GM-Relative MASE, recomputed against the committed score file")
    print(f"{'cell':<27} {'n':>4} {'recomputed':>11} {'committed':>10}  ok")
    bad = 0
    for name, tag in CELLS.items():
        path = csv_for(tag)
        if not path.exists():
            print(f"{name:<27} MISSING {path}")
            bad += 1
            continue
        arm = read_mase(path)
        common = sorted(set(arm) & set(sn))
        val = gm(list(log_rel(arm, sn, common).values()))
        sf = RES / f"score_{tag}.txt"
        ref = sf.read_text().strip() if sf.exists() else "-"
        ok = (ref != "-") and abs(float(ref) - val) < 5e-5
        bad += 0 if ok else 1
        print(f"{name:<27} {len(common):>4} {val:>11.4f} {ref:>10}  "
              f"{'yes' if ok else 'NO'}")

    # ---- 2. the contrasts ----------------------------------------------
    print("\nPaired dataset-cluster bootstrap, "
          f"{args.iters} resamples, seed {args.seed}")
    print("negative delta = the second arm scores better\n")
    rows = []
    for label, tag_a, tag_b in CONTRASTS:
        a, b = read_mase(csv_for(tag_a)), read_mase(csv_for(tag_b))
        common = sorted(set(a) & set(b) & set(sn))
        la, lb = log_rel(a, sn, common), log_rel(b, sn, common)
        clusters = {}
        for ds in common:
            clusters.setdefault(ds.rsplit("/", 1)[0], []).append(ds)
        keys = sorted(clusters)
        rng = random.Random(args.seed)
        for sub, members in split(common).items():
            if not members:
                continue
            obs, lo, hi, share = bootstrap(la, lb, members, keys, clusters,
                                           args.iters, rng)
            rows.append({"label": label.strip(), "subset": sub,
                         "n": len(members), "delta": f"{obs:.4f}",
                         "ci_lo": f"{lo:.4f}", "ci_hi": f"{hi:.4f}",
                         "p_improved": f"{share:.3f}"})
            if sub == "all":
                print(f"{label}  n={len(members):3d}  d={obs:+.4f}  "
                      f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
                      f"better in {share * 100:5.1f}% of resamples")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "subset", "n", "delta",
                                           "ci_lo", "ci_hi", "p_improved"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    # ---- 3. item 3's split ---------------------------------------------
    d = {r["label"]: float(r["delta"]) for r in rows if r["subset"] == "all"}
    print("\nItem 3 — how B1's k=0 -> k=3 gain divides")
    for head in ("student", "teacher"):
        tot = d[f"item3 k3_vs_k0        {head}"]
        rw = d[f"item3 alignx4_vs_k0   {head}"]
        dep = d[f"item3 k3_vs_alignx4   {head}"]
        print(f"  {head}: total {tot:+.4f} = re-weight {rw:+.4f} "
              f"({100 * rw / tot:.0f}%) + depth {dep:+.4f} "
              f"({100 * dep / tot:.0f}%)")

    print(f"\n{'PASS' if bad == 0 else 'FAIL'}: {bad} score mismatch(es)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
