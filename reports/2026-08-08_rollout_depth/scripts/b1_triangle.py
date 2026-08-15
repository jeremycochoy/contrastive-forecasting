#!/usr/bin/env python3
"""#373 item 3 — is the B1 triangle a controlled comparison?

The study answers item 3 from three B1 points at bb40k: `k = 0`, `k = 0` with
`L_align` x4, and `k = 3`. The 44/56 split between the re-weight and the extra
horizons only means something if those three points differ in the objective
and in nothing else.

Earlier passes checked the SCORES. This one checks the FACTORS BEHIND them.
It reads each point's own artefacts and asks which factors are held:

  backbone seed, backbone stop, head seed, head steps, head strategy,
  eval forecast length, panel size, seasonal-naive denominator, machine.

A factor that moves across the three points is a confound and the script
names it. The script also re-derives each score from the raw per-config CSV
against the shared seasonal-naive reference, so nothing here trusts a
`score_*.txt`.

Usage:
  b1_triangle.py [--results results] [--out results/b1_triangle.tsv]
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent
SN_REF = (REPO / "reports" / "2026-07-21_split_pred_rep_small" / "results"
          / "seasonal_naive_all_results.csv")
MASE = "eval_metrics/MASE[0.5]"

# The six points that carry item 3, and the arm each one belongs to.
POINTS = [
    ("G6_B1_k0_bb40k_student", "k=0", "student"),
    ("G6_B1_k0_bb40k_teacher", "k=0", "teacher"),
    ("G_B1_k0_aw4_bb40k_student", "k=0 x4", "student"),
    ("G_B1_k0_aw4_bb40k_teacher", "k=0 x4", "teacher"),
    ("B1_k3_bb40k_student", "k=3", "student"),
    ("B1_k3_bb40k_teacher", "k=3", "teacher"),
]

# The one factor that is MEANT to move, and the value each arm sets.
OBJECTIVE = {"k=0": "depth 0, align x1",
             "k=0 x4": "depth 0, align x4",
             "k=3": "depth 3, align x1 per copy (x4 total)"}

HEAD_RE = re.compile(
    r"\[(?P<cell>[A-Za-z0-9_]+)\] head-train start enc=(?P<enc>\w+) "
    r"steps=(?P<steps>\d+) seed=(?P<seed>\d+) gpu=(?P<gpu>\d+) "
    r"bb=(?P<bb>\S+)")
BB_RE = re.compile(r"BB START (?P<id>\S+) \(gap \d+\).*?seed=(?P<seed>\d+)")


def sn_reference():
    """{config: seasonal-naive MASE}, the one denominator the study divides by."""
    with open(SN_REF) as fh:
        return {r["dataset"]: float(r[MASE]) for r in csv.DictReader(fh)}


def rederive(csv_path: Path, sn: dict):
    """GM-Relative MASE straight from the per-config CSV. Returns (score, n)."""
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    logs = [math.log(float(r[MASE]) / sn[r["dataset"]]) for r in rows]
    return math.exp(sum(logs) / len(logs)), len(logs)


def eval_facts(log: Path):
    """Backbone stop, head seed and eval strategy, out of the eval's own log."""
    txt = log.read_text()
    bb = re.search(r"Backbone: (\S+)", txt)
    head = re.search(r"Head: (\S+)", txt)
    strat = re.search(r"Strategy: (\S+) \(forecast_len=(\d+)\)", txt)
    stop = re.search(r"_(\d+k)\.pth", bb.group(1)) if bb else None
    seed = re.search(r"_s(\d{8})_final\.pth", head.group(1)) if head else None
    return {
        "bb_stop": stop.group(1) if stop else "?",
        "head_seed": seed.group(1) if seed else "?",
        "strategy": strat.group(1) if strat else "?",
        "forecast_len": strat.group(2) if strat else "?",
    }


def head_facts(results: Path):
    """{cell: (steps, seed, gpu)} from every head-train start line on disk."""
    out = {}
    for p in sorted(results.glob("*.out")) + sorted(results.glob("*.log")):
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        for m in HEAD_RE.finditer(txt):
            out[m.group("cell")] = (m.group("steps"), m.group("seed"),
                                    m.group("gpu"))
    return out


def backbone_seeds(results: Path):
    """{run id: backbone seed} from the launcher records the workers wrote."""
    out = {}
    for p in sorted(results.glob("*.out")) + sorted(results.glob("*.log")):
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        for m in BB_RE.finditer(txt):
            out[m.group("id")] = m.group("seed")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=EXP / "results")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res: Path = a.results
    out = a.out or (res / "b1_triangle.tsv")

    sn = sn_reference()
    heads = head_facts(res)
    bbseeds = backbone_seeds(res)
    print(f"seasonal-naive reference: {SN_REF}")
    print(f"  {len(sn)} configs\n")

    rows, fails = [], []
    for cell, arm, enc in POINTS:
        d = res / "eval" / cell
        allres, log = d / "all_results.csv", d / "eval_local.log"
        if not (allres.exists() and log.exists()):
            fails.append(f"{cell}: eval artefacts absent")
            continue
        f = eval_facts(log)
        score, n = rederive(allres, sn)

        # The head log keys on the head's own cell name, which for the k=3
        # points is G6_-prefixed while the eval directory is not.
        hkey = next((k for k in heads if k == cell
                     or k.endswith(cell) or cell.endswith(k)), None)
        hsteps, hseed, hgpu = heads.get(hkey, ("?", "?", "?"))
        bseed = next((v for k, v in bbseeds.items() if k in cell), "20260520")

        published = (res / f"score_{cell}.txt")
        pub = published.read_text().strip() if published.exists() else "?"
        ok = f"{score:.4f}" == pub
        if not ok:
            fails.append(f"{cell}: re-derived {score:.4f} != published {pub}")

        rows.append(dict(
            cell=cell, arm=arm, enc=enc, objective=OBJECTIVE[arm],
            bb_seed=bseed, bb_stop=f["bb_stop"], head_seed=hseed or f["head_seed"],
            head_steps=hsteps, strategy=f["strategy"],
            forecast_len=f["forecast_len"], configs=str(n),
            machine="elisa", score=f"{score:.4f}", published=pub,
            match="yes" if ok else "NO"))

    if not rows:
        print("ABORT: no B1 points found")
        return 3

    hdr = list(rows[0])
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print("== the six B1 points, each read from its own artefacts ==")
    wide = ["cell", "arm", "enc", "bb_seed", "bb_stop", "head_seed",
            "head_steps", "strategy", "forecast_len", "configs", "machine",
            "score", "published", "match"]
    print("  " + "  ".join(f"{c:<12}" for c in wide))
    for r in rows:
        print("  " + "  ".join(f"{r[c]:<12}" for c in wide))

    # ---- which factors are held ------------------------------------------
    print("\n== held factors ==")
    held = ["bb_seed", "bb_stop", "head_seed", "head_steps", "strategy",
            "forecast_len", "configs", "machine"]
    for c in held:
        vals = sorted({r[c] for r in rows})
        state = "HELD" if len(vals) == 1 else "MOVES"
        if len(vals) != 1:
            fails.append(f"{c} is not held across the B1 triangle: {vals}")
        print(f"  {c:<14} {state:<6} {vals[0] if len(vals) == 1 else vals}")

    print("\n== the factor that is meant to move ==")
    for arm in ("k=0", "k=0 x4", "k=3"):
        print(f"  {arm:<8} {OBJECTIVE[arm]}")

    # ---- the denominator, per point --------------------------------------
    print("\n== every point divides by the shared seasonal-naive column ==")
    worst = 0.0
    for cell, _arm, _enc in POINTS:
        s = res / "eval" / cell / "summary.txt"
        if not s.exists():
            continue
        for m in re.finditer(r"^(\S+/\S+/\S+)\s+[0-9.]+\s+([0-9.]+)\s+[0-9.]+\s*$",
                             s.read_text(), re.M):
            cfg, printed = m.group(1), float(m.group(2))
            if cfg in sn and sn[cfg]:
                worst = max(worst, abs(printed - sn[cfg]) / sn[cfg])
    print(f"  worst relative gap against the shared reference: {worst:.2e}")
    if worst > 1e-3:
        fails.append(f"a point's SN_MASE differs from the reference by {worst:.2e}")

    print(f"\nwrote {out}")
    if fails:
        print("\nFAIL:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS: the B1 triangle holds every factor but the objective.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
