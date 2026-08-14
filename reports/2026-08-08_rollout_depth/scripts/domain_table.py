#!/usr/bin/env python3
"""#373 — the per-family table that goes with the radar.

GM-Relative MASE = 1.0 is parity with seasonal naive. The card names the four
families where the k = 0 model loses to it, and the two where it already
wins. This script prints, per family, the k = 0 value, the k = 3 value, the
difference, and which side of 1.0 the depth left the family on.

It draws the same three pairs the radar's first, second and fourth panels
draw, and reads the same two files, so a number in the table and a vertex in
the figure cannot disagree.

Writes `results/domain_table.md` and, with `--inject`, the report's
`<!-- DOMAIN:BEGIN -->` block.

Usage: domain_table.py [--results DIR] [--inject report.md]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_ladder as L                                     # noqa: E402

# (cell, stop, head, k0 file, k0 key, k3 key, one line saying what it is)
PAIRS = [
    ("A4", 100, "student", "k0", "A4_k0_bb100k_student",
     "A4_k3_bb100k_student",
     "The cell that sets this study's frontier, at the deepest stop its "
     "parent published. Published `k = 0`."),
    ("B1", 40, "student", "study", "G6_B1_k0_bb40k_student",
     "G6_B1_k3_bb40k_student",
     "The pair whose `k = 0` side this study trained, so the depth is the "
     "only change."),
    ("B2", 200, "student", "k0", "B2_k0_bb200k_student",
     "B2_k3_bb200k_student",
     "The arm and stop the card quotes its own per-family numbers from. "
     "Published `k = 0`."),
]

HARD = ["Energy", "Econ/Fin", "Web/CloudOps", "Healthcare"]


def load(path):
    out, counts = {}, {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "domain":
                continue
            out.setdefault(r["stop"], {})[r["name"]] = float(r["gm_rel_mase"])
            counts[r["name"]] = int(r["n"])
    return out, counts


def lands(k0, k3):
    """Where k = 3 left the family, against seasonal-naive parity at 1.0."""
    if k0 > 1.0 and k3 <= 1.0:
        return "**past 1.0**"
    if k0 <= 1.0 and k3 > 1.0:
        return "**crosses above 1.0**"
    if k0 <= 1.0:
        return "stays below 1.0" + (", lower" if k3 < k0 else ", higher")
    return "toward 1.0" if k3 < k0 else "away from 1.0"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=L.RESULTS)
    ap.add_argument("--inject")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = Path(a.results)

    study, counts = load(res / "splits.csv")
    pub, c2 = load(res / "splits_k0.csv")
    counts.update(c2)
    src = {"study": study, "k0": pub}

    out = []
    for cell, stop, head, f0, k0key, k3key, why in PAIRS:
        d0 = src[f0].get(k0key)
        d3 = study.get(k3key)
        if not d0 or not d3:
            print(f"skip {cell} bb{stop}k: one side missing")
            continue
        arm, align, ema = L.CELL_ARM[cell]
        tgt = "no L_align" if align == "none" else f"L_align on the {align}"
        out += [f"**{cell}  {arm} · {tgt}, bb{stop}k, {head}-encoder head.** "
                f"{why}", "",
                "| family | configs | k = 0 | k = 3 | difference | "
                "where k = 3 leaves it |",
                "|---|---:|---:|---:|---:|---|"]
        for d in sorted(d0, key=lambda d: -counts[d]):
            v0, v3 = d0[d], d3[d]
            flag = " ⚑" if d in HARD else ""
            out.append(f"| {d}{flag} | {counts[d]} | {v0:.3f} | {v3:.3f} | "
                       f"{v3 - v0:+.3f} | {lands(v0, v3)} |")
        out += ["", ""]
    out += ["⚑ marks the four families the card names as the ones seasonal "
            "naive wins by the largest margin: " + ", ".join(HARD) + ".", ""]

    dst = Path(a.out) if a.out else res / "domain_table.md"
    dst.write_text("\n".join(out) + "\n")
    print(f"wrote {dst}")

    if a.inject:
        md = Path(a.inject)
        text = md.read_text()
        b, e = "<!-- DOMAIN:BEGIN -->", "<!-- DOMAIN:END -->"
        if b in text and e in text:
            head_, rest = text.split(b, 1)
            _old, tail = rest.split(e, 1)
            md.write_text(f"{head_}{b}\n\n" + "\n".join(out) + f"\n{e}{tail}")
            print(f"injected DOMAIN into {md}")
        else:
            print(f"NOTE: {md} carries no DOMAIN markers; not injecting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
