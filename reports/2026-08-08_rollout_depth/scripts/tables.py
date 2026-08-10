#!/usr/bin/env python3
"""#373 — every table the report carries, from the score files and the splits.

Six tables, in the order the report asks its questions:

  1. coverage        which of the card's 14 cells this study trained.
  2. reproduction    this study's retrained k = 0 against the published one.
  3. depth response  each arm's own k = 0 against its deeper runs, on the
                     full 97 and on the card's horizon criterion.
  4. backbone seed   B5 trained twice, same recipe, two seeds.
  5. EMA regime      one loss shape, two EMA regimes.
  6. A3 controls     the depth ladder beside the re-weighting control.

Every delta is against the SAME arm's own k = 0. No delta in this file is
computed against a published number or against another backbone seed.

The same text goes two places: `results/scores.md`, and the report itself
between its `<!-- TABLES:BEGIN -->` and `<!-- TABLES:END -->` markers. The
report standard puts the tables in the canonical report; writing them there
by hand would let them drift from the score files, so the rebuild writes
them.

Usage: tables.py --results <results dir> --out <scores.md> [--inject <md>]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs as R                                          # noqa: E402
from published import (PUBLISHED as PUB_ALL, GATE,             # noqa: E402
                       NOISE_BAND, PRINTED_PRECISION, verdict)

CARD_CELLS = ["A1", "A2", "A3", "A4"] + [f"B{i}" for i in range(1, 11)]


def read_scores(results):
    out = {}
    for p in Path(results).glob("score_*.txt"):
        try:
            out[p.name[len("score_"):-len(".txt")]] = float(p.read_text().strip())
        except ValueError:
            continue
    return out


def read_splits(results):
    """`{tag: {split name: value}}`."""
    out = {}
    path = Path(results) / "splits.csv"
    if not path.is_file():
        return out
    for r in csv.DictReader(open(path)):
        out.setdefault(r["stop"], {})[r["name"]] = float(r["gm_rel_mase"])
    return out


def fmt(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"


def pct(a, b):
    return "—" if None in (a, b) else f"{100.0 * (b / a - 1.0):+.1f}%"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inject", help="report to write the tables into, "
                                     "between its TABLES markers")
    args = ap.parse_args(argv)

    sc = read_scores(args.results)
    sp = read_splits(args.results)
    tags = sorted(sc)
    reg = R.resolve_all(tags)
    L = []

    # ---- 1. coverage -------------------------------------------------------
    trained = sorted({r.cell for r in reg.values()})
    missing = [c for c in CARD_CELLS if c not in trained]
    L += ["### Coverage", "",
          f"The card names 14 cells. This study trained **{len(trained)} of "
          f"them**: {', '.join(sorted(trained))}. It never ran "
          f"**{len(missing)}**: {', '.join(missing)}.", "",
          "| cell | f-bearing term | EMA α | depths trained |",
          "|---|---|---|---|"]
    for cell in CARD_CELLS:
        ks = sorted({r.k for r in reg.values()
                     if r.cell == cell and r.role == "depth"})
        L.append(f"| {cell} | {R.CELL_TERM.get(cell, '—')} | "
                 f"{R.CELL_EMA.get(cell, '—')} | "
                 f"{', '.join(f'k = {k}' for k in ks) if ks else '**never ran**'} |")
    L += ["", "Every trained stop is bb40k. No cell reached bb100k or "
          "bb200k, so the card's extend rule never fired and this study "
          "publishes one stop.", ""]

    # ---- 2. reproduction ---------------------------------------------------
    L += ["### Reproduction of the published k = 0", "",
          "Same cell, same recipe, same head seed 20260722, same 97-config "
          "B4 eval. The only thing that differs from the parent report is "
          "the code snapshot.", "",
          "| arm | published k = 0 | retrained k = 0 | \\|Δ\\| | "
          f"verdict (threshold {GATE}) |", "|---|---|---|---|---|"]
    for arm, heads in R.ladders(tags).items():
        r0 = heads.get("student", {}).get(0)
        if r0 is None:
            continue
        pub = PUB_ALL.get(r0.cell, {}).get("student", {}).get(40)
        got = sc.get(r0.tag)
        if pub is None or got is None:
            continue
        d = abs(got - pub)
        L.append(f"| {arm} | {pub:.4f} | {got:.4f} | {d:.4f} | "
                 f"{verdict(d)} |")
    L += ["", "The parents print four decimals, so a difference below "
          f"{PRINTED_PRECISION} is the smallest the published table can "
          f"resolve. The card's gate of {GATE} is stricter than that.", ""]
    g1 = sc.get("G1_B5pub_bb40k_student")
    if g1 is not None:
        L += ["", "And one control that changes the backbone instead of the "
              "code: #379's own published B5 backbone, re-headed and "
              "re-scored by this study.", "",
              "| backbone | head + eval | GM-Relative MASE |", "|---|---|---|",
              f"| #379's published B5 bb40k | this study | {g1:.4f} |",
              "| #379's published B5 bb40k | as published | 1.2748 |"]
    L.append("")

    # ---- 3. depth response -------------------------------------------------
    L += ["### Depth response, against each arm's own k = 0", "",
          "| arm | EMA α | f-bearing term | head | k | k = 0 | this k | Δ | "
          "all | short | med+long | criterion |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, head, k, base, deep in R.pairs(tags):
        a, b = sc.get(base.tag), sc.get(deep.tag)
        A, B = sp.get(base.tag, {}), sp.get(deep.tag, {})
        ok = "—"
        if A and B:
            dm = 100.0 * (B["medium_long"] / A["medium_long"] - 1.0)
            ds = 100.0 * (B["short"] / A["short"] - 1.0)
            ok = "**MET**" if (dm <= -5.0 and ds < 2.0) else "not met"
        L.append(
            f"| {arm} | {deep.ema} | {deep.term} | {head} | {k} | {fmt(a)} | "
            f"{fmt(b)} | {f'{b - a:+.4f}' if None not in (a, b) else '—'} | "
            f"{pct(A.get('all'), B.get('all'))} | "
            f"{pct(A.get('short'), B.get('short'))} | "
            f"{pct(A.get('medium_long'), B.get('medium_long'))} | {ok} |")
    L += ["", "Criterion, from the card: medium+long (42 configs) at least "
          "5% better, short (55 configs) losing less than 2%.", "",
          f"Head-seed band ±{NOISE_BAND} (`ema_sched_ladder.md`, pooled). It "
          "bounds the head seed alone. The backbone-seed table below "
          "measures the backbone seed, which is larger.", ""]

    # ---- 4. backbone seed --------------------------------------------------
    L += ["### Two backbone seeds of one cell", "",
          "B5 (`arm4_combab_fix09`) trained twice. Same code, same recipe, "
          "same head seed, same eval; the backbone seed is the only "
          "difference.", "",
          "| head | k | seed 20260520 | seed 20260521 | seed spread |",
          "|---|---|---|---|---|"]
    for head in ("student", "teacher"):
        for k, t1, t2 in ((0, "B5_k0", "G5_B5_s2_k0"),
                          (3, "B5_k3", "G5_B5_s2_k3")):
            a, b = sc.get(f"{t1}_bb40k_{head}"), sc.get(f"{t2}_bb40k_{head}")
            if a is None or b is None:
                continue
            L.append(f"| {head} | {k} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    L += ["", "| head | seed | k = 0 | k = 3 | k = 3 − k = 0 |",
          "|---|---|---|---|---|"]
    for head in ("student", "teacher"):
        for seed, t0, t3 in (("20260520", "B5_k0", "B5_k3"),
                             ("20260521", "G5_B5_s2_k0", "G5_B5_s2_k3")):
            a, b = sc.get(f"{t0}_bb40k_{head}"), sc.get(f"{t3}_bb40k_{head}")
            if a is None or b is None:
                continue
            L.append(f"| {head} | {seed} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    L.append("")

    # ---- 5. EMA regime at one loss shape -----------------------------------
    L += ["### One loss shape, two EMA regimes", "",
          "B1 and A3 train the same f-bearing term, `rep_only` + `L_align`, "
          "on the same `arm6_v2 combab` arm. They differ in the EMA "
          "schedule.", "",
          "| arm | EMA α | head | k = 0 | k = 3 | Δ | Δ% |",
          "|---|---|---|---|---|---|---|"]
    for arm, t0, t3 in (("B1", "G6_B1_k0", "G6_B1_k3"),
                        ("A3", "A3_k0", "A3_k3")):
        for head in ("student", "teacher"):
            a, b = sc.get(f"{t0}_bb40k_{head}"), sc.get(f"{t3}_bb40k_{head}")
            if a is None or b is None:
                continue
            L.append(f"| {arm} | {R.CELL_EMA[arm]} | {head} | {a:.4f} | "
                     f"{b:.4f} | {b - a:+.4f} | {100 * (b / a - 1):+.1f}% |")
    L.append("")

    # ---- 6. A3 controls ----------------------------------------------------
    L += ["### A3: is the damage the depth, or the weight?", "",
          "Summing the depths multiplies `L_align`'s weight against the "
          "f-free terms by k + 1. The `L_align x4` row applies that "
          "re-weighting at k = 0, with no depth at all.", "",
          "| head | k = 0 | k = 0, `L_align` x4 | k = 1 | k = 3 | "
          "share of the k = 3 damage the re-weighting explains |",
          "|---|---|---|---|---|---|"]
    for head in ("student", "teacher"):
        v = [sc.get(f"{t}_bb40k_{head}") for t in
             ("A3_k0", "G3_A3_k0_aw4", "G3_A3_k1", "A3_k3")]
        if v[0] is None or v[3] is None:
            continue
        share = ("—" if v[1] is None or v[3] == v[0]
                 else f"{100.0 * (v[1] - v[0]) / (v[3] - v[0]):.0f}%")
        L.append("| " + head + " | " + " | ".join(fmt(x) for x in v)
                 + f" | {share} |")
    L.append("")

    body = "\n".join(L) + "\n"
    Path(args.out).write_text(body)
    print(f"wrote {args.out} ({len(reg)} run(s), {len(trained)} cell(s))")

    if args.inject:
        md = Path(args.inject)
        text = md.read_text()
        a, b = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"
        if a not in text or b not in text:
            print(f"NOTE: {md} carries no TABLES markers; not injecting")
            return 0
        head, rest = text.split(a, 1)
        _old, tail = rest.split(b, 1)
        md.write_text(f"{head}{a}\n\n{body}\n{b}{tail}")
        print(f"injected the tables into {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
