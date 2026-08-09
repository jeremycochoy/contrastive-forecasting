#!/usr/bin/env python3
"""#373 — the tables for the runs the experiment review asked for.

`score_table.py` covers the card's own cells, whose score files are named
`score_<cell>_k<k>_bb40k_<head>.txt`. The review runs are not cells: one is
a head on a PUBLISHED backbone, one is a re-weighted control, two are a
second backbone seed. They carry their own ids, and each answers a
different question, so each gets its own small table rather than a row in
one wide one.

Missing runs are skipped, not faked: this prints whatever has finished.

Usage: gap_table.py --results <results dir> --out <gap_scores.md>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from published import NOISE_BAND                      # noqa: E402

# Published bb40k numbers of the cells these runs speak to, from the parent
# reports. B5 and B9 carry no L_align, so both parents print the same value.
PUB = {"B5": 1.2748, "B9": 1.5579, "A3": 1.1895, "B1": 1.2025}


def scores(results):
    out = {}
    for p in Path(results).glob("score_*.txt"):
        try:
            out[p.stem[len("score_"):]] = float(p.read_text().strip())
        except ValueError:
            continue
    return out


def get(sc, *names):
    for n in names:
        if n in sc:
            return sc[n]
    return None


def fmt(v, digits=4):
    return "—" if v is None else f"{v:.{digits}f}"


def delta(a, b):
    return "—" if a is None or b is None else f"{b - a:+.4f}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    sc = scores(args.results)
    L = []

    # ---- gap 1 -------------------------------------------------------------
    pub_bb = get(sc, "G1_B5pub_bb40k_student")
    own_k0 = get(sc, "B5_k0_bb40k_student")
    L += ["### Gap 1 — where the 0.1169 came from", "",
          "One head, one eval, one difference: the backbone. Every row uses "
          "this study's head trainer, head seed 20260722, 15,000 steps, and "
          "its 97-config B4 GIFT-Eval on the student encoder.", "",
          "| backbone | head + eval | GM-Relative MASE |", "|---|---|---|",
          f"| #379's published B5 bb40k | this study | {fmt(pub_bb)} |",
          f"| #379's published B5 bb40k | as published | {PUB['B5']:.4f} |",
          f"| this study's B5 k = 0 bb40k | this study | {fmt(own_k0)} |", ""]
    if pub_bb is not None:
        d_pub = pub_bb - PUB["B5"]
        L += [f"On the published backbone this code scores {pub_bb:.4f} "
              f"against a published {PUB['B5']:.4f}, a gap of {d_pub:+.4f}.", ""]

    # ---- gap 2 -------------------------------------------------------------
    L += ["### Gap 2 — B9 gets a same-code baseline", "",
          "| head | k = 0 (this study) | k = 3 | Δ | published k = 0 |",
          "|---|---|---|---|---|"]
    for head in ("student", "teacher"):
        a = get(sc, f"G2_B9_k0_bb40k_{head}")
        b = get(sc, f"B9_k3_bb40k_{head}")
        if a is None and b is None:
            continue
        L.append(f"| {head} | {fmt(a)} | {fmt(b)} | {delta(a, b)} | "
                 f"{PUB['B9']:.4f} |")
    L.append("")

    # ---- gap 3 -------------------------------------------------------------
    L += ["### Gap 3 — is it the depth or the weight?", "",
          "At k = 3 the depths are summed, so `L_align` carries four times "
          "its k = 0 weight against the f-free terms. The `aw4` row is that "
          "re-weighting with no depth at all; k = 1 is one depth at twice "
          "the weight.", "",
          "| head | k = 0 | k = 0, L_align x4 | k = 1 | k = 3 |",
          "|---|---|---|---|---|"]
    for head in ("student", "teacher"):
        row = [get(sc, f"A3_k0_bb40k_{head}"),
               get(sc, f"G3_A3_k0_aw4_bb40k_{head}"),
               get(sc, f"G3_A3_k1_bb40k_{head}"),
               get(sc, f"A3_k3_bb40k_{head}")]
        if all(v is None for v in row):
            continue
        L.append(f"| {head} | " + " | ".join(fmt(v) for v in row) + " |")
    L.append("")

    # ---- gap 5 -------------------------------------------------------------
    L += ["### Gap 5 — how far apart are two backbone trainings?", "",
          "Same recipe, same code, same head seed, same eval. The only "
          "difference between the two columns of a depth is the backbone "
          "seed: 20260520 against 20260521.", "",
          "| head | depth | seed 20260520 | seed 20260521 | seed spread |",
          "|---|---|---|---|---|"]
    for head in ("student", "teacher"):
        for k, s2 in ((0, "G5_B5_s2_k0"), (3, "G5_B5_s2_k3")):
            a = get(sc, f"B5_k{k}_bb40k_{head}")
            b = get(sc, f"{s2}_bb40k_{head}")
            if a is None and b is None:
                continue
            L.append(f"| {head} | k = {k} | {fmt(a)} | {fmt(b)} | "
                     f"{delta(a, b)} |")
    L += ["", "| head | seed | k = 0 | k = 3 | k = 3 − k = 0 |",
          "|---|---|---|---|---|"]
    for head in ("student", "teacher"):
        for seed, a_id, b_id in (("20260520", f"B5_k0_bb40k_{head}",
                                  f"B5_k3_bb40k_{head}"),
                                 ("20260521", f"G5_B5_s2_k0_bb40k_{head}",
                                  f"G5_B5_s2_k3_bb40k_{head}")):
            a, b = get(sc, a_id), get(sc, b_id)
            if a is None and b is None:
                continue
            L.append(f"| {head} | {seed} | {fmt(a)} | {fmt(b)} | "
                     f"{delta(a, b)} |")
    L += ["", f"Head-seed band, for scale: ±{NOISE_BAND} "
          "(`ema_sched_ladder.md`, pooled).", ""]

    # ---- gap 6 -------------------------------------------------------------
    L += ["### Gap 6 — the loss-shape contrast inside one EMA regime", "",
          "B1 and B5 both hold the EMA α at 0.9 and both take a MoCo "
          "teacher. B5's f-bearing term is a pooled InfoNCE with `f` in the "
          "numerator and every denominator family; B1's is `L_align`, which "
          "has no denominator.", "",
          "| cell | f-bearing term | head | k = 0 | k = 3 | Δ |",
          "|---|---|---|---|---|---|"]
    for cell, term, k0id, k3id in (
            ("B5 `arm4_combab_fix09`", "pooled `xshh_allt`",
             "B5_k0_bb40k_{h}", "B5_k3_bb40k_{h}"),
            ("B1 `arm6_v2_combab_alignS_fix09`", "`L_align` only",
             "G6_B1_k0_bb40k_{h}", "G6_B1_k3_bb40k_{h}")):
        for head in ("student", "teacher"):
            a = get(sc, k0id.format(h=head))
            b = get(sc, k3id.format(h=head))
            if a is None and b is None:
                continue
            L.append(f"| {cell} | {term} | {head} | {fmt(a)} | {fmt(b)} | "
                     f"{delta(a, b)} |")
    L.append("")

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out} ({len(sc)} score file(s) read)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
