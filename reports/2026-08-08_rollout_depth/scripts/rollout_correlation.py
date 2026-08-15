#!/usr/bin/env python3
"""#373 — does the eval's own rollout depth predict what `k = 3` is worth?

Two Spearman rank correlations per pair, over the 97 GIFT-Eval configs,
student head:

    rho(rollout steps, k = 0)          does a long rollout make a config hard?
    rho(rollout steps, k = 3 - k = 0)  does the depth pay more where the eval
                                       rolls out further?

Both are on RELATIVE MASE — the config's MASE over its seasonal-naive MASE —
because that is the quantity every score in this report is built from. The
second column is the number `rollout_count.png` prints in its legend, and
this script imports that figure's own pair list so the two cannot disagree.

`B5·s1` is here and is retracted: its `k = 0` trained on a rented box and
misses its published value by 0.1169. It is the one pair whose second column
is positive, so it is printed rather than dropped.

Writes `results/rollout_correlation.csv` and, with `--inject`, the report's
`<!-- ROLLOUTCORR:BEGIN -->` block.

Usage: rollout_correlation.py [--results DIR] [--inject report.md]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cell_config as CC                                   # noqa: E402
import paired_bootstrap as PB                              # noqa: E402
import plot_rollout_count as RC                            # noqa: E402
import published_bootstrap as PBOOT                        # noqa: E402
import r2_ladder as L                                      # noqa: E402
import runs as R                                           # noqa: E402

# The figure's five pairs, plus the retracted B5 backbone. Same tuple shape:
# (cell, label, k = 0 tag or None, k = 3 tag, stop, this study trained k = 0).
EXTRA = [("B5", "B5·s1 ✗  bb40k", "B5_k0_bb40k_student", "B5_k3_bb40k_student",
          40, True)]
PAIRS = list(RC.PAIRS) + EXTRA


def read_counts(path):
    with path.open() as fh:
        return {r["config"]: int(r["rollout_steps"]) for r in csv.DictReader(fh)}


def rows(res):
    counts = read_counts(res / "rollout_count.csv")
    sn = PB.read_mase(PB.SN_REF)

    def rel(path):
        m = PB.read_mase(path)
        return {d: m[d] / sn[d] for d in m if d in sn}

    out = []
    for cell, label, k0tag, k3tag, stop, own in PAIRS:
        p3 = res / "eval" / k3tag / "all_results.csv"
        p0 = (res / "eval" / k0tag / "all_results.csv") if k0tag \
            else PBOOT.parent_csv(cell, stop, "student", res)
        if not p3.is_file() or p0 is None or not Path(p0).is_file():
            print(f"skip {label}: one side missing")
            continue
        r0, r3 = rel(p0), rel(str(p3))
        ds = sorted(set(r0) & set(r3) & set(counts))
        if len(ds) != 97:
            print(f"skip {label}: {len(ds)} configs, want 97")
            continue
        x = [counts[d] for d in ds]
        out.append({
            "cell": cell,
            "arm": label.split("  ")[0],
            "configuration": CC.name(cell),
            "short": CC.name(cell, short=True),
            "stop": f"bb{stop}k",
            "k0_side": "this study" if own else "published",
            "n": len(ds),
            "rho_k0": f"{RC.spearman(x, [r0[d] for d in ds]):+.3f}",
            "rho_delta": f"{RC.spearman(x, [r3[d] - r0[d] for d in ds]):+.3f}",
            "retracted": "yes" if "✗" in label else "no",
        })
    return out


def block(rs):
    out = ["| configuration | cell, stop | rho(rollout steps, `k = 0`) | "
           "rho(rollout steps, `k = 3` minus `k = 0`) |",
           "|---|---|---:|---:|"]
    for r in rs:
        dag = "" if r["k0_side"] == "this study" else " †"
        out.append(f"| {r['short']} | {r['arm']}{dag}, {r['stop']} | "
                   f"{r['rho_k0']} | {r['rho_delta']} |")
    k0 = sorted(float(r["rho_k0"]) for r in rs)
    live = [r for r in rs if r["retracted"] == "no"]
    pos = [r for r in rs if float(r["rho_delta"]) > 0]
    out += ["",
            "Spearman rank correlation over the 97 configs, on relative MASE, "
            "student head, n = 97 on every row. The right column reads: the "
            "further the eval rolls out on a config, the more `k = 3` "
            "improves that config.", "",
            f"Left column: every pair is positive, {k0[0]:+.3f} to "
            f"{k0[-1]:+.3f}, so a config the eval rolls out further is a "
            "harder config at `k = 0` as well.", "",
            "† this pair reads a published `k = 0`; every other row trained "
            "both sides here. ✗ a retracted backbone.", ""]
    if pos and all(r["retracted"] == "yes" for r in pos):
        names = ", ".join(r["arm"] for r in pos)
        out += [f"Right column: the one positive value is {names}, the "
                "backbone this report retracts. The pairs it carries all "
                f"run one way, {max(float(r['rho_delta']) for r in live):+.3f} "
                f"to {min(float(r['rho_delta']) for r in live):+.3f}.", ""]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=L.RESULTS)
    ap.add_argument("--inject")
    a = ap.parse_args(argv)
    res = Path(a.results)
    if not (res / "rollout_count.csv").is_file():
        print("no rollout_count.csv — no rollout correlation")
        return 0

    rs = rows(res)
    if not rs:
        print("no finished pair — no rollout correlation")
        return 0

    dst = res / "rollout_correlation.csv"
    with dst.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)
    print(f"wrote {dst}")
    for r in rs:
        print(f"  {r['cell']:<3} {r['arm']:<12} k0 {r['rho_k0']}  "
              f"delta {r['rho_delta']}")

    if a.inject:
        md = Path(a.inject)
        text = md.read_text()
        b, e = "<!-- ROLLOUTCORR:BEGIN -->", "<!-- ROLLOUTCORR:END -->"
        if b in text and e in text:
            head, rest = text.split(b, 1)
            _old, tail = rest.split(e, 1)
            md.write_text(f"{head}{b}\n\n" + "\n".join(block(rs)) +
                          f"\n{e}{tail}")
            print(f"injected ROLLOUTCORR into {md}")
        else:
            print(f"NOTE: {md} carries no ROLLOUTCORR markers; not injecting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
