#!/usr/bin/env python3
"""#407 review gap 9 — the project's three goal metrics, at no extra compute.

The card's deliverable is GM-Relative MASE. `CLAUDE.md` names GM-MASE,
GM-MAPE_SN and GM-CRPS_SN as the project's goal, and every one of them is
already inside the `all_results.csv` each stop wrote. So this reads them
back rather than running an eval again.

  GM-Relative MASE  geometric mean of MASE / SN_MASE over the 97 configs.
                    The same number `summary.txt` prints, and the card's
                    deliverable.
  GM-MASE           geometric mean of raw MASE. No denominator, so a change
                    here is not a change against seasonal naive.
  GM-MAPE_SN        geometric mean of MAPE / SN_MAPE.
  GM-CRPS_SN        geometric mean of the mean weighted sum quantile loss
                    over seasonal naive's. This is the study's CRPS.

The aggregation is `experiments/2026-07-03_b1024_traj_ckpts/scripts/
_compute_gm.py`, called rather than copied. The seasonal-naive denominator
is the one file the whole project reads, so this study cannot disagree with
`summary.txt` about GM-Relative MASE. The table prints that agreement as a
check column.

Usage:
  metrics_table.py [--results DIR] [--root ROOT] [--csv OUT] [--md OUT]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import full_pass  # noqa: E402

GM = os.path.join(full_pass.REPO_ROOT, "experiments",
                  "2026-07-03_b1024_traj_ckpts", "scripts")
sys.path.insert(0, GM)

import _compute_gm  # noqa: E402

COLUMNS = ["gm_rel_mase", "gm_mase", "gm_mape_sn", "gm_crps_sn"]
TITLES = {"gm_rel_mase": "GM-Relative MASE", "gm_mase": "GM-MASE",
          "gm_mape_sn": "GM-MAPE_SN", "gm_crps_sn": "GM-CRPS_SN"}


def rows_path(tag, results, root):
    """The 97 per-config rows of one tag, in the study or on the root."""
    for path in (os.path.join(str(results), "eval", tag, "all_results.csv"),
                 os.path.join(str(root), "eval", tag, "gift",
                              "all_results.csv")):
        if os.path.isfile(path):
            return path
    return None


def n_configs(path) -> int:
    with open(path) as fh:
        return len({r["dataset"] for r in csv.DictReader(fh)
                    if r.get("dataset")})


def table(stops, results, root, parent, seeds=None):
    """One row per (stop, head, seed) whose 97 rows are on disk."""
    out = []
    for stop in stops:
        for head in full_pass.HEADS:
            for seed in ([None] + list(seeds or [])):
                tag = full_pass.tag(stop, head)
                if seed is not None:
                    tag = f"{tag}_s{seed}"
                path = rows_path(tag, results, root)
                if path is None:
                    continue
                n = n_configs(path)
                if n != 97:
                    continue
                got = _compute_gm.compute(path)
                # The score file is what every other table in this study and
                # in #373 reads. Print it beside the recomputed value: a
                # mismatch means two denominators are in play.
                for directory in (results, parent):
                    published = full_pass.score(stop, head, directory) \
                        if seed is None else None
                    if published is not None:
                        break
                out.append({"stop": stop, "head": head,
                            "seed": seed or full_pass.HEAD_SEED,
                            "n": got["n"],
                            **{c: got[c] for c in COLUMNS},
                            "published_gm_rel_mase": published})
    return out


def as_markdown(rows) -> str:
    head = ("| stop | head | seed | " +
            " | ".join(TITLES[c] for c in COLUMNS) + " |")
    rule = "|---:|:---|---:|" + "---:|" * len(COLUMNS)
    lines = [head, rule]
    for r in rows:
        cells = " | ".join(f"{r[c]:.4f}" for c in COLUMNS)
        lines.append(f"| {r['stop'] // 1000}k | {r['head']} | "
                     f"{r['seed']} | {cells} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", type=int, action="append", dest="stops")
    ap.add_argument("--seed", type=int, action="append", dest="seeds",
                    help="a replicate head seed to include. Repeatable.")
    ap.add_argument("--results", default=full_pass.RESULTS)
    ap.add_argument("--parent", default=full_pass.PARENT_RESULTS)
    ap.add_argument("--root", default="/home/jupyter/cf373_r3/sync")
    ap.add_argument("--csv")
    ap.add_argument("--md")
    a = ap.parse_args(argv)

    stops = a.stops or ([full_pass.RESUME_STEP] + full_pass.STOPS)
    seeds = a.seeds if a.seeds is not None else [20260723, 20260724]
    rows = table(stops, a.results, a.root, a.parent, seeds)
    if not rows:
        print("no 97-config eval on disk yet")
        return 0

    print(f"{'stop':>7} {'head':<8} {'seed':>9} " +
          " ".join(f"{TITLES[c]:>17}" for c in COLUMNS) + "   published")
    for r in rows:
        pub = "-" if r["published_gm_rel_mase"] is None \
            else f"{r['published_gm_rel_mase']:.4f}"
        print(f"{r['stop']:>7} {r['head']:<8} {r['seed']:>9} " +
              " ".join(f"{r[c]:>17.4f}" for c in COLUMNS) + f"   {pub}")

    bad = [r for r in rows if r["published_gm_rel_mase"] is not None
           and abs(r["gm_rel_mase"] - r["published_gm_rel_mase"]) > 5e-4]
    if bad:
        print("WARNING: the recomputed GM-Relative MASE disagrees with the "
              "published score, so two denominators are in play:")
        for r in bad:
            print(f"  {r['stop']} {r['head']}: {r['gm_rel_mase']:.4f} != "
                  f"{r['published_gm_rel_mase']:.4f}")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(as_markdown(rows) + "\n")
        print(f"wrote {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
