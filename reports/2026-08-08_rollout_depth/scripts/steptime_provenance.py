#!/usr/bin/env python3
"""#373 — step time per run, and whether the card was that run's alone.

train.py prints `timing: data=… fwd=… bwd=… total=…` every `--log-every`
steps, averaged over that window. A median over those windows is the cost of
a step only when nothing else held the card. For most of this study's runs
something did, so a bare median is not a cost of the depth.

Two kinds of sharing, and each needs its own detector.

  Another PROCESS on the same card. `run_provenance.py` reads it off the
  driver logs: which runs overlapped in wall clock on which card, and how
  much head training ran beside them. Every elisa backbone in this study was
  overlapped by a second one for 43% to 100% of its life; every rented-box
  backbone had its card to itself.

  A CLONE of the run itself. `CSVLogger` and the trainer log both open in
  append mode, so when two identical A3 processes shared box b, BOTH wrote
  their windows into one file: the log carries 272 windows for a 200-window
  run. The step number separates them — two processes on one file produce
  each step number twice, and everything after the last duplicated step is
  the survivor, alone. That stretch is a solo measurement and the rest is
  not.

A run is `solo` only when it passes both. `compute_ms` is filled for a solo
run and blank otherwise; `compute_ms_contended` carries the shared number so
nothing is thrown away. The report publishes a depth ratio only where both
depths of a cell are solo.

`data` is the input pipeline. It is network-bound on a rented box and swings
by an order of magnitude between windows, so the depth overhead is read off
`fwd + bwd` and `data` is reported beside it rather than folded in.

Usage:
    steptime_provenance.py --out results/steptime_solo.csv \\
        --provenance results/run_provenance.csv \\
        --log B5·s1:0=path/to/run.log [--log ...]
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs as R                                            # noqa: E402

STEP = re.compile(r"^\[\s*(\d+)\]")
TIMING = re.compile(
    r"timing:\s*data=([\d.]+)ms\s+fwd=([\d.]+)ms\s+bwd=([\d.]+)ms\s+"
    r"total=([\d.]+)ms")

FIELDS = ["arm", "k", "machine", "card", "solo", "why_not_solo",
          "windows_total", "windows_solo", "windows_contended",
          "first_solo_step", "data_ms", "fwd_ms", "bwd_ms", "compute_ms",
          "total_ms", "compute_ms_contended", "backbone_overlap",
          "head_overlap", "neighbours"]


def windows(path):
    """[(step, data, fwd, bwd, total)] in file order.

    The `timing:` line is printed under the step line it belongs to, so the
    most recent step number is the window's.
    """
    out, step = [], None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = STEP.match(line)
            if m:
                step = int(m.group(1))
                continue
            m = TIMING.search(line)
            if m and step is not None:
                out.append((step, *(float(g) for g in m.groups())))
    return out


def split_clone(rows):
    """(after the clone, during it). Splits at the last duplicated step."""
    counts = Counter(r[0] for r in rows)
    dup = [s for s, n in counts.items() if n > 1]
    if not dup:
        return rows, []
    cut = max(dup)
    return [r for r in rows if r[0] > cut], [r for r in rows if r[0] <= cut]


def read_provenance(path):
    """`{(arm, k): row}` from run_provenance.csv, or {} if it is not there."""
    if not path or not Path(path).is_file():
        return {}
    return {(r["arm"], int(r["k"])): r for r in csv.DictReader(open(path))}


def med(rows):
    """Median (data, fwd, bwd, total) over a list of windows."""
    return [statistics.median(col) for col in zip(*rows)][1:]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--log", action="append", required=True, metavar="ARM:K=LOG")
    p.add_argument("--provenance", help="results/run_provenance.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=1,
                   help="leading windows to drop from an uncontended run")
    args = p.parse_args(argv)

    prov = read_provenance(args.provenance)
    if not prov:
        print("  NOTE: no run_provenance.csv — every run is assumed shared,"
              " because that is the safe way to be wrong here",
              file=sys.stderr)
    out_rows = []
    for spec in args.log:
        head, path = spec.split("=", 1)
        arm, ktxt = head.split(":")
        k = int(ktxt)
        if not Path(path).is_file():
            print(f"  skip {spec}: no {path}", file=sys.stderr)
            continue
        rows = windows(path)
        if not rows:
            print(f"  skip {spec}: no timing window", file=sys.stderr)
            continue

        run = R.find_run(arm, k)
        pv = prov.get((arm, k))
        shared = []
        if pv is None:
            shared.append("no provenance row")
        else:
            if float(pv["backbone_overlap"]) > 0:
                shared.append(f"another backbone for "
                              f"{float(pv['backbone_overlap']):.0%} of the run")
            if float(pv["head_overlap"]) > 0:
                shared.append(f"head training for "
                              f"{float(pv['head_overlap']):.0%} of it")

        after, during = split_clone(rows)
        if during:
            shared.append(f"a clone of itself over {len(during)} window(s)")
        solo_rows = after if during else rows[args.warmup:]
        solo = not shared or (len(shared) == 1 and during and solo_rows)

        m_all = med(rows)
        m_solo = med(solo_rows) if solo and solo_rows else None
        m_cont = med(during) if during else (None if solo else m_all)

        out_rows.append({
            "arm": arm, "k": k,
            "machine": run.machine if run else "?",
            "card": run.card if run else "?",
            "solo": "yes" if solo else "no",
            "why_not_solo": "" if solo else " and ".join(shared),
            "windows_total": len(rows),
            "windows_solo": len(solo_rows) if solo else 0,
            "windows_contended": len(during) if during else
                                 (0 if solo else len(rows)),
            "first_solo_step": solo_rows[0][0] if solo and solo_rows else "",
            "data_ms": f"{m_solo[0]:.1f}" if m_solo else "",
            "fwd_ms": f"{m_solo[1]:.1f}" if m_solo else "",
            "bwd_ms": f"{m_solo[2]:.1f}" if m_solo else "",
            "compute_ms": f"{m_solo[1] + m_solo[2]:.1f}" if m_solo else "",
            "total_ms": f"{m_solo[3]:.1f}" if m_solo else "",
            "compute_ms_contended": (f"{m_cont[1] + m_cont[2]:.1f}"
                                     if m_cont else ""),
            "backbone_overlap": pv["backbone_overlap"] if pv else "",
            "head_overlap": pv["head_overlap"] if pv else "",
            "neighbours": pv["backbone_neighbours"] if pv else "",
        })
        tail = ("alone" if solo and not during else
                "alone after a clone" if solo else " and ".join(shared))
        got = (f"{m_solo[1] + m_solo[2]:6.1f}" if m_solo else "     —")
        print(f"{arm:<6} k={k} {out_rows[-1]['machine']:<11} "
              f"{out_rows[-1]['card']:<9} solo n={out_rows[-1]['windows_solo']:4d}"
              f"  fwd+bwd={got} ms   ({tail})")

    if not out_rows:
        raise SystemExit("ABORT: no log produced a window")

    by = {(r["arm"], r["k"]): r for r in out_rows}
    print()
    for (arm, k) in sorted(by):
        if k == 0 or (arm, 0) not in by:
            continue
        a, b = by[(arm, 0)], by[(arm, k)]
        if not a["compute_ms"] or not b["compute_ms"]:
            print(f"{arm} k=0 -> k={k}: NO RATIO — "
                  f"{'k=0' if not a['compute_ms'] else f'k={k}'} was shared")
            continue
        c0, c3 = float(a["compute_ms"]), float(b["compute_ms"])
        same_card = a["card"] == b["card"]
        same_box = a["machine"] == b["machine"]
        note = ("" if same_box else
                f"  [{a['machine']} -> {b['machine']}"
                f"{'' if same_card else ', DIFFERENT CARD MODELS'}]")
        print(f"{arm}: fwd+bwd {c0:.1f} -> {c3:.1f} ms "
              f"({c3 / c0 - 1:+.0%}){note}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {args.out} ({len(out_rows)} run(s), "
          f"{sum(r['solo'] == 'yes' for r in out_rows)} of them solo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
