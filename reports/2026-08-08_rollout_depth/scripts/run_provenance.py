#!/usr/bin/env python3
"""#373 — what else held each backbone's card while it trained.

The cost table reads step times off the trainer logs. A step time is only a
cost of the depth if the card was doing nothing else, and for most of this
study's runs it was not:

  * elisa ran TWO of this study's backbones on GPU 0 at a time, and handed
    each finished backbone's heads to a background subshell on top of that.
  * one rented box ran two identical A3 processes for 45 minutes.
  * the rented boxes were otherwise one run per card, in a queue.

Nothing in a trainer log says which of those it was — train.py prints no
wall clock. The drivers do. This reads their start and end lines, works out
which runs shared a card, and writes the overlap per run.

Inputs, all committed:

  results/gaps_driver.log     elisa: `BB START <id> ... gpu=N` / `BB END <id>`
                              and the same pair for `HEAD`.
  sync/<box>/queue.log        rented boxes: `START <cell> k=<k>` / `END ...`.

Each machine keeps its own clock, and the vast.ai containers run an hour
behind elisa. That costs nothing here: an overlap is only ever computed
between two runs on the same machine.

Usage:
    run_provenance.py --driver results/gaps_driver.log \\
        --queue d=sync/d/queue.log [--queue ...] \\
        --out results/run_provenance.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs as R                                            # noqa: E402

YEAR = 2026
TS = r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\]"
BB = re.compile(TS + r".*BB (START|END)\s+(\S+)")
BB_GPU = re.compile(r"gpu=(\d+)")
HEAD = re.compile(TS + r".*HEAD (START|END)\s+(\S+)\s+(student|teacher)")
HEAD_GPU = re.compile(r"gpu (\d+)\)")
Q = re.compile(TS + r".*\[queue\]\s+(START|END)\s+(\S+)\s+k=(\d+)")


def when(text):
    return datetime.strptime(f"{YEAR}-{text}", "%Y-%m-%d %H:%M:%S")


def spans(events):
    """([(key, start, end, extra)], {key: n clones}) from START / END events.

    A driver writes one START and one END per key, in order. An END with no
    START is dropped; a START with no END is closed at the last event seen,
    so a run killed mid-flight still contributes its overlap.

    A SECOND START before the first key's END is a second process on the
    same card. That happened once, on box b, and the clone ran 45 minutes
    before it was killed. The kill is in no log, so the count is what this
    reports; `steptime_provenance.py` measures the stretch exactly, off the
    duplicated step numbers the two processes wrote into one file.
    """
    open_, out, clones, last = {}, [], {}, None
    for ts, kind, key, extra in events:
        last = ts
        if kind == "START":
            if key in open_:
                clones[key] = clones.get(key, 0) + 1
                continue
            open_[key] = (ts, extra)
        elif key in open_:
            start, ex = open_.pop(key)
            out.append((key, start, ts, ex))
    for key, (start, ex) in open_.items():
        out.append((key, start, last or start, ex))
    return out, clones


def read_driver(path):
    """(backbone spans, head spans) off one elisa driver log."""
    bb, hd = [], []
    for line in open(path, errors="replace"):
        m = BB.search(line)
        if m:
            g = BB_GPU.search(line)
            bb.append((when(m.group(1)), m.group(2), m.group(3),
                       g.group(1) if g else "?"))
            continue
        m = HEAD.search(line)
        if m:
            g = HEAD_GPU.search(line)
            hd.append((when(m.group(1)), m.group(2),
                       f"{m.group(3)}:{m.group(4)}", g.group(1) if g else "?"))
    bb_spans, clones = spans(bb)
    return bb_spans, spans(hd)[0], clones


def read_queue(box, path, stem_of):
    """Backbone spans off one rented box's queue log.

    A box log names a run by (cell, k). Two boxes ran the same cell at
    different depths, so the box is part of the key.
    """
    ev = []                       # `Q` groups: (ts, kind, cell, k)
    for line in open(path, errors="replace"):
        m = Q.search(line)
        if not m:
            continue
        stem = stem_of(box, m.group(3), int(m.group(4)))
        if stem:
            ev.append((when(m.group(1)), m.group(2), stem, "0"))
    return spans(ev)


def overlap(a0, a1, b0, b1):
    """Seconds two intervals share."""
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, (hi - lo).total_seconds())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--driver", action="append", default=[],
                   help="an elisa gap-worker driver log")
    p.add_argument("--queue", action="append", default=[],
                   metavar="BOX=LOG", help="a rented box's queue log")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    by_stem = {}                       # stem -> (machine, card, start, end)
    heads = []                         # (machine, card, start, end, label)
    clones = {}                        # stem -> second processes on its card

    for path in args.driver:
        if not Path(path).is_file():
            print(f"  skip {path}: not a file", file=sys.stderr)
            continue
        bb, hd, cl = read_driver(path)
        clones.update(cl)
        for stem, s, e, gpu in bb:
            run = R.resolve(f"{stem}_bb40k_student")
            machine = run.machine if run else "elisa"
            card = f"{run.card if run else '?'} #{gpu}"
            by_stem[stem] = (machine, card, s, e)
        for key, s, e, gpu in hd:
            heads.append(("elisa", f"RTX 4090 #{gpu}", s, e, key))

    def stem_of(box, cell, k):
        want = f"vast box {box}"
        for st, (mach, _card) in R._MACHINE.items():
            if mach != want:
                continue
            run = R.resolve(f"{st}_bb40k_student")
            if run and run.cell == cell and run.k == k:
                return st
        return None

    for spec in args.queue:
        box, path = spec.split("=", 1)
        if not Path(path).is_file():
            print(f"  skip {spec}: no {path}", file=sys.stderr)
            continue
        qs, qc = read_queue(box, path, stem_of)
        clones.update(qc)
        for stem, s, e, _gpu in qs:
            run = R.resolve(f"{stem}_bb40k_student")
            by_stem[stem] = (run.machine if run else f"vast box {box}",
                             run.card if run else "?", s, e)

    rows = []
    for stem, (machine, card, s, e) in by_stem.items():
        run = R.resolve(f"{stem}_bb40k_student")
        length = max(1.0, (e - s).total_seconds())
        nb, bb_sec = [], 0.0
        for other, (m2, c2, s2, e2) in by_stem.items():
            if other == stem or m2 != machine or c2 != card:
                continue
            ov = overlap(s, e, s2, e2)
            if ov > 0:
                nb.append(f"{other}({ov / length:.0%})")
                bb_sec = max(bb_sec, ov)
        if clones.get(stem):
            nb.append(f"{clones[stem]} clone(s) of itself")
        hd_sec = sum(overlap(s, e, s2, e2)
                     for m2, c2, s2, e2, _k in heads
                     if m2 == machine and c2 == card)
        rows.append({
            "stem": stem,
            "arm": run.arm if run else "?",
            "k": run.k if run else "?",
            "role": run.role if run else "?",
            "machine": machine,
            "card": card,
            "start": s.strftime("%m-%d %H:%M:%S"),
            "end": e.strftime("%m-%d %H:%M:%S"),
            "minutes": f"{length / 60:.0f}",
            "backbone_neighbours": " ".join(sorted(nb)) or "none",
            "backbone_overlap": f"{bb_sec / length:.2f}",
            "head_overlap": f"{min(1.0, hd_sec / length):.2f}",
            # A clone's stretch is in no driver log, so it carries no
            # overlap fraction. A consumer deciding "was this run alone"
            # must read this column as well as the two above.
            "clones": clones.get(stem, 0),
        })

    if not rows:
        raise SystemExit("ABORT: no driver or queue log produced a span")

    order = {a: i for i, a in enumerate(R.ARM_ORDER)}
    rows.sort(key=lambda r: (order.get(r["arm"], 99), r["k"], r["role"]))
    for r in rows:
        print(f"{r['stem']:<13} {r['arm']:<6} k={r['k']} {r['machine']:<11} "
              f"{r['card']:<12} {r['minutes']:>4} min  "
              f"backbone {float(r['backbone_overlap']):.0%} / "
              f"head {float(r['head_overlap']):.0%}  "
              f"[{r['backbone_neighbours']}]")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} run(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
