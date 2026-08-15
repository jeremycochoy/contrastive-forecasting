#!/usr/bin/env python3
"""#373 — read a losses CSV, one row per step.

`CSVLogger` opens its file in append mode, so two trainers pointed at one
save dir interleave their rows into it and every step appears twice. That
happened once in this study: box b is in `Default` compute mode rather than
the `Exclusive_Process` a vast.ai box usually comes up in, so a queue start
that had already succeeded behind a timed-out ssh did not collide with the
second one, and two identical A3 k = 3 runs shared the card for 45 minutes.

The duplicate was killed. Its rows are still in the file, and they are rows
of the same run — same seed, same data order — so the fix is to keep one
row per step rather than to throw the file away.

Keeping the FIRST row per step, not the last, so the series is the earlier
writer's throughout and does not switch trainers mid-curve.
"""
from __future__ import annotations

import csv


def read_by_step(path, cols):
    """`{col: [values]}` plus `step`, one row per step, in step order.

    A row missing or malforming a requested column contributes nothing for
    that column and still contributes its step.
    """
    seen, rows = set(), []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                step = float(r["step"])
            except (KeyError, ValueError, TypeError):
                continue
            if step in seen:
                continue
            seen.add(step)
            rows.append((step, r))
    rows.sort(key=lambda sr: sr[0])

    out = {"step": [s for s, _r in rows]}
    for c in cols:
        vals = []
        for _s, r in rows:
            v = r.get(c)
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(None)
        if any(v is not None for v in vals):
            out[c] = vals
    return out


def series(data, col):
    """`(steps, values)` for one column, dropping the steps it is blank at."""
    if col not in data:
        return [], []
    xs, ys = [], []
    for s, v in zip(data["step"], data[col]):
        if v is not None:
            xs.append(s)
            ys.append(v)
    return xs, ys
