"""The parts of the training loss over the steps, for the loss figures (#462).

The trainer writes one row for each step in ``<run>_losses.csv``. A resume
writes a new file (``<run>_r2_losses.csv``, ...) that logs again from the
step of its checkpoint. A value-space run (#415) logs its loss term at each
rollout depth j in the column ``val_err_d<j>``, and the column ``loss`` is
their sum.
"""

import csv
import re

import numpy as np

DEPTH_COLUMN = re.compile(r"val_err_d(\d+)$")


def depth_columns(header) -> list:
    """The depth columns of a value-space run (``val_err_d0``,
    ``val_err_d1``, ...), in depth order."""
    depths = {int(m.group(1)): name for name in header
              if (m := DEPTH_COLUMN.match(name))}
    return [depths[d] for d in sorted(depths)]


def read_loss_legs(paths, columns):
    """``(steps, {column: values})`` of the losses CSVs of one run, in leg
    order. A later leg logs again from the step of its checkpoint, so its
    rows replace the rows of the earlier legs from its first step on."""
    rows = {}
    for path in paths:
        with open(path) as f:
            leg = [(int(r["step"]), [float(r[c]) for c in columns])
                   for r in csv.DictReader(f)]
        if leg:
            first = leg[0][0]
            rows = {s: v for s, v in rows.items() if s < first}
            rows.update(leg)
    steps = np.array(sorted(rows), dtype=np.int64)
    table = np.array([rows[s] for s in steps], dtype=float).reshape(
        len(steps), len(columns))
    return steps, {c: table[:, i] for i, c in enumerate(columns)}


def log_binned_means(steps, values: dict, first, last, bins):
    """``(centres, {name: means})``: the mean of each array of ``values``
    over each of ``bins`` log-spaced bins of steps from ``first`` to
    ``last``. The last bin holds its top edge. A bin with no logged step has
    no point. The centre of a bin is its geometric centre, and a partial
    last bin ends at the last logged step."""
    edges = np.geomspace(first, last, bins + 1)
    centres, means = [], {name: [] for name in values}
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        top_in = steps <= high if i == bins - 1 else steps < high
        inside = (steps >= low) & top_in
        if not inside.any():
            continue
        centres.append(np.sqrt(low * min(high, steps.max())))
        for name, column in values.items():
            means[name].append(column[inside].mean())
    return np.array(centres), {n: np.array(m) for n, m in means.items()}
