#!/usr/bin/env python3
"""Which columns separate two arms, computed from `arms.tsv`.

Three times on this card a comparison was called single-axis while two
settings moved. Prose hides that; a column diff does not. Run this before
any sentence of the form "A against B moves <one thing>".

Usage:  arm_diff.py <arm> <arm>      the columns that differ
        arm_diff.py --pairs          every pair that differs in ONE column
"""
import sys, pathlib, itertools

COLS = ["arm", "k", "reduce", "tau", "end", "ramp", "seed", "decay", "lr"]

# `end` and `ramp` encode ONE experimental setting. `cf412_ema_args` reads
# `end == "-"` as a constant momentum and ignores `ramp`, so the trainer sees
# either `--ema-tau T` or `--ema-tau T --ema-tau-end E --ema-tau-ramp-steps R`.
# Counting the columns calls that pair two axes. It is one.
# Two levels. PARAM is the thing a comparison can isolate. OBJECT is the
# thing several parameters belong to.
#
# `end` and `ramp` are ONE parameter, because `cf412_ema_args` reads
# `end == "-"` as a constant momentum and drops `ramp`: together they encode
# whether a ramp exists and how long it is. `tau` is a separate parameter of
# the same object, the momentum the schedule starts from.
#
# A pair differing in one PARAM is single axis. A pair differing in several
# params of ONE object still answers "does that object matter" cleanly, and
# answers nothing about which part. A column count alone cannot tell those
# apart, and a tool cannot choose the question a report is asking.
PARAM = {"end": "ema_ramp_spec", "ramp": "ema_ramp_spec"}
OBJECT = {"ema_ramp_spec": "ema_schedule", "tau": "ema_schedule"}
TSV = pathlib.Path(__file__).with_name("arms.tsv")


def load():
    rows = {}
    for line in TSV.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        rows[f[0]] = dict(zip(COLS, f))
    return rows


def differ(a, b):
    """The PARAMETERS that separate two arms, not the columns."""
    seen, out = set(), []
    for c in COLS[1:]:
        if a[c] == b[c]:
            continue
        prm = PARAM.get(c, c)
        if prm in seen:
            continue
        seen.add(prm)
        out.append(prm)
    return out


def cols_of(param):
    return [c for c in COLS[1:] if PARAM.get(c, c) == param]


def verdict(params):
    if not params:
        return "identical"
    if len(params) == 1:
        return "SINGLE AXIS"
    objs = {OBJECT.get(p, p) for p in params}
    if len(objs) == 1:
        o = objs.pop()
        return f"ONE OBJECT, {len(params)} PARAMETERS: {o} — clean for \"does {o} matter\", not for which part"
    return "CONFOUNDED"


def main():
    rows = load()
    if sys.argv[1:2] == ["--pairs"]:
        for x, y in itertools.combinations(sorted(rows), 2):
            d = differ(rows[x], rows[y])
            if len(d) == 1:
                print(f"{d[0]:7s} {x} -> {y}")
        return 0
    try:
        a, b = sys.argv[1], sys.argv[2]
        ra, rb = rows[a], rows[b]
    except (IndexError, KeyError) as e:
        print(f"usage: arm_diff.py <arm> <arm> | --pairs   ({e})", file=sys.stderr)
        return 2
    d = differ(ra, rb)
    for axis in d:
        for c in cols_of(axis):
            if ra[c] != rb[c]:
                print(f"{axis:13s} {c:6s} {ra[c]:>8s} -> {rb[c]:>8s}")
    print(f"{len(d)} parameter(s) differ — {verdict(d)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
