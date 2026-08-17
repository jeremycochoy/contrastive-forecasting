#!/usr/bin/env python3
"""#401 — what a leg of THIS objective costs, out of the leg's own log.

`smoke_depth.sh` measures a depth on a free card, 400 steps at a time. This
reads the same numbers off a leg that already ran: every `timing:` window the
trainer wrote, the first one dropped as warm-up, the median of the rest. Same
rule, same columns, one row per log.

Why both. The summed arm's committed table (`results/smoke_k16.csv`) measures
the OTHER objective, and the run plan of this card must not be sized from it:
the mean adds one pass over the f-bearing terms at depth 0
(`docs/train_rollout_depth.md`). A free card is what the smoke needs, and the
mean's legs measured themselves while they ran.

The row says where its number came from, because two legs on one card is not
the same measurement as one leg alone. `concurrent_legs` is that count and
`source` is the log. A step time measured beside a second leg is an UPPER bound
on the same leg alone, so a plan sized from it is over-sized and never
under-sized.

The reduction is checked, not assumed. train.py writes its command line as the
first line of its log, so a log that names another reduction is refused rather
than recorded under the wrong word.

Usage:
  cost_from_log.py --reduce mean --concurrent-legs 2 \\
      --leg 8=results/mean/run_..._cf373k8_mean.log \\
      --leg 32=results/mean/run_..._cf373k32_mean.log \\
      --used-mib 8=5414 --used-mib 32=5532 \\
      --out results/mean/leg_cost.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent

# The stop this card plans against. `hours_200k` is the wall clock one arm
# needs to reach it at the measured step time.
PLAN_STEPS = 200_000
CELL = "arm6_v2_combab_alignS"

FIELDS = ["cell", "reduce", "k", "windows", "data_ms", "fwd_ms", "bwd_ms",
          "total_ms", "sps", "used_mib", "concurrent_legs", "hours_200k",
          "source"]

TIMING = re.compile(r"^\s*timing:\s*(.*)$")
FIELD = re.compile(r"(\w+)=([\d.]+)ms")
SPS = re.compile(r"([\d.]+)\s+sps\b")
CMDLINE = "Command line:"


def reduce_of_cmdline(line: str) -> str:
    """The reduction a trainer command line names.

    `sum` when it carries no flag: `sum` is train.py's own default, so a line
    without the flag trains it. Both `--flag value` and `--flag=value`.
    """
    args = line.split()
    for i, arg in enumerate(args):
        if arg == "--train-rollout-reduce" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--train-rollout-reduce="):
            return arg.split("=", 1)[1]
    return "sum"


def log_reduce(text: str) -> str | None:
    """The reduction the LAST command line in this log names, or None.

    The runner appends to one log per cell, so a resumed cell's log holds one
    command line per leg. The last is the leg that wrote the newest windows.
    """
    lines = [ln for ln in text.splitlines() if ln.startswith(CMDLINE)]
    if not lines:
        return None
    return reduce_of_cmdline(lines[-1][len(CMDLINE):])


def windows(text: str) -> list[dict]:
    """Every `timing:` window of a log, with the `sps` reported above it.

    The first window is warm-up — CUDA context, autotune, stream fill — and
    the caller drops it. Every window is kept here so the count is the log's
    own.
    """
    out, sps = [], None
    for line in text.splitlines():
        hit = SPS.search(line)
        if hit:
            sps = float(hit.group(1))
        timing = TIMING.match(line)
        if not timing:
            continue
        row = {k: float(v) for k, v in FIELD.findall(timing.group(1))}
        if row:
            row["sps"] = sps
            out.append(row)
    return out


def measure(path: Path, k: int, reduce: str, concurrent: int,
            used_mib: int, cell: str) -> dict:
    """One row: the median of every window after the first."""
    text = path.read_text()
    named = log_reduce(text)
    if named is not None and named != reduce:
        raise SystemExit(
            f"ABORT: {path.name} names --train-rollout-reduce {named!r}, "
            f"and this table is {reduce!r}. One log, one objective.")
    rows = windows(text)[1:]
    if not rows:
        raise SystemExit(
            f"ABORT: {path.name} holds {len(windows(text))} timing window(s). "
            "The first is warm-up, so at least two are needed to measure one.")

    def med(name):
        vals = [r[name] for r in rows if r.get(name) is not None]
        return round(median(vals), 1) if vals else ""

    total = med("total")
    return {"cell": cell, "reduce": reduce, "k": k, "windows": len(rows),
            "data_ms": med("data"), "fwd_ms": med("fwd"), "bwd_ms": med("bwd"),
            "total_ms": total, "sps": med("sps"), "used_mib": used_mib,
            "concurrent_legs": concurrent,
            "hours_200k": round(total * PLAN_STEPS / 3_600_000, 1),
            "source": path.name}


def pairs(values, what):
    """`K=VALUE` arguments, as `{int(K): VALUE}`."""
    out = {}
    for item in values or ():
        if "=" not in item:
            raise SystemExit(f"ABORT: --{what} takes K=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[int(key)] = value
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", action="append", metavar="K=LOG", required=True,
                    help="the trainer log of the arm at depth K")
    ap.add_argument("--used-mib", action="append", metavar="K=MIB",
                    help="the card memory that arm's process held")
    ap.add_argument("--reduce", default="mean")
    ap.add_argument("--concurrent-legs", type=int, default=1,
                    help="how many legs shared the card while this ran")
    ap.add_argument("--cell", default=CELL)
    ap.add_argument("--out", help="write here instead of stdout")
    a = ap.parse_args(argv)

    legs = pairs(a.leg, "leg")
    mem = pairs(a.used_mib, "used-mib")
    rows = [measure(Path(log), k, a.reduce, a.concurrent_legs,
                    mem.get(k, ""), a.cell)
            for k, log in sorted(legs.items())]

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {a.out}  ({len(rows)} row(s))")
    else:
        writer = csv.DictWriter(sys.stdout, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
