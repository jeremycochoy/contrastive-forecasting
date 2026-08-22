#!/usr/bin/env python3
"""#404 — the eval shards, checked row by row.

Why this file exists. On 2026-08-19 at 19:35 a `pkill` pattern for the sync
loop also matched the four running eval shards, because an eval command line
carries the sync root. Three shard CSVs of each arm were empty and `--resume`
removed them. The fourth shard of each arm kept its rows and the eval resumed
on top of them.

`eval_local.sh` counts 97 configs in the merged CSV, so a lost row cannot pass.
It does not check the rows it KEPT. A process killed while it writes a row
leaves a short line, and a short line that still holds the dataset name and
enough numbers reads as a config that was scored.

So this reads every shard of every arm and asks four questions:

  1. Does each row carry the header's field count? The last row of a shard is
     the one a kill can truncate, so its number is printed by name.
  2. Does every numeric field parse as a float?
  3. Does each shard name each dataset once?
  4. Is the merged CSV the union of the shards, line for line, at 97 rows and
     97 distinct datasets?

Question 4 is what makes questions 1 to 3 count: a malformed row that never
reached the merge cannot reach a score.

Usage:
  check_shards.py --root ~/cf404_sync/box_a/sync --out results/shard_check.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Which columns of a GIFT-Eval row hold a number. The eval writes
# `dataset`, `model` and `domain` as text and every other column as a float,
# so the rule is stated over the metric prefix rather than over a list of
# names that a new metric would silently leave out.
NUMERIC_PREFIXES = ("eval_metrics/",)
NUMERIC_NAMES = ("num_variates",)
N_CONFIGS = 97


def is_numeric(column: str) -> bool:
    return (column.startswith(NUMERIC_PREFIXES)
            or column in NUMERIC_NAMES)


def read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """The header and the data rows of one CSV, unparsed."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def check_csv(path: Path, label: str) -> tuple[list[str], set[str], list[str]]:
    """`(problems, datasets, raw lines)` of one shard or merged CSV."""
    problems: list[str] = []
    header, rows = read_rows(path)
    if not header:
        return [f"{label}: empty file"], set(), []
    width = len(header)
    seen: set[str] = set()
    lines = [",".join(r) for r in rows]

    for i, row in enumerate(rows):
        where = f"{label} row {i + 1} of {len(rows)}"
        if i == len(rows) - 1:
            where += " (the last row)"
        if len(row) != width:
            problems.append(f"{where}: {len(row)} fields, header has {width}")
            continue
        for col, value in zip(header, row):
            if not is_numeric(col):
                if not value.strip():
                    problems.append(f"{where}: column '{col}' is empty")
                continue
            try:
                float(value)
            except ValueError:
                problems.append(
                    f"{where}: column '{col}' is not a number: {value!r}")
        name = row[0]
        if name in seen:
            problems.append(f"{where}: dataset '{name}' appears twice")
        seen.add(name)
    return problems, seen, lines


def check_eval(gift: Path) -> tuple[bool, list[str]]:
    """One arm's eval directory. `(ok, report lines)`."""
    out: list[str] = []
    problems: list[str] = []
    merged = gift / "all_results.csv"
    shards = sorted(gift.glob("shard_*/all_results.csv"))

    if not merged.is_file():
        return False, [f"  MISSING {merged}"]
    if not shards:
        return False, [f"  no shard CSV under {gift}"]

    shard_lines: set[str] = set()
    shard_names: set[str] = set()
    for s in shards:
        label = s.parent.name
        probs, names, lines = check_csv(s, label)
        problems += probs
        overlap = shard_names & names
        if overlap:
            problems.append(
                f"{label}: {len(overlap)} dataset(s) also in another shard: "
                f"{sorted(overlap)[:3]}")
        shard_names |= names
        shard_lines |= set(lines)
        out.append(f"  {label}: {len(lines)} row(s), last row complete"
                   if not probs else f"  {label}: {len(probs)} problem(s)")

    probs, merged_names, merged_lines = check_csv(merged, "merged")
    problems += probs
    if len(merged_lines) != N_CONFIGS:
        problems.append(f"merged: {len(merged_lines)} rows, want {N_CONFIGS}")
    if len(merged_names) != N_CONFIGS:
        problems.append(
            f"merged: {len(merged_names)} distinct configs, want {N_CONFIGS}")
    # Every merged row came from a shard, unchanged. A row the merge invented,
    # or a row it altered, would not be in the shard set.
    stray = [ln for ln in merged_lines if ln not in shard_lines]
    if stray:
        problems.append(
            f"merged: {len(stray)} row(s) are in no shard: {stray[0][:60]}")
    missing = shard_names - merged_names
    if missing:
        problems.append(
            f"merged: {len(missing)} shard config(s) did not reach it: "
            f"{sorted(missing)[:3]}")
    out.append(f"  merged: {len(merged_lines)} row(s), "
               f"{len(merged_names)} distinct config(s)")

    for p in problems:
        out.append(f"  PROBLEM {p}")
    return not problems, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="the runs root, <root>/<arm>/eval/<tag>/gift/")
    ap.add_argument("--out", help="write the report here as well as to stdout")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser()
    gifts = sorted(root.glob("*/eval/*/gift"))
    if not gifts:
        print(f"ABORT: no eval directory under {root}", file=sys.stderr)
        return 2

    lines: list[str] = [f"#404 shard check over {root}", ""]
    bad = 0
    for gift in gifts:
        tag = gift.parent.name
        ok, report = check_eval(gift)
        lines.append(f"{tag}: {'PASS' if ok else 'FAIL'}")
        lines += report
        lines.append("")
        bad += 0 if ok else 1

    lines.append(f"{len(gifts) - bad} of {len(gifts)} eval(s) pass")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
