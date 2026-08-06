#!/usr/bin/env python3
"""#393 — copy every run's training-time latent-drift CSV into `results/`.

The trainer writes `<run>_latent_drift.csv` once per leg. A leg holds the
adjacent-checkpoint movement of the two encoders, measured every 20k steps:

    drift_cos = 1 - cos(h(step_ref), h(step))   mean over (b, t, c)

Same measure as the parent studies' latent-movement figure, so the two can be
read on one scale. Legs live on the durable root and in the per-machine sync
trees; this script unions both, keeps the longest copy of a duplicated leg,
and writes one pooled file plus a verbatim copy of each source leg.

Writes `results/latent_drift.csv` and `results/latent_drift/<cell>_leg<N>k.csv`.

Usage: python3 collect_latent_drift.py [--out-dir <results dir>]
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
SOURCES = [Path("/home/jupyter/checkpoints_backup/cf-393"),
           *sorted(Path("/home/jupyter").glob("cf393_sync*"))]
HEADER = ["step", "latent", "kind", "step_ref", "delta_step",
          "drift_cos", "drift_cos_aligned", "rot_gap", "cka"]


def find_legs() -> dict[tuple[str, str], Path]:
    """`(cell, leg) -> path`, keeping the copy with the most rows."""
    best: dict[tuple[str, str], Path] = {}
    for root in SOURCES:
        for path in root.rglob("*_latent_drift.csv"):
            leg = path.parent.name                       # leg_40k
            cell = path.parent.parent.name               # arm5_combab_alignS
            key = (cell, leg)
            if key not in best or _rows(path) > _rows(best[key]):
                best[key] = path
    return best


def _rows(path: Path) -> int:
    with open(path, newline="") as fh:
        return sum(1 for _ in fh) - 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    copies = out_dir / "latent_drift"
    copies.mkdir(parents=True, exist_ok=True)

    legs = find_legs()
    pooled: list[list[str]] = []
    seen: set[tuple] = set()
    for (cell, leg), path in sorted(legs.items()):
        shutil.copyfile(path, copies / f"{cell}_{leg}.csv")
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != HEADER:
                raise SystemExit(f"{path}: unexpected header {reader.fieldnames}")
            for row in reader:
                key = (cell, row["step"], row["latent"], row["kind"],
                       row["step_ref"])
                if key in seen:
                    continue
                seen.add(key)
                pooled.append([cell, leg] + [row[c] for c in HEADER])

    pooled.sort(key=lambda r: (r[0], int(r[2]), r[3], r[4]))
    out = out_dir / "latent_drift.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "leg"] + HEADER)
        w.writerows(pooled)
    cells = sorted({r[0] for r in pooled})
    print(f"wrote {out}: {len(pooled)} rows, {len(legs)} legs, {len(cells)} cells")
    for cell in cells:
        steps = sorted({int(r[2]) for r in pooled if r[0] == cell})
        print(f"  {cell:24s} steps {steps[0]}..{steps[-1]}  ({len(steps)} stops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
