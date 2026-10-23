"""Merge the score tables of several result folders of the forecast waves
into results/scores.tsv and results/config_mase.tsv of this report.

The waves of this report ran on elisa
(/home/jupyter/cf_runs/freq_family/results) and, from 10-14, on three vast
boxes (their results pulled to /home/jupyter/cf_runs/box_heads/ff_results,
box_heads2/ff_results and box_heads3/ff_results). Each folder holds a scores.tsv (one row for each run, stop and arm) and a
config_mase.tsv (one row for each run, stop, arm and config).

    python3 merge_scores.py <result folder> [<result folder> ...]

A row of a later folder with the key of an earlier row is left out, so the
order of the folders decides. The rows keep the order of their folders.
"""
import csv
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
# table -> the columns of its key
KEYS = {"scores.tsv": ("run", "stop_k", "arm"),
        "config_mase.tsv": ("run", "stop_k", "arm", "config")}


def merge(table, folders):
    """The header and the rows of ``table`` of each folder, with no repeated key."""
    header, rows, seen = None, [], set()
    for folder in folders:
        path = Path(folder) / table
        if not path.is_file():
            continue
        with open(path) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            header = header or reader.fieldnames
            for row in reader:
                key = tuple(row[column] for column in KEYS[table])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    return header, rows


def main():
    for table in KEYS:
        header, rows = merge(table, sys.argv[1:])
        with open(RESULTS / table, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
