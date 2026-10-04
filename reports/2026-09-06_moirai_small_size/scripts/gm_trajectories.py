"""Collect every scored stop into results/gm_trajectories.tsv.

One row per stop: arm, stop in thousands of steps, GM-Relative MASE. The
source is the score file that each head and eval writes,
results/score_<arm>_bb<stop>k_h30k_student.txt.
"""
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
SCORE_FILE = re.compile(r"score_(.+)_bb(\d+)k_h30k_student\.txt")


def scored_stops():
    for path in RESULTS.glob("score_*_h30k_student.txt"):
        match = SCORE_FILE.fullmatch(path.name)
        text = path.read_text().strip()
        if match and text:
            yield match.group(1), int(match.group(2)), float(text)


rows = sorted(scored_stops())
with open(RESULTS / "gm_trajectories.tsv", "w") as out:
    for arm, stop_k, score in rows:
        out.write(f"{arm}\t{stop_k}\t{score:.4f}\n")
print(f"{len(rows)} scored stops -> {RESULTS / 'gm_trajectories.tsv'}")
