"""#425: the scores of the box, in the tables of the report.

Reads what sync_box.sh brings to elisa:

* the box results (CF425_RESULTS_MIRROR, default
  ~/checkpoints_backup/cf-425/box_results): one score file per job,
  ``score_<arm>_bb<stop>k_h30k_recon.txt`` for a reconstruction and
  ``score_<arm>_bb<stop>k_h30k_student.txt`` for a B4 forecast of this card.
* the per-config table of each reconstruction score, in elisa's mirror of the
  box (CF425_MIRROR, default ~/checkpoints_backup/cf-412/vast_lr100x).

Writes, in the results directory of the report:

* ``recon_trajectories.tsv`` and ``forecast_425.tsv``: arm, stop in
  thousands of steps, GM-Relative MASE. The layout of #412's
  ``gm_trajectories.tsv``.
* ``per_config/<tag>.csv``: the 97 rows of each reconstruction score.
"""
import os
import re
import shutil
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
RESULTS = STUDY / "results"
HOME = Path.home() / "checkpoints_backup"
BOX_RESULTS = Path(os.environ.get("CF425_RESULTS_MIRROR",
                                  HOME / "cf-425" / "box_results"))
MIRROR = Path(os.environ.get("CF425_MIRROR",
                             HOME / "cf-412" / "vast_lr100x")) / "cf-425"
SCORE = re.compile(r"score_(.+)_bb(\d+)k_h30k_(recon|student)\.txt")


def scores(folder):
    """``{kind: [(arm, stop_k, score)]}`` from the score files of
    ``folder``. An empty file is no score."""
    found = {"recon": [], "student": []}
    for path in folder.glob("score_*.txt"):
        match = SCORE.fullmatch(path.name)
        text = path.read_text().strip()
        if match and text:
            arm, stop_k, kind = match.groups()
            found[kind].append((arm, int(stop_k), float(text)))
    return {kind: sorted(rows) for kind, rows in found.items()}


def write_table(rows, path):
    with open(path, "w") as out:
        for arm, stop_k, score in rows:
            out.write(f"{arm}\t{stop_k}\t{score:.4f}\n")
    print(f"{len(rows)} scores -> {path}")


def copy_per_config(rows):
    """The per-config table of each reconstruction score, when elisa holds
    it."""
    folder = RESULTS / "per_config"
    folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    for arm, stop_k, _ in rows:
        tag = f"{arm}_bb{stop_k}k_h30k_recon"
        table = MIRROR / "recon" / "eval" / tag / "gift_r" / "all_results.csv"
        if table.is_file():
            shutil.copyfile(table, folder / f"{tag}.csv")
            copied += 1
    print(f"{copied} per-config tables -> {folder}")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    found = scores(BOX_RESULTS)
    write_table(found["recon"], RESULTS / "recon_trajectories.tsv")
    write_table(found["student"], RESULTS / "forecast_425.tsv")
    copy_per_config(found["recon"])


if __name__ == "__main__":
    main()
