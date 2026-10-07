"""#425: the scores of the box, in the tables of the report, and the raw
artefacts of each job.

Reads what sync_box.sh brings to elisa:

* the box results (CF425_RESULTS_MIRROR, default
  ~/checkpoints_backup/cf-425/box_results): one score file per job,
  ``score_<arm>_bb<stop>k_h30k_recon.txt`` for a reconstruction and
  ``score_<arm>_bb<stop>k_h30k_student.txt`` for a B4 forecast of this card,
  the logs of the queue, and the log and the job list of each wave.
* the folder of each job, in elisa's mirror of the box (CF425_MIRROR,
  default ~/checkpoints_backup/cf-412/vast_lr100x): its per-config table,
  its logs and the loss CSV of its head.

Writes, in the results directory of the report:

* ``recon_trajectories.tsv`` and ``forecast_425.tsv``: arm, stop in
  thousands of steps, GM-Relative MASE. The layout of #412's
  ``gm_trajectories.tsv``.
* ``scores/score_<tag>.txt``: the score file of each job.
* ``per_config/<tag>.csv``: the 97 rows of each score.
* ``logs/``: the logs of the queue, the elisa sync log, the log and the job
  list of each wave, and for each job its stop log, its eval log and the
  summary of its eval.
* ``head_losses/<tag>_losses.csv.gz``: the loss of each step of each head.
  The gzip has no time stamp, so the same CSV gives the same bytes.

* ``snapshots/``: the R score of other snapshots of some heads
  (snapshot_score.sh). ``scores.tsv`` gives the arm, the stop, the snapshot,
  its head step and its score. Beside it, the per-config table and the logs
  of each snapshot.

It copies no head file: the heads stay on elisa.
"""
import gzip
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
SYNC_LOG = Path(os.environ.get("CF425_SYNC_LOG", HOME / "cf-425" / "sync.log"))
SCORE = re.compile(r"score_(.+)_bb(\d+)k_h30k_(recon|student)\.txt")
SNAPSHOT = re.compile(r"score_(.+)_bb(\d+)k_h30k_(best|final)_recon\.txt")
# The last line of a head in the log of its wave: the step of its best head.
BEST_STEP = r"\[qhead_{job}_s\d+\] Done in .* at step (\d+)"
HEAD_STEPS = 30000
QUEUE_LOGS = ["queue.log", "scores.log", "heads.log", "stops.log",
              "forecast_scores.log"]
# The folder of a job in the mirror, and the folder of its eval, by kind.
JOB_DIR = {"recon": ("recon", "gift_r", "eval_local_r.log"),
           "student": ("forecast", "gift", "eval_local.log")}


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


def tag_of(arm, stop_k, kind):
    return f"{arm}_bb{stop_k}k_h30k_{kind}"


def copy(source, target):
    """Copy a file when it exists. Returns 1 for a copy, else 0."""
    if not source.is_file():
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return 1


def gzip_copy(source, target):
    """A gzip copy with no time stamp in it. Returns 1 for a copy, else 0."""
    if not source.is_file():
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as raw, open(target, "wb") as out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=out,
                           mtime=0) as packed:
            shutil.copyfileobj(raw, packed)
    return 1


def copy_jobs(found):
    """The score file, the per-config table, the logs and the head loss
    CSV of each job with a score."""
    counts = {"scores": 0, "per_config": 0, "logs": 0, "head_losses": 0}
    for kind, rows in found.items():
        root, gift, eval_log = JOB_DIR[kind]
        for arm, stop_k, _ in rows:
            tag = tag_of(arm, stop_k, kind)
            job = MIRROR / root / "eval" / tag
            counts["scores"] += copy(BOX_RESULTS / f"score_{tag}.txt",
                                     RESULTS / "scores" / f"score_{tag}.txt")
            counts["per_config"] += copy(job / gift / "all_results.csv",
                                         RESULTS / "per_config" / f"{tag}.csv")
            logs = RESULTS / "logs" / "jobs" / tag
            counts["logs"] += copy(job / "stop.log", logs / "stop.log")
            counts["logs"] += copy(job / eval_log, logs / eval_log)
            counts["logs"] += copy(job / gift / "summary.txt",
                                   logs / "summary.txt")
            for losses in job.glob("qhead_*_losses.csv"):
                counts["head_losses"] += gzip_copy(
                    losses, RESULTS / "head_losses" / f"{tag}_losses.csv.gz")
    return counts


def copy_logs():
    """The logs of the queue and of the sync, and of each wave."""
    copied = sum(copy(BOX_RESULTS / name, RESULTS / "logs" / name)
                 for name in QUEUE_LOGS)
    copied += copy(SYNC_LOG, RESULTS / "logs" / "sync.log")
    for wave in sorted((BOX_RESULTS / "waves").glob("*/")):
        for name in ("train.log", "jobs.jsonl"):
            copied += copy(wave / name,
                           RESULTS / "logs" / "waves" / wave.name / name)
    return copied


def best_step(job):
    """The step of the best head of a job, from the log of its wave, or 0."""
    pattern = re.compile(BEST_STEP.format(job=re.escape(job)))
    for log in sorted((BOX_RESULTS / "waves").glob("*/train.log")):
        found = pattern.findall(log.read_text(errors="replace"))
        if found:
            return int(found[-1])
    return 0


def copy_snapshots():
    """The scores of snapshot_score.sh: one table, and the per-config table
    and the logs of each snapshot. Returns the number of scores."""
    rows = []
    for path in sorted((BOX_RESULTS / "snapshots").glob("score_*.txt")):
        match = SNAPSHOT.fullmatch(path.name)
        text = path.read_text().strip()
        if not (match and text):
            continue
        arm, stop_k, snapshot = match.groups()
        job = tag_of(arm, stop_k, "recon")
        step = HEAD_STEPS if snapshot == "final" else best_step(job)
        rows.append((arm, int(stop_k), snapshot, step, float(text)))
        tag = f"{arm}_bb{stop_k}k_h30k_{snapshot}_recon"
        source = MIRROR / "snapshots" / "eval" / tag
        target = RESULTS / "snapshots"
        copy(source / "gift_r" / "all_results.csv",
             target / "per_config" / f"{tag}.csv")
        for name in ("stop.log", "gift_r/summary.txt"):
            copy(source / name, target / "logs" / tag / Path(name).name)
    if rows:
        (RESULTS / "snapshots").mkdir(parents=True, exist_ok=True)
        with open(RESULTS / "snapshots" / "scores.tsv", "w") as out:
            out.write("arm\tstop_k\tsnapshot\thead_step\tscore\n")
            for arm, stop_k, snapshot, step, score in sorted(rows):
                out.write(f"{arm}\t{stop_k}\t{snapshot}\t{step}\t{score:.4f}\n")
    return len(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    found = scores(BOX_RESULTS)
    write_table(found["recon"], RESULTS / "recon_trajectories.tsv")
    write_table(found["student"], RESULTS / "forecast_425.tsv")
    counts = copy_jobs(found)
    counts["queue and wave logs"] = copy_logs()
    counts["snapshot scores"] = copy_snapshots()
    print(", ".join(f"{n} {what}" for what, n in counts.items()),
          f"-> {RESULTS}")


if __name__ == "__main__":
    main()
