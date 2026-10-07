"""#425: one check line for each job of jobs.tsv, from the raw artefacts
that collect.py copies and from elisa's mirror of the box.

A job passes when each of these holds:

* its score file holds a number;
* its per-config table holds 97 configs, each with a finite MASE;
* the geometric mean of its relative MASE gives the score. The MASE comes
  from the per-config table, and the seasonal-naive MASE from the eval
  summary;
* its eval ran strategy R on 97 configs (the stop log);
* the loss CSV of its head ends at step 30,000;
* elisa holds its final head (CF425_MIRROR, default
  ~/checkpoints_backup/cf-412/vast_lr100x).

Writes ``results/checks.tsv``: the code of the run, the stop in thousands
of steps, the score, the value of each check, and ``ok`` or the names of
the checks that fail. Exits with 1 when a job does not pass.

With CF425_HEAD_ARCH=linear, it checks the linear head of each job (the tag
``..._recon_lin``, the heads in ~/checkpoints_backup/cf-425-lin/ckpt) and
writes ``results/checks_lin.tsv``.
"""
import csv
import gzip
import math
import os
import re
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
RESULTS = STUDY / "results"
JOBS = STUDY / "scripts" / "jobs.tsv"
BACKUP = Path.home() / "checkpoints_backup"
# For each head: the end of its tags, its table, and the folder of the heads
# on elisa (CF425_MIRROR/<name> when CF425_MIRROR is set).
SUFFIX, TABLE, NAME, HEADS = {
    "transformer": ("recon", "checks.tsv", "cf-425",
                    BACKUP / "cf-412" / "vast_lr100x" / "cf-425"),
    "linear": ("recon_lin", "checks_lin.tsv", "cf-425-lin",
               BACKUP / "cf-425-lin" / "ckpt"),
}[os.environ.get("CF425_HEAD_ARCH", "transformer")]
MIRROR = (Path(os.environ["CF425_MIRROR"]) / NAME
          if os.environ.get("CF425_MIRROR") else HEADS)
CONFIGS, HEAD_STEPS = 97, 30000
MASE = "eval_metrics/MASE[0.5]"
# A row of the eval summary: the config, its MASE, its seasonal-naive MASE
# and the ratio of the two.
SUMMARY_ROW = re.compile(r"(\S+/\S+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s*")
EVAL_START = re.compile(r"eval start \((\d+) configs, (\w+),")
COLUMNS = ["code", "stop_k", "score", "configs", "gm", "strategy",
           "head_steps", "head_bytes", "result"]


def jobs(path):
    """``(code, arm, stop in thousands)`` of each row of jobs.tsv."""
    for row in open(path):
        if not row.startswith("#"):
            code, arm, stop_k = row.split("\t")[:3]
            yield code, arm, int(stop_k)


def read_score(path):
    """The score of a score file, or None."""
    try:
        return float(path.read_text())
    except (OSError, ValueError):
        return None


def config_mase(path):
    """``{config: MASE}`` of a per-config table. {} with no table."""
    if not path.is_file():
        return {}
    return {row["dataset"]: float(row[MASE])
            for row in csv.DictReader(open(path))}


def naive_mase(path):
    """``{config: seasonal-naive MASE}`` of an eval summary."""
    if not path.is_file():
        return {}
    matches = (SUMMARY_ROW.fullmatch(row.rstrip("\n")) for row in open(path))
    return {m.group(1): float(m.group(2)) for m in matches if m}


def geometric_mean(mase, naive):
    """The GM of MASE / seasonal-naive MASE, or None when a config has no
    finite positive ratio."""
    try:
        logs = [math.log(mase[config] / naive[config]) for config in mase]
        return math.exp(sum(logs) / len(logs))
    except (KeyError, ValueError, ZeroDivisionError):
        return None


def eval_start(path):
    """``(configs, strategy)`` of the last eval of a stop log."""
    found = EVAL_START.findall(path.read_text()) if path.is_file() else []
    return (int(found[-1][0]), found[-1][1]) if found else (0, "")


def last_step(path):
    """The last step of a head loss CSV, or 0."""
    if not path.is_file():
        return 0
    with gzip.open(path, "rt") as rows:
        step = rows.read().strip().rsplit("\n", 1)[-1].split(",")[0]
    return int(step) if step.isdigit() else 0


def head_bytes(folder):
    """The size of the final head of a job in elisa's mirror, or 0."""
    return sum(path.stat().st_size for path in folder.glob("*_final.pth"))


def check(code, arm, stop_k):
    """One row of checks.tsv."""
    tag = f"{arm}_bb{stop_k}k_h30k_{SUFFIX}"
    score = read_score(RESULTS / "scores" / f"score_{tag}.txt")
    mase = config_mase(RESULTS / "per_config" / f"{tag}.csv")
    logs = RESULTS / "logs" / "jobs" / tag
    gm = geometric_mean(mase, naive_mase(logs / "summary.txt"))
    configs, strategy = eval_start(logs / "stop.log")
    steps = last_step(RESULTS / "head_losses" / f"{tag}_losses.csv.gz")
    size = head_bytes(MIRROR / "recon" / "eval" / tag)
    # A job with no score has no eval yet: the other checks say nothing more.
    failed = ["score"] if score is None else [name for name, passed in [
        ("configs", len(mase) == CONFIGS and configs == CONFIGS
         and all(math.isfinite(v) for v in mase.values())),
        ("gm", gm is not None and abs(gm - score) < 1e-4),
        ("strategy", strategy == "R"),
        ("head_steps", steps == HEAD_STEPS),
        ("head_bytes", size > 0),
    ] if not passed]
    return {"code": code, "stop_k": stop_k,
            "score": "" if score is None else f"{score:.4f}",
            "configs": len(mase), "gm": "" if gm is None else f"{gm:.4f}",
            "strategy": strategy, "head_steps": steps, "head_bytes": size,
            "result": "ok" if not failed else "FAIL " + ",".join(failed)}


def main():
    rows = [check(*job) for job in jobs(JOBS)]
    with open(RESULTS / TABLE, "w", newline="") as out:
        writer = csv.DictWriter(out, COLUMNS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    bad = [row for row in rows if row["result"] != "ok"]
    for row in bad:
        print(f"{row['code']} {row['stop_k']}k: {row['result']}")
    print(f"{len(rows) - len(bad)} of {len(rows)} jobs pass "
          f"-> {RESULTS / TABLE}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
