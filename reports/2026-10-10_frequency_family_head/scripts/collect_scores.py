"""freq_family: the score tables of the waves.

Reads, under ``--base`` (default /home/jupyter/cf_runs/freq_family):

    results/arms.tsv                         one line for each arm of each
                                             wave (run_wave.sh)
    results/score_<tag>.txt                  the GM-Relative MASE of an arm
    heads/eval/<tag>/gift/all_results.csv    the MASE of each config
    heads/eval/<tag>/gift/shard_*/shard.log  the family member of each config

Writes:

    results/scores.tsv       one row for each scored (run, stop, arm)
    results/config_mase.tsv  one row for each config of each scored arm

``ratio_to_control`` is the score of an arm divided by the score of the
control of its wave: a family is compared with that control only.

The eval log names the family member of each config. A family arm whose log
gives a config another member than the member of its frequency gets no row,
and the script ends with code 1: such a score did not select the decoder by
the frequency.

Usage: PYTHONPATH=<code folder> python3 collect_scores.py [--base <folder>]
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys

from src.freq_family import FAMILY_MEMBERS, family_member

BASE = "/home/jupyter/cf_runs/freq_family"
MASE = "eval_metrics/MASE[0.5]"
ARM_ORDER = ("control", "shared_strict", "shared_draw", "heads_strict",
             "heads_draw")
MEMBER_LINE = re.compile(
    r"^\s*\[eval\] (\S+): frequency \S+, family member (\d+)\s*$")
SCORE_COLUMNS = ("run", "stop_k", "arm", "gm_rel_mase", "ratio_to_control",
                 "configs", "members", "head_steps", "tag")
CONFIG_COLUMNS = ("run", "stop_k", "arm", "config", "mase", "member")


def read_arms(base):
    """The lines of ``results/arms.tsv``, as dicts."""
    path = os.path.join(base, "results", "arms.tsv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_score(base, tag):
    """The GM-Relative MASE of an arm, as the eval wrote it, or None."""
    path = os.path.join(base, "results", f"score_{tag}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read().strip() or None


def config_mase(gift):
    """``{config: MASE}`` of one eval, with the digits of its table."""
    with open(os.path.join(gift, "all_results.csv")) as f:
        return {row["dataset"]: row[MASE] for row in csv.DictReader(f)}


def config_members(gift):
    """``{config: member key}`` from the shard logs of one eval."""
    members = {}
    for name in sorted(os.listdir(gift)):
        log = os.path.join(gift, name, "shard.log")
        if not name.startswith("shard_") or not os.path.exists(log):
            continue
        with open(log, errors="replace") as f:
            for line in f:
                match = MEMBER_LINE.match(line)
                if match:
                    members[match.group(1)] = int(match.group(2))
    return members


def arm_members(arm):
    """The member keys of a family arm (`<body>_<rule>[_m<keys>]`), or None
    for the control."""
    if arm == "control":
        return None
    keys = re.search(r"_m([0-9-]+)$", arm)
    if keys is None:
        return FAMILY_MEMBERS
    return tuple(int(k) for k in keys.group(1).split("-"))


def wrong_members(arm, configs, logged):
    """The configs of a family arm whose logged member is not the member of
    their frequency. The frequency is the second part of a config name."""
    family = arm_members(arm)
    return [c for c in configs
            if logged.get(c) != family_member(c.split("/")[1], family)]


def arm_key(row):
    """Sort key: the run, the stop, the head steps, then the arm."""
    arm = row["arm"]
    order = ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER)
    return row["run"], int(row["stop_k"]), int(row["head_steps"]), order, arm


def scored_arm(base, arm_row):
    """``(score row, config rows, wrong configs)`` of one arm, or None for
    an arm with no score."""
    tag, arm = arm_row["tag"], arm_row["arm"]
    score = read_score(base, tag)
    if score is None:
        return None
    gift = os.path.join(base, "heads", "eval", tag, "gift")
    mase = config_mase(gift)
    family = arm_members(arm) is not None
    logged = config_members(gift) if family else {}
    wrong = wrong_members(arm, mase, logged) if family else []
    counts = collections.Counter(logged[c] for c in mase if c in logged)
    row = dict(arm_row, gm_rel_mase=score, configs=len(mase),
               members=" ".join(f"{k}:{counts[k]}" for k in sorted(counts)))
    configs = [dict(arm_row, config=c, mase=m, member=logged.get(c, ""))
               for c, m in sorted(mase.items())]
    return row, configs, wrong


def add_ratios(rows):
    """``ratio_to_control`` of each score row: its score over the score of
    the control of its wave, or nothing when that control has no score."""
    control = {arm_key(r)[:3]: float(r["gm_rel_mase"]) for r in rows
               if r["arm"] == "control"}
    for row in rows:
        base = control.get(arm_key(row)[:3])
        row["ratio_to_control"] = (
            "" if base is None else f"{float(row['gm_rel_mase']) / base:.4f}")


def write_tsv(path, columns, rows):
    """Write ``rows`` to ``path`` in one step."""
    with open(path + ".tmp", "w", newline="") as f:
        writer = csv.DictWriter(f, columns, delimiter="\t",
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(path + ".tmp", path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", default=BASE)
    base = parser.parse_args(argv).base
    scores, configs, bad = [], [], 0
    for arm_row in sorted(read_arms(base), key=arm_key):
        scored = scored_arm(base, arm_row)
        if scored is None:
            continue
        row, config_rows, wrong = scored
        if wrong:
            bad += 1
            print(f"ERROR {row['tag']} ({row['arm']}): {len(wrong)} of "
                  f"{row['configs']} configs have no log line with the "
                  f"member of their frequency: {', '.join(wrong[:5])}")
            continue
        scores.append(row)
        configs.extend(config_rows)
    add_ratios(scores)
    results = os.path.join(base, "results")
    write_tsv(os.path.join(results, "scores.tsv"), SCORE_COLUMNS, scores)
    write_tsv(os.path.join(results, "config_mase.tsv"), CONFIG_COLUMNS,
              configs)
    print(f"{len(scores)} scored arms -> {results}/scores.tsv, "
          f"{len(configs)} config rows -> {results}/config_mase.tsv")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
