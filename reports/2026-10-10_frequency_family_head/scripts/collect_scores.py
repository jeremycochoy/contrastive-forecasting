"""freq_family: the score tables of the waves.

Reads, under ``--base`` (default /home/jupyter/cf_runs/freq_family):

    results/arms.tsv                         one line for each arm of each
                                             wave (run_wave.sh)
    results/score_<tag>.txt                  the GM-Relative MASE of an arm
    heads/eval/<tag>/gift/all_results.csv    the MASE of each config
    heads/eval/<tag>/gift/shard_*/shard.log  the family member of each config
    heads/eval/<tag>/gift/protocol.txt       the device and the config count
                                             of the eval (run_wave.sh)
    heads/eval/<tag>/stop.log                the device of an eval with no
                                             protocol file (the code of
                                             ad1984a6)

Writes:

    results/scores.tsv       one row for each scored (run, stop, arm), with
                             its config count and its eval device
    results/config_mase.tsv  one row for each config of each scored arm

``ratio_to_control`` is the score of an arm divided by the score of the
control of its wave: a family is compared with that control only. The two
scores must have the same configs and the same device. If not, the arm gets
no ratio, and the script names it.

An arm gets no row, and the script ends with code 1, in three cases:

* The eval log of a family arm gives a config another member than the
  member of its frequency. Such a score did not select the decoder by the
  frequency.
* The eval table does not hold the config count of its protocol file: 97,
  or the count of a test filter.
* A stop left a score with no eval table, or a protocol file with no text.

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
# The line of run_wave.sh in a protocol file, and the line of
# head_eval_bb.sh at each eval start.
PROTOCOL_LINE = re.compile(r"device=(\S+) configs=(\d+) filter=")
EVAL_START_LINE = re.compile(r"\] eval start \(.*, ([a-z]+)\)\s*$")
DEVICES = ("cpu", "cuda")
SCORE_COLUMNS = ("run", "stop_k", "arm", "gm_rel_mase", "ratio_to_control",
                 "configs", "device", "members", "head_steps", "tag")
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


def eval_protocol(gift):
    """``(device, config count)`` of the protocol file that run_wave.sh
    wrote for the eval of ``gift``, or None for an eval with none. Raises
    ValueError for a file that names no device: a stop cut it."""
    path = os.path.join(gift, "protocol.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        match = PROTOCOL_LINE.match(f.read())
    if match is None:
        raise ValueError(f"{path} names no device and no config count")
    return match.group(1), int(match.group(2))


def logged_devices(gift):
    """The devices of the eval starts in the runner log beside ``gift``,
    joined with ``+``. One device for an eval that ran on one device, and
    nothing for a log that names none."""
    path = os.path.join(os.path.dirname(gift), "stop.log")
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as f:
        starts = (EVAL_START_LINE.search(line) for line in f)
        return "+".join(sorted({m.group(1) for m in starts if m}))


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


def score_errors(arm, mase, logged, protocol):
    """Why the eval of an arm is not its score: a family config with
    another member than the member of its frequency, or a table with
    another config count than its protocol file. Nothing for a good eval."""
    errors = []
    family = arm_members(arm) is not None
    wrong = wrong_members(arm, mase, logged) if family else []
    if wrong:
        errors.append(f"{len(wrong)} of {len(mase)} configs have no log line "
                      f"with the member of their frequency: "
                      f"{', '.join(wrong[:5])}")
    if protocol is not None and protocol[1] != len(mase):
        errors.append(f"the eval table holds {len(mase)} configs, and its "
                      f"protocol file names {protocol[1]}")
    return errors


def scored_arm(base, arm_row):
    """``(score row, config rows, errors)`` of one arm, or None for an arm
    with no score."""
    tag, arm = arm_row["tag"], arm_row["arm"]
    score = read_score(base, tag)
    if score is None:
        return None
    gift = os.path.join(base, "heads", "eval", tag, "gift")
    try:
        mase, protocol = config_mase(gift), eval_protocol(gift)
    except (OSError, ValueError) as error:
        return dict(arm_row), [], [str(error)]
    logged = config_members(gift) if arm_members(arm) is not None else {}
    counts = collections.Counter(logged[c] for c in mase if c in logged)
    row = dict(arm_row, gm_rel_mase=score, configs=len(mase),
               device=protocol[0] if protocol else logged_devices(gift),
               members=" ".join(f"{k}:{counts[k]}" for k in sorted(counts)),
               config_set=frozenset(mase))
    configs = [dict(arm_row, config=c, mase=m, member=logged.get(c, ""))
               for c, m in sorted(mase.items())]
    return row, configs, score_errors(arm, mase, logged, protocol)


def same_eval(row, control):
    """The two scores have the same configs and one known device."""
    return (row["device"] in DEVICES and row["device"] == control["device"]
            and row["config_set"] == control["config_set"])


def add_ratios(rows):
    """``ratio_to_control`` of each score row: its score over the score of
    the control of its wave. Nothing when that control has no score, other
    configs or another device. Returns the rows of the last two cases."""
    control = {arm_key(r)[:3]: r for r in rows if r["arm"] == "control"}
    apart = []
    for row in rows:
        base = control.get(arm_key(row)[:3])
        row["ratio_to_control"] = ""
        if base is not None and same_eval(row, base):
            ratio = float(row["gm_rel_mase"]) / float(base["gm_rel_mase"])
            row["ratio_to_control"] = f"{ratio:.4f}"
        elif base is not None:
            apart.append((row, base))
    return apart


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
        row, config_rows, errors = scored
        if errors:
            bad += 1
            print(f"ERROR {row['tag']} ({row['arm']}): {'. '.join(errors)}")
            continue
        scores.append(row)
        configs.extend(config_rows)
    for row, control in add_ratios(scores):
        print(f"NOTE {row['tag']}: no ratio to its control. Its eval has "
              f"{row['configs']} configs on the device '{row['device']}'. "
              f"The control has {control['configs']} on "
              f"'{control['device']}'. A ratio needs the same configs and "
              f"one device of {DEVICES}.")
    results = os.path.join(base, "results")
    write_tsv(os.path.join(results, "scores.tsv"), SCORE_COLUMNS, scores)
    write_tsv(os.path.join(results, "config_mase.tsv"), CONFIG_COLUMNS,
              configs)
    print(f"{len(scores)} scored arms -> {results}/scores.tsv, "
          f"{len(configs)} config rows -> {results}/config_mase.tsv")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
