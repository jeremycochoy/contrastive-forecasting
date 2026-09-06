#!/usr/bin/env python3
"""The tables of the report, built from `results/`, in Markdown.

WHY THIS SCRIPT EXISTS. A number typed into a report by hand is a number that
can differ from the artefact it comes from. Every table the report carries is
built here and pasted in whole, so the report and `results/` cannot disagree.

It writes four tables:

  1. the scores, each 11.4M arm beside its 1.1M twin and the head budget of
     each
  2. the contrastive AUC verdict of every run
  3. the training loss by term at each stop
  4. what each arm cost, in backbone steps and in wall-clock hours

Usage:  report_tables.py [--out results/tables.md]
"""
from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import plot_style as S

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RESULTS = STUDY / "results"
HEAD_STEPS = 30000


def table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def scores_table(arms, scores):
    rows = []
    for (arm, stop), value in sorted(scores.items(), key=lambda kv: kv[1]):
        row = arms.get(arm, {})
        ref = S.reference(arm, stop)
        if ref:
            gap = value - ref[0]
            matched = "yes" if ref[1] == HEAD_STEPS else \
                f"no, {ref[1]:,}-step head"
            ref_text, gap_text = f"{ref[0]:.4f}", f"{gap:+.4f}"
            spread = S.reference_spread(arm, stop)
            if spread:
                ref_text += f" ({spread[0]:.4f} to {spread[1]:.4f})"
        else:
            matched, ref_text, gap_text = "—", "never run", "—"
        rows.append((arm, row.get("k", "?"), row.get("reduce", "?"),
                     row.get("seed", "?"),
                     "yes" if row.get("decay", "-") != "-" else "no",
                     f"{stop:,}", f"{value:.4f}", ref_text, gap_text, matched))
    return table(rows, ["arm", "k", "reduce", "seed", "L_rep decay", "stop",
                        "11.4M", "1.1M twin (parent seed range)", "gap",
                        "head-matched"])


def auc_table(path):
    rows = []
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                run = r.get("run", "")
                arm = run.split("_cf412_")[-1].replace("_losses.csv", "")
                if r.get("verdict") == "error":
                    continue
                rows.append((arm, r.get("verdict", "?"), r.get("floor", "-"),
                             r.get("floor_step", "-"), r.get("last", "-"),
                             r.get("last_step", "-")))
    except OSError:
        return "_no AUC verdict yet._"
    if not rows:
        return "_no AUC verdict yet._"
    return table(sorted(rows), ["arm", "verdict", "AUC floor", "at step",
                                "AUC last", "at step"])


def terms_table(path):
    rows = []
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                def num(key, nd=4):
                    v = r.get(key)
                    try:
                        return f"{float(v):.{nd}f}"
                    except (TypeError, ValueError):
                        return "—"
                rows.append((r["arm"], f"{int(r['stop']):,}",
                             f"{int(r['step']):,}", num("loss"), num("l_rep"),
                             num("l_align"), num("rep_w", 2),
                             num("ema_tau", 4), num("auc", 4)))
    except OSError:
        return "_no loss row yet._"
    if not rows:
        return "_no loss row yet._"
    return table(sorted(rows), ["arm", "stop", "last step", "total loss",
                                "L_rep", "L_align", "L_rep weight",
                                "EMA momentum", "AUC"])


def cost_table(paths):
    """Wall-clock hours of each stage, from the timestamps the lanes wrote.

    The backbone lines land in `arms.log` and the head and evaluation lines in
    `stops.log`, so this reads both and sorts them into one stream.
    """
    stamp = re.compile(r"^\[(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\]")
    events, text = [], []
    for path in paths:
        try:
            text += Path(path).read_text().splitlines()
        except OSError:
            continue
    if not text:
        return "_no lane log yet._"
    for line in text:
        m = stamp.match(line)
        if not m:
            continue
        month, day, hh, mm, ss = (int(x) for x in m.groups())
        events.append((datetime(2026, month, day, hh, mm, ss), line))
    rows, open_at = [], {}
    for when, line in sorted(events, key=lambda e: e[0]):
        m = re.search(r"\[#412\] arm (\S+) arch=", line)
        if m:
            open_at[("backbone", m.group(1))] = when
        m = re.search(r"\[#412\] arm (\S+) stop=(\d+) rc=", line)
        if m and ("backbone", m.group(1)) in open_at:
            t0 = open_at.pop(("backbone", m.group(1)))
            rows.append((m.group(1), "backbone",
                         f"{int(m.group(2)):,}",
                         f"{(when - t0).total_seconds() / 3600:.1f}"))
        m = re.search(r"\[(\S+)\] head-train start", line)
        if m:
            open_at[("head", m.group(1))] = when
        m = re.search(r"\[(\S+)\] head-train rc=0", line)
        if m and ("head", m.group(1)) in open_at:
            t0 = open_at.pop(("head", m.group(1)))
            rows.append((m.group(1), "head", f"{HEAD_STEPS:,}",
                         f"{(when - t0).total_seconds() / 3600:.1f}"))
        m = re.search(r"\[(\S+)\] eval start", line)
        if m:
            open_at[("eval", m.group(1))] = when
        m = re.search(r"\[(\S+)\] eval rc=0", line)
        if m and ("eval", m.group(1)) in open_at:
            t0 = open_at.pop(("eval", m.group(1)))
            rows.append((m.group(1), "GIFT-Eval, 97 configs", "—",
                         f"{(when - t0).total_seconds() / 3600:.1f}"))
    if not rows:
        return "_no finished stage yet._"
    return table(rows, ["run", "stage", "steps", "hours"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(RESULTS / "tables.md"))
    args = p.parse_args()

    arms = {r["arm"]: r for r in S.read_arms(HERE / "arms.tsv")}
    scores = S.read_scores(RESULTS / "scores.csv")
    text = "\n\n".join([
        "### The scores", scores_table(arms, scores),
        "### The contrastive AUC", auc_table(RESULTS / "auc_verdicts.tsv"),
        "### The loss by term", terms_table(RESULTS / "loss_terms.csv"),
        "### The cost", cost_table([RESULTS / "arms.log",
                                    RESULTS / "stops.log"]),
    ]) + "\n"
    Path(args.out).write_text(text)
    print(f"wrote {args.out} ({len(scores)} scored pair(s))")


if __name__ == "__main__":
    main()
