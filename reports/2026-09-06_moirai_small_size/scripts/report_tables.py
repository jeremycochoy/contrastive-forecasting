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
                     row.get("seed", "?"), row.get("lr", "?"),
                     "yes" if row.get("decay", "-") != "-" else "no",
                     f"{stop:,}", f"{value:.4f}", ref_text, gap_text, matched))
    return table(rows, ["arm", "k", "reduce", "seed", "lr", "L_rep decay",
                        "stop", "11.4M", "1.1M twin (parent seed range)",
                        "gap", "head-matched"])


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


# The steps the AUC table reads. 2,000 is where the decay ramp ends and
# 40,000 is the first stop.
#
# THERE IS NO MARK NEAR A COLLAPSE, on purpose. `k32_r100_09_dec` puts its
# rolling median under 0.55 at step 17,313, back above 0.75 by 18,000, and
# under for good from 18,634. No single mark inside that 1,300-step band is a
# readable number, and an earlier version of this table put one at 18,600.
# The `verdict` column carries the collapse instead, from `auc_verdicts.tsv`,
# which is the statement the card asks for: whether a run lost the task, and
# at which step.
# THE MARKS ARE DENSE BETWEEN 8,000 AND 15,000 ON PURPOSE. Two sessions drew a
# conclusion from this table and had it overturned by the next two rows, twice.
# At 8,000 it showed a clean fall with the depth. At 10,000 it showed a
# crossing that read as "no separation". By 15,000 the shallower arm was 0.076
# worse, which inverts both readings. A table that stops early does not say
# "unknown", it says something wrong.
AUC_MARKS = (2000, 5000, 8000, 10000, 12000, 15000, 18600, 25000, 28000,
             40000)
AUC_ROOT = "/home/jupyter/checkpoints_backup/cf-412"
# The gate's own window, from `study.sh`. One number, so the table and the
# verdicts cannot drift apart.
AUC_WINDOW = 500


def rolling_median(series, window):
    """`[(step, median of the last `window` values)]`, the gate's statistic."""
    from collections import deque
    from statistics import median
    run, out = deque(maxlen=window), []
    for step, value in series:
        run.append(value)
        out.append((step, median(run)))
    return out


# The columns that make two arms a controlled pair. `lr` is in the list
# because the rate bracket moves it and nothing else.
PAIR_COLUMNS = ("k", "reduce", "tau", "end", "ramp", "seed", "decay", "lr")
COLUMN_NAMES = {"k": "the rollout depth", "reduce": "the reduction",
                "decay": "the L_rep decay", "seed": "the seed",
                "lr": "the learning rate"}


def momentum(row):
    """One arm's EMA schedule, short enough for a table cell."""
    if row.get("end", "-") == "-":
        return f"{row['tau']} fixed"
    return f"{row['tau']} to {row['end']} at {int(row['ramp']) // 1000}k"


def controlled_pairs(arms, drawn):
    """Every pair of drawn arms that differs in exactly ONE column.

    WHY THIS EXISTS. The table puts four k = 32 rows together, and two of
    them differ from a third in the decay and two in the momentum. A reader
    who subtracts any other two is reading a diagonal: a difference that
    moves two things at once and therefore measures neither. Both sessions
    working this card made that subtraction once. Naming the pairs the table
    supports is cheaper than describing them beside it.
    """
    out = []
    names = sorted(drawn)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = arms.get(a), arms.get(b)
            if not ra or not rb:
                continue
            moved = [c for c in PAIR_COLUMNS if ra.get(c) != rb.get(c)]
            if len(moved) == 1:
                col = moved[0]
                if col in ("tau", "end", "ramp"):
                    col = "the EMA momentum"
                else:
                    col = COLUMN_NAMES.get(col, col)
                out.append(f"`{a}` against `{b}`, which moves {col}")
    # The momentum lives in three columns, so two arms on different schedules
    # differ in more than one of them. Fold those into one comparison.
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = arms.get(a), arms.get(b)
            if not ra or not rb:
                continue
            moved = [c for c in PAIR_COLUMNS if ra.get(c) != rb.get(c)]
            if len(moved) > 1 and set(moved) <= {"tau", "end", "ramp"}:
                out.append(f"`{a}` against `{b}`, which moves the EMA momentum")
    return sorted(set(out))


def auc_verdict_map(path):
    """`{arm: 'held' or 'lost at <step>'}` from the gate's own table."""
    out = {}
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                v = r.get("verdict", "")
                if v not in ("held", "lost"):
                    continue
                arm = r["run"].split("_cf412_")[-1].replace("_losses.csv", "")
                text = "held" if v == "held" else f"lost at {int(r['lost_at']):,}"
                # A later leg of one arm must not overwrite a `lost`.
                if out.get(arm, "held") == "held":
                    out[arm] = text
    except OSError:
        pass
    return out


def auc_by_step_table(arms, root=AUC_ROOT, verdicts=None):
    """The contrastive AUC of every run at a fixed set of steps.

    WHY THIS TABLE EXISTS. The verdict table says held or lost, and that hides
    the thing this card measures. The rollout depth erodes the contrastive
    task on its own at 11.4M parameters, and the `L_rep` decay is fatal only
    where the depth has already spent the margin. One row per arm over the
    same steps shows both, and a held-or-lost column cannot.

    THE STATISTIC IS THE GATE'S: a rolling MEDIAN over the last
    `CF412_AUC_WINDOW` rows, which is 500. `auc_guard.sh` stops a run on it
    and `auc_verdicts.tsv` reports it, so a cell here and a floor there are
    the same kind of number. An earlier version of this table took a trailing
    MEAN over 200 rows and this docstring called it a median, which invited a
    reader to reconcile two different statistics as one.

    A dash is a step the arm never reached.
    """
    verdicts = verdicts or {}
    rows = []
    for arm, row in arms.items():
        paths = losses_csvs(root, arm, row["k"])
        if not paths:
            continue
        series = rolling_median(S.read_run(paths, ["auc"], 1)["auc"],
                                AUC_WINDOW)
        if not series:
            continue
        cells = []
        for mark in AUC_MARKS:
            near = [(abs(s - mark), v) for s, v in series if abs(s - mark) < 400]
            cells.append(f"{min(near)[1]:.3f}" if near else "—")
        ref = S.reference_auc(arm)
        ref_text = f"{ref[0]:.3f} ({ref[2]})" if ref else "—"
        rows.append((arm, row["k"], momentum(row),
                     "yes" if row.get("decay", "-") != "-" else "no",
                     *cells, verdicts.get(arm, "—"), ref_text))
    if not rows:
        return "_no losses CSV yet._"
    rows.sort(key=lambda r: (int(r[1]), r[3]))
    return table(rows, ["arm", "k", "EMA momentum", "L_rep decay"]
                 + [f"{m:,}" for m in AUC_MARKS]
                 + ["verdict", "1.1M twin at 40,000"])


def losses_csvs(root, arm, k):
    """Every losses CSV one arm wrote, over every leg."""
    return S.losses_csvs(root, arm, k)


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


def pairs_note(arms):
    """The only subtractions this table supports, named."""
    drawn = {a for a in arms if losses_csvs(AUC_ROOT, a, arms[a]["k"])}
    pairs = controlled_pairs(arms, drawn)
    if not pairs:
        return "_no two runs of this card differ in one column alone._"
    return ("**Read this table by row and by column, never on the diagonal.** "
            "Two rows are comparable only when they differ in ONE column. "
            "These are the pairs, and there are no others:\n\n"
            + "\n".join(f"- {p}" for p in pairs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(RESULTS / "tables.md"))
    args = p.parse_args()

    arms = {r["arm"]: r for r in S.read_arms(HERE / "arms.tsv")}
    scores = S.read_scores(RESULTS / "scores.csv")
    band, band_src = S.effective_band(
        {a: v for (a, s), v in scores.items() if s == S.STOP})
    text = "\n\n".join([
        "### The scores",
        f"The band is **{band:.4f}** ({band_src}). Two numbers closer than "
        "that are not ranked.",
        scores_table(arms, scores),
        "### The contrastive AUC", auc_table(RESULTS / "auc_verdicts.tsv"),
        "### The contrastive AUC, step by step",
        "Lower is worse. A run at 0.5 has lost the task. The last column is "
        "what the same cell, seed and stop reached at 1.1M parameters.",
        auc_by_step_table(arms,
                          verdicts=auc_verdict_map(RESULTS / "auc_verdicts.tsv")),
        pairs_note(arms),
        "### The loss by term", terms_table(RESULTS / "loss_terms.csv"),
        "### The cost", cost_table([RESULTS / "arms.log",
                                    RESULTS / "stops.log"]),
    ]) + "\n"
    Path(args.out).write_text(text)
    print(f"wrote {args.out} ({len(scores)} scored pair(s))")


if __name__ == "__main__":
    main()
