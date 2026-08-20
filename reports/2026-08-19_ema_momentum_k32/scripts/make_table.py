#!/usr/bin/env python3
"""#404 deliverable 4 — the table of scores, and the one statement that reads it.

The table holds every arm and the five published references. The statement
names the momentum that wins, gives its distance to the k = 3 score at bb40k,
and says whether that arm goes below it.

It then gives the card's own repeat spread. Four arms of this table are ONE arm
at four backbone seeds, and the distance between their scores is what says
whether the winner is ahead of the next arm or level with it. Round 1 named a
winner with no such number, and the review of PR #405 asked for one.

ONE OF THOSE FOUR MUST NOT COUNT. `s08b` did not measure noise: its backbone
fell to chance while it trained. The distance between a collapsed run and a
healthy one is not a spread, and quoting it as one calls every arm of the card
unranked. So `--sync-root` lets this script read the contrastive AUC of every
arm, drop the collapsed runs, and quote the spread over the rest.
`seed_report.py` holds the one definition of a collapse in this study.

1.0862 is the comparison, not 1.0660: both this card's arms and that number
stop at 40,000 backbone steps.

Usage:
  make_table.py --scores results/scores.csv --out results/table.md \
                --sync-root ~/cf404_sync
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = _load("cf404_refs", "references.py")
REPEAT = _load("cf404_repeat", "repeat_spread.py")
SEEDS = _load("cf404_seeds", "seed_report.py")

def schedule_text(r: dict) -> str:
    """How one arm's momentum moves, with its OWN ramp length.

    This was a constant, `{"ramp": "to 1.0 at 200k"}`, while every ramp of
    this card ran 200,000 steps. It is a function now: the card added a
    100,000-step ramp, and a constant would have printed 200k for it.
    """
    if r["schedule"] != "ramp" or not r.get("ramp"):
        return "fixed"
    return f"to 1.0 at {r['ramp'] // 1000}k"


def holds_at(r: dict, stop: int) -> float:
    """The momentum the arm HOLDS at `stop`, not the one it starts at.

    Two ramp lengths now share a start value: `s08` and `r100_08` both start
    at 0.8 and hold 0.840 and 0.880 at 40,000 steps. Linear over the ramp and
    clamped, the same formula as `src.models.ema_tau_at_step`.
    `scripts/test_momentum_at.sh` holds the shell copy against the trainer's.
    """
    if r["schedule"] != "ramp" or not r.get("ramp"):
        return float(r["alpha"])
    frac = min(max(stop / r["ramp"], 0.0), 1.0)
    return float(r["alpha"]) + frac * (1.0 - float(r["alpha"]))


def read_scores(path) -> list[dict]:
    """The rows of collect.sh's scores.csv, typed and ordered by score."""
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({"arm": r["arm"], "alpha": float(r["alpha"]),
                             "schedule": r["schedule"],
                             "ramp": int(float(r.get("ramp") or 0)),
                             "seed": r.get("seed", ""),
                             "score": float(r["score"])})
            except (KeyError, ValueError, TypeError):
                continue
    return sorted(rows, key=lambda r: r["score"])


def best(rows: list[dict]) -> dict:
    """The arm with the lowest GM-Relative MASE."""
    if not rows:
        raise SystemExit("ABORT: no arm is scored yet")
    return rows[0]


def beats_k3(rows: list[dict]) -> bool:
    """True when the best arm goes below the k = 3 score at bb40k."""
    return best(rows)["score"] < REF.K3_BB40K


def table_markdown(rows: list[dict], stop: int = 40000) -> str:
    """The card's table: this card's arms first, then the references.

    The backbone seed is a column because four arms of this card are ONE arm at
    four seeds, and the contrastive AUC is a column because one of them fell to
    chance while it trained. A score table without the AUC shows a collapsed
    run as a bad arm.
    """
    out = [f"| arm | EMA momentum | holds at {stop // 1000}k | backbone seed "
           "| AUC at the stop | GM-Relative MASE | vs k = 3 at bb40k |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        alpha = f"{r['alpha']:g}, {schedule_text(r)}"
        auc = r.get("auc")
        auc_text = "?" if auc is None else f"{auc:.3f}"
        if r.get("collapsed"):
            auc_text += " (collapsed)"
        out.append(f"| {r['arm']} | {alpha} | {holds_at(r, stop):.3f} | "
                   f"{r.get('seed') or '?'} | "
                   f"{auc_text} | {r['score']:.4f} | "
                   f"{r['score'] - REF.K3_BB40K:+.4f} |")
    out += ["", "| reference | GM-Relative MASE |", "|---|---|"]
    for label, value in REF.TABLE:
        out.append(f"| {label} | {value:.4f} |")
    return "\n".join(out)


def statement(rows: list[dict], stop: int = 40000) -> str:
    """The one sentence the card asks for.

    It names the ARM as well as the momentum. Two arms of this card start at
    0.8 and two start at 0.9, so a sentence that gives the start value alone
    does not say which arm won.
    """
    win = best(rows)
    delta = win["score"] - REF.K3_BB40K
    verdict = ("goes below" if delta < 0 else "does not go below")
    band = (" It lands inside the k = 3 repeat band."
            if REF.enters_band(win["score"]) else "")
    return (f"`{win['arm']}` wins, at {win['score']:.4f}. Its momentum starts "
            f"at {win['alpha']:g} ({schedule_text(win)}) and holds "
            f"{holds_at(win, stop):.3f} at {stop:,} steps. It sits "
            f"{delta:+.4f} from the k = 3 score at bb40k, "
            f"{REF.K3_BB40K:.4f}, so it {verdict} that score.{band}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sync-root",
                   help="the sync tree, to read each arm's contrastive AUC")
    p.add_argument("--stop", type=int, default=40000)
    args = p.parse_args(argv)
    if not Path(args.scores).is_file():
        raise SystemExit(f"ABORT: no scores table at {args.scores}")
    rows = read_scores(args.scores)

    # The AUC of every arm at the stop, when a sync tree is here. It fills the
    # AUC column and it decides which runs the spread below is measured over.
    rep = None
    if args.sync_root:
        root = Path(args.sync_root).expanduser()
        for r in rows:
            r["auc"] = SEEDS.auc_at(root, r["arm"], args.stop)
            r["collapsed"] = SEEDS.collapsed(r["auc"])
        rep = SEEDS.report(rows, root, args.stop)

    parts = [table_markdown(rows, args.stop),
             statement(rows, args.stop)]
    if rep is not None and rep["spread"] is not None:
        parts.append(SEEDS.spread_sentence(rows, rep))
        parts.append(rep["separation"])
    elif rep is None:
        measured = REPEAT.sentence(rows)
        if measured:
            parts.append(measured)
    parts = [x for x in parts if x]
    text = "\n\n".join(parts) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
