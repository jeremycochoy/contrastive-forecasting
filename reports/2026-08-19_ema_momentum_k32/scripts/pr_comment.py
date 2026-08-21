#!/usr/bin/env python3
"""#404 — the ExperimentRunner comment, built from the tables on disk.

The comment carries every scored arm and the repeat spread this card
measures. Both come from `results/scores.csv`, which `collect.sh` writes from
the score files, so the comment cannot disagree with the figures.

`repeat_spread.py` finds a repeat family by (alpha, schedule, ramp). Four arms
of this card share all three and differ in the backbone seed alone, so the
distance between their scores IS the run-to-run spread of this cell.

ONE OF THOSE FOUR MUST NOT COUNT. `s08b` did not measure noise: its backbone
fell to chance while it trained, AUC 0.91 at 10,000 steps to 0.57 at 40,000. A
collapsed run is a different event from a noisy one, and its distance from a
healthy run is not a spread. So `--sync-root` lets this script read the AUC of
every arm and report the spread over the seeds that did NOT collapse.
`seed_report.py` holds the one definition of a collapse in this study.

Usage:  python3 scripts/pr_comment.py --scores results/scores.csv \\
          --sync-root ~/cf404_sync \\
          --agent "ExperimentRunner claude-opus-5" \\
          --dir reports/2026-08-19_ema_momentum_k32
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import references  # noqa: E402
import repeat_spread  # noqa: E402
import seed_report  # noqa: E402


def read_scores(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "arm": r["arm"],
                "alpha": float(r["alpha"]),
                "schedule": r["schedule"],
                "ramp": int(r["ramp"]),
                "score": float(r["score"]),
                "seed": r.get("seed", ""),
                "align_w": float(r.get("align_w") or 1.0),
            })
    return sorted(rows, key=lambda r: r["score"])


def momentum(r: dict) -> str:
    if r["schedule"] == "fixed":
        return f"{r['alpha']:.2f}, fixed"
    return f"{r['alpha']:.2f}, to 1.0 at {r['ramp'] // 1000}k"


def holds_at(r: dict, stop: int) -> float:
    """The momentum the arm HOLDS at `stop`, not the one it starts at.

    Two ramp lengths now share a start value: `s08` and `r100_08` both start
    at 0.8 and hold 0.840 and 0.880 at 40,000 steps. A comment that prints the
    start value alone gives those two arms one momentum and reads as a repeat.

    Linear over the ramp and clamped, the same formula as
    `src.models.ema_tau_at_step` and `cf404_momentum_at` in `study.sh`.
    `scripts/test_momentum_at.sh` holds the shell copy against the trainer's.
    """
    if r["schedule"] != "ramp" or not r["ramp"]:
        return float(r["alpha"])
    frac = min(max(stop / r["ramp"], 0.0), 1.0)
    return float(r["alpha"]) + frac * (1.0 - float(r["alpha"]))


def arms_of(tsv: pathlib.Path) -> list[dict]:
    """Every row of the arms table, typed. Empty when the file is absent."""
    out = []
    if not tsv.is_file():
        return out
    for line in tsv.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 4:
            continue
        out.append({"arm": f[0], "alpha": float(f[1]),
                    "schedule": "fixed" if f[2] == "-" else "ramp",
                    "ramp": 0 if f[3] == "-" else int(f[3]),
                    "seed": f[4] if len(f) > 4 else ""})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, type=pathlib.Path)
    ap.add_argument("--sync-root", type=pathlib.Path,
                    help="the sync tree, to read each arm's contrastive AUC")
    ap.add_argument("--stop", type=int, default=40000)
    ap.add_argument("--agent", default="ExperimentRunner claude-opus-5")
    ap.add_argument("--dir", default="reports/2026-08-19_ema_momentum_k32")
    ap.add_argument("--runs", type=int, default=None,
                    help="arms trained this round, for the runs line")
    ap.add_argument("--cost", default="")
    ap.add_argument("--arms-tsv", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent
                    / "arms.tsv",
                    help="the arms table, to name backbones that trained but "
                         "carry no score")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    rows = read_scores(a.scores)
    if not rows:
        print("ABORT: no scored arm in", a.scores, file=sys.stderr)
        return 2

    # The one number every arm is measured against: k = 3 at the SAME 40,000
    # steps. The card compares at bb40k, because that is where the arms stop.
    k3_40k = references.K3_BB40K
    best = rows[0]
    out = []
    out.append(f"«Agent {a.agent} writing»")
    out.append("")
    out.append(f"The experiment directory is `{a.dir}`.")
    out.append("")
    out.append(f"## The {len(rows)} scored arms")
    out.append("")
    out.append(f"| arm | EMA momentum | holds at {a.stop // 1000}k | "
               "L_align weight | backbone seed | GM-Relative MASE "
               "| vs k = 3 at bb40k |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(f"| {r['arm']} | {momentum(r)} | "
                   f"{holds_at(r, a.stop):.3f} | {r['align_w']:g} | "
                   f"{r['seed'] or '?'} | "
                   f"{r['score']:.4f} | {r['score'] - k3_40k:+.4f} |")
    out.append("")
    out.append(f"`holds at {a.stop // 1000}k` is the momentum the backbone "
               "trains against at the stop. A ramp arm does not hold the "
               "value it starts at.")
    out.append("")
    out.append("`L_align weight` is `--align-loss-weight`. The rollout depth "
               "duplicates the align term and not the repel term, and the "
               "reduction is a mean, so this flag sets the balance between "
               "one h-anchored repel term and the mean of k + 1 f-anchored "
               "pull terms.")
    out.append("")

    # The repeat spread. When a sync tree is given, the AUC of every arm at
    # the stop decides which runs count: a backbone that fell to chance is a
    # collapse, not a draw from the noise, and its distance from a healthy run
    # is not a spread.
    d = None
    rep = None
    if a.sync_root is not None:
        rep = seed_report.report(rows, a.sync_root.expanduser(), a.stop)
        d = rep["spread"]
        out.append("## The repeat family, seed by seed")
        out.append("")
        out.append(seed_report.markdown(rep, a.stop))
        out.append("")
    else:
        sentence = repeat_spread.sentence(rows)
        pairs = repeat_spread.pairs(rows)
        out.append("## The measured repeat spread")
        out.append("")
        if pairs:
            p = max(pairs, key=lambda q: repeat_spread.spread(*q)[0])
            d, rel = repeat_spread.spread(*p)
            out.append(f"**{d:.4f} ({rel:.1%})**, from `{p[0]['arm']}` "
                       f"{p[0]['score']:.4f} against `{p[1]['arm']}` "
                       f"{p[1]['score']:.4f}.")
            out.append("")
            out.append(sentence)
        else:
            out.append("No repeat pair is scored yet.")
        out.append("")

    # A backbone can exist without a score. `s08c` and `s08d` trained to the
    # stop and their heads were dropped on purpose, so the card holds their
    # contrastive AUC and no GM-Relative MASE. A comment that lists only the
    # scored arms hides two trained backbones, and one of them answers a
    # question the card asked.
    if a.sync_root is not None:
        scored = {r["arm"] for r in rows}
        trained = []
        for t in arms_of(a.arms_tsv):
            if t["arm"] in scored:
                continue
            auc = seed_report.auc_at(a.sync_root.expanduser(), t["arm"],
                                     a.stop)
            if auc is not None:
                trained.append((t, auc))
        if trained:
            out.append("## Backbones that trained but carry no score")
            out.append("")
            out.append("| arm | EMA momentum | backbone seed | "
                       f"contrastive AUC at {a.stop // 1000}k | verdict |")
            out.append("|---|---|---|---|---|")
            for t, auc in trained:
                verdict = ("fell to chance"
                           if seed_report.collapsed(auc) else "healthy")
                out.append(f"| {t['arm']} | {momentum(t)} | "
                           f"{t['seed'] or '?'} | {auc:.4f} | {verdict} |")
            out.append("")
            out.append("These arms trained no head and ran no eval, so they "
                       "have no GM-Relative MASE. Their AUC still says "
                       "whether the backbone lived.")
            out.append("")

    # The card's own question: is the distance between the two fixed-momentum
    # arms larger than one repeat of the same cell? Both numbers come from
    # scores.csv, so the answer cannot disagree with the table above.
    out.append("## Does the spread separate 0.90 fixed from 0.95 fixed?")
    out.append("")
    answer = repeat_spread.separation(rows, d, 0.90, 0.95) if d is not None \
        else ""
    out.append(answer or "The card cannot answer this yet: it has fewer than "
                         "two stable seeds of one arm, or one of the two arms "
                         "has no score.")
    out.append("")
    out.append("## The verdict")
    out.append("")
    out.append(f"`{best['arm']}` wins at **{best['score']:.4f}**, EMA momentum "
               f"{momentum(best)}. It sits {best['score'] - k3_40k:+.4f} from "
               f"the k = 3 score at the same 40,000 steps, {k3_40k:.4f}, so it "
               f"does NOT go below that score.")

    # The card's own goal is a LOWER GM-Relative MASE, and the line it has to
    # cross is the k = 0 parent of this cell. A comment that names only the
    # winner leaves the reader to subtract. This block reads the same
    # `references` file the figures draw the line from, so the two agree.
    k0 = references.K0_PARENT_BB40K
    under = [r for r in rows if r["score"] < k0]
    out.append("")
    if under:
        out.append(f"{len(under)} arm(s) go below the k = 0 parent of this "
                   f"cell, {k0:.4f} at the same 40,000 steps:")
        out.append("")
        for r in under:
            out.append(f"- `{r['arm']}`, {r['score']:.4f}, "
                       f"{k0 - r['score']:.4f} under it. It holds "
                       f"{holds_at(r, a.stop):.3f} at the stop.")
    else:
        out.append(f"No arm goes below the k = 0 parent of this cell, "
                   f"{k0:.4f} at the same 40,000 steps. The best arm sits "
                   f"{best['score'] - k0:+.4f} from it.")
    if rep is not None and rep["spread"] is not None:
        near = repeat_spread.unresolved(rows, rep["spread"])
        out.append("")
        if len(near) > 1:
            out.append(f"{len(near)} arms sit within one repeat spread of that "
                       f"score: " + ", ".join(f"`{n}`" for n in near) +
                       ". This card does not rank them.")
        else:
            out.append(f"No other arm sits within one repeat spread "
                       f"({rep['spread']:.4f}) of it.")
    out.append("")
    if a.runs is not None:
        out.append(f"Runs completed this round: {a.runs}.")
    if a.cost:
        out.append(f"Cost: {a.cost}.")

    text = "\n".join(out) + "\n"
    if a.out:
        a.out.write_text(text)
        print(f"wrote {a.out}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
