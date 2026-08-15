#!/usr/bin/env python3
"""#373 review gap 3 — write the B1 control's prose from the B1 control's numbers.

B1 carries ``L_align`` as its only f-bearing term. ``align_loss`` adds one
complete copy of that term per rollout depth at the SAME weight
(``src/loss.py``), and B1's main term is ``rep_only``, which holds no ``f``
and is added once at any depth. So B1's ``k = 3`` run multiplies L_align's
weight against the f-free terms by 4 as well as adding depth, and its
-0.1175 cannot say which of the two paid.

The ``L_align x4`` run applies the re-weighting at ``k = 0``. Three columns
then split the total move in two:

    re-weighting   x4 - k0
    depth          k3 - x4
    total          k3 - k0

THE RULE BELOW IS PRE-REGISTERED. It was committed while the control's
backbone was still training and before either score existed, so the verdict
is a reading of the numbers and not a choice made after seeing them.

    re-weighting   share >= 0.60 AND the x4 - k0 interval excludes 0
    depth          share <= 0.25 OR (x4 - k0 covers 0 while k3 - k0 does not)
    split          anything else

``share = (x4 - k0) / (k3 - k0)``. A control that lands the WRONG way (x4
worse than k0, against a k3 that is better) gives a negative share, falls in
the depth branch, and is reported as such. The two heads are judged apart
and must agree; heads that disagree give `split`.

One thing the control cannot separate. At ``k = 3`` the four copies sit at
four horizons (t+1 .. t+4); at ``k = 0`` x4 the four sit on t+1 alone. So
`depth` here means "the extra horizons", not "depth net of every other
difference".

Usage: gap3_item3.py --results <dir> [--check]
  --check  exit 1 if any input is missing, write nothing
"""
import argparse
import csv
import pathlib
import sys

HEADS = ("student", "teacher")
COLS = {"k0": "G6_B1_k0_bb40k_{h}",
        "x4": "G_B1_k0_aw4_bb40k_{h}",
        "k3": "G6_B1_k3_bb40k_{h}"}
# Pre-registered thresholds. Do not edit after the scores land.
SHARE_REWEIGHT = 0.60
SHARE_DEPTH = 0.25


def read_scores(res):
    out = {}
    for head in HEADS:
        for key, pat in COLS.items():
            p = res / f"score_{pat.format(h=head)}.txt"
            if not p.exists() or not p.read_text().strip():
                return None, f"missing {p.name}"
            out[key, head] = float(p.read_text().strip())
    return out, None


def read_boot(res):
    """label -> (delta, lo, hi) over the 'all' subset."""
    out = {}
    p = res / "bootstrap.csv"
    if not p.exists():
        return out
    for r in csv.DictReader(p.open()):
        if r["subset"] == "all":
            out[r["label"]] = (float(r["delta"]), float(r["ci_lo"]),
                               float(r["ci_hi"]))
    return out


def missing_boot(bs):
    """The interval labels the rule reads. A missing one is not 'covers 0'.

    ``verdict_for`` treats an absent interval as one that covers zero, and
    the depth branch fires on exactly that. So a bootstrap that never ran
    would hand back `depth` and read like a measurement. Demand the labels
    instead, the same way a missing score file stops the run.
    """
    want = [f"B1_alignx4_{h}" for h in HEADS]
    want += [f"B1_alignx4_vs_k3_{h}" for h in HEADS]
    want += [f"B1_k3_{h}" for h in HEADS]
    return [w for w in want if w not in bs]


def verdict_for(share, ci_x4, ci_k3):
    excl = lambda ci: ci is not None and (ci[1] > 0 or ci[2] < 0)
    if share >= SHARE_REWEIGHT and excl(ci_x4):
        return "reweighting"
    if share <= SHARE_DEPTH or (not excl(ci_x4) and excl(ci_k3)):
        return "depth"
    return "split"


def ci_str(ci):
    return f"[{ci[1]:+.4f}, {ci[2]:+.4f}]" if ci else "no interval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=pathlib.Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    res = a.results

    sc, err = read_scores(res)
    if err:
        print(f"gap3_item3: {err} — nothing written", file=sys.stderr)
        return 1
    bs = read_boot(res)
    gone = missing_boot(bs)
    if gone:
        print(f"gap3_item3: bootstrap.csv has no {', '.join(gone)} — "
              f"run make_report_assets.sh first. Nothing written.",
              file=sys.stderr)
        return 3
    if a.check:
        print("gap3_item3: every input in hand")
        return 0

    rows, verdicts = [], {}
    for head in HEADS:
        k0, x4, k3 = (sc["k0", head], sc["x4", head], sc["k3", head])
        rw, dp, tot = x4 - k0, k3 - x4, k3 - k0
        share = rw / tot if tot else float("nan")
        v = verdict_for(share, bs.get(f"B1_alignx4_{head}"),
                        bs.get(f"B1_k3_{head}"))
        verdicts[head] = v
        rows.append(dict(head=head, k0=k0, x4=x4, k3=k3, rw=rw, dp=dp,
                         tot=tot, share=share, v=v))

    agree = len(set(verdicts.values())) == 1
    overall = rows[0]["v"] if agree else "split"

    L = []
    L.append("| head | k = 0 | k = 0, `L_align` x4 | k = 3 | the re-weighting"
             "<br>k = 0 → x4 | the depth<br>x4 → k = 3 | share |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['head']} | {r['k0']:.4f} | {r['x4']:.4f} | "
                 f"{r['k3']:.4f} | {r['rw']:+.4f} | {r['dp']:+.4f} | "
                 f"{100 * r['share']:.0f}% |")
    L.append("")
    L.append("Intervals, 95% paired dataset-cluster over the 97 eval configs:")
    L.append("")
    for head in HEADS:
        L.append(f"- {head}: re-weighting "
                 f"{ci_str(bs.get(f'B1_alignx4_{head}'))}, depth "
                 f"{ci_str(bs.get(f'B1_alignx4_vs_k3_{head}'))}"
                 f", total {ci_str(bs.get(f'B1_k3_{head}'))}")
    L.append("")

    st = rows[0]
    if overall == "reweighting":
        L.append(f"**The win is the re-weighting.** Raising `L_align`'s weight "
                 f"to 4 at depth 0 reproduces {100 * st['share']:.0f}% of the "
                 f"student's {st['tot']:+.4f}. The extra horizons carry "
                 f"{st['dp']:+.4f} of it.")
    elif overall == "depth":
        L.append(f"**The win is the depth.** The re-weighting alone moves the "
                 f"student by {st['rw']:+.4f} of the {st['tot']:+.4f} total, "
                 f"{100 * st['share']:.0f}% of it. The extra horizons carry "
                 f"the rest, {st['dp']:+.4f}.")
    else:
        L.append(f"**Both pay.** The re-weighting carries "
                 f"{100 * st['share']:.0f}% of the student's {st['tot']:+.4f} "
                 f"and the extra horizons carry the rest. Neither alone "
                 f"accounts for the win.")
    if not agree:
        L.append("")
        L.append(f"The two heads do not agree — student `{verdicts['student']}`"
                 f", teacher `{verdicts['teacher']}`. Read this as split.")
    L.append("")
    L.append("Every column trained on elisa at backbone seed 20260520 on the "
             "same head budget: 15,000 head steps at seed 20260722, then 97 "
             "GIFT-Eval configs. This is the study's one machine-held, "
             "seed-held, head-budget-matched set, so it may divide one column "
             "by another. The two cards are both RTX 4090s of the one box.")
    L.append("")
    L.append("What it cannot separate: `k = 3` puts its four copies of "
             "`L_align` on four horizons and `k = 0` x4 puts all four on "
             "t+1. So the depth column is the extra HORIZONS at a held total "
             "weight, not depth net of everything else.")
    (res / "gap_close_item3.md").write_text("\n".join(L) + "\n")

    can = (f"On B1, the one cell where the depth wins, the re-weighting that "
           f"comes with it carries {100 * st['share']:.0f}% of the student's "
           f"{st['tot']:+.4f}. Holding `L_align`'s total weight at 4 and "
           f"dropping the depth to 0 reads {st['x4']:.4f} against "
           f"{st['k0']:.4f}.")
    if overall == "reweighting":
        cannot = ("That the depth is what wins on B1. The re-weighting that "
                  "rides along with `k = 3` reproduces most of the move on "
                  "its own, at depth 0.")
    elif overall == "depth":
        cannot = ("That the re-weighting explains B1's win. It does not: it "
                  f"carries {100 * st['share']:.0f}% of the move. But the "
                  "control holds the WEIGHT and not the horizons, so what is "
                  "left is the extra horizons and not depth in general.")
    else:
        cannot = ("That either the depth or the re-weighting alone wins on "
                  "B1. The control splits the move between them and one "
                  "cell cannot say which generalises.")

    tmpl = (res / "gap_close_verdict.tmpl.md").read_text()
    out = tmpl.replace("@@ITEM3_CAN@@", can).replace("@@ITEM3_CANNOT@@", cannot)
    if "@@" in out:
        print("gap3_item3: an unfilled marker remains", file=sys.stderr)
        return 2
    (res / "gap_close_verdict.md").write_text(out)

    print(f"gap3_item3: verdict={overall} "
          f"share student={100 * rows[0]['share']:.0f}% "
          f"teacher={100 * rows[1]['share']:.0f}%")
    print(f"  wrote {res / 'gap_close_item3.md'}")
    print(f"  wrote {res / 'gap_close_verdict.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
