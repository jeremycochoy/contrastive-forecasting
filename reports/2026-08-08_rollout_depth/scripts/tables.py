#!/usr/bin/env python3
"""#373 — every table the report carries, from the score files and the splits.

Eight tables, in the order the report asks its questions:

  1. coverage        which of the card's 14 cells this study trained.
  2. reproduction    this study's retrained k = 0 against the published one,
                     grouped by the machine it trained on.
  3. depth response  each arm's own k = 0 against its deeper runs, on the
                     full 97 and on the card's horizon criterion.
  4. bootstrap       the paired dataset-cluster interval behind every one of
                     those deltas, per horizon subset.
  5. B5's backbones  one cell trained three times: two seeds, two machines.
  6. A3 controls     the depth ladder beside the re-weighting control.
  7. cost            step time, and which runs had a card to themselves.
  8. depth-0 gap     the depth-0 forecast error of each deeper run against
                     its own k = 0, over four end-of-run windows.
  9. glossary        every term this report uses that is not standard in
                     the field.

Every delta is against the SAME arm's own k = 0. No delta in this file is
computed against a published number or against another backbone.

Two things every delta table carries, because the study cannot separate
them from the depth and a reader must see them where the number is:

  machine   whether the two sides trained on the same box. The reproduction
            table separates on the machine and not on the seed.
  ✗         a retracted row. B5·s1's k = 0 misses its published value by
            0.1169, so its depth delta stands on a baseline the parents do
            not recognise.

Numbers come from `splits.csv`, not from the 4-decimal `score_*.txt` files,
so a Δ here and a Δ in `bootstrap.csv` are the same difference of the same
two full-precision values.

The same text goes two places: `results/scores.md`, and the report itself
between its `<!-- TABLES:BEGIN -->` and `<!-- TABLES:END -->` markers. The
report standard puts the tables in the canonical report; writing them there
by hand would let them drift from the score files, so the rebuild writes
them.

Usage: tables.py --results <results dir> --out <scores.md> [--inject <md>]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs as R                                          # noqa: E402
from published import (PUBLISHED as PUB_ALL, GATE,             # noqa: E402
                       NOISE_BAND, PRINT_QUANT, PUBLISHED_SEED,
                       REEVAL_FLOOR, RESOLUTION, SEED_BAND, verdict)

# What every bootstrap interval in this file is over, said where the
# intervals are. Both B5 contrasts are ONE run pair each.
INTERVAL_SCOPE = (
    "Every interval here is a paired dataset-cluster bootstrap over the 97 "
    "eval configs of ONE run pair. It bounds the eval sample: how far the "
    "difference between these two runs could move if the datasets had been "
    "drawn again. It does not bound run-to-run variance, and neither "
    "contrast has a replicate to bound it with. No two of B5's three "
    "backbones share both a seed and a machine.")

CARD_CELLS = ["A1", "A2", "A3", "A4"] + [f"B{i}" for i in range(1, 11)]


def read_scores(results):
    out = {}
    for p in Path(results).glob("score_*.txt"):
        try:
            out[p.name[len("score_"):-len(".txt")]] = float(p.read_text().strip())
        except ValueError:
            continue
    return out


def read_splits(results):
    """`{tag: {split name: value}}`."""
    out = {}
    path = Path(results) / "splits.csv"
    if not path.is_file():
        return out
    for r in csv.DictReader(open(path)):
        out.setdefault(r["stop"], {})[r["name"]] = float(r["gm_rel_mase"])
    return out


def read_bootstrap(results):
    """`{(label, subset): row}` from bootstrap.csv."""
    path = Path(results) / "bootstrap.csv"
    if not path.is_file():
        return {}
    return {(r["label"], r["subset"]): r for r in csv.DictReader(open(path))}


def read_steptime(results):
    """Rows of steptime_solo.csv, keyed by (arm, k)."""
    path = Path(results) / "steptime_solo.csv"
    if not path.is_file():
        return {}
    return {(r["arm"], int(r["k"])): r for r in csv.DictReader(open(path))}


def boot_label(arm, k, head):
    """The label `find_artefacts.py --what pairs` gave this comparison."""
    return f"{arm.replace(chr(183), '_')}_k{k}_{head}"


def fmt(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"


def pct(a, b):
    return "—" if None in (a, b) else f"{100.0 * (b / a - 1.0):+.1f}%"


def mark(arm):
    """The retraction marker, where the report has withdrawn an arm."""
    return " ✗" if arm in R.RETRACTED else ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inject", help="report to write the tables into, "
                                     "between its TABLES markers")
    args = ap.parse_args(argv)

    sc = read_scores(args.results)
    sp = read_splits(args.results)
    bs = read_bootstrap(args.results)
    st = read_steptime(args.results)
    tags = sorted(sc)
    reg = R.resolve_all(tags)
    L = []

    def val(tag, name="all"):
        """The full-precision score for a tag, or the 4-decimal fallback."""
        v = sp.get(tag, {}).get(name)
        return v if v is not None else (sc.get(tag) if name == "all" else None)

    # ---- 1. coverage -------------------------------------------------------
    trained = sorted({r.cell for r in reg.values()})
    missing = [c for c in CARD_CELLS if c not in trained]
    L += ["### Coverage", "",
          f"The card names 14 cells. This study trained **{len(trained)} of "
          f"them**: {', '.join(sorted(trained))}. It never ran "
          f"**{len(missing)}**: {', '.join(missing)}.", "",
          "| cell | f-bearing term | EMA α | depths trained |",
          "|---|---|---|---|"]
    for cell in CARD_CELLS:
        ks = sorted({r.k for r in reg.values()
                     if r.cell == cell and r.role == "depth"})
        L.append(f"| {cell} | {R.CELL_TERM.get(cell, '—')} | "
                 f"{R.CELL_EMA.get(cell, '—')} | "
                 f"{', '.join(f'k = {k}' for k in ks) if ks else '**never ran**'} |")
    L += ["", "Every trained stop is bb40k. No cell reached bb100k or "
          "bb200k, so the card's extend rule never fired and this study "
          "publishes one stop.", ""]

    # ---- 2. reproduction ---------------------------------------------------
    # The seed band, live from the bootstrap that measured it, so re-running
    # the bootstrap moves the gate with it.
    seed_ci = bs.get(("B5_seed_k0_student", "all"))
    seed_band = (max(abs(float(seed_ci["ci_lo"])), abs(float(seed_ci["ci_hi"])))
                 if seed_ci else SEED_BAND)
    L += ["### Reproduction of the published k = 0", "",
          "Same cell, same recipe, same head seed 20260722, same 97-config "
          "B4 eval, student head. Rows are grouped by machine.", "",
          f"A row at the parents' own backbone seed {PUBLISHED_SEED} takes "
          f"the card's {GATE}; a row at any other seed takes the seed "
          "band.", "",
          "| backbone | seed | machine | published k = 0 | retrained k = 0 | "
          "\\|Δ\\| | gate | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    repro = [r for r in R.reproductions(tags) if r.head == "student"]
    repro.sort(key=lambda r: (0 if r.machine == "elisa" else
                              1 if r.run else 2,
                              abs((val(r.tag) or 0)
                                  - (PUB_ALL.get(r.cell, {})
                                     .get("student", {}).get(40) or 0))))
    cross_seed = []
    for r in repro:
        pub = PUB_ALL.get(r.cell, {}).get("student", {}).get(40)
        got = val(r.tag)
        if pub is None or got is None:
            continue
        same = r.seed == PUBLISHED_SEED
        if not same:
            cross_seed.append(r.arm)
        gate = (f"{GATE}, the card" if same
                else f"{seed_band:.4f}, the seed band")
        L.append(f"| {r.arm}{mark(r.arm)} | {r.seed} | {r.machine} | "
                 f"{pub:.4f} | {got:.4f} | {abs(got - pub):.4f} | {gate} | "
                 f"{verdict(abs(got - pub), same, seed_band)} |")
    L += ["", f"Two things this comparison cannot resolve, added: {REEVAL_FLOOR} "
          "for the head and the eval, which is what `B5·pub` moves the "
          "score by while training nothing, and "
          f"{PRINT_QUANT} for the parents' four printed decimals. A |Δ| at "
          f"or below {RESOLUTION:.4f} is a run this pipeline cannot separate "
          f"from the published one. The card's gate of {GATE} is stricter "
          "than that.", ""]
    if cross_seed:
        L += [f"The seed band is {seed_band:.4f}, the far end of the 95% "
              "interval on this study's one measurement of a seed change: "
              "`B5·s2` against `B5·s3`, one machine, one recipe, +0.0035 "
              "[-0.0183, +0.0230]. It is one run pair, and the interval is "
              "over that pair's eval sample rather than over seeds, so the "
              "band is a floor on what a seed can move and not a bound on "
              f"it. {', '.join(sorted(set(cross_seed)))} is the only row it "
              "gates; every other row here carries the parents' own seed.", ""]
    L += ["`B5·pub` is not a training: it takes the parent report's own "
          "published B5 checkpoint and puts this study's head and eval on "
          "it, so its row bounds the head and the eval rather than the "
          "trainer. `B5·s3` is a training, at the protocol seed, on elisa, "
          "and its 97-config eval output is byte-identical to `B5·pub`'s "
          "(`results/eval/G7_B5_k0_e_bb40k_student/all_results.csv` against "
          "`results/eval/G1_B5pub_bb40k_student/all_results.csv`): the "
          "elisa retrain reproduced the parent's backbone exactly, and the "
          f"{REEVAL_FLOOR} both rows carry is the head and the eval.", ""]

    # ---- 3. depth response -------------------------------------------------
    L += ["### Depth response, against each arm's own k = 0", "",
          "| arm | seed | machine held | head | k | k = 0 | this k | Δ | "
          "all | short | med+long | criterion |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, head, k, base, deep in R.pairs(tags):
        a, b = val(base.tag), val(deep.tag)
        A, B = sp.get(base.tag, {}), sp.get(deep.tag, {})
        ok = "—"
        if A and B:
            dm = 100.0 * (B["medium_long"] / A["medium_long"] - 1.0)
            ds = 100.0 * (B["short"] / A["short"] - 1.0)
            ok = "**MET**" if (dm <= -5.0 and ds < 2.0) else "not met"
        held = R.machine_held(base, deep)
        where = ("yes, " + base.machine if held else
                 f"no, {base.machine} → {deep.machine}")
        L.append(
            f"| {arm}{mark(arm)} | {deep.seed} | {where} | {head} | {k} | "
            f"{fmt(a)} | "
            f"{fmt(b)} | {f'{b - a:+.4f}' if None not in (a, b) else '—'} | "
            f"{pct(A.get('all'), B.get('all'))} | "
            f"{pct(A.get('short'), B.get('short'))} | "
            f"{pct(A.get('medium_long'), B.get('medium_long'))} | {ok} |")
    L += ["", "Criterion, from the card: medium+long (42 configs) at least "
          "5% better, short (55 configs) losing less than 2%.", "",
          "`machine held` = did the two sides train on the same box. A `no` "
          "row carries a machine change as well as a depth change. The B5 "
          "table below measures the machine alone, at one seed, at 0.1166, "
          "so a `no` row carries a term larger than most of the deltas in "
          "this table. Only the `yes` rows report the depth and nothing "
          "else.", "",
          "✗ marks a retracted row: " + R.RETRACTED_WHY + ".", "",
          f"Head-seed band ±{NOISE_BAND} (`ema_sched_ladder.md`, pooled). It "
          "bounds the head seed alone. It does not bound the machine.", ""]

    # ---- 4. the interval behind every one of those deltas ------------------
    L += ["### Paired dataset-cluster bootstrap, per horizon subset", "",
          "The resampling unit is the dataset: `<ds>/short`, `/medium` and "
          "`/long` are three configs of one series and are not independent "
          "draws. 95% percentile interval over 10,000 resamples. Each "
          "interval is over one run pair's 97 configs, so it bounds the "
          "eval sample and not run-to-run variance.", "",
          "| arm | head | k | subset | n | Δ | 95% CI | resamples improved |",
          "|---|---|---|---|---|---|---|---|"]
    for arm, head, k, _base, _deep in R.pairs(tags):
        for subset in ("all", "short", "medium_long"):
            r = bs.get((boot_label(arm, k, head), subset))
            if r is None:
                continue
            L.append(f"| {arm}{mark(arm)} | {head} | {k} | {subset} | "
                     f"{r['n']} | {float(r['delta']):+.4f} | "
                     f"[{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}] | "
                     f"{100 * float(r['p_improved']):.1f}% |")
    L.append("")

    # ---- 5. B5's three backbones -------------------------------------------
    L += ["### One cell, three backbones", "",
          "B5 (`arm4_combab_fix09`) trained three times on one recipe, one "
          "code snapshot, one head seed and one eval. They differ by backbone "
          "seed and by machine, and each contrast below names which of the "
          "two it changes. The machine moves the score and the seed does "
          "not.", "",
          "| backbone | seed | machine | k = 0 | k = 3 | k = 3 − k = 0 |",
          "|---|---|---|---|---|---|"]
    b5_arms = [a for a in R.arms_of("B5") if a != "B5·pub"]
    b5 = {}
    for arm in b5_arms:
        row = []
        for k in (0, 3):
            run = R.find_run(arm, k, "depth") or R.find_run(arm, k, "control")
            v = val(f"{run.stem}_bb40k_student") if run else None
            row.append(v)
            if v is not None:
                b5[(arm, k)] = v
        if row[0] is None and row[1] is None:
            continue
        d = ("—" if None in row else f"{row[1] - row[0]:+.4f}")
        L.append(f"| {arm}{mark(arm)} | {R.arm_seed(arm)} | "
                 f"{R.arm_where(arm)} | {fmt(row[0])} | {fmt(row[1])} | {d} |")

    L += ["", "| contrast | what changes | k | Δ | 95% CI |",
          "|---|---|---|---|---|"]
    for a1, a2, what, lab in (
            ("B5·s1", "B5·s3", "the machine, at one seed", "machine"),
            ("B5·s2", "B5·s3", "the seed, on one machine", "seed"),
            ("B5·s1", "B5·s2", "the seed AND the machine", "seed_and_machine")):
        for k in (0, 3):
            if (a1, k) not in b5 or (a2, k) not in b5:
                continue
            ci = bs.get((f"B5_{lab}_k{k}_student", "all"))
            L.append(f"| {a1} against {a2} | {what} | {k} | "
                     f"{b5[(a2, k)] - b5[(a1, k)]:+.4f} | "
                     + (f"[{float(ci['ci_lo']):+.4f}, "
                        f"{float(ci['ci_hi']):+.4f}] |" if ci else "— |"))
    L += ["", "Student head, 97 configs. `B5·s3` holds `B5·s1`'s seed and "
          "`B5·s2`'s machine, so the first two rows separate what the third "
          "confounds: the machine moves `k = 0` by 0.1166 and the seed by "
          "0.0035.", "",
          INTERVAL_SCOPE, ""]

    early = list(csv.DictReader(open(Path(args.results) / "early_loss.csv"))) \
        if (Path(args.results) / "early_loss.csv").is_file() else []
    if early:
        cols = [a for a in R.ARM_ORDER if any(r["arm"] == a for r in early)]
        steps = sorted({int(r["step"]) for r in early})
        cell_of = {(r["arm"], int(r["step"])): r for r in early}
        L += ["`mixup` counts the examples the mixer touched in the "
              "200-step window, so one count at every step is one data "
              "order. `B5·s1` and `B5·s3` carry one seed, print one count "
              "at every step, and their losses still part.", "",
              "| step | " + " | ".join(
                  f"{a}<br>seed {R.arm_seed(a)}, {R.arm_where(a)}"
                  for a in cols) + " |",
              "|---" * (len(cols) + 1) + "|"]
        for s in steps:
            row = []
            for a in cols:
                r = cell_of.get((a, s))
                row.append(f"{r['loss']}  `{r['mixup']}`" if r else "—")
            L.append(f"| {s} | " + " | ".join(row) + " |")
        L.append("")

    # ---- 7. A3 controls ----------------------------------------------------
    L += ["### A3: is the damage the depth, or the weight?", "",
          "Summing the depths multiplies `L_align`'s weight against the "
          "f-free terms by k + 1. The `L_align x4` row applies that "
          "re-weighting at k = 0, with no depth at all.", "",
          "| head | k = 0 | k = 0, `L_align` x4 | k = 1 | k = 3 |",
          "|---|---|---|---|---|"]
    A3_COLS = ("A3_k0", "G3_A3_k0_aw4", "G3_A3_k1", "A3_k3")
    A3_BOOT = (None, "A3_alignx4_{h}", "A3_k1_{h}", "A3_k3_{h}")
    for head in ("student", "teacher"):
        v = [val(f"{t}_bb40k_{head}") for t in A3_COLS]
        if v[0] is None or v[3] is None:
            continue
        cells = []
        for value, lab in zip(v, A3_BOOT):
            if value is None:
                cells.append("—")
                continue
            r = bs.get((lab.format(h=head), "all")) if lab else None
            cells.append(f"{value:.4f}" + (
                f"<br>{float(r['delta']):+.4f} "
                f"[{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}]"
                if r else ""))
        L.append("| " + head + " | " + " | ".join(cells) + " |")
    machines = " · ".join(
        f"{c}: {r.machine}" for c, r in
        ((c, R.resolve(f"{c}_bb40k_student")) for c in A3_COLS) if r)
    L += ["", "Second line of each cell: the difference against `k = 0` "
          "and its 95% paired dataset-cluster interval.", "",
          "Every column trained on a different box from at least one "
          f"other. {machines}. The machine alone is worth 0.1166 on this "
          "study's one controlled measurement of it, which is more than "
          "either control's own size, so read the two controls as direction "
          "and not as magnitude. This table therefore does not divide one "
          "column by another.", ""]

    # ---- 8. what the depth costs -------------------------------------------
    L += ["### What the depth costs", "",
          "Median `fwd + bwd` per step, from each run's own trainer log. A "
          "median is a cost of the depth only where the run had the card to "
          "itself, so the table says which did. `run_provenance.py` reads "
          "that off the driver logs and "
          "[`results/steptime_solo.csv`](results/steptime_solo.csv) carries "
          "it per run.", "",
          "| arm | f-bearing term | k | machine | card | fwd+bwd | alone? |",
          "|---|---|---|---|---|---|---|"]
    order = {a: i for i, a in enumerate(R.ARM_ORDER)}
    for (arm, k) in sorted(st, key=lambda x: (order.get(x[0], 99), x[1])):
        r = st[(arm, k)]
        run = R.find_run(arm, k)
        ms = (f"{float(r['compute_ms']):.1f} ms" if r["compute_ms"] else
              f"{float(r['compute_ms_contended']):.1f} ms, shared"
              if r["compute_ms_contended"] else "—")
        # No ✗ here. The retraction is of B5·s1's depth DELTA, which rests on
        # a k = 0 the parents do not recognise. Its step time is a wall clock
        # and nothing about the score touches it.
        L.append(f"| {arm} | {run.term if run else '?'} | {k} | "
                 f"{r['machine']} | {r['card']} | {ms} | "
                 f"{'yes' if r['solo'] == 'yes' else 'no — ' + r['why_not_solo']} |")
    L += ["", "The ratios that survive that test:", "",
          "| arm | f-bearing term | k = 0 | k = 3 | change | both sides |",
          "|---|---|---|---|---|---|"]
    for arm in R.ARM_ORDER:
        a, b = st.get((arm, 0)), st.get((arm, 3))
        if not a or not b or not a["compute_ms"] or not b["compute_ms"]:
            continue
        c0, c3 = float(a["compute_ms"]), float(b["compute_ms"])
        run = R.find_run(arm, 3)
        same = ("one box" if a["machine"] == b["machine"] else
                f"{a['machine']} → {b['machine']}"
                + ("" if a["card"] == b["card"] else ", DIFFERENT CARDS"))
        L.append(f"| {arm} | {run.term if run else '?'} | "
                 f"{c0:.1f} ms | {c3:.1f} ms | {c3 / c0 - 1:+.0%} | {same} |")
    L.append("")

    # ---- 8. the depth-0 forecast-error gap ---------------------------------
    gap_path = Path(args.results) / "depth0_gap.csv"
    if gap_path.is_file():
        gap = list(csv.DictReader(open(gap_path)))
        L += ["### The depth-0 forecast error, deeper run minus its own k = 0",
              "",
              "`1 - cos(f_t, h_{t+1})` during training: the same quantity on "
              "both runs, unlike the loss. Negative means the deeper run "
              "forecasts one step ahead better. Four end-of-run windows, "
              "because a gap that changes sign between them is not a "
              "result.", "",
              "| arm | k | last 50% | last 25% | last 10% | final step | "
              "one sign over all four |",
              "|---|---|---|---|---|---|---|"]
        for r in sorted(gap, key=lambda r: (order.get(r["arm"], 99),
                                            int(r["k"]))):
            stable = r["sign_stable_across_windows"].strip().lower() == "yes"
            L.append(f"| {r['arm']}{mark(r['arm'])} | {r['k']} | "
                     f"{r['last_50pct']} | {r['last_25pct']} | "
                     f"{r['last_10pct']} | {r['final_step']} | "
                     f"{'yes' if stable else '**no**'} |")
        L.append("")

    # ---- 9. glossary -------------------------------------------------------
    L += ["### Glossary", "",
          "| term | what it means here |",
          "|---|---|",
          "| the card | the issue this study answers, and the 14 cells, "
          "stops and criteria it names |",
          "| cell | one of those 14 recipes, `A1`..`A4` and `B1`..`B10` |",
          "| arm | a (cell, backbone seed, machine) triple. B5 trained "
          "three, so the cell is not the unit a delta lives in |",
          "| bb40k | backbone step 40,000, the one stop every run here "
          "reached |",
          "| GM-Relative MASE | geometric mean over the 97 GIFT-Eval "
          "configs of each config's MASE divided by the seasonal-naive "
          "MASE. Lower is better; 1.0 is seasonal-naive parity |",
          "| B4 eval strategy | GIFT-Eval's official evaluation strategy, "
          "the one the parent reports use |",
          "| student / teacher head | the quantile head is trained twice "
          "per backbone, once on the student encoder and once on its EMA "
          "copy, the teacher. The two are separate measurements of one "
          "backbone |",
          "| f-bearing term | the loss term that the forecast operator `f` "
          "enters. `--train-rollout-depth K` duplicates it at depth 1..K |",
          "| `rep_only` | the representation loss with no forecast term |",
          "| `L_align` | the term that aligns `f`'s output with the future "
          "latent |",
          "| `L_pred` | the predictive contrastive term, split from the "
          "representation term |",
          "| `xshh_allt` | negatives pooled across the batch and across "
          "channels, taken over every time index |",
          "| `arm4`, `arm6_v2 combab` | the launcher recipes the cells run; "
          "the Coverage table gives each cell's |",
          "| head-seed band ±0.0384 | how far the head seed alone moved a "
          "score in `ema_sched_ladder.md`, pooled. It bounds the head seed "
          "and nothing else |",
          "| `mixup` | the count of examples the batch mixer touched in a "
          "200-step window. Two runs on one data order print one count |",
          ""]

    body = "\n".join(L) + "\n"
    Path(args.out).write_text(body)
    print(f"wrote {args.out} ({len(reg)} run(s), {len(trained)} cell(s))")

    if args.inject:
        md = Path(args.inject)
        text = md.read_text()
        a, b = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"
        if a not in text or b not in text:
            print(f"NOTE: {md} carries no TABLES markers; not injecting")
            return 0
        head, rest = text.split(a, 1)
        _old, tail = rest.split(b, 1)
        md.write_text(f"{head}{a}\n\n{body}\n{b}{tail}")
        print(f"injected the tables into {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
