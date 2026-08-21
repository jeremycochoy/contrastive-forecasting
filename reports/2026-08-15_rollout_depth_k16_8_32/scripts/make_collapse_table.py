#!/usr/bin/env python3
"""#401 — one table: GM-Relative MASE beside the latent measurements.

Joins, on (k, step):

    results/diag/collapse_all.csv    rank and cosine ACROSS SERIES
    results/diag/time_rank.csv       rank and cosine ALONG TIME
    results/diag/scalar_readout.csv  what the top direction still carries
    results/diag/curve_state.csv     the trainer's AUC at the same step
    results/stops.log                the GM-Relative MASE of the cell

`stops.log` is the authority for the score. Its `DONE` lines name the tag
and the number the eval printed, so a cell can not pick up a score from a
run it did not have.

Every join carries the REDUCTION. The two arms share their depths and their
stops, and their only difference is how a k-depth run combines its k + 1
depth copies. Keyed on (depth, step) alone, one arm's rank would land beside
the other arm's score. Each arm also keeps its own `stops.log`, under its own
results directory, for the same reason.

Writes a Markdown table and a CSV, and prints the Spearman rank correlation
between the score and each latent measurement over the scored cells.

Usage:
    python3 make_collapse_table.py
"""
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RES = STUDY / "results"
DIAG = RES / "diag"

# The PHASE-1 head budget, pinned. This table has one row per backbone
# checkpoint, so its score column must mean one thing across every row: the
# cell with the study's fixed 30,000-step head. With `h\d+k` the phase-2 cell
# at the same stop matched too, and the later DONE line replaced the earlier
# one — so `k = 8` at bb40k read 1.2543, its 40,000-step head, in a column
# every other row filled from a 30,000-step head.
#
# A variant tag (`k32_ema30k_bb40k_h30k_student`) carries its schedule between
# the depth and the stop, so `_bb` right after the digits keeps it out. It is
# a second schedule at a stop the base cell already holds, and this table has
# one row for that checkpoint.
HEAD_STEPS_PHASE1 = 30_000
DONE = re.compile(
    r"\[k(\d+)_bb(\d+)k_h%dk_\w+\] DONE — GM-Relative MASE ([0-9.]+)"
    % (HEAD_STEPS_PHASE1 // 1000))


# Each arm's results directory, by the reduction it holds. `sum` is the
# stopped comparison arm and it writes at the study root. A k = 0 parent is
# neither arm's, so it has no score here and takes no directory.
ARM_RES = {"sum": RES, "mean": RES / "mean"}


def scores():
    """{(reduce, k, stop_k): GM-Relative MASE} from the eval's own DONE lines.

    A variant cell (`k32_ema30k_bb40k_...`) is a second training schedule at a
    stop the base cell already holds. Its DONE line does not match the
    pattern, so it never lands on the base cell's key.
    """
    out = {}
    for red, res in ARM_RES.items():
        p = res / "stops.log"
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            m = DONE.search(line)
            if m:
                out[(red, int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    return out


def read(path, key_cols=("k", "step_k")):
    rows = {}
    if not path.is_file():
        return rows
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (r.get("reduce") or "sum",) \
                + tuple(int(r[c]) for c in key_cols) + (r["label"],)
            rows[key] = r
    return rows


def curve():
    """{(reduce, k, step): auc} from the trainer curves."""
    out = {}
    p = DIAG / "curve_state.csv"
    if not p.is_file():
        return out
    with p.open() as f:
        for r in csv.DictReader(f):
            out[(r.get("reduce") or "sum", int(r["k"]),
                 int(r["step"]) // 1000)] = float(r["auc"])
    return out


def spearman(xs, ys):
    """Rank correlation, with the mean rank for ties."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    sc = scores()
    coll = read(DIAG / "collapse_all.csv")
    time = read(DIAG / "time_rank.csv")
    rdo = read(DIAG / "scalar_readout.csv")
    auc = curve()

    by_key = {}
    for key, r in coll.items():
        red, k, step, label = key
        by_key[key] = dict(
            k=k, reduce=red, step_k=step, label=label,
            eff_rank=float(r["eff_rank"]), pair_cos=float(r["pair_cos"]),
            dim_std=float(r["dim_std"]), cos_err_d0=float(r["cos_err_d0"]),
        )
        t = time.get(key)
        if t:
            by_key[key].update(
                time_eff_rank=float(t["time_eff_rank"]),
                time_pair_cos=float(t["time_pair_cos"]))
        d = rdo.get(key)
        if d:
            by_key[key].update(readout_r=float(d["readout_r"]),
                               top_dir_share=float(d["top_dir_share"]))
        # #379's B5pub is another study's run. This study holds no curve
        # for it, so it gets no AUC rather than the parent's.
        by_key[key]["auc"] = None if "379" in label else \
            auc.get((red, k, step))
        # only a leg's last step is a stop the study scored
        by_key[key]["score"] = sc.get((red, k, step)) \
            if step in (40, 100, 200) else None

    rows = sorted(by_key.values(),
                  key=lambda r: (r["reduce"], r["k"], r["step_k"]))

    cols = ["reduce", "k", "step_k", "label", "score", "eff_rank", "pair_cos",
            "time_eff_rank", "time_pair_cos", "dim_std", "readout_r",
            "top_dir_share", "cos_err_d0", "auc"]
    with (DIAG / "collapse_vs_score.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in cols})

    def fmt(v, n=4):
        return "-" if v is None else f"{v:.{n}f}"

    md = []
    md.append("| reduction | k | backbone step | GM-Relative MASE | eff. rank "
              "(series) | mean cos (series) | eff. rank (time) | "
              "mean cos (time) | readout r | train AUC |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        star = " *" if r["score"] is not None else ""
        md.append(
            f"| {r['reduce']} | {r['k']} | {r['step_k']}k{star} | "
            f"{fmt(r['score'])} | "
            f"{fmt(r['eff_rank'], 3)} | {fmt(r['pair_cos'], 5)} | "
            f"{fmt(r.get('time_eff_rank'), 3)} | "
            f"{fmt(r.get('time_pair_cos'), 5)} | "
            f"{fmt(r.get('readout_r'), 3)} | {fmt(r['auc'], 4)} |")
    md.append("")
    md.append("`*` marks a stop the study scored. A row without a star is a "
              "periodic checkpoint, measured but not scored.")
    (DIAG / "collapse_vs_score.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # The rank correlation is reported PER ARM. The two arms sit in two
    # separate clusters of rank — every summed checkpoint near 1, every mean
    # one between 4 and 8 — so one correlation over both would measure the
    # gap between the arms and report it as a relation inside each.
    for red in ("sum", "mean"):
        scored = [r for r in rows
                  if r["score"] is not None and r["k"] > 0
                  and r["reduce"] == red]
        if not scored:
            continue
        print(f"\nScored #401 cells, {red} arm: {len(scored)}")
        for name in ("eff_rank", "time_eff_rank", "dim_std", "readout_r",
                     "top_dir_share"):
            have = [r for r in scored if r.get(name) is not None]
            rho = spearman([r[name] for r in have], [r["score"] for r in have])
            print(f"  Spearman(score, {name:<14}) = {rho:+.3f}   "
                  f"n={len(have)}")

    print(f"\n-> {DIAG / 'collapse_vs_score.csv'}")
    print(f"-> {DIAG / 'collapse_vs_score.md'}")


if __name__ == "__main__":
    main()
