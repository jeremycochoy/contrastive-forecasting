"""Shared cell/metric loading for the #379 plots.

One place that knows how to find a cell's evaluation, so the progression
figure and the per-domain radars cannot disagree about which value a cell has.
Only full-97 aggregates are returned: a partial config count yields a value
that is not comparable to the others.

The cell name comes from `scripts/eval_cell_identity.py`, the grammar
`eval_arm.sh` writes the names with. Spelled here instead, the lookup sees
only the untagged form, so a cell measured on a resumed backbone
(`<arm>_bb100k_r2_hd30000s` — the wave-2 and wave-3 backbones of this
experiment are all resumes) reads as missing and drops out of every figure
at once.
"""
from pathlib import Path
import csv
import math
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import eval_cell_identity as cid  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
REP = RESULTS / "eval_gm_mase"
EXP = HERE.parent.parent.parent / "experiments" / "2026-08-01_lalign_teacher" / "eval_gm_mase"
SN_REF = RESULTS / "seasonal_naive_all_results.csv"

ARMS = ["arm1", "arm3", "arm4", "arm5", "arm6_v2", "bimoco"]
VARIANTS = ["", "_tr1", "_nse", "_ncpc", "_combab"]
# (backbone step k, head steps) in increasing order of backbone step.
HORIZONS = [(40, 15000), (100, 30000), (200, 30000)]

ARM_COLOR = {"arm1": "#2a78d6", "arm3": "#eb6834", "arm4": "#008300",
             "arm5": "#8b1e8b", "arm6_v2": "#b8860b", "bimoco": "#00a3a3"}
ARM_LOSS = {
    "arm1":    "L_pred + L_rep",
    "arm3":    "L_pred_moco + L_rep",
    "arm4":    "pooled + MoCo",
    "arm5":    "L_align + L_rep",
    "arm6_v2": "L_align + L_rep_moco",
    "bimoco":  "L_pred_moco + L_rep_moco",
}
VAR_SHORT = {"": "base", "_tr1": "tr1", "_nse": "nse",
             "_ncpc": "ncpc", "_combab": "combab"}
VAR_STYLE = {
    "":        {"ls": "-",  "marker": "o", "lw": 2.0, "ms": 7},
    "_tr1":    {"ls": "--", "marker": "s", "lw": 1.5, "ms": 6},
    "_nse":    {"ls": ":",  "marker": "^", "lw": 1.5, "ms": 6},
    "_ncpc":   {"ls": "-.", "marker": "D", "lw": 1.5, "ms": 6},
    "_combab": {"ls": "-",  "marker": "P", "lw": 2.0, "ms": 8},
}

def variant_knobs(arm, var):
    if var == "":        return "tau=0.10, cpc=1, sigreg_e=1"
    if var == "_tr1":    return "tau=1.0"
    if var == "_nse":    return "sigreg_e=0"
    if var == "_ncpc":   return "cpc=0"
    parts = ["tau=1.0", "cpc=0"]
    if arm in ("arm1", "arm3", "arm4"): parts.append("sigreg_e=0")
    return " + ".join(parts)

def label(arm, var):
    return f"{arm} {VAR_SHORT[var]}"

def one_cell(root, slug, bb, hd, suffix=""):
    """The one `<root>/<cell><suffix>` of this coordinate, or None.

    Two measured replicates is the choice `resolve_eval_checkpoint.sh`
    refuses to make; a figure must not make it silently by taking whichever
    it listed first.
    """
    hits = cid.cell_paths(root, slug, bb, hd, suffix=suffix)
    if len(hits) > 1:
        raise SystemExit(
            f"{slug} at bb{bb}k/hd{hd}s: {len(hits)} replicate cells in "
            f"{root} ({', '.join(p.name for p in hits)}) — name the one the "
            "figures should cite.")
    return hits[0] if hits else None


def read_aggregate(slug, bb, hd):
    """Aggregate GM-Relative MASE for one cell, or None unless it covers 97 configs."""
    flat = one_cell(REP, slug, bb, hd, suffix="_summary.txt")
    nested = one_cell(EXP, slug, bb, hd)
    for p in (flat, nested / "summary.txt" if nested else None):
        if p is None or not p.exists(): continue
        m = re.search(r"Aggregate.*\((\d+) configs\):\s+([0-9.]+)", p.read_text())
        if m and m.group(1) == "97": return float(m.group(2))
    return None

def all_cells():
    """[(arm, var, [agg@40k, agg@100k, agg@200k])] for every cell with any value."""
    out = []
    for arm in ARMS:
        for var in VARIANTS:
            vals = [read_aggregate(f"{arm}{var}", bb, hd) for bb, hd in HORIZONS]
            if any(v is not None for v in vals):
                out.append((arm, var, vals))
    return out

def last_horizon(vals):
    """(index, value) of the highest backbone step this cell was evaluated at."""
    for i in range(len(vals) - 1, -1, -1):
        if vals[i] is not None: return i, vals[i]
    return None, None

def _load_mase(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try: out[row["dataset"]] = float(row["eval_metrics/MASE[0.5]"])
            except (KeyError, ValueError, TypeError): continue
    return out

def _load_domains(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row.get("domain")
            if d: out[row["dataset"]] = d
    return out

def per_domain_relative_mase(slug, bb, hd):
    """{domain: GM of MASE(model)/MASE(seasonal-naive) over that domain's configs}.

    Same geometric mean the headline aggregate uses, restricted to one domain.
    Returns (mapping, n_configs_per_domain) or (None, None) if the cell has no
    complete per-config file.
    """
    flat = one_cell(REP, slug, bb, hd)
    nested = one_cell(EXP, slug, bb, hd)
    for base in (flat, nested / "gift" if nested else None):
        if base is None: continue
        csv_path = base / "all_results.csv"
        if csv_path.exists(): break
    else:
        return None, None
    ours = _load_mase(csv_path)
    doms = _load_domains(csv_path)
    sn = _load_mase(SN_REF)
    logs, counts = {}, {}
    for ds, mase in ours.items():
        ref, dom = sn.get(ds), doms.get(ds)
        if not ref or ref <= 0 or not dom or mase <= 0: continue
        logs.setdefault(dom, []).append(math.log(mase / ref))
        counts[dom] = counts.get(dom, 0) + 1
    if not logs: return None, None
    return ({d: math.exp(sum(v) / len(v)) for d, v in logs.items()}, counts)
