#!/usr/bin/env python3
"""#373 — the study's runs, one registry every table and figure reads.

A run's evidence is a directory `results/eval/<tag>/`. The tag is built by
whichever launcher produced the run, so it carries the launcher's history
rather than the run's meaning: `B5_k0_bb40k_student` and
`G5_B5_s2_k0_bb40k_student` are the same cell at the same depth on two
backbone seeds, and nothing in the two names says so.

This file is where a tag becomes a measurement. Every consumer resolves
tags through `resolve()` and none of them parses a tag itself.

Vocabulary:

  cell      one of the card's 14, `A1`..`A4` and `B1`..`B10`.
  arm       a (cell, backbone seed, machine) triple. The card assumed one
            training per cell; B5 has three, they disagree, and they differ
            on the seed and on the machine. So the arm is the unit a depth
            delta is computed within.
  k         `--train-rollout-depth`.
  role      `depth` for a run that is one point of an arm's depth ladder,
            `control` for a run that answers a specific question and is not
            on any ladder.
  machine   the box the backbone trained on, and its card. The reproduction
            table separates on this and not on the seed, so it is a field of
            the run rather than a footnote.

Usage:  python3 runs.py      # prints the registry
"""
from __future__ import annotations

import re
import sys

# ---------------------------------------------------------------------------
# The registry. One row per backbone; the two heads share it.
#
# `run` is the name the launcher gave the training run. It is what ties a
# score back to a trainer log and a losses CSV, and it is not derivable from
# the tag: the launchers build it from their own `NAME=` blocks. An empty
# `run` means the run has no trainer log of its own — G1 trains no backbone,
# it re-heads one #379 already published.
#
# tag stem                cell  arm       k  seed      role     note
_ROWS = [
    ("G6_B1_k0",          "B1", "B1",     0, 20260520, "depth",   "",
     "bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k"
     "_sigreg_ema_qk_aon_cpc_tau090_cf373k0"),
    ("G6_B1_k3",          "B1", "B1",     3, 20260520, "depth",   "",
     "bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k"
     "_sigreg_ema_qk_aon_cpc_tau090_cf373k3"),
    ("G2_B9_k0",          "B9", "B9",     0, 20260520, "depth",   "",
     "bb_small_arm1_nse_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k0"),
    ("B9_k3",             "B9", "B9",     3, 20260520, "depth",   "",
     "bb_small_arm1_nse_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3"),
    ("B5_k0",             "B5", "B5·s1",  0, 20260520, "depth",   "",
     "bb_small_arm4_combab_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k0"),
    ("B5_k3",             "B5", "B5·s1",  3, 20260520, "depth",   "",
     "bb_small_arm4_combab_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3"),
    ("G5_B5_s2_k0",       "B5", "B5·s2",  0, 20260521, "depth",   "",
     "bb_small_arm4_combab_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k0_s2"),
    ("G5_B5_s2_k3",       "B5", "B5·s2",  3, 20260521, "depth",   "",
     "bb_small_arm4_combab_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k3_s2"),
    ("A3_k0",             "A3", "A3",     0, 20260520, "depth",   "",
     "cf393_arm6_v2_combab_alignT_cf373k0"),
    ("G3_A3_k1",          "A3", "A3",     1, 20260520, "depth",   "",
     "cf393_arm6_v2_combab_alignT_cf373k1"),
    ("A3_k3",             "A3", "A3",     3, 20260520, "depth",   "",
     "cf393_arm6_v2_combab_alignT_cf373k3"),
    ("G3_A3_k0_aw4",      "A3", "A3",     0, 20260520, "control",
     "L_align x4, no depth", "cf393_arm6_v2_combab_alignT_cf373k0_aw4"),
    # The same control on the cell where k = 3 WINS. A3's control sits on the
    # cell where k = 3 does the most damage, and every column of that table
    # crosses a machine. B1 is the study's one machine-held, seed-held,
    # head-budget-matched pair, so this row splits a delta the study measured
    # rather than one it screened.
    ("G_B1_k0_aw4",       "B1", "B1",     0, 20260520, "control",
     "L_align x4, no depth",
     "bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k"
     "_sigreg_ema_qk_aon_cpc_tau090_cf373k0_aw4"),
    # Its own arm, not B5·s1's: this control swaps the BACKBONE for the one
    # #379 published, so folding it into B5·s1 would put a box this study
    # never rented into B5·s1's machine list.
    ("G1_B5pub",          "B5", "B5·pub", 0, 20260520, "control",
     "the parent report's published backbone, this study's head and eval", ""),
    ("G7_B5_k0_e",        "B5", "B5·s3",  0, 20260520, "control",
     "the protocol seed on elisa",
     "bb_small_arm4_combab_xshh_allt_moco_enc3l3_b64_200k"
     "_sigreg_ema_qk_aon_cpc_tau090_cf373k0_e"),
]

# Where each backbone trained, and on what card.
#
# The study's reproduction table separates PERFECTLY on this column and not
# on the seed: every rented-box `k = 0` missed its published value and every
# elisa one hit it, while three runs at seed 20260520 land 0.1169 apart. A
# comparison that crosses a machine therefore carries a term this study
# cannot bound, and the figures have to be able to say which ones do.
#
# Sources, all committed: `results/gap_worker0.log` and `gap_worker1.log`
# for the elisa runs (`BB START <id> ... gpu=0`), `sync/<box>/queue.log` for
# the rented ones, `results/box_gpu.tsv` for each box's card.
_MACHINE = {
    "G6_B1_k0":      ("elisa", "RTX 4090"),
    "G6_B1_k3":      ("elisa", "RTX 4090"),
    "G2_B9_k0":      ("elisa", "RTX 4090"),
    "B9_k3":         ("vast box c", "RTX 4090"),
    "B5_k0":         ("vast box d", "RTX 5090"),
    "B5_k3":         ("vast box a", "RTX 5090"),
    "G5_B5_s2_k0":   ("elisa", "RTX 4090"),
    "G5_B5_s2_k3":   ("elisa", "RTX 4090"),
    "A3_k0":         ("vast box d", "RTX 5090"),
    "G3_A3_k1":      ("elisa", "RTX 4090"),
    "A3_k3":         ("vast box b", "RTX 5090"),
    "G3_A3_k0_aw4":  ("elisa", "RTX 4090"),
    "G_B1_k0_aw4":   ("elisa", "RTX 4090"),
    "G1_B5pub":      ("the parent report's box", "—"),
    "G7_B5_k0_e":    ("elisa", "RTX 4090"),
}

# The f-bearing term each cell trains, and its EMA regime. Both are read off
# the launcher the cell runs (`cells.tsv` names it), not off the card's prose.
CELL_TERM = {
    "A3": "rep_only + L_align",
    "B1": "rep_only + L_align",
    "B5": "pooled xshh_allt",
    "B9": "split L_pred",
}
CELL_EMA = {
    "A3": "scheduled 0.9 -> 1.0",
    "B1": "fixed 0.9",
    "B5": "fixed 0.9",
    "B9": "fixed 0.9",
}

# Draw order: the two arms that improve, then the three B5 backbones that
# disagree, then the arm that degrades. Figures read it so a reader meets
# the same sequence everywhere.
ARM_ORDER = ["B9", "B1", "B5·s1", "B5·s2", "B5·s3", "B5·pub", "A3"]

# The retracted rows. Section 4 drops B5·s1's depth verdict, so every table
# and figure that still draws the arm has to say so where the number is,
# not four sections later.
RETRACTED = {"B5·s1"}
RETRACTED_WHY = ("B5·s1's `k = 0` trained on a rented box and misses its "
                 "published value by 0.1169; `B5·s3` retrains it at the same "
                 "seed on elisa and lands 0.0003 away, so the baseline the "
                 "-5.1% rests on is a rented-box artefact and the delta is "
                 "retracted")

_TAG_RE = re.compile(r"^(?P<stem>.+)_bb(?P<stop>\d+)k_(?P<head>student|teacher)$")
_BY_STEM = {r[0]: r for r in _ROWS}


class Run:
    """One (backbone, head) measurement."""

    __slots__ = ("tag", "stem", "cell", "arm", "k", "seed", "role", "note",
                 "run", "stop", "head")

    def __init__(self, tag, stem, row, stop, head):
        self.tag, self.stem = tag, stem
        (_, self.cell, self.arm, self.k, self.seed, self.role, self.note,
         self.run) = row
        self.stop, self.head = stop, head

    @property
    def term(self):
        return CELL_TERM.get(self.cell, "?")

    @property
    def ema(self):
        return CELL_EMA.get(self.cell, "?")

    @property
    def machine(self):
        return _MACHINE.get(self.stem, ("?", "?"))[0]

    @property
    def card(self):
        return _MACHINE.get(self.stem, ("?", "?"))[1]

    @property
    def retracted(self):
        return self.arm in RETRACTED

    def label(self):
        base = f"{self.arm} k = {self.k}"
        if self.retracted:
            base += " (retracted)"
        return f"{base} ({self.note})" if self.note else base

    def __repr__(self):
        return f"<Run {self.tag} arm={self.arm} k={self.k} head={self.head}>"


def resolve(tag):
    """A `Run` for an eval tag, or None if the tag names no known run."""
    m = _TAG_RE.match(tag)
    if not m:
        return None
    row = _BY_STEM.get(m.group("stem"))
    if row is None:
        return None
    return Run(tag, m.group("stem"), row, int(m.group("stop")), m.group("head"))


def resolve_all(tags):
    """`{tag: Run}` over the tags this registry knows. Unknown tags are
    dropped, so a stray directory cannot enter a figure unnamed."""
    out = {}
    for t in tags:
        r = resolve(t)
        if r is not None:
            out[t] = r
    return out


def index(tags):
    """`{(arm, k, head, role): Run}` — the key every consumer joins on."""
    return {(r.arm, r.k, r.head, r.role): r for r in resolve_all(tags).values()}


def ladders(tags):
    """`{arm: {head: {k: Run}}}` over the depth runs only, in ARM_ORDER."""
    out = {}
    for r in resolve_all(tags).values():
        if r.role != "depth":
            continue
        out.setdefault(r.arm, {}).setdefault(r.head, {})[r.k] = r
    return {a: out[a] for a in ARM_ORDER if a in out}


def pairs(tags, base_k=0):
    """Every (baseline, deeper) pair a depth delta can be computed from.

    Yields `(arm, head, k, base Run, deep Run)`. The baseline is always the
    SAME arm's own `k = base_k`, never a published number and never another
    seed: those two comparisons are what the study got wrong the first time.
    """
    for arm, heads in ladders(tags).items():
        for head, byk in sorted(heads.items()):
            base = byk.get(base_k)
            if base is None:
                continue
            for k in sorted(byk):
                if k != base_k:
                    yield arm, head, k, base, byk[k]


# The runs that re-run a published cell's own recipe at k = 0, and so
# belong in the reproduction check. A control that CHANGES the recipe —
# `G3_A3_k0_aw4` multiplies L_align by 4 — must never drift into a table
# about reproducing it, so the depth ladder's k = 0 runs are joined by an
# explicit list rather than by "every k = 0 run".
REPRO_CONTROLS = {"G1_B5pub", "G7_B5_k0_e"}


def reproductions(tags):
    """Every run whose score answers "does this code reproduce the parent?"."""
    out = [r for r in resolve_all(tags).values()
           if r.k == 0 and (r.role == "depth" or r.stem in REPRO_CONTROLS)]
    out.sort(key=lambda r: (ARM_ORDER.index(r.arm) if r.arm in ARM_ORDER
                            else 99, r.stem))
    return out


def retrainings(k=0):
    """`[(arm, k, role, run name)]` for every backbone that retrains a cell
    this study already trained once at this depth.

    B5 is the only such cell today: three backbones, two seeds, two
    machines. A control belongs here when it re-runs the cell's own recipe;
    one that changes the objective does not.
    """
    out = [(arm, kk, role, run)
           for stem, cell, arm, kk, _seed, role, _note, run in _ROWS
           if kk == k and run and len(arms_of(cell)) > 1
           and (role == "depth" or stem in REPRO_CONTROLS)]
    return sorted(out, key=lambda r: ARM_ORDER.index(r[0]))


def arm_seed(arm):
    """The backbone seed an arm trained at, or None."""
    for row in _ROWS:
        if row[2] == arm:
            return row[4]
    return None


def arm_machines(arm):
    """Every machine an arm's runs trained on, in registry order.

    More than one means the arm's own depth ladder crosses a machine.
    """
    out = []
    for row in _ROWS:
        if row[2] != arm:
            continue
        host = _MACHINE.get(row[0], ("?",))[0]
        if host not in out:
            out.append(host)
    return out


def arm_where(arm):
    """One phrase for where an arm trained: `elisa`, `a rented box`, or both."""
    hosts = arm_machines(arm)
    elisa = "elisa" in hosts
    rented = any(h.startswith("vast box") for h in hosts)
    if elisa and rented:
        return "elisa and a rented box"
    if elisa:
        return "elisa"
    if rented:
        return "a rented box"
    return hosts[0] if len(hosts) == 1 else "?"


def arms_of(cell):
    """Every arm the registry holds for a cell, in ARM_ORDER."""
    seen = [row[2] for row in _ROWS if row[1] == cell]
    return [a for a in ARM_ORDER if a in seen]


def find_run(arm, k, role="depth"):
    """The `Run` for an (arm, k, role), or None. Stop and head are unset.

    The registry is keyed by eval tag, and a trainer log has no eval tag.
    This is the way in from the training side.
    """
    for row in _ROWS:
        if row[2] == arm and row[3] == k and row[5] == role:
            return Run(row[0], row[0], row, None, None)
    return None


def machine_held(a, b):
    """Did these two runs train on the same box?

    A `k = 3` against a `k = 0` on two boxes measures the depth AND the box.
    The reproduction check separates on the box at up to 0.1169 GM-Relative
    MASE, so a comparison that does not hold it fixed carries a term this
    study cannot bound. Every table that prints a delta prints this beside
    it.
    """
    return a.machine == b.machine


def ckpt_step(run, filename):
    """The step a periodic checkpoint of `run` holds, or None.

    Prefix matching is not enough. Group A's launcher writes every depth of
    a cell AND its re-weighting control into one directory, so
    `cf393_..._cf373k0` is a prefix of `cf393_..._cf373k0_aw4_40k.pth` and a
    `startswith` test silently folds the control into the cell. What may
    follow the run name is `_<N>k.pth`, optionally after train.py's `_rN`
    re-fire infix, and nothing else.
    """
    m = re.match(rf"^{re.escape(run)}(?:_r\d+)?_(\d+)k\.pth$", filename)
    return int(m.group(1)) * 1000 if m else None


def backbones():
    """`[(arm, k, role, run name)]` for every row that trained a backbone,
    in ARM_ORDER. The curve and checkpoint figures join on this."""
    rows = [(arm, k, role, run)
            for _s, _c, arm, k, _seed, role, _n, run in _ROWS if run]
    return sorted(rows, key=lambda r: (ARM_ORDER.index(r[0]), r[1], r[2]))


if __name__ == "__main__":
    w = max(len(r[0]) for r in _ROWS)
    for stem, cell, arm, k, seed, role, note, run in _ROWS:
        host, card = _MACHINE.get(stem, ("?", "?"))
        print(f"{stem:<{w}}  {cell:<3} {arm:<7} k={k}  seed={seed}  "
              f"{host:<11} {card:<9} {role:<7} {note or run}")
    missing = [r[0] for r in _ROWS if r[0] not in _MACHINE]
    if missing:
        print(f"\nNO MACHINE RECORDED: {', '.join(missing)}")
        sys.exit(1)
    sys.exit(0)
