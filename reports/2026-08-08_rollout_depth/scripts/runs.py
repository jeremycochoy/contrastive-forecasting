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
  arm       a (cell, backbone seed) pair. The card assumed one training per
            cell; B5 has two, and they disagree, so the arm is the unit a
            depth delta is computed within.
  k         `--train-rollout-depth`.
  role      `depth` for a run that is one point of an arm's depth ladder,
            `control` for a run that answers a specific question and is not
            on any ladder.

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
    ("G1_B5pub",          "B5", "B5·s1",  0, 20260520, "control",
     "#379's published backbone, this study's head and eval", ""),
]

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

# Draw order: the two arms that improve, then the two that do not, then the
# arm whose two seeds disagree. Figures read it so a reader meets the same
# sequence everywhere.
ARM_ORDER = ["B9", "B1", "B5·s1", "B5·s2", "A3"]

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

    def label(self):
        base = f"{self.arm} k = {self.k}"
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
        print(f"{stem:<{w}}  {cell:<3} {arm:<7} k={k}  seed={seed}  "
              f"{role:<7} {note or run}")
    sys.exit(0)
