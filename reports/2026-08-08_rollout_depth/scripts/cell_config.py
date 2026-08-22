#!/usr/bin/env python3
"""#373 — what each cell TRAINS ON, derived from the launcher sources.

A reader who meets `arm6_v2 combab` cannot tell what the model optimises.
This module turns a cell id into the configuration in words, and every
table, figure and sentence in the report reads its name from here. Nothing
downstream hand-types a loss term, so nothing downstream can drift.

A configuration name needs four parts, because three of them collide:

    the arm            `arm6_v2`
    its loss terms     `L_rep MoCo, tau_rep 1 + L_align, no CPC`
    the align target   `on the student`   (A4 against A3, B1 against B2)
    the EMA regime     `EMA 0.9 to 1.0`   (A4 against B1)

Drop the align target and A4 collides with A3. Drop the EMA regime and A4
collides with B1. `name()` carries all four; `check()` asserts the 14 names
are distinct.

Nothing here is typed from the card's prose. `cells.tsv` says which
launcher each cell runs and with which argument; this module reads that
launcher's `case` block for the cell's `LOSS_ARGS` / `ALIGN_ARGS` /
`EXTRA_ARGS`, reads the shared trainer invocation for the defaults those
override, and replays argparse's last-wins rule over the two. So
`--cpc-infonce-weight 0.0` in EXTRA_ARGS beats the `1.0` in the shared
block, exactly as the trainer saw it.

A launcher can also hold a group of shared flags in one array, so that a
later card replaces the group as a unit. #404 sweeps the EMA momentum over
cell A3 that way. Those arrays carry their default on their own
`read -r -a` line, and this module reads it there. The module stops when
neither source gives an array a value. An array that expands to nothing
takes its flags with it, and the name that comes out still looks correct.

`tau_rep` is the one term that is not a flag the launcher writes. The
trainer's `--tau-rep` defaults to None and the loss then falls back to
`--tau`, so `arm6_v2 combab` and `arm6_v2 ncpc` — identical in every other
flag — differ only in that fallback: 1.0 against 0.10. Both names carry it,
so the two never read as one configuration.

Usage:
    python3 cell_config.py            # print the 14 configurations
    python3 cell_config.py --check    # assert against r2_ladder.CELL_ARM
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The launcher's own array names, and the placeholder each one appears as
# inside the trainer invocation. A per-arm `case` block assigns these three.
_ARRAYS = ("LOSS_ARGS", "ALIGN_ARGS", "EXTRA_ARGS")
# Empty for every cell of the card, and the launcher declares no default for
# it: a fresh leg leaves `RESUME=()`, and only a resumed leg fills it.
_EMPTY = ("RESUME",)


# --------------------------------------------------------------------------
# reading the launchers
# --------------------------------------------------------------------------
def _unwrap(text):
    """Join backslash-continued shell lines and drop comment lines."""
    text = text.replace("\\\n", " ")
    keep = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    return "\n".join(keep)


def _case_blocks(text):
    """`{case label: block text}` for the launcher's per-arm dispatch."""
    m = re.search(r'case\s+"\$(?:ARM|CELL)"\s+in\n(.*?)\nesac', text,
                  re.S)
    if not m:
        raise SystemExit("no per-arm case block in launcher")
    body, out = m.group(1), {}
    for chunk in re.split(r"\n\s*;;\s*\n?", body):
        head = re.match(r"\s*([A-Za-z0-9_|*]+)\)\s*\n?(.*)", chunk, re.S)
        if not head or head.group(1) == "*":
            continue
        out[head.group(1)] = head.group(2)
    return out


def _arrays(block):
    """`{array name: [tokens]}` for the arrays this case block assigns."""
    out = {}
    for name in _ARRAYS:
        m = re.search(rf"{name}=\((.*?)\)\s*(?:\n|$)", block, re.S)
        out[name] = _tokens(m.group(1)) if m else []
    return out


def _tokens(s):
    return [t.strip('"') for t in s.split() if t.strip('"')]


def _shell_defaults(text):
    """`{array name: [tokens]}` for the launcher's own top-level defaults.

    A launcher can hold a group of flags in one array, so that a caller
    replaces the group as a unit. #404 sweeps the EMA momentum over #373's
    cell that way, and the launcher writes:

        read -r -a EMA_ARGS_ARR <<<"${EMA_ARGS:---ema-tau 0.9 ...}"

    The default in that line is the command line every published leg ran,
    so this function takes it from the launcher and never from prose. A
    caller that overrides the array runs a different card, and that card
    names its own arms.
    """
    out = {}
    for name, default in re.findall(
            r'read\s+-r\s+-a\s+(\w+)\s+<<<"\$\{\w+:-(.*?)\}"', text):
        out[name] = _tokens(default)
    return out


def _invocation(text):
    """The trainer command line, as one token list with placeholders kept."""
    m = re.search(r'python3 -u "\$TRAIN"(.*?)>>"\$tlog"', text, re.S)
    if not m:
        raise SystemExit("no trainer invocation in launcher")
    return _tokens_keep(m.group(1))


def _tokens_keep(s):
    """Tokenise, keeping `${ARRAY[@]}` placeholders whole."""
    out = []
    for raw in s.split():
        t = raw.strip('"')
        if not t:
            continue
        m = re.fullmatch(r"\$\{(\w+)\[@\]\}", t)
        out.append(("@", m.group(1)) if m else t)
    return out


def _flags(invocation, arrays, defaults=None):
    """Replay argparse over the real token order. Last value wins.

    Each `${ARRAY[@]}` placeholder takes its tokens from the case block
    first, then from the launcher's own default for that array. The module
    stops if neither source has the array. A silent expansion to nothing
    drops every flag the array carries, and the name that comes out still
    looks right.
    """
    defaults = defaults or {}
    flat = []
    for t in invocation:
        if isinstance(t, tuple):
            name = t[1]
            if name in arrays:
                flat.extend(arrays[name])
            elif name in defaults:
                flat.extend(defaults[name])
            elif name not in _EMPTY:
                raise SystemExit(f"{name} in the trainer invocation, but the "
                                 f"launcher gives it no value")
        else:
            flat.append(t)
    out, i = {}, 0
    while i < len(flat):
        tok = flat[i]
        if tok.startswith("--"):
            nxt = flat[i + 1] if i + 1 < len(flat) else None
            if nxt is not None and not nxt.startswith("--"):
                out[tok] = nxt
                i += 2
                continue
            out[tok] = True
        i += 1
    return out


def _read_cells():
    """`{cell: (launcher file, launcher argument)}` from `cells.tsv`."""
    out = {}
    for ln in (HERE / "cells.tsv").read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        cell, _slug, launcher, arg = ln.split("\t")
        out[cell] = (launcher, arg)
    return out


def _resolve():
    """`{cell: flag dict the trainer actually saw}`."""
    cache, out = {}, {}
    for cell, (launcher, arg) in _read_cells().items():
        if launcher not in cache:
            text = _unwrap((HERE / launcher).read_text())
            cache[launcher] = (_case_blocks(text), _invocation(text),
                               _shell_defaults(text))
        blocks, inv, defaults = cache[launcher]
        if arg not in blocks:
            raise SystemExit(f"{launcher} has no case for '{arg}'")
        out[cell] = _flags(inv, _arrays(blocks[arg]), defaults)
        out[cell]["_arg"] = arg
    return out


FLAGS = _resolve()


# --------------------------------------------------------------------------
# flags to words
# --------------------------------------------------------------------------
def _num(v):
    """`1.0` -> `1`, `0.10` -> `0.1`. Names read better without the zeros."""
    f = float(v)
    return f"{f:g}"


# One arm's launcher description disagrees with the flags it passes.
# `arm4 combab` is labelled τ=1.0 and puts `--tau 1.0` in LOSS_ARGS, which
# the trainer invocation expands BEFORE the shared `--tau 0.10`; argparse
# keeps the last value, so the run trained at 0.10. Every name here carries
# the value the trainer received. `--tau-rep` is not in the shared block, so
# the combab arms' `--tau-rep 1.0` does survive.
TAU_NOTE = {
    "arm4 combab": "its launcher's own label says tau 1.0, and its "
                   "`--tau 1.0` sits before the shared `--tau 0.10`, so "
                   "argparse kept 0.10",
}


def arm(cell):
    """The launcher recipe, for example `arm6_v2 combab`. It matches the
    run names."""
    a = re.sub(r"_align[ST]$", "", FLAGS[cell]["_arg"])
    m = re.match(r"(arm\d+(?:_v\d+)?)(?:_(\w+))?$", a)
    base, abl = m.group(1), m.group(2)
    return f"{base} {abl}" if abl else base


def base_arm(cell):
    """The arm alone, for example `arm6_v2`."""
    return arm(cell).split(" ")[0]


def align_target(cell):
    """`student`, `teacher`, or `none` where the recipe carries no L_align."""
    f = FLAGS[cell]
    if "--align-loss-weight" not in f:
        return "none"
    return f.get("--align-target", "student")


def ema(cell):
    """`scheduled` or `fixed 0.9`, off the EMA flags the launcher passed."""
    f = FLAGS[cell]
    return "scheduled" if "--ema-tau-end" in f else "fixed 0.9"


def ema_words(cell):
    f = FLAGS[cell]
    if "--ema-tau-end" not in f:
        return f"EMA {f['--ema-tau']}"
    return f"EMA {f['--ema-tau']} to {f['--ema-tau-end']}"


def terms(cell, target=True, short=False):
    """The loss terms, in short names.

    `target` puts L_align's target in. `short` trims the pooled term's
    wording for a figure legend, which has no room for the long form.
    """
    f = FLAGS[cell]
    shape = f["--loss-shape"]
    tau = f.get("--tau", "0.07")
    tau_rep = f.get("--tau-rep", tau)          # unset: L_rep falls back to tau
    parts = []

    if shape.endswith("split_pred_rep"):
        parts.append(f"split L_pred + L_rep{_moco(f)}, tau {_num(tau)}")
    elif shape.endswith("rep_only"):
        core = f"L_rep{_moco(f)}, tau_rep {_num(tau_rep)}"
        if "--align-loss-weight" in f:
            core += " + L_align"
            if target:
                core += f" on the {align_target(cell)}"
        parts.append(core)
    elif "xshh_allt" in shape:
        core = ("pooled contrastive" if short
                else "pooled contrastive over batch and channels")
        core += _moco(f, sep=", ")
        if "--subtract-contrastive-floor" in f:
            core += ", floor subtracted"
        parts.append(f"{core}, tau {_num(tau)}")
    else:
        parts.append(shape)

    parts.append("CPC" if float(f.get("--cpc-infonce-weight", 0)) > 0
                 else "no CPC")
    if float(f.get("--sigreg-embedding-weight", 0)) == 0:
        parts.append("no SIGReg on e")
    return ", ".join(parts)


def _moco(f, sep=" "):
    bits = []
    if "--moco-rep-keys" in f:
        bits.append("MoCo keys")
    if "--moco-negatives" in f:
        bits.append("MoCo negatives")
    return sep + ", ".join(bits) if bits else ""


# --------------------------------------------------------------------------
# the names the report uses
# --------------------------------------------------------------------------
def name(cell, short=False):
    """The full configuration: arm, loss terms, align target, EMA regime."""
    return (f"{base_arm(cell)} ({terms(cell, short=short)}, "
            f"{ema_words(cell)})")


def name_id(cell):
    """The configuration, with the card's short cell id for traceability."""
    return f"{name(cell)} [{cell}]"


def short_name(cell):
    """The three parts that separate this cell from every other one.

    The loss terms are the longest part of `name()` and the least
    discriminating: 4 cells share `arm6_v2 combab`'s terms. The arm, the
    align target and the EMA regime separate all 14, so a second mention in
    prose carries those and drops the rest.
    """
    tgt = ("no L_align" if align_target(cell) == "none"
           else f"L_align on the {align_target(cell)}")
    return f"{base_arm(cell)}, {tgt}, {ema_words(cell)}"


def bracket(cell, target=True):
    """The loss-term bracket that follows an arm name at its first use."""
    return f"({terms(cell, target=target)})"


def arm_bracket(cell):
    """`arm6_v2 combab (L_rep MoCo keys, tau_rep 1 + L_align, no CPC)`."""
    return f"{arm(cell)} {bracket(cell, target=False)}"


def recipe(cell, with_cell=True):
    """One figure-width line: the four parts, compressed.

    An axis label has room for about seventy characters, so the long words
    of `terms()` go and the discriminating ones stay — MoCo, the two
    temperatures, CPC, SIGReg, the align target and the EMA regime.
    """
    f = FLAGS[cell]
    shape = f["--loss-shape"]
    tau = f.get("--tau", "0.07")
    if shape.endswith("split_pred_rep"):
        core = f"split L_pred + L_rep{_moco(f)}, tau {_num(tau)}"
    elif shape.endswith("rep_only"):
        core = (f"L_rep{_moco(f)} + L_align->{align_target(cell)}, "
                f"tau_rep {_num(f.get('--tau-rep', tau))}")
    else:
        core = f"pooled{_moco(f)}, tau {_num(tau)}"
    if "--subtract-contrastive-floor" in f:
        core += ", floor"
    core += ", CPC" if float(f.get("--cpc-infonce-weight", 0)) > 0 \
        else ", no CPC"
    if float(f.get("--sigreg-embedding-weight", 0)) == 0:
        core += ", no SIGReg e"
    head = f"{cell}  " if with_cell else ""
    return (f"{head}{base_arm(cell)}  {core}, "
            f"{ema_words(cell).replace(' to ', '->')}")


CELLS = list(FLAGS)


# --------------------------------------------------------------------------
REPORT = HERE.parent / "rollout_depth.md"


def check(report=REPORT):
    """The names separate the 14 cells, agree with the ladder's table, and
    the report's own prose spells them the way this module builds them."""
    sys.path.insert(0, str(HERE))
    import r2_ladder as L                                   # noqa: E402

    bad = []
    for cell in CELLS:
        want = L.CELL_ARM[cell]
        got = (arm(cell), align_target(cell), ema(cell))
        if got != want:
            bad.append(f"{cell}: launcher says {got}, CELL_ARM says {want}")
    seen = {}
    for cell in CELLS:
        seen.setdefault(name(cell), []).append(cell)
    for n, cs in seen.items():
        if len(cs) > 1:
            bad.append(f"name collision on {cs}: {n}")

    # Prose is hand-written and the injected blocks are not, so the drift
    # this guards is in the prose: every `[<cell>]` in the report must have
    # that cell's arm in the words just before it.
    named = 0
    if report and Path(report).is_file():
        text = Path(report).read_text()
        for cell in CELLS:
            for m in re.finditer(rf"\[(?:cell )?{cell}\]", text):
                named += 1
                lead = text[max(0, m.start() - 240):m.start()]
                if base_arm(cell) not in lead:
                    bad.append(f"{cell}: '[{cell}]' in the report is not "
                               f"preceded by {base_arm(cell)}")
    for line in bad:
        print("MISMATCH " + line)
    print(f"{len(CELLS)} cells, {len(seen)} distinct names, {named} named in "
          f"the report, {len(bad)} mismatch(es)")
    return 1 if bad else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--check" in argv:
        return check()
    for cell in CELLS:
        print(f"{cell:<4} {name(cell)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
