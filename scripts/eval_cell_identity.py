"""What an evaluation cell is called — the Python half of the grammar.

`eval_cell_identity.sh` is the half the evaluation scripts source; this is
the half the readers import: the wave-3 gate, the table builder, the
bootstrap, the report's figures. Two bindings, not two grammars —
`tests/test_eval_cell_identity.py` generates every name both ways and
requires them to agree character for character.

A cell name is::

    <slug>[_s<head-seed>]_bb<K>k[_r<N>]_hd<H>s

Two things decide the number and so appear in the name. `_r<N>` is the
backbone replicate: a resumed run keeps its own `_r<N>` run name, so one
(arm, step) pair can leave several different backbones. `_s<seed>` is the
head seed, the head trainer's `--seed`. Both tokens are empty at the values
every wave ran — the base run, and seed 20260722 — so the cell names the
report cites do not move.

A reader that spells the name itself sees only the untagged form, so a
replicate- or seed-backed cell reads as missing: dropped from the CI table,
dropped from the figures, re-run from scratch by the batch script. Hence
`cell_paths`, which answers a coordinate with every cell that measured it.
More than one is an ambiguity for the caller to refuse, the same one
`resolve_eval_checkpoint.sh` refuses.

    import eval_cell_identity as cid

    cid.cell_name("arm5_nse", 200, "_r3", 30000)     # wave seed implied
    cid.cell_paths(root, "arm5", 40, 15000, suffix="_summary.txt")
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import NamedTuple

# The head seed every wave, and so every committed cell, was measured under.
# Pinned to `EVAL_DEFAULT_HEAD_SEED` in eval_cell_identity.sh by the parity
# test; moving one without the other renames every cell.
DEFAULT_HEAD_SEED = "20260722"

# `_r[0-9]+` rather than `_r.*`, so a sibling recipe suffix (`_revin_`) does
# not read as a resume. Anchored on both sides, so no other step and no
# `_hd` variant matches.
CELL_RE = re.compile(
    r"^(?P<slug>.+?)_bb(?P<bb>\d+)k(?P<repl>_r\d+)?_hd(?P<hd>\d+)s$")

# The head-seed token, as it appears at the end of a slug. `\d+`, because
# that is what `head_seed_tag` in eval_cell_identity.sh writes one for: a
# rule that only read six digits or more parsed `arm5_s7` as an arm called
# `arm5_s7` measured at the *wave* seed, so the table carried the wrong seed
# and the delta script collided that cell with the real wave cell.
HEAD_SEED_RE = re.compile(r"_s(?P<seed>\d+)$")

# What a seed may be, on both sides of the grammar.
SEED_RE = re.compile(r"^\d+$")


class Cell(NamedTuple):
    """The coordinate a cell name spells out."""
    slug: str          # still carries the head-seed token, if any
    bb_k: int
    replicate: str     # "" for the base run, "_r<N>" for a resume
    head_steps: int


def head_seed_tag(head_seed: str | int = DEFAULT_HEAD_SEED) -> str:
    """`""` for the seed every wave ran, `_s<seed>` for any other.

    A seed that is not a run of digits is refused rather than written into a
    name, because `split_head_seed` could not read it back out of one: the
    same refusal the bash binding makes, for the same reason.
    """
    if not SEED_RE.match(str(head_seed)):
        raise ValueError(
            f"head seed {head_seed!r} is not a run of digits; the cell name "
            "would carry a token no reader can read back")
    return "" if str(head_seed) == DEFAULT_HEAD_SEED else f"_s{head_seed}"


def cell_name(slug: str, bb_k: str | int, replicate: str,
              head_steps: str | int,
              head_seed: str | int = DEFAULT_HEAD_SEED) -> str:
    """The output cell, with each token beside the thing it qualifies."""
    return (f"{slug}{head_seed_tag(head_seed)}"
            f"_bb{int(bb_k)}k{replicate}_hd{int(head_steps)}s")


def parse_cell(cell: str) -> Cell | None:
    """The coordinate `cell` names, or None if it is not a cell name."""
    m = CELL_RE.match(cell)
    if not m:
        return None
    return Cell(m.group("slug"), int(m.group("bb")), m.group("repl") or "",
                int(m.group("hd")))


def split_head_seed(slug: str) -> tuple[str, str]:
    """`arm5_s20260723` -> `("arm5", "20260723")`; a bare slug -> the default.

    The inverse of the token `cell_name` appends, so a reader recovers the
    seed a cell was measured under from its name alone.
    """
    m = HEAD_SEED_RE.search(slug)
    if not m:
        return slug, DEFAULT_HEAD_SEED
    return slug[: m.start()], m.group("seed")


def cell_pattern(slug: str, bb_k: str | int, head_steps: str | int,
                 head_seed: str | int = DEFAULT_HEAD_SEED,
                 suffix: str = "") -> re.Pattern:
    """Matches every cell of this coordinate — base run and replicates."""
    return re.compile(
        re.escape(f"{slug}{head_seed_tag(head_seed)}_bb{int(bb_k)}k")
        + r"(?P<repl>_r\d+)?"
        + re.escape(f"_hd{int(head_steps)}s{suffix}") + "$")


def cell_paths(root: str | os.PathLike, slug: str, bb_k: str | int,
               head_steps: str | int,
               head_seed: str | int = DEFAULT_HEAD_SEED,
               suffix: str = "") -> list[Path]:
    """Sorted `<root>/<cell><suffix>` that exist, whichever replicate wrote
    them. The base run sorts first. More than one is the caller's ambiguity
    to refuse."""
    pattern = cell_pattern(slug, bb_k, head_steps, head_seed, suffix)
    try:
        names = os.listdir(root)
    except OSError:
        return []
    return sorted(Path(root) / n for n in names if pattern.match(n))
