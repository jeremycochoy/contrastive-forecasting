"""--skip-nan-samples (#421): drop only the rows that make a step non-finite.

A value-space step reads each batch row on its own. The model mixes no rows
(attention and the patch encoders run per series, and there is no batch
statistic), and the loss links rows only through its mean over the kept
values. So each row, after every transform (mixup, synthetic rows, crossfade
triplets), is one unit.

When a step gives a non-finite loss or gradient, the trainer finds the rows
that cause it by bisection. Each pass runs the step again with a subset of
rows active and the others inert: zero inputs, no target, and no share of the
loss. The batch keeps its shape, and each pass first restores the random
state from before the first pass, so each row draws the same dropout and
DropKey masks as before. The split, the patch size and the mixup of each row
are drawn once, before any pass. A last pass with the culprits inert gives
the gradient the step uses. When no row alone explains the fault, the step
is skipped.
"""

from __future__ import annotations

import math

import torch

# The largest number of bisection passes one step may take. One culprit in
# 257 rows takes 16 passes, two take 18 to 32, and eight at most 94.
MAX_PASSES = 128

# Steps skipped in a row after which the trainer stops: a fault that no row
# explains, step after step, is not in the data.
MAX_SKIPPED_IN_A_ROW = 10


def rng_state(device):
    """The CPU and device random state, to replay a pass with the same draws."""
    state = {"cpu": torch.get_rng_state()}
    if torch.device(device).type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def weights_are_finite(model):
    """True when every weight of ``model`` is finite."""
    return bool(torch.stack([torch.isfinite(p).all()
                             for p in model.parameters()]).all())


def is_finite_step(loss_value, model):
    """True when the loss and every gradient of ``model`` are finite."""
    if not math.isfinite(loss_value):
        return False
    flags = [torch.isfinite(p.grad).all() for p in model.parameters()
             if p.grad is not None]
    return bool(torch.stack(flags).all()) if flags else True


class _OutOfPasses(Exception):
    pass


def find_culprits(is_bad, rows, max_passes=MAX_PASSES):
    """The rows that make a pass non-finite on their own, by bisection.

    ``is_bad(rows)`` runs one pass with only ``rows`` active, and the pass
    with all of ``rows`` active is known to be bad. Each bad half is split
    again, down to single rows. Returns ``(culprits, passes)``. ``culprits``
    is None when a bad set has two clean halves (the fault needs rows of
    both at once), or when the passes run out.
    """
    passes = 0

    def bad(part):
        nonlocal passes
        if passes == max_passes:
            raise _OutOfPasses
        passes += 1
        return is_bad(part)

    def search(part):
        if len(part) == 1:
            return list(part)
        half = len(part) // 2
        halves = [(p, bad(p)) for p in (part[:half], part[half:])]
        if not any(flag for _, flag in halves):
            return None
        found = []
        for p, flag in halves:
            if flag:
                sub = search(p)
                if sub is None:
                    return None
                found += sub
        return found

    try:
        return search(list(rows)), passes
    except _OutOfPasses:
        return None, passes
