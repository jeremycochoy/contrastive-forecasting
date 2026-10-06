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
the gradient the step uses. The search has no pass budget: it always ends
with the rows to drop, and it skips the step only when every row is bad.

--skip-spike-samples (#421) uses the same search for a step whose gradient
norm is far above the recent median: it drops the rows that cause the spike.
"""

from __future__ import annotations

import math

import torch

# Steps skipped in a row after which the trainer stops. A step is skipped
# only when every one of its rows is bad.
MAX_SKIPPED_IN_A_ROW = 10

# --skip-spike-samples: the gradient norms of the last HISTORY steps give the
# median, and the guard waits for MIN_HISTORY of them.
HISTORY, MIN_HISTORY = 200, 50

# The search first drops the 1, 2, 4, ... MAX_RANKED most suspect rows. On
# the #421f resume, steps near 21,850 held 7 to 16 bad rows.
MAX_RANKED = 64

# A row whose input gradient is above OUTLIER_FACTOR times the median row's
# is an outlier. The search first tries the pass without every outlier. On
# the #421f resume near step 21,850, 40 to 60 rows of a step exploded, and
# a search one row at a time took about 100 passes.
OUTLIER_FACTOR = 100.0


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


def row_grad_norms(input_grad):
    """The norm of each row's input gradient, in float64. A non-finite norm
    is inf. An exploding gradient in a patch encoder shows in it."""
    norms = torch.linalg.vector_norm(
        input_grad.detach().flatten(1).double(), dim=1)
    return torch.nan_to_num(norms, nan=float("inf"))


def rank_rows(input_grad):
    """The rows by the norm of their input gradient, the largest first."""
    if input_grad is None:
        return []
    return row_grad_norms(input_grad).argsort(descending=True).tolist()


def outlier_rows(input_grad, factor=OUTLIER_FACTOR):
    """The rows whose input gradient norm is above ``factor`` times the
    median row's, or not finite."""
    if input_grad is None:
        return []
    norms = row_grad_norms(input_grad)
    limit = factor * norms.median()
    return (~(norms <= limit)).nonzero().view(-1).tolist()


def search_order(input_grad, faults=None):
    """``(ranked, outliers)``: the order in which the row search tries the
    rows, and the rows it first drops together (find_culprits).

    ``faults`` (bool, one per row, #412) marks the rows whose forward is
    not finite, on an objective whose terms couple the rows. Through those
    terms one such row makes the input gradient of every row it meets NaN,
    so only its forward names it. The faults are then the only outliers
    from a non-finite gradient, and they rank first. A row with a
    non-finite input gradient only ranks high, and the search checks it on
    its own. None: the input gradient alone, as for rows that the objective
    reads one by one.
    """
    ranked, outliers = rank_rows(input_grad), outlier_rows(input_grad)
    if faults is None:
        return ranked, outliers
    bad = faults.nonzero().view(-1).tolist()
    finite = ([] if input_grad is None
              else torch.isfinite(row_grad_norms(input_grad)).tolist())
    outliers = bad + [r for r in outliers if finite[r] and r not in bad]
    return bad + [r for r in ranked if r not in bad], outliers


def total_grad_norm(model):
    """The L2 norm of every gradient of ``model`` together, before the clip."""
    norms = [torch.linalg.vector_norm(p.grad.detach()) for p in model.parameters()
             if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.0


class SpikeGuard:
    """--skip-spike-samples: a step is a spike when its gradient norm is above
    ``factor`` times the reference: the median of the last HISTORY clean
    steps. Until the guard holds MIN_HISTORY of them, the reference is
    ``prior``, the clip norm: without a prior the first spikes would set the
    median. A checkpoint keeps the history, so a resume reads it back. A pass with n of
    the B rows active has a noisier mean gradient, so its threshold grows
    with sqrt(B / n)."""

    def __init__(self, factor, prior):
        self.factor, self.prior, self.norms = factor, prior, []

    def reference(self):
        if len(self.norms) < MIN_HISTORY:
            return self.prior
        return sorted(self.norms)[len(self.norms) // 2]

    def threshold(self, active, total):
        return self.factor * self.reference() * math.sqrt(total / active)

    def is_spike(self, norm, total):
        return not math.isfinite(norm) or norm > self.threshold(total, total)

    def record(self, norm):
        if math.isfinite(norm):
            self.norms = (self.norms + [norm])[-HISTORY:]


def find_culprits(is_bad, rows, ranked=(), outliers=()):
    """The rows to drop so that the pass with the other rows is clean.

    ``is_bad(active)`` runs one pass with exactly the rows of ``active``, and
    the pass with all of ``rows`` is known to be bad. Bisection: each bad half
    is split again, down to single rows. When both halves are clean on their
    own, the fault needs rows of both, so the second half is searched with the
    first one active. After the rows found are dropped, one more pass checks
    the rest, and a bad rest is searched again.

    ``ranked`` lists the rows from the most suspect (rank_rows). The search
    first tries the passes without the top 1, 2, 4, ... MAX_RANKED of them.
    When such a pass is clean, it looks for the culprits among these rows
    only, with the other rows active. The ranking only sets the order of the
    passes: a row is dropped only when the pass without it is clean.

    ``outliers`` lists the rows whose input gradient is far above the
    median row's (outlier_rows). The search first tries the pass without
    all of them. When it is clean, it looks for the culprits among these
    rows only, with the other rows active, as for the ranked rows: a large
    input gradient alone drops no row.

    Returns ``(culprits, passes, clean)``: ``clean`` is True when the last
    pass, with the culprits out, was clean, and the model then holds its
    gradients. It is False only when every row was dropped.
    """
    passes = 0

    def bad(active):
        nonlocal passes
        passes += 1
        return is_bad(active)

    def search(part, context):
        # The pass with context + part is bad, and the pass with context alone
        # is clean (or context is empty).
        if len(part) == 1:
            return list(part)
        half = len(part) // 2
        first, second = part[:half], part[half:]
        first_bad, second_bad = bad(context + first), bad(context + second)
        if not (first_bad or second_bad):
            return search(second, context + first)
        return ((search(first, context) if first_bad else [])
                + (search(second, context) if second_bad else []))

    kept, culprits = list(rows), []
    part, context = kept, []
    outliers = [r for r in outliers if r in set(rows)]
    if outliers and len(outliers) < len(kept):
        rest = [r for r in kept if r not in set(outliers)]
        if not bad(rest):
            if len(outliers) == 1:
                return outliers, passes, True
            part, context, ranked = outliers, rest, ()
    ranked = [r for r in ranked if r in set(rows)]
    k = 1
    while k <= min(MAX_RANKED, len(ranked), len(kept) - 1):
        top = set(ranked[:k])
        rest = [r for r in kept if r not in top]
        if not bad(rest):
            if k == 1:
                return ranked[:1], passes, True
            part, context = ranked[:k], rest
            break
        k *= 2
    while True:
        found = search(part, context)
        culprits += found
        dropped = set(found)
        kept = [r for r in kept if r not in dropped]
        if not kept:
            return culprits, passes, False
        if not bad(kept):
            return culprits, passes, True
        part, context = kept, []
