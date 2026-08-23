# The second answer, and what measures it

The card of #409 asks for two answers:

1. A new best GM-Relative MASE.
2. A backbone that can improve more with longer training. "Say what supports
   that, and what does not."

Every arm of this card stops at 40,000 steps. So no arm measures the second
answer. This file holds the decision on that gap, for the agent that runs the
study and for the agent that writes the report.

## The decision

**No extra backbone. Every arm stops at 40,000 steps. The report answers the
second question from four named measurements, and states in one sentence that
the evidence is indirect.**

Two reasons. First, the direct answer costs a card of its own. One arm to
200,000 steps adds 160,000 steps, which is more than half of this card's whole
backbone budget. Second, the arm to extend is not known until the first answer
arrives, at the end of this card. So the direct measurement belongs to the card
after this one.

## The four measurements

The report gives all four. Each one is a fact this card measures.

**1. The loss by term at the stop, and its slope.** Read `l_rep`, the reduced
align term and `cos_err_d0` to `cos_err_d32` from each `<run>_losses.csv`. Give
the value at step 40,000 and the change over steps 30,000 to 40,000. A term
that still falls at the stop is headroom. A flat term is not. Read the align
term as the residual `loss - rep_w * l_rep - sigreg_e - sigreg_h`, not from
`l_align` alone: `l_align` is the depth-0 copy and this cell aligns on the
teacher. `notes/loss_decomposition.md` gives the formula.

**2. The contrastive AUC of every run.** Read `results/auc_verdicts.tsv`, which
`collect.sh` writes from `auc_watch.py`. An arm that lost the contrastive task
has nothing left to learn from more steps, whatever its loss does. The AUC gate
stops such an arm, so a stopped arm has a verdict, a step, and no score. The
report gives its AUC and its loss to the step it reached.

**3. The measured trajectory of this cell.** The EMA momentum sweep ran the
best arm of this cell to 40,000 steps only, so no longer stop exists for it.
The nearest measured curve is #373's k = 3 cell, which is another cell. Say so.
The report must not borrow it as this cell's own trajectory.

**4. The seed spread of this card.** Six arms are the same decay at six
backbone seeds, so their range IS this treatment's run-to-run spread, measured
and not borrowed. The reference has its own range, 0.0016 over two seeds. A
score difference under the larger of the two is not a rank.

## What the report must not say

Do not write that an arm "will improve with longer training". No arm of this
card measures a score past 40,000 steps. Give the four measurements, name them
as indirect, and stop there.

## The follow-up

The decay's mean over six seeds can beat 1.1491 by more than the two spreads of
measurement 4. The next card then trains the decay to 200,000 steps and
compares it against a reference at the same stop. That is the direct answer. Do
not run it inside this card.
