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

**4. The seed spread of this card.** Arm 1 is one EMA schedule at three
backbone seeds, so its range IS this treatment's run-to-run spread, measured
and not borrowed. The other seven arms are one seed each, because the axis is
the schedule. The reference has its own range, 0.0016 over two seeds. A score
difference under the larger of the two is not a rank.

## What the report must not say

Do not write that an arm "will improve with longer training". No arm of this
card measures a score past 40,000 steps. Give the four measurements, name them
as indirect, and stop there.

## The follow-up

The best schedule under the decay can beat 1.1491 by more than the two spreads
of measurement 4. The next card then trains the decay to 200,000 steps and
compares it against a reference at the same stop. That is the direct answer. Do
not run it inside this card.

---

# The measured answer, 2026-08-25

The four measurements above are now taken. `results/loss_terms_at_stop.csv`,
`results/loss_terms_trajectory.csv` and `results/loss_slope.csv` hold them.

## The answer

**No backbone of this card is named as one that will improve more with longer
training.** Every arm still learns at the stop, and nothing this card measured
separates one arm from another on that.

## What supports more training, on every arm

**1. No arm converged at 40,000 steps.** Over steps 20,000 to 40,000 the total
loss of every scored arm still falls, by 0.079 to 0.235 per 10,000 steps. The
mean `cos_err` falls with it, by 0.060 to 0.134. `results/loss_slope.csv` fits
those slopes to 1,000-step block means, and `results/loss_terms_trajectory.csv`
gives the same terms at every 5,000 steps.

**2. Five of the six scored runs held the contrastive task** to the stop, at
AUC 0.9166 to 0.9924. A run that held the task has a task left to learn.
`dec_m050_fix` lost it at step 10,162 and has no score.

**3. The decay reached 0.0 on every arm, and the loss that is left still
moves.** `l_rep` goes blank at step 9,999 on all nine drawn runs, where the
weight reaches 0.0 and the trainer computes no L_rep. Past step 10,000 the
total loss IS the reduced align term, within 0.003 on every arm. So the steps
after the ramp go to the term that moves.

## What does NOT support naming one backbone

**1. The slope at the stop is inside the seed spread.** Over steps 30,000 to
40,000 the arms split in two: `dec_m070_fix` at -0.258, `dec_m099_fix` at
-0.256 and `dec_m080_r200` at -0.088 keep falling, while `dec_s20` at +0.106,
`dec_s24` at +0.071 and `dec_s22` at +0.009 do not. But those last three are
ONE schedule at three seeds, so their span of 0.097 is this measurement's own
noise floor. `dec_m070_fix` and `dec_m099_fix` sit 0.002 apart. The slope
separates the two groups. It ranks nothing inside a group.

**2. The slope does not track the score.** `dec_m070_fix` falls fastest at the
stop and scores the WORST of the six, 1.3534. `dec_m080_r200` falls the most
slowly of the three falling arms and scores the BEST, 1.2352. The two readings
disagree, and this card holds nothing that settles them.

**3. No arm measures a score past 40,000 steps.** The card holds no
measurement of what a longer run would score.

## The arm a follow-up should extend

`dec_m080_r200`, on four of the five measurements: the lowest loss at the stop
(0.208), the lowest mean `cos_err` (0.151), the highest AUC (0.9924) and the
best score (1.2352). Its slope at the stop is the fifth, and it argues the
other way. That is a choice for the next card, not a result of this one.
