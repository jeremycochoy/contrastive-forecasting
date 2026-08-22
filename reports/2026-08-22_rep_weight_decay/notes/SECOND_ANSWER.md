# The second answer, and what measures it

The card of #409 asks for two answers:

1. A new best GM-Relative MASE.
2. A backbone that can improve more with longer training. "Say what supports
   that, and what does not."

Every arm of this card stops at 40,000 steps. So no arm measures the second
answer. This file holds the decision on that gap, for the agent that runs the
study and for the agent that writes the report.

## The decision

**No ninth backbone. Every arm stops at 40,000 steps. The report answers the
second question from four named measurements, and states in one sentence that
the evidence is indirect.**

Two reasons. First, the direct answer costs a card of its own. This card's
backbone budget is 8 x 40,000 steps. One arm to 200,000 steps adds 160,000
steps on top, which is half of that budget again. Second, the arm to extend
is not known until the first answer arrives, at the end of this card. So the
direct measurement belongs to the card after this one.

## The four measurements

The report gives all four. Each one is a fact this card measures.

**1. The loss by term at the stop, and its slope.** Read `l_rep`, `l_align`
and `cos_err_d0` to `cos_err_d3` from each `<run>_losses.csv`. Give the value
at step 40,000 and the slope over steps 30,000 to 40,000. A term that still
falls at the stop is headroom. A flat term is not. Read the forecast error
from the `cos_err_d*` columns, not from `l_align` alone: `l_align` is the
depth-0 copy, and this card runs k = 3.

**2. The contrastive AUC of every run.** Read `results/auc_verdicts.tsv`, which
`collect.sh` writes from `auc_watch.py`. An arm that lost the contrastive task
has nothing left to learn from more steps, whatever its loss does. The AUC gate
stops such an arm, so a stopped arm has a verdict, a step, and no score. The
report gives its AUC and its loss to the step it reached.

**3. The measured trajectory of the control cell.** #373 ran this same cell at
k = 3 to three stops:

| backbone steps | GM-Relative MASE |
|---|---|
| 40,000 | 1.0862 |
| 100,000 | 1.0801 |
| 200,000 | 1.0660 |

Source: `reports/2026-08-08_rollout_depth/rollout_depth.md`, row A4, student
encoder. This is the only measured value of "more training" for this
configuration, and it belongs to the control. A treated arm that beats the
control at 40,000 steps has no such curve of its own.

**4. The seed spread of this card.** Three arms are a repeat at a second seed:
`ctrl_s24`, `dec0_s24` and `flr05_s24`. Their ranges against `ctrl_s20`,
`dec0_s20` and `flr05_s20` measure this cell's run-to-run spread. A score
difference under the largest of those three ranges is not a rank.

## What the report must not say

Do not write that an arm "will improve with longer training". No arm of this
card measures a score past 40,000 steps. Give the four measurements, name them
as indirect, and stop there.

## The follow-up

One arm can beat `ctrl_s20` and `ctrl_s24` by more than the largest seed range
of measurement 4. The next card then trains that arm to 200,000 steps, and
compares it against 1.0660. That is the direct answer. Do not run it inside
this card.
