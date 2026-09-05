# One issue, one PR, one report

Issue #409, PR #410, and `rep_weight_decay.md` in this directory. Every result
of this card goes in that one file.

Do not open a second experiment directory. Do not write a second report. Do not
open a second PR.

## What the one report holds

1. The decay of the `L_rep` weight at k = 32, align on the teacher: the EMA
   momentum axis, the decay ramp axis, and the ramp-by-momentum grid.
2. The two continuations to 80,000 steps.
3. The fixed-momentum search: one (momentum, ramp) pair per cell, and the two
   continuations of its best two.
4. The A4 run at k = 3, align on the student, resumed from its 40,000-step
   checkpoint with the `L_rep` coefficient at 0.0, scored at 200,000 steps and
   at each later stop the user asks for.

Item 4 uses a different cell from items 1 to 3. Say so where it appears. It
belongs here because this card asks how to beat that model, and the reference
for it is `reports/2026-08-20_a4_full_pass/a4_full_pass.md`.
