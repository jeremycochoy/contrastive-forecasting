# What this card can rank, and whether its reference is comparable

Two measurements for the agent that writes the report. Both are run-phase
facts, and both have a script and a table behind them. Neither is a report.

## 1. The gate: 0.0219

`scripts/rank_gate.py` writes `results/rank_gate.tsv`.

Only ONE schedule of this card ran at more than one seed: 0.9 to 1.0 at 100k,
at seeds 20260520, 20260522 and 20260524. Their scores are 1.2670, 1.2593 and
1.2812, so the range is **0.0219**. That range IS this treatment's whole
measured run-to-run spread. This card measured no other, and a spread from
another study would not be this treatment's.

### What clears the gate

**Every arm against the no-decay reference does.** The six scored arms sit
+0.0570 to +0.1305 above the same schedule with no decay, and +0.0861 to
+0.2043 above the card's target of 1.1491. The smallest of those, +0.0570, is
2.6 times the gate. So the report CAN state that the decay costs the score.

**The decay schedules against each other mostly do not.** Of the 15 arm-to-arm
pairs, 5 fall under the gate: `dec_s22` vs `dec_s20` (0.0077), `dec_s20` vs
`dec_s24` (0.0142), `dec_s20` vs `dec_m099_fix` (0.0179), `dec_s24` vs
`dec_m099_fix` (0.0037) and `dec_s22` vs `dec_s24` (0.0219).

### One correction to the review

The review states that `dec_m080_r200`'s "0.024 lead over `dec_s22` sits
inside that spread". It does not, quite: the lead is **0.0241** and the gate is
**0.0219**, so it clears by 0.0022.

Do not read that as a rank. A range over three seeds is a crude estimator and
it runs low at small n. A lead that clears it by one tenth of itself is a weak
separation, not a result. The safe sentence stands either way: this card ranks
the decay against the no-decay reference, and it does not rank the decay
schedules against each other.

## 2. The reference is comparable: 11 of 11 items match

`scripts/reference_match.sh` writes `results/reference_match.tsv`.

1.1491 and 1.1507 come from `reports/2026-08-19_ema_momentum_k32/`, not from
any arm of this card. A gap between two studies is a result only when the two
measure the same thing. Every item matches:

| item | value |
|---|---|
| cell | `arm6_v2_combab_alignT` |
| rollout depth k | 32 |
| depth reduction | `mean` |
| align target | `teacher` |
| backbone stop | 40,000 steps |
| head steps | 30,000 |
| head seed | 20260722 |
| head encoder | `student` |
| head runner | `2026-08-08_rollout_depth/scripts/head_eval_bb.sh` |
| eval | 97 configs, B4, forecast-len 16 |
| score tag | `_bb40k_h30k_student` |

The head runner row is the strongest. It is not two settings that agree, it is
ONE file that both studies call. The align target is the same shape: it rides
the CELL in `run_leg_k.sh`, and both studies name that cell.

### The one weakness, and say it

`/home/jupyter/checkpoints_backup/cf-404` is DELETED. The sweep's head seed and
align target therefore rest on its scripts and its report, not on a surviving
qhead file. The `evidence` column of the table marks which rows are `script`
and which are `artefact`. This card's own side of both rows is an artefact: its
qhead files carry `_s20260722` in their names.
