# How the arms get chosen

The search chooses its arms one round at a time. Each round reads the scores
of the round before it. `scripts/arms.tsv` is a catalogue of candidate arms, not a
queue. Some of its rows never ran.

The cell never moves: k = 32, `mean`, align on the EMA teacher. The decay keeps
one shape, one extra factor in front of `L_rep` that falls linearly from 1.0 to
0.0. Two things move. The EMA schedule is the first axis. The length of the
decay ramp is the second, and it is column 5 of the catalogue.

The backbone seed is 20260520. A round can spend a backbone on a REPEAT SEED in
place of a new treatment, because a repeat gives the error bar of a headline
number.

## Round 0, measured

| schedule | momentum at 40k | ramp | GM-Relative MASE |
|---|---|---|---|
| 0.9 to 1.0 at 100k | 0.940 | no decay | 1.1491, 1.1507 |
| 0.9 to 1.0 at 100k | 0.940 | 10,000 | 1.2670, 1.2593, 1.2812 |

The decay costs 0.12 at the sweep's best schedule. The three seeds under the
decay span 0.0219, and the two without it span 0.0016. So the cost is large
against both.

## Round 1, measured. Does more EMA, or less EMA, recover the cost?

Round 1 takes the two ends of the axis, so one round tells the direction.

| arm | momentum at 40k | GM-Relative MASE |
|---|---|---|
| `dec_m080_r200` | 0.840 | 1.2352 |
| `dec_m099_fix` | 0.990 | 1.2849 |

The fast end beat every seed of round 0. The slow end lost to all of them. So
round 2 goes further down the fast end.

## Round 2, measured. Where does the fast end turn?

| arm | momentum at 40k | GM-Relative MASE |
|---|---|---|
| `dec_m070_fix` | 0.700 | 1.3534 |
| `dec_m050_fix` | 0.500 | lost the contrastive task at step 10,162 |

0.840 is the turn, and it still loses to the reference of 1.1491 by 0.0861.
That is four times the seed range. No schedule on this axis closes that gap, so
the EMA axis stops here and five of its rows stay unspent.

## Round 3, measured. Does the decay cost less when it takes longer?

Round 3 moves the RAMP at the best schedule, 0.8 to 1.0 at 200k, and holds the
seed.

| arm | ramp | GM-Relative MASE |
|---|---|---|
| `dec_ramp5k_m080` | 5,000 | 1.2727 |
| `dec_m080_r200` | 10,000 | 1.2352 |
| `dec_ramp20k_m080` | 20,000 | 1.3178 |
| `dec_ramp30k_m080` | 30,000 | 1.3623 |

The card's own 10,000 steps is the best of the four ramps.

## Round 4, running. How large is the error bar on the headline?

Both axes hold their best value inside the tested range, so the headline is
`dec_m080_r200` at 1.2352. `dec_m080_r200_s24` repeats that arm at seed
20260524 and gives its error bar.
