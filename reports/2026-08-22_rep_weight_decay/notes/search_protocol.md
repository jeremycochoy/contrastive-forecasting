# How the arms get chosen

The arms are chosen one round at a time. Each round reads the scores of the
round before it. `scripts/arms.tsv` is a catalogue of candidate schedules, not
a queue.

The decay never moves. It is the card's decay: one extra factor in front of
`L_rep`, linear from 1.0 to 0.0 at step 10,000. The cell never moves either:
k = 32, `mean`, align on the EMA teacher. Only the EMA schedule moves.

One seed per schedule, 20260520. The budget of eight backbones buys eight
schedules.

## Round 0, measured

| schedule | momentum at 40k | decay | GM-Relative MASE |
|---|---|---|---|
| 0.9 to 1.0 at 100k | 0.940 | no | 1.1491, 1.1507 |
| 0.9 to 1.0 at 100k | 0.940 | yes | 1.2670, 1.2593 |

The decay costs 0.11 at the sweep's best schedule. The seed spread of the pair
with the decay is 0.008, and of the pair without it 0.0016. So the cost is
large against both.

## Round 1, running

The question: does more EMA, or less EMA, recover the 0.11?

Round 1 takes the two ends of the axis, so one round tells the direction.

| arm | schedule | momentum at 40k |
|---|---|---|
| `dec_m080_r200` | 0.8 to 1.0 at 200k | 0.840 |
| `dec_m099_fix` | 0.99 fixed | 0.990 |

Round 2 refines toward whichever end moves the score down. If neither end
moves it, the decay is what costs the 0.11, and the remaining backbones stay
unspent.
