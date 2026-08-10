# Rollout depth k: training the composed forecaster

At eval the forecaster runs on its own output, up to about 45 times. Training
never did. `--train-rollout-depth K` duplicates every f-bearing loss term at
depth 1..K, so training composes the operator the eval composes.

**The flag helps two of the four cells this study trained.** B1 improves
9.8% and B9 improves 17.9%, both meeting the card's primary criterion. A3
degrades 11.7%. B5 has no verdict.

Only one of those comparisons trained both of its sides on the same machine
and reproduced its published `k = 0` exactly, and that one is B1. Section 2
says why nothing else here is as strong.

Four of the card's 14 cells ran. The compute bought no more.

---

## 1. The answer

![depth response](plots/depth_response.png)

Two arms improve, one degrades, and one is undecided between its backbone
trainings.

The hatch is the thing to read first. A hatched bar is a delta whose `k = 0`
and `k = 3` sides trained on **different machines**. Section 6 finds that
every rented-box `k = 0` missed its published value, by up to 0.1169, and
every elisa one hit it. Two bars per head are not hatched. They are B1 and
B5·s2, and they are the only two comparisons in this study that change the
depth and nothing else.

The two shaded spans are not the same kind of thing. The narrow one is the
parents' pooled head-seed band, which bounds the head seed alone. The wide
one is the largest disagreement this study measured between two trainings of
one cell at one depth, and section 5 says what that span is and is not.

## 2. B1 is the study's one clean improvement

Three things have to hold before a delta here means the depth.

| | B9 | **B1** | B5·s2 | A3 |
|---|---|---|---|---|
| same machine on both sides | no | **yes** | yes | no |
| its own `k = 0` reproduces the published value | 0.0004 | **0.0000** | 0.0032 | 0.0294 |
| meets the card's primary criterion | yes | **yes** | no | no |

B1 is the only column with all three. B9's −17.9% is the study's largest
number, but its `k = 0` trained on elisa and its `k = 3` on a rented box, so
it carries a machine change as well as a depth change. B5·s2 holds the
machine fixed and does not improve. A3 fails all three.

So the study's strongest claim is B1's, and it is one training at one seed at
one stop with no `k = 3` replicate.

## 3. Where the change lands

![horizon split, student head](plots/horizon_split_student.png)

The mechanism predicts the gain concentrates on medium and long. For the two
arms that improve it does: B1 gains 15.2% on the 42 medium and long configs
against 5.4% on the 55 short ones, and B9 gains 24.4% against 12.6%. Every
one of those four intervals excludes zero (section 14).

The two arms that do not improve lose most on short, which is the opposite
pattern.

Teacher head: [`horizon_split_teacher.png`](plots/horizon_split_teacher.png).
It says the same thing.

## 4. B1 at k = 3 is the lowest number in the protocol, and it is unreplicated

![ladder](plots/ladder.png)

B1's star sits below its own published trajectory at every stop the parent
reached: 1.2025 at bb40k, 1.1616 at bb100k, 1.1652 at bb200k. It also sits
below the best published number of all 14 cells, A4's teacher head at 1.1544.

**1.0850 is one backbone training, at one seed, at bb40k, and no second run
of it exists.** The one replication this study did attempt reversed a sign.
Section 15 puts replicating it first.

Per domain, B1 at `k = 3` improves on six of seven and holds level on the
seventh: Econ/Fin −17%, Energy −14%, Transport −13%, Web/CloudOps −6%,
Nature −5%, Healthcare −2%, Sales +0%. It beats seasonal naive on three
domains (Nature 0.840, Sales 0.775, Transport 0.907).

![domain radar, student head](plots/domain_radar_student.png)

Each panel holds one arm against its own `k = 0`, and its title says whether
the two sides trained on one machine. B1 pulls inward on 6 of the 7 domains
and B9 on 5; B5·s2 pulls inward on 1 and A3 on none. Teacher head:
[`domain_radar_teacher.png`](plots/domain_radar_teacher.png).

## 5. One cell, three backbones: the machine or the seed

![seed spread](plots/seed_spread.png)

B5 trained twice at first: same code, same recipe, same head seed, same
eval. The two runs give opposite answers about the depth, and they land
0.1200 apart at `k = 0` and 0.0088 apart at `k = 3`. The disagreement is a
`k = 0` disagreement.

They differ by **two** things, not one. B5·s1 ran at seed 20260520 on a
rented RTX 5090. B5·s2 ran at seed 20260521 on elisa. So neither of them
says whether the seed or the machine moved the score, and section 6 says the
machine sorts every other row of this study.

B5·s3 is the third corner of that square: the seed of s1, the machine of s2.
It is the run that separates them, and it is the reason this section carries
no verdict on B5's depth yet.

Whichever way it lands, B5·s1's `k = 0` misses #379's published 1.2748 by
0.1169, which is three times the parents' pooled head-seed band. Its depth
delta stands on a baseline the parents do not recognise, so **the −5.1% that
row shows is retracted** and is marked ✗ wherever it still appears.

## 6. The code reproduces the published k = 0 on elisa and not on a rented box

![reproduction](plots/reproduction.png)

The card gates every group-B delta on this check, because a delta against a
published number crosses a code snapshot as well as the flag.

The rows sort on the machine and not on the seed. Every retrain on elisa
landed on its published value: B1 exactly, B9 0.0004 away, B5·s2 0.0032 away.
Both retrains on a rented box missed it: A3 by 0.0294 and B5·s1 by 0.1169.
Three of those five runs carry the same backbone seed, 20260520, and they
land 0.0000, 0.0004 and 0.1169 from their published values. The seed does not
sort them.

The `B5·pub` row is the control that changes the backbone instead of the
code. It takes #379's own published B5 checkpoint, trains this study's head
on it and runs this study's eval: 1.2751 against a published 1.2748. The
head and the eval are therefore not what moved.

Every delta in this report is against the same arm's own retrained `k = 0`.
No published number is used as a baseline.

## 7. The loss shape does not decide the sign

B1 and A3 train the same f-bearing term, `rep_only` + `L_align`, on the same
`arm6_v2 combab` arm. They differ in the EMA schedule.

| arm | EMA α | machine held | student Δ% | teacher Δ% |
|---|---|---|---|---|
| B1 | fixed 0.9 | yes | **−9.8%** | **−8.8%** |
| A3 | scheduled 0.9 → 1.0 | no | +11.7% | +11.0% |

Holding the shape fixed and changing the EMA regime flips the sign. So the
shape alone does not predict it.

The regime alone does not predict it either. B9, B1 and B5·s2 all hold α at
0.9, and B5·s2 does not improve. Four cells is not enough to name the factor
that decides this, and this report does not name one. A3's row also crosses
a machine, so it is not clean evidence for anything.

## 8. A3: the depth, not the weight it carries

![A3 depth against weight](plots/a3_depth.png)

Summing the depths multiplies the f-bearing term's weight against the f-free
terms by `k + 1`, so a `k = 3` run changes two things at once. The control
applies the ×4 re-weighting with no depth at all.

Re-weighting alone costs +0.0401, which is 28% of `k = 3`'s +0.1429. The
remaining +0.1028 is the depth. Re-weighting does not explain A3.

The ladder is not monotonic. `k = 1` scores 1.1995 against `k = 0`'s 1.2189,
an improvement of 1.6%; its bootstrap CI is [−0.0537, +0.0148], so it is
inside the noise rather than a win. `k = 3` then costs 11.7%. Whatever breaks
in A3 does not break at the first depth.

Both of these controls trained on elisa against a `k = 0` on a rented box.
Under section 6's machine effect that is worth up to 0.1169, which is more
than either control's own size, so read them as direction and not as
magnitude.

## 9. The composed operator does get more faithful

![rollout fidelity](plots/rollout_fidelity.png)

On #379's fixed diagnostic batch, `cos` between the rolled latent and the
true `h_{t+d}` for `d = 1..16`, with no head in the way. **Every arm improves
at `k = 3` at every one of the 16 depths, including the two that score
worse.** The smallest gain is A3's +0.003 and the largest is B5·s2's +0.545.

So the fixed-point approximation is not what failed. The flag does what it
says: it makes the model better at consuming its own output. On A3 and B5 a
more faithful latent rollout did not turn into a better forecast.

The batch is #379's committed `_latent_movement_batch.pt`, the same one the
two parent reports' latent-movement figures use. It is a fixed batch, not a
held-out one: nothing here establishes it is disjoint from
`gift-pretrain-full-4096 / small_v1`, which is what these backbones trained
on. It holds every curve on one scale, and that is what it is for.

## 10. Depth 0 does not pay, where the depth helps

![per-depth forecast error](plots/cos_err_depth.png)

`1 − cos(f^(j)_t, h_{t+1+j})` per step, one line per depth, against the
`k = 0` run's single line. The deeper depths carry more error than depth 0 in
every run, as expected.

The `k = 3` run's own depth-0 curve is the diagnostic. On B9 and B1 it sits
**below** the `k = 0` run's: depth 0 got better, not worse. On A3 and B5 it
sits above. The sign of that gap matches the sign of the eval result in all
four cells.

No run destabilised. Training loss is flat on every depth run
([`per_run_loss.png`](plots/per_run_loss.png)), and the latent-movement and
dimension-usage diagnostics show nothing unusual
([`latent_movement.png`](plots/latent_movement.png),
[`dim_usage_per_arm.png`](plots/dim_usage_per_arm.png),
[`cos_error_per_arm.png`](plots/cos_error_per_arm.png)).

## 11. Which head encoder to train on does not matter

![encoder delta](plots/encoder_delta.png)

Teacher minus student, per arm per depth. Every value is inside ±0.0198,
which is 0.52 of the head-seed band. The choice does not change any
conclusion here.

## 12. What the depth costs

The extra depths cost forward and backward passes, and how much depends on
the loss shape. A shape whose `f` sits in the numerator and in every
denominator family rebuilds all of that per depth. `L_align` has no
denominator.

The pooled shape pays **+157%** for three extra depths and the `L_align`
shape pays **+13%**. Section 14 has both, per run, with their provenance.

**Only two of this study's eleven backbones can give a ratio.** elisa ran two
of this study's backbones on GPU 0 at a time and trained heads beside them,
so every elisa run was contended for 43% to 100% of its life and its median
is not a cost of the depth. The four rented-box runs had a card each.
[`results/steptime_solo.csv`](results/steptime_solo.csv) carries the
contention per run.

Both surviving ratios cross two rented boxes of the same GPU model. The one
measurement that holds the card fixed is a controlled probe: B5 alternating
`k = 0` and `k = 3` on elisa's GPU 1, 3 reps of 600 steps, 190.2 ms against
509.9 ms, +168%. That card carried another session's job throughout
(8946 MiB at the start, 44% mean utilisation), so the probe alternates on a
shared card rather than owning one; alternating is what makes it comparable,
not exclusivity.

B9 gives no ratio for the same reason as B1 and B5·s2: its `k = 0` ran
contended on elisa. Its `k = 3` ran alone on a rented RTX 4090 at 425.2 ms.
Its `L_pred` puts `f` in the numerator and in every denominator family, so
its overhead should be the pooled kind rather than the `L_align` kind.

## 13. What did not run

Ten of the card's 14 cells never trained: **A1, A2, A4, B2, B3, B4, B6, B7,
B8, B10**. The four that did are A3, B1, B5 and B9.

Every run stopped at bb40k. No cell reached bb100k or bb200k, so the card's
extend rule never fired and every number here is one stop.

The card's secondary criterion asks for the full-97 score to beat `k = 0` by
more than the head-seed band. B9 and B1 clear it. It also asks for a bb100k
and bb200k comparison, which this study cannot give.

## 14. Tables

Generated from the score files, the per-config CSVs and the driver logs by
`scripts/tables.py`, which writes this section and
[`results/scores.md`](results/scores.md) from the same pass.

<!-- TABLES:BEGIN -->

### Coverage

The card names 14 cells. This study trained **4 of them**: A3, B1, B5, B9. It never ran **10**: A1, A2, A4, B2, B3, B4, B6, B7, B8, B10.

| cell | f-bearing term | EMA α | depths trained |
|---|---|---|---|
| A1 | — | — | **never ran** |
| A2 | — | — | **never ran** |
| A3 | rep_only + L_align | scheduled 0.9 -> 1.0 | k = 0, k = 1, k = 3 |
| A4 | — | — | **never ran** |
| B1 | rep_only + L_align | fixed 0.9 | k = 0, k = 3 |
| B2 | — | — | **never ran** |
| B3 | — | — | **never ran** |
| B4 | — | — | **never ran** |
| B5 | pooled xshh_allt | fixed 0.9 | k = 0, k = 3 |
| B6 | — | — | **never ran** |
| B7 | — | — | **never ran** |
| B8 | — | — | **never ran** |
| B9 | split L_pred | fixed 0.9 | k = 0, k = 3 |
| B10 | — | — | **never ran** |

Every trained stop is bb40k. No cell reached bb100k or bb200k, so the card's extend rule never fired and this study publishes one stop.

### Reproduction of the published k = 0

Same cell, same recipe, same head seed 20260722, same 97-config B4 eval, student head. The rows are sorted by the machine, because that is what the check separates on.

| backbone | seed | machine | published k = 0 | retrained k = 0 | \|Δ\| | verdict (threshold 0.0002) |
|---|---|---|---|---|---|---|
| B1 | 20260520 | elisa | 1.2025 | 1.2025 | 0.0000 | PASS |
| B9 | 20260520 | elisa | 1.5579 | 1.5583 | 0.0004 | at printed precision |
| B5·s2 | 20260521 | elisa | 1.2748 | 1.2716 | 0.0032 | FAIL |
| A3 | 20260520 | vast box d | 1.1895 | 1.2189 | 0.0294 | FAIL |
| B5·s1 ✗ | 20260520 | vast box d | 1.2748 | 1.3917 | 0.1169 | FAIL |
| B5·pub | 20260520 | #379's box | 1.2748 | 1.2751 | 0.0003 | at printed precision |

The parents print four decimals, so a difference below 0.0005 is the smallest the published table can resolve. The card's gate of 0.0002 is stricter than that.

`B5·pub` is not a training: it takes #379's own published B5 checkpoint and puts this study's head and eval on it, so its row bounds the head and the eval rather than the trainer. `B5·s3` is a training, at the protocol seed, on elisa.

### Depth response, against each arm's own k = 0

| arm | seed | machine held | head | k | k = 0 | this k | Δ | all | short | med+long | criterion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B9 | 20260520 | no, elisa → vast box c | student | 3 | 1.5583 | 1.2791 | -0.2791 | -17.9% | -12.6% | -24.4% | **MET** |
| B9 | 20260520 | no, elisa → vast box c | teacher | 3 | 1.5599 | 1.2728 | -0.2871 | -18.4% | -12.8% | -25.2% | **MET** |
| B1 | 20260520 | yes, elisa | student | 3 | 1.2025 | 1.0850 | -0.1175 | -9.8% | -5.4% | -15.2% | **MET** |
| B1 | 20260520 | yes, elisa | teacher | 3 | 1.2001 | 1.0948 | -0.1053 | -8.8% | -5.1% | -13.4% | **MET** |
| B5·s1 ✗ | 20260520 | no, vast box d → vast box a | student | 3 | 1.3917 | 1.3204 | -0.0713 | -5.1% | -6.4% | -3.4% | not met |
| B5·s1 ✗ | 20260520 | no, vast box d → vast box a | teacher | 3 | 1.3719 | 1.3216 | -0.0503 | -3.7% | -4.4% | -2.6% | not met |
| B5·s2 | 20260521 | yes, elisa | student | 3 | 1.2716 | 1.3292 | +0.0575 | +4.5% | +7.0% | +1.4% | not met |
| B5·s2 | 20260521 | yes, elisa | teacher | 3 | 1.2661 | 1.3260 | +0.0599 | +4.7% | +8.1% | +0.5% | not met |
| A3 | 20260520 | no, vast box d → elisa | student | 1 | 1.2189 | 1.1995 | -0.0195 | -1.6% | -2.6% | -0.2% | not met |
| A3 | 20260520 | no, vast box d → vast box b | student | 3 | 1.2189 | 1.3618 | +0.1429 | +11.7% | +17.1% | +5.1% | not met |
| A3 | 20260520 | no, vast box d → elisa | teacher | 1 | 1.2184 | 1.2063 | -0.0121 | -1.0% | -1.5% | -0.4% | not met |
| A3 | 20260520 | no, vast box d → vast box b | teacher | 3 | 1.2184 | 1.3521 | +0.1337 | +11.0% | +15.8% | +4.9% | not met |

Criterion, from the card: medium+long (42 configs) at least 5% better, short (55 configs) losing less than 2%.

`machine held` = did the two sides train on the same box. A `no` row carries a machine change as well as a depth change, and the reproduction table separates on the machine at up to 0.1169.

✗ marks a retracted row: B5·s1's `k = 0` misses its published value by 0.1169 and trained on a rented box; its depth delta is retracted.

Head-seed band ±0.0384 (`ema_sched_ladder.md`, pooled). It bounds the head seed alone. It does not bound a retraining, which the B5 table below measures at 0.1200.

### Paired dataset-cluster bootstrap, per horizon subset

The resampling unit is the dataset: `<ds>/short`, `/medium` and `/long` are three configs of one series and are not independent draws. 95% percentile interval over 10,000 resamples.

| arm | head | k | subset | n | Δ | 95% CI | resamples improved |
|---|---|---|---|---|---|---|---|
| B9 | student | 3 | all | 97 | -0.2791 | [-0.3548, -0.1980] | 100.0% |
| B9 | student | 3 | short | 55 | -0.1736 | [-0.2470, -0.1038] | 100.0% |
| B9 | student | 3 | medium_long | 42 | -0.4472 | [-0.5655, -0.3382] | 100.0% |
| B9 | teacher | 3 | all | 97 | -0.2871 | [-0.3644, -0.2032] | 100.0% |
| B9 | teacher | 3 | short | 55 | -0.1751 | [-0.2501, -0.1018] | 100.0% |
| B9 | teacher | 3 | medium_long | 42 | -0.4670 | [-0.5952, -0.3523] | 100.0% |
| B1 | student | 3 | all | 97 | -0.1175 | [-0.1801, -0.0615] | 100.0% |
| B1 | student | 3 | short | 55 | -0.0556 | [-0.1017, -0.0184] | 99.9% |
| B1 | student | 3 | medium_long | 42 | -0.2244 | [-0.3504, -0.1243] | 100.0% |
| B1 | teacher | 3 | all | 97 | -0.1053 | [-0.1661, -0.0515] | 100.0% |
| B1 | teacher | 3 | short | 55 | -0.0523 | [-0.0980, -0.0146] | 99.7% |
| B1 | teacher | 3 | medium_long | 42 | -0.1963 | [-0.3129, -0.1047] | 100.0% |
| B5·s1 ✗ | student | 3 | all | 97 | -0.0713 | [-0.1327, -0.0267] | 100.0% |
| B5·s1 ✗ | student | 3 | short | 55 | -0.0848 | [-0.1732, -0.0068] | 98.3% |
| B5·s1 ✗ | student | 3 | medium_long | 42 | -0.0504 | [-0.0972, -0.0137] | 99.7% |
| B5·s1 ✗ | teacher | 3 | all | 97 | -0.0503 | [-0.0965, -0.0108] | 99.4% |
| B5·s1 ✗ | teacher | 3 | short | 55 | -0.0571 | [-0.1215, +0.0086] | 95.8% |
| B5·s1 ✗ | teacher | 3 | medium_long | 42 | -0.0395 | [-0.0882, +0.0028] | 96.6% |
| B5·s2 | student | 3 | all | 97 | +0.0575 | [+0.0173, +0.1094] | 0.2% |
| B5·s2 | student | 3 | short | 55 | +0.0809 | [+0.0231, +0.1549] | 0.2% |
| B5·s2 | student | 3 | medium_long | 42 | +0.0199 | [-0.0215, +0.0745] | 19.0% |
| B5·s2 | teacher | 3 | all | 97 | +0.0599 | [+0.0214, +0.1105] | 0.1% |
| B5·s2 | teacher | 3 | short | 55 | +0.0925 | [+0.0345, +0.1701] | 0.0% |
| B5·s2 | teacher | 3 | medium_long | 42 | +0.0074 | [-0.0268, +0.0543] | 38.7% |
| A3 | student | 1 | all | 97 | -0.0195 | [-0.0537, +0.0148] | 86.9% |
| A3 | student | 1 | short | 55 | -0.0294 | [-0.0652, +0.0007] | 97.1% |
| A3 | student | 1 | medium_long | 42 | -0.0029 | [-0.0565, +0.0628] | 55.8% |
| A3 | student | 3 | all | 97 | +0.1429 | [+0.0893, +0.2122] | 0.0% |
| A3 | student | 3 | short | 55 | +0.1899 | [+0.1254, +0.2739] | 0.0% |
| A3 | student | 3 | medium_long | 42 | +0.0698 | [+0.0226, +0.1415] | 0.0% |
| A3 | teacher | 1 | all | 97 | -0.0121 | [-0.0479, +0.0261] | 74.0% |
| A3 | teacher | 1 | short | 55 | -0.0163 | [-0.0596, +0.0275] | 77.9% |
| A3 | teacher | 1 | medium_long | 42 | -0.0052 | [-0.0572, +0.0602] | 59.5% |
| A3 | teacher | 3 | all | 97 | +0.1337 | [+0.0839, +0.2004] | 0.0% |
| A3 | teacher | 3 | short | 55 | +0.1760 | [+0.1177, +0.2537] | 0.0% |
| A3 | teacher | 3 | medium_long | 42 | +0.0673 | [+0.0197, +0.1414] | 0.0% |

### One cell, three backbones

B5 (`arm4_combab_fix09`) trained three times on one recipe, one code snapshot, one head seed and one eval. They differ by backbone seed and by machine, and each contrast below names which of the two it changes.

| backbone | seed | machine | k = 0 | k = 3 | k = 3 − k = 0 |
|---|---|---|---|---|---|
| B5·s1 ✗ | 20260520 | a rented box | 1.3917 | 1.3204 | -0.0713 |
| B5·s2 | 20260521 | elisa | 1.2716 | 1.3292 | +0.0575 |

| contrast | what changes | k | Δ | 95% CI |
|---|---|---|---|---|
| B5·s1 against B5·s2 | the seed AND the machine | 0 | -0.1200 | [-0.1825, -0.0742] |
| B5·s1 against B5·s2 | the seed AND the machine | 3 | +0.0088 | [-0.0306, +0.0520] |

Student head, 97 configs. `B5·s3` is this study's answer to the third row: it holds `B5·s1`'s seed and `B5·s2`'s machine.

### One loss shape, two EMA regimes

B1 and A3 train the same f-bearing term, `rep_only` + `L_align`, on the same `arm6_v2 combab` arm. They differ in the EMA schedule — and, since A3's two depths trained on two boxes, in the machine as well.

| arm | EMA α | machine held | head | k = 0 | k = 3 | Δ | Δ% |
|---|---|---|---|---|---|---|---|
| B1 | fixed 0.9 | yes, elisa | student | 1.2025 | 1.0850 | -0.1175 | -9.8% |
| B1 | fixed 0.9 | yes, elisa | teacher | 1.2001 | 1.0948 | -0.1053 | -8.8% |
| A3 | scheduled 0.9 -> 1.0 | no | student | 1.2189 | 1.3618 | +0.1429 | +11.7% |
| A3 | scheduled 0.9 -> 1.0 | no | teacher | 1.2184 | 1.3521 | +0.1337 | +11.0% |

### A3: is the damage the depth, or the weight?

Summing the depths multiplies `L_align`'s weight against the f-free terms by k + 1. The `L_align x4` row applies that re-weighting at k = 0, with no depth at all.

| head | k = 0 | k = 0, `L_align` x4 | k = 1 | k = 3 | share of the k = 3 damage the re-weighting explains |
|---|---|---|---|---|---|
| student | 1.2189 | 1.2590 | 1.1995 | 1.3618 | 28% |
| teacher | 1.2184 | 1.2558 | 1.2063 | 1.3521 | 28% |

Every column trained on a different box from at least one other. A3_k0: vast box d · G3_A3_k0_aw4: elisa · G3_A3_k1: elisa · A3_k3: vast box b.

### What the depth costs

Median `fwd + bwd` per step, from each run's own trainer log. A median is a cost of the depth only where the run had the card to itself, so the table says which did. `run_provenance.py` reads that off the driver logs and [`results/steptime_solo.csv`](results/steptime_solo.csv) carries it per run.

| arm | f-bearing term | k | machine | card | fwd+bwd | alone? |
|---|---|---|---|---|---|---|
| B9 | split L_pred | 0 | elisa | RTX 4090 | 212.6 ms, shared | no — another backbone for 96% of the run; head training for 4% of it |
| B9 | split L_pred | 3 | vast box c | RTX 4090 | 425.2 ms | yes |
| B1 | rep_only + L_align | 0 | elisa | RTX 4090 | 178.6 ms, shared | no — another backbone for 100% of the run; head training for 100% of it |
| B1 | rep_only + L_align | 3 | elisa | RTX 4090 | 235.1 ms, shared | no — another backbone for 68% of the run; head training for 100% of it |
| B5·s1 | pooled xshh_allt | 0 | vast box d | RTX 5090 | 117.6 ms | yes |
| B5·s1 | pooled xshh_allt | 3 | vast box a | RTX 5090 | 301.9 ms | yes |
| B5·s2 | pooled xshh_allt | 0 | elisa | RTX 4090 | 201.1 ms, shared | no — another backbone for 100% of the run; head training for 98% of it |
| B5·s2 | pooled xshh_allt | 3 | elisa | RTX 4090 | 500.9 ms, shared | no — another backbone for 43% of the run; head training for 100% of it |
| A3 | rep_only + L_align | 0 | vast box d | RTX 5090 | 115.9 ms | yes |
| A3 | rep_only + L_align | 1 | elisa | RTX 4090 | 214.7 ms, shared | no — another backbone for 72% of the run; head training for 100% of it |
| A3 | rep_only + L_align | 3 | vast box b | RTX 5090 | 131.5 ms | yes |

The ratios that survive that test:

| arm | f-bearing term | k = 0 | k = 3 | change | both sides |
|---|---|---|---|---|---|
| B5·s1 | pooled xshh_allt | 117.6 ms | 301.9 ms | +157% | vast box d → vast box a |
| A3 | rep_only + L_align | 115.9 ms | 131.5 ms | +13% | vast box d → vast box b |

No ✗ in this table. The retraction is of B5·s1's depth delta, which rests on a `k = 0` the parents do not recognise; its wall clock is unaffected.


<!-- TABLES:END -->

Full paired dataset-cluster bootstraps, including the per-domain splits:
[`results/bootstrap.csv`](results/bootstrap.csv). The resampling unit is the
dataset, because `<ds>/short`, `/medium` and `/long` are three configs of one
series and are not independent draws.

## 15. What to run next

1. **Replicate B1 at `k = 3` on a second backbone seed, on elisa.** It is the
   study's best result and the only unreplicated side of it. One training.
2. **B1 to bb100k and bb200k.** The published B1 trajectory reaches 1.1616 at
   bb100k; whether the depth's 0.1175 survives that is the question the
   card's extend rule would have answered.
3. **`k = 1` and `k = 2` on B1 and B9.** A3's ladder is not monotonic, so the
   best depth for the arms that improve is not known either.

Every one of those runs both of its sides on one machine. Do not spend the
next run on the ten cells that did not train.

## Protocol

Backbone `d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3,
batch_size=64`, seed 20260520 (B5's second training uses 20260521); dataset
`gift-pretrain-full-4096 / small_v1`; `--ema-embedding --ema-encoder`. Group
B holds EMA α at 0.9; group A raises it linearly from 0.9 to 1.0 by step
100k. Every cell starts fresh at step 0. Two heads per checkpoint, student
and teacher, trained separately on their own encoder, 15,000 steps, head seed
20260722, `--grad-clip 1.0` on the head for comparability with the parents.
97 GIFT-Eval configs, official B4 strategy, forecast horizon 16, one shared
seasonal-naive denominator file.

**Deviation from the card.** The card's default is to compute the h-anchored
negative families once and reuse them unshifted at every depth. This
implementation takes the card's stated alternative and **shifts them with the
depth**, so a depth-`j` copy is a literal copy of the depth-0 objective under
one rule: every `h` index moves by `j`. It touches exactly one of the 14
cells. B5 (`arm4`, pooled `xshh_allt`) is the only cell whose f-bearing
denominator holds h-anchored families; B9's `L_pred` denominator is
f-anchored only, and the other twelve cells' f-bearing term is `L_align`,
which has no denominator.

`bash scripts/make_report_assets.sh` rebuilds every figure and table in this
report from the artefacts under `results/`. Operational events are in
[`results/execution_log.md`](results/execution_log.md).
