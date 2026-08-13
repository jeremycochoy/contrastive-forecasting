«Agent ExperimentRunner claude-opus-5 writing»

**Experiment directory:** `reports/2026-08-08_rollout_depth/`
(results `results/`, plots `plots/`, scripts `scripts/`, run tree `/home/jupyter/cf373_r3`)

## Coverage, 07:20Z

```
cell      40k S     40k T     100k S    100k T    200k S    200k T
A1        1.1305    1.1318    1.1676    1.1565    stop      stop
A2        1.2735    1.2753    1.2479    1.2514    1.2507    1.2500
A3        1.3618    1.3521    1.3010    1.3151    run       run
A4        1.0862    1.0855    1.0801    1.0874    bb-run    stop
B1        1.0850    1.0948    1.0881    1.0897    run       run
B2        1.3976    1.4041    1.3443    1.3117    1.2904    run
B3        1.1305    1.1343    1.1676    1.1618    stop      stop
B4        1.3334    1.3339    1.2804    1.2748    1.3182    1.3202
B5        1.3204    1.3216    1.3383    1.3428    stop      stop
B6        1.2297    1.2184    1.2151    1.2110    1.2207    1.2339
B7        1.2617    1.2444    1.3205    1.2780    stop      stop
B8        1.2857    1.2865    1.3157    1.3239    stop      stop
B9        1.2791    1.2728    1.3299    1.3094    stop      stop
B10       1.2669    1.2730    1.2403    1.2499    1.2624    1.2440

deliverables 71   done 65   running 6   queued 0   NOT STARTED 0   (+13 stops)
done=number in hand  run=own head/eval running  bb-run=backbone training now
stop=the five cells the rule holds at 100k, plus A4's teacher at 200k
```

## Runs completed

```
queue     47 jobs   37 done   6 running   4 queued   0 failed
backbone  8/9 done; A4 157.4k of 200k at 3.05 sps, ETA ~11:15Z
heads     18/20 done at 30,000 steps, seed 20260722, --grad-clip 1.0
evals     11/18 done over 97 GIFT-Eval configs, strategy B4, horizon 16
```

## Headline numbers

**200k does not pay.** Five cells now carry a 100k -> 200k move, and four of
the five get worse on the student:

```
cell    100kS   200kS       dS   100kT   200kT       dT
A2     1.2479  1.2507  +0.0028  1.2514  1.2500  -0.0014
B2     1.3443  1.2904  -0.0539  1.3117       -        -
B4     1.2804  1.3182  +0.0378  1.2748  1.3202  +0.0454
B6     1.2151  1.2207  +0.0056  1.2110  1.2339  +0.0229
B10    1.2403  1.2624  +0.0221  1.2499  1.2440  -0.0059
```

B2 is the one clear gain, -0.0539, and B2 is the worst cell at 100k. The best
cells stay where they were at 100k: **A4 1.0801** and **B1 1.0881** student.

**The head budget differs by column, and the report now says so.** Every bb40k
head trains 15,000 steps, the round-1 standard; every bb100k and bb200k head
trains 30,000. Read off the head losses CSVs, not from memory. So a comparison
DOWN a column is head-matched and a comparison ACROSS columns is not — part of
any 40k -> 100k move is the head's own extra 15,000 steps. The Protocol
section said 15,000 for every head; corrected.

This also settles the card's note on B1's 40k. B1's 15,000-step head is the
same budget every other cell's 40k head carries, so B1 is not the exception:
no cell's 40k number is head-matched to its own 100k number.

**A1/B3 stays closed.** `arm5_combab_alignS` aligns on the student and passes
no `--moco-rep-keys`, so the EMA regime moves the teacher alone
(`train.py:1861`). Student weights equal, 110/110 tensors, max abs diff 0 at
both stops; 0 of 97 GIFT-Eval rows differ on the student, all 97 differ on the
teacher. One trajectory by construction, not by a path fault. The report does
not read A1 against B3 on the student column.

**Naming.** 65/65 cell scores under `score_<CELL>_k3_bb<stop>k_<head>.txt`.

## New: the success tail has an owner

`q_super.sh` covers a dead dispatcher and `q_guard.sh` covers the credit floor.
Nothing covered the round simply finishing. Added `scripts/q_finish.sh`,
detached: it waits for every backbone and head to go terminal, waits one full
sync tick, then verifies EVERY artefact a done job produced on this disk by
name and size — backbone plus optimizer sidecar over 4 MB, head final over
300 KB — and destroys the box only if that passes. One miss and the box lives
and `results/FINISH_BLOCKED` says so. Then it waits for the queue to drain,
takes the last publish tick and posts the completion comment here. Gate
dry-run at 07:18Z: 8 backbones and 16 heads verified, `bad=0`.

This matters for money: the box holds backbones and heads, every eval runs on
elisa cores, so the box is dead weight from the last head to the last eval,
about 1.5 h at $0.81/h.

## Spend

Credit **$15.00** at 07:15Z. Box 47557391 (2x RTX 4090, on-demand, rel 0.9973)
has run 16.0 h and spent $13.04 at $0.8144/h. It is needed until A4's student
head lands, about 6 h and about $4.9, which leaves about $10 against the $5.50
guard floor. Every eval runs on elisa cores and costs nothing.
