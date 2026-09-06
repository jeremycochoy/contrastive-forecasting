# The Fable opinion on the cell, and what the user decided

The card of #409 says: "Ask a Fable agent for an opinion before you choose the
next backbone." This file holds that opinion, as it came back.

Model: `claude-fable-5`. Date: 2026-08-22. The agent read
`reports/2026-08-19_ema_momentum_k32/`, `reports/2026-08-04_ema_sched_ladder/`,
`reports/2026-08-08_rollout_depth/`, `reports/2026-08-15_rollout_depth_k16_8_32/`
and `reports/2026-08-08_rollout_depth/scripts/run_leg_k.sh`.

## The decision, which overrules the opinion

The user rejected the opinion's cell. The card gives the cell and the decay,
and the study runs those:

- **The cell is k = 32 under `mean`, with `--align-target teacher`**, at an EMA
  momentum of 0.9 rising to 1.0 at step 100,000. That is the sweep's best arm,
  `r100_09`. The opinion's k = 3 answers a different question.
- **One decay: 1.0 falling linearly to 0.0 at step 10,000.** No floor at 0.5,
  at 0.2, or at any other value. No second shape.
- **No control arm.** The sweep already scored this cell at two seeds, 1.1491
  and 1.1507.
- **Every backbone goes to the decay arm, at its own seed.**

So sections 1, 3 and 4 below are VOID. Their floors come from a 1:4
repel-to-pull ratio that k = 3 under `sum` produces, and this cell does not
produce it. Sections 2 and 5 still hold: the EMA schedule does not move, the
AUC gate runs, and seed 20260521 stays out.

Kept below because the reasoning that was rejected is worth reading once.

---

## 1. Cell: (a), k = 3 `arm6_v2_combab_alignS` — REJECTED

Three reasons. First, information lands on the frontier. The issue wants "a new
best". The k = 32 cell's best (1.1491) sits 0.063 above the k = 3 cell at the
same 40k stop (1.0862). No plausible decay gain closes that from the k = 32
side. Second, the premise transfers: I recomputed the sweep's loss CSVs; L_rep
is 92-93 percent of the total and flat from step 100 (11.54 to 11.63 over 40k
steps) while L_align moves. Both cells share
`cosine_similarity_batch_rep_only`, so this is a property of the objective, not
of k = 32. Third, k = 3 trains 4 rollout copies per step, not 33, so eight
backbones fit the budget.

Carry one asymmetry: k = 3 runs the default `sum` reduction, so its align term
is 4 summed copies. Baseline repel:align effective weight is 1:4, not ~1:1 as
at k = 32 mean. This sets the floors below.

## 2. EMA schedule: keep 0.9 to 1.0 over 100k — KEPT

Do not change it. It is already the k = 3 frontier cell's schedule
(`run_leg_k.sh` default), so the decay is the only moved flag against the
control. It is also the k = 32 sweep winner. And after the decay completes, the
teacher leaves the alignS loss entirely (it enters only through the MoCo keys
inside L_rep), so a new schedule cannot act after step 10k in seven of eight
arms.

## 3. The eight arms — REJECTED

The opinion proposed two control arms, three floors (0.0, 0.2, 0.5) and one
teacher-target arm, at two seeds. The card allows one decay shape and no
control, so the study runs six seeds of the single decay instead. See
`scripts/arms.tsv`.

## 4. Risk — REJECTED as stated

The opinion argued from three summed k = 16 / 8 / 32 arms that ran repel:align
ratios of 1:9, 1:17 and 1:33 from step 0 and lost the contrastive task at steps
4,404, 347 and 1,343. It concluded that a decay through 1:9 near step 5,600 is
the danger, and that a floor at 0.5 or 0.2 is safer than 0.0.

That arithmetic is the `sum` reduction's. This cell reduces by the MEAN of 33
align copies, so the baseline ratio is about 1:1 and the walk is different. The
risk the card keeps is the plain one: past step 10,000 the weight is 0.0, so
nothing pushes the representations apart. The AUC gate watches it.

## 5. Mistakes in the issue — KEPT in part

- "The best score today is 1.1491." That is the best of the SWEEP, not of the
  project. The k = 3 cell scores 1.0862 at the same stop and 1.0660 at 200,000
  steps. The card measures the k = 32 cell, so 1.1491 is the number its arms
  are read against.
- "Most of the objective stopped learning." Magnitude is not gradient. L_rep at
  tau_rep 1.0 is a logsumexp over thousands of negatives; about 11.6 is mostly
  its log-N constant floor. Flat means equilibrium, not inert. The sweep
  already tested a rebalance: w3_s08 (align x3) scored worse, 1.2060 against
  1.1782, AUC 0.957 to 0.936. The issue does not cite it.
- Decay ends at 10k but runs stop at 40k: a collapsed arm burns 30k dead steps.
  Gate each arm on the trainer AUC column (500-step median below 0.55: kill,
  respend the budget). The study does this, in `scripts/auc_guard.sh`.
- Keep seed 20260521 out of treated arms. It collapsed once at full weight; a
  collapse there is unreadable. The study does this.
