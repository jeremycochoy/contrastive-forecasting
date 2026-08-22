# The Fable opinion on the cell, the schedule and the eight arms

The card of #409 says: "Ask a Fable agent for an opinion before you choose the
next backbone." This file holds that opinion, as it came back, so the arms
table beside it can be read against the reasoning that made it.

Model: `claude-fable-5`. Date: 2026-08-22. The agent read
`reports/2026-08-19_ema_momentum_k32/`, `reports/2026-08-04_ema_sched_ladder/`,
`reports/2026-08-08_rollout_depth/`, `reports/2026-08-15_rollout_depth_k16_8_32/`
and `reports/2026-08-08_rollout_depth/scripts/run_leg_k.sh`.

---

## 1. Cell: (a), k = 3 `arm6_v2_combab_alignS`

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

## 2. EMA schedule: keep 0.9 to 1.0 over 100k

Do not change it. It is already the k = 3 frontier cell's schedule
(`run_leg_k.sh` default), so the decay is the only moved flag against the
control. It is also the k = 32 sweep winner. And after the decay completes, the
teacher leaves the alignS loss entirely (it enters only through the MoCo keys
inside L_rep), so a new schedule cannot act after step 10k in seven of eight
arms.

## 3. The eight arms

All: k = 3, sum, EMA 0.9 to 1.0 at 100k, 40k backbone steps, 30k head, head
seed 20260722. Decay is linear over steps 0 to 10k. `--rep-loss-weight` exists
but is static; add a small ramp flag like the EMA ramp.

| # | arm | change vs control | backbone seed | buys |
|---|---|---|---|---|
| 1 | ctrl_s20 | none (w = 1.0) | 20260520 | in-study baseline on this path and head budget (published 1.0862 used a 15k head) |
| 2 | ctrl_s24 | none | 20260524 | k = 3 backbone-seed spread, never measured; the denominator of every delta |
| 3 | dec0_s20 | w to 0.0 | 20260520 | the issue's hypothesis |
| 4 | dec0_s24 | w to 0.0 | 20260524 | repeat; collapse frequency across seeds |
| 5 | flr05_s20 | w to 0.5 | 20260520 | probes the unmeasured 1:4-to-1:8 gap; likeliest score gain |
| 6 | flr05_s24 | w to 0.5 | 20260524 | repeat of the likely headline arm |
| 7 | flr02_s20 | w to 0.2 | 20260520 | with 0.0/0.2/0.5/1.0, a dose-response that brackets the collapse boundary |
| 8 | dec0T_s20 | align target teacher, w to 0.0 | 20260520 | BYOL form; the arm most likely to keep AUC alive at w = 0 |

## 4. Risk

The sum-reduction runs are the precedent: same align form, same alignS target,
SIGReg on at weight 1.0, repel:align 1:9 / 1:17 / 1:33 from step 0. AUC fell
below 0.55 at steps 4,404 / 347 / 1,343, latent rank went to about 1.0, scores
1.8 to 12.5. SIGReg did not save them, and at 0.0002 it cannot; its gradient is
negligible against align at about 0.9. Known-safe is 1:4, this cell's own
baseline. The decay crosses the known-dead 1:9 at w about 0.44, step about
5,600. Expect dec0 AUC to leave the 0.95-0.97 band near step 6-8k and sit near
0.5 by 10-25k; the warm start buys some delay. Two settings can keep AUC alive:
the alignT arm (post-decay it is exactly BYOL, predictor to a detached EMA
target) and a floor at 0.5. A 0.1 floor at this cell gives 1:40, far past the
dead boundary; 0.1 was implicitly calibrated on the k = 32 mean cell where
align is about 1x. So a floor buys more than 0.0, but at 0.5/0.2, not 0.1. The
0.0 arms still run: whether a 10k contrastive warm start plus stop-grad rescues
the SimSiam form is the one unmeasured bit.

## 5. Mistakes in the issue

- "The best score today is 1.1491." False. 1.0862 at the same stop, 1.0660 at
  200k, both k = 3.
- "Most of the objective stopped learning." Magnitude is not gradient. L_rep at
  tau_rep 1.0 is a logsumexp over thousands of negatives; about 11.6 is mostly
  its log-N constant floor. Flat means equilibrium, not inert. The sweep
  already tested a rebalance: w3_s08 (align x3) scored worse, 1.2060 against
  1.1782, AUC 0.957 to 0.936. The issue does not cite it.
- Decay ends at 10k but runs stop at 40k: a collapsed arm burns 30k dead steps.
  Gate each arm on the trainer AUC column (500-step median below 0.55: kill,
  respend the budget).
- Keep seed 20260521 out of treated arms. It collapsed once at full weight; a
  collapse there is unreadable.
