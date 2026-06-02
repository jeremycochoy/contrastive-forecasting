# #307 cross-branch ablation — execution log

Continuation of #303 (`experiments/2026-05-19_crossed_loss_ablation`).
Agent: Edward. Autonomous run.

## Arms (only `--loss-shape` changes vs #296/#303 of-record recipe)

| short | loss_shape | meaning |
|---|---|---|
| `hhff`   | `cosine_similarity_batch_full_hh_ff_negs`     | (B)+(C): all-time h↔h **and** f↔f; no all-time f↔h; cross-batch = standard f↔h (as #303) |
| `fhhhff` | `cosine_similarity_batch_full_fh_hh_ff_negs`  | (A)+(B)+(C): all three all-time crossed terms; cross-batch = standard f↔h |
| `hhxbf`  | `cosine_similarity_batch_full_hh_negs_xbfree` | (B) cross-branch-negative-free: all-time h↔h + square-style within-branch cross-batch (f↔f, h↔h), **drop** f↔h cross-batch ⇒ no f↔h negative anywhere; f↔h positive retained |

Code + 124 loss tests green (closed-form orthonormal pins incl. B=2 for
`hhxbf`). Committed `7984946`.

## Resource reality

`vastrun-balance` = **$10.88** at start (user stated "plenty"; tracking
spend tightly regardless). Frugal plan keeps est. total ≈ $7–9 (3 arms,
prosumer 2-GPU, parallel backbones). Issue #307's snapshot offers were
all gone live (offers drift); selecting from live `vastrun-forward
--raw` search. Datacenter 2-GPU pool currently thin/expensive → using
prosumer per issue #307's explicit recommendation for this task
(overrides the general datacenter-only default for this card).

## Orchestration

- Backbones: 3 boxes (2-GPU each), one 50k DDP backbone per box, in
  parallel. Validate box 1 fully (setup + DDP + sync tick) before
  fanning out (user instruction).
- Downstream (q-head 30k + GIFT-Eval triage11/full97, single-GPU):
  packed so **no GPU sits idle while a q-head runs** — a 2-GPU box runs
  2 arms concurrently (1/GPU); the odd 3rd arm on a cheap 1-GPU box.
  Backbone boxes destroyed once their backbone+optimizer are synced
  (don't pay 2-GPU rates for single-GPU work — issue #307 note).
- Sync: detached `sync_loop.sh` on elisa, 15-min ticks, atomic
  `safe_pull.sh` into the MAIN checkout
  `/home/jupyter/contrastive-forecasting/.../sync_<arm>/` (CLAUDE.md
  rule 4). Backbone+optimizer committed (resume-capable) + eval CSVs.

## Timeline
(appended as it happens)

## 2026-05-19 ~15:52Z — all 3 backbones live

| arm | inst | GPU | $/h | loss_shape | status |
|---|---|---|---|---|---|
| hhxbf  | 37075647 | RTX PRO 4000 (BW) | 0.553 | full_hh_negs_xbfree    | ~5.1 sps, auth, 1 DDP |
| hhff   | 37075919 | RTX 5070 Ti  (BW) | 0.310 | full_hh_ff_negs        | ~2.8 sps, auth, 1 DDP |
| fhhhff | 37076887 | RTX 4070S Ti (Ada)| 0.396 | full_fh_hh_ff_negs     | live, auth, 1 DDP |

Incidents handled: (a) hf_token.txt is gitignored & absent from the
worktree → first launches streamed HF unauthenticated (~2.4 sps);
copied token into worktree, pushed to boxes, rebuilt tarball, restarted
authenticated (data 189ms→8ms). (b) overlapping provision/restart
attempts left 2 duplicate DDP jobs on 37075647 + a self-matching
`pkill -f train.py` (killed its own shell); fixed with a script-file
hard-killer (`_hardkill.sh`) → single clean run. (c) duplicate
xbranch-hhxbf label → 37075919 relabelled xbranch-hhff.
Burn rate ≈ $1.26/h (3 boxes). sync_loop detached pid in state/.

## 2026-05-19 17:29Z — backbones ~16-17% (healthy)
hhxbf step8400 loss2.83 3.7sps | hhff step7300 loss2.40 2.6sps |
fhhhff step8400 loss2.50 3.8sps. No NaN, single DDP each, sync robust.
Credit $9.34 (burn ≈$1.5/h incl. ~1.7GB/h sync egress). ETA: hhxbf/fhhhff
~20:30, hhff ~22:00. Budget plan: destroy 37075647 (PRO4000 $0.55,
priciest) immediately after its backbone syncs; consolidate downstream
on 37076887 (4070STi $0.41) to cap spend.

## 2026-05-19 18:13Z — backbones ~30-36%
hhxbf step18.2k loss2.78 ETA2.4h | hhff 14.8k loss2.35 ETA3.5h |
fhhhff 18.2k loss2.45 ETA2.3h. No NaN, single DDP, sync robust.
Credit $8.18 (burn ≈$1.58/h). First completions ~20:30 (hhxbf/fhhhff),
hhff ~21:45. Budget discipline: destroy 37075647 (PRO4000 $0.55) the
moment its backbone FINAL syncs; downstream consolidated on 37076887.

## 2026-05-19 19:47Z — PLAN REVISION (budget): downstream on elisa, FREE
Slow prosumer downstream would cost ~$3-5 (budget $5.60). elisa GPU1 is
fully free (GPU0 has 3.8GB parked by another session, 0% util). Revised:
backbones finish on vast → commit resume-capable ckpt+optimizer+losses,
**destroy each vast box immediately** (zero post-backbone vast spend) →
run all downstream (q-head 30k + GIFT-Eval triage/full, recipe-identical
to #303) on elisa's free 4090 from the synced backbone via
`scripts/local_downstream.sh`. Faster (4090 = of-record card) and free.
Backbones: hhxbf/fhhhff ~78% (ETA 0.8h, done ~20:30), hhff ~62%
(ETA 1.8h, ~21:35). No NaN. Credit $5.60 — now comfortably covers
backbone-only vast spend.

## 2026-05-19 20:40Z — hhxbf+fhhhff DONE, downstream on elisa (FREE)
Backbones hhxbf (loss 2.726) + fhhhff (2.395) hit 50k, BB DONE. Resume
bundles (FINAL.pth+FINAL_optimizer.pth+losses+log) pulled via safe_pull,
verified (45M/88M/15M), **committed** (artifacts/<arm>/, git add -f) —
the key deliverable. Destroyed 37075647 (PRO4000, spent $2.31) +
37076887 (4070STi, $1.60). Only 37075919 (hhff, 5070Ti) remains
(~21:40, $0.31/h). Credit $4.21. Downstream q-head 30k + GIFT-Eval
(triage/full, recipe-identical to #303) launched FREE on elisa:
hhxbf→GPU1, fhhhff→GPU0, concurrent (~2h50m → ~23:30). hhff downstream
queues when a GPU frees (~23:30).

## 2026-05-19 21:41Z — ALL 3 BACKBONES DONE + committed; zero vast spend
hhff hit 50k (loss 2.300). All 3 resume bundles (FINAL.pth +
FINAL_optimizer.pth + losses + log) committed to artifacts/<arm>/.
Destroyed 37075919 (5070Ti, $1.61). **No vast instances running.**
Total vast spend ≈ $7.08 (start $10.88 → credit $3.80; PRO4000 $2.31 +
4070STi $1.60 + 5070Ti $1.61 + early throttled-restart waste ~$0.6).
Downstream FREE on elisa: hhxbf/fhhhff q-heads ~21:50 → evals → ~23:25;
hhff downstream deferred (auto-starts when a GPU frees ~23:25).

## 2026-05-19 22:07Z — first triage GMs (sanity: code works end-to-end)
q-heads done cleanly; triage-11 GM: hhxbf(B-xbfree)=1.4843,
fhhhff(A+B+C)=1.6315 (err=0). vs #303 triage: A 1.5611, B 1.4461,
C 1.5185, A+B 1.5426. hhxbf ≈ #303 best (B); fhhhff (all-3 incl f↔h)
worst — directionally consistent with "f↔h negative is harmful".
Triage is ~7-10% noisy (#303); full-97 (trusted) running, ETA ~23:20.
hhff downstream still deferred (auto when GPU frees).

## 2026-05-20 02:11Z — all 3 full-evals done, PR #308 opened
Full-97 GMs (trusted):
  hhxbf (B-xbfree)  = 1.3681  (B + 0.8%, inside #296 noise)
  hhff  (B+C)       = 1.3982  (B + 3.0%, NOT additive — slightly worse)
  fhhhff (A+B+C)    = 1.4465  (A+B - 0.4%, A's harm robust to C)
Verdict: (B) alone (#303, 1.357) remains unique best in this family.
Cross-branch f↔h *negative* contributes nothing measurable — B-xbfree
is structurally simpler and equally good. Within-branch siblings are
NOT additive (B+C slightly worse than B). A's harm is robust to
compositional context (A+B+C ≈ A+B).

PR #308 → experiments: https://github.com/jeremycochoy/contrastive-forecasting/pull/308
All artifacts committed (3× resume-capable backbone+optimizer+losses+log,
3× q-head FINAL, 3× triage + full GIFT-Eval CSVs/summaries).
Final state: vast 0 instances (~$7.10 total), elisa procs clean,
124-test loss suite green (incl. 11 #307 closed-form pins).

End-to-end (code → eval → PR): ≈14 h on 2026-05-19 (start 12:30Z) /
2026-05-20 (end 02:15Z). 3 vast boxes destroyed promptly post-FINAL,
downstream ran free on elisa.

## 2026-05-20 15:58Z — variance follow-up complete (n=3 B, n=2 B-xbfree)
Multi-seed pipeline (elisa 4090 + 2 vast prosumer Blackwells in parallel):
  B-s18 (elisa) full=1.3331  B-s19 (vast 4070STi) full=1.4368
  B-xbfree-s19 (vast PRO 4000) full=1.4244
B-xbfree-s18 attempted but destroyed mid-training (HF throttling on
4070STi → 2.2 sps → would push budget past $5 cap); N=2 for B-xbfree.

Stats:
  B (n=3):       mean 1.376  std 0.054  CV 3.94 %  range 0.104
  B-xbfree (n=2): mean 1.396  std 0.040  CV 2.85 %  range 0.056

VERDICT REVISED. The #307 single-seed "(B) is unique winner" was
overconfident — within-arm CV (≈4 %, Econ/Fin volatility) is LARGER
than the across-arm gaps inside the non-A cluster (B, C, B+C, B-xbfree).
These four arms are statistically indistinguishable at this recipe.
Only the (A)-is-harmful finding survives: A 1.438 is +4.5 % above B's
3-seed mean (1.376), outside its ±1σ band.

Cost: vast ≈$5.25 (Box A + Box B + Box C + orphan). End-to-end variance
work: ≈8 h on 2026-05-20 (~07:47Z → ~15:55Z).

## 2026-05-21 02:12Z — B-xfree-s18 done on elisa (N=3 for both arms)
Finished the missing variance arm by re-running on free elisa 4090s
overnight (no vast spend). B-xfree-s18 (hhxbf seed 20260518) full GM =
1.3779 (triage 1.4679). Now N=3 for both arms:

  B (n=3):       mean 1.376  std 0.054  CV 3.94 %  range 0.104
  B-xbfree (n=3): mean 1.390  std 0.030  CV 2.16 %  range 0.056

The +1.0 % gap between means is well inside either's ±1σ band —
distributions overlap fully. Verdict unchanged: A is harmful (+4.5 %
above B's mean, outside ±1σ); non-A cluster (B, C, B+C, B-xbfree)
statistically indistinguishable at this recipe.
