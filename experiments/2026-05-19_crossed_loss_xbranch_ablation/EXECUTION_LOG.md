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
