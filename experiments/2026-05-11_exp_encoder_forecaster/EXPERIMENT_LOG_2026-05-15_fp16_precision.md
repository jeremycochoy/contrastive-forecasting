# Experiment log — fp16/bf16 precision stability + sweep details

Detailed companion to [`RESULTS.md`](RESULTS.md). Everything verbose
that does not belong in the concise report lives here: run-by-run
divergence trajectories, amplitude instrumentation tables, the
apples-to-apples snapshot/12L sets, and the git branch-divergence note.

Branch context: all work below is on `exp/encoder-forecaster-v9-explore`
(precision-API @ 09bad2c + bottleneck/instrumentation @ 5fb3628).

---

## 1. fp16 / bf16 stability — run by run

Recipe baseline = v11c (enc6 / fcst1 / dropkey 0.9 shared/shared) or
v16 (dropkey 0.7), 50k steps, GRU patch-emb. "Body fp16" = residual +
attention + FFN in fp16; patch-emb (GRU/RevEWMNorm) governed
separately. Loss is always computed in fp32 (09bad2c).

| Run | Recipe / precision | Init | Outcome | Diverge step |
|-----|--------------------|------|---------|--------------|
| **v18** | v11c recipe, all-fp16 body, dropkey 0.9 | fresh | diverged | ~2,800 |
| **v19** | v16 recipe, all-fp16 body, dropkey 0.7 | fresh | diverged (delayed) | ~38–40k |
| **v20** | v11c recipe, 5k fp32 warmup → fp16 body | fresh seed | **healthy** | — (≥41k stable) |
| **v22** | v11c recipe, attn fp32 / rest fp16 | fresh | diverged | ~1,700–2,800 |
| **v23** | v22 recipe + (old) instrumentation | fresh | diverged | ~900 |
| **v24** | attn fp32 / rest fp16, proper instrumentation | fresh | diverged | ~2,800 |
| **v25** | residual fp32 / attn+ffn+pemb fp16 | fresh | diverged | ~1,400–1,500 |

Trajectory detail:

- **v18** — clean low-loss attractor (loss ~3.6, R²~0.91, AUC/Top1 1.0)
  through ~2,800, then NaN/explosion. All-fp16 body at fresh init fails
  almost immediately.
- **v19** — dropkey 0.7 *delays but does not prevent*. Stable loss
  ~2.03 through 32k, first wobble at 36k (loss 2.33), full explosion
  by 40k (loss 6.7 → 10.8, R² → negative, Top1 → 0.01). The extra
  dropkey noise buys ~35k steps; the failure mode is identical.
- **v20** — the fix. Phase A: fresh fp32 v11c-recipe 0→5,000 (best
  loss 2.306 @ 5k, 0.5 h). Phase B: resume `_5k.pth` with fp16 body,
  5,000→50,000. Healthy and stable past **41,300** steps at report
  time (loss ~2.117, gap ~1.118, flat). Fresh seed ≠ v11c's init, so
  this *also* doubles as the lucky-init falsification for v11c.
- **v22 / v23 / v24** — attention-in-fp32 (only resid+ffn+pemb fp16)
  is **not** sufficient at fresh init. All three diverge in the
  ~900–2,800 window from the same clean attractor. v24 carries the
  proper amplitude instrumentation (section 2).
- **v25** — residual-fp32 + attn/ffn/patch-emb fp16. Diverges
  ~1,400–1,500. **This falsifies the "keeping the residual stream in
  fp32 is sufficient" hypothesis** — the residual cast is not the
  single fragile site; fp16 anywhere in the body at fresh init is
  enough to blow up.

**Conclusion.** Fresh-init partial-fp16 is unsafe in every axis tested
(all-body, attn-fp32-rest, resid-fp32-rest). The only robust speedup
is a short fp32 warmup before the fp16 cast (v20).

## 2. Amplitude instrumentation (per-layer max-abs)

Opt-in logger (commit 5fb3628; see
`enc_fcst_v24_DIAGDONE_2800_attn_amplitude.csv`,
`enc_fcst_v25_DIVERGED_1500_attn_amplitude.csv`,
`enc_fcst_v23_OLDINSTR_900_attn_amplitude.csv`). Columns: per (step,
layer, block) `qk_logit_maxabs`, `sa_in_maxabs`, `sa_out_maxabs`,
`resid_post_sa_maxabs`, `resid_post_ffn_maxabs`.

### v24 — forecaster block (layer 0, `fcst`)

| Step | qk_logit | sa_out | resid_post_sa | resid_post_ffn |
|-----:|---------:|-------:|--------------:|---------------:|
|  200 |     26.7 |   58.4 |          55.3 |           79.5 |
|  400 |     29.8 |   56.1 |          62.8 |           98.0 |
|  600 |     34.3 |   61.2 |          70.6 |          110.4 |
|  800 |     39.1 |   79.8 |          90.0 |          140.8 |
| 1000 |     52.2 |  184.8 |         210.5 |          361.8 |
| 1200 |     54.5 |  248.9 |         273.9 |          475.2 |
| 1400 |     59.1 |  342.0 |         342.1 |          553.6 |
| 1600 |     51.2 |  317.8 |         389.9 |          605.2 |
| 2000 |     51.3 |  413.0 |         463.6 |          692.6 |
| 2400 |     55.0 |  730.5 |         784.2 |         1047.2 |
| 2800 |     54.7 |  936.5 |         962.7 |         1073.8 |

**QK logits stay bounded** (~27→55, ceiling ~60) across the whole run.
**The residual stream blows up >8×** (resid_post_ffn 79 → 1074) and
sa_out blows up ~16× (58 → 936). Encoder layers show the same shape at
smaller magnitude (deeper enc layer → larger amplitude even at step
200: enc-0 ~11 vs enc-5 ~52). The growth is in the residual/value
path, not the softmax/attention-score path.

### v25 — residual-fp32 variant, forecaster block @ step 1000

`qk_logit 41.8 / sa_out 173.0 / resid_post_sa 181.5 /
resid_post_ffn 300.3`. Even with the residual cast forced to fp32,
sa_out (173) and resid_post_ffn (300) are already in the regime where
the fp16 attn/ffn intermediate products overflow before they reach the
fp32 residual — consistent with v25 diverging ~1,400.

**Mechanism (supported by the tables, stated as the working model).**
Residual-stream amplitude grows monotonically with depth and training
and is unbounded; fp16's 10-bit mantissa cannot represent values in the
hundreds–thousands without catastrophic relative error in the
attention/FFN intermediates. A short fp32 warmup lets the network
settle into a lower-amplitude basin before the cast, after which fp16
is adequate (section 3 corroborates).

## 3. Precision-envelope warm-resume sweep

Resume the **trained** `v11c_5k` checkpoint (not fresh) and continue
5k→15k under five precision axes. All five reached 15k cleanly:

| Axis | log | loss @15k |
|------|-----|----------:|
| ffn bf16        | `run_enc_fcst_precenv_ffn_bf16_v11c_5k_15k.log`     | 2.169 |
| ffn fp16        | `run_enc_fcst_precenv_ffn_fp16_v11c_5k_15k.log`     | 2.173 |
| attn+ffn fp16   | `run_enc_fcst_precenv_attnffn_fp16_v11c_5k_15k.log` | 2.235 |
| all-body fp16   | `run_enc_fcst_precenv_allbody_fp16_v11c_5k_15k.log` | 2.254 |
| patch-emb fp16  | `run_enc_fcst_precenv_pemb_fp16_v11c_5k_15k.log`    | 2.251 |

Contrast with section 1: the *identical* all-body-fp16 precision that
diverges at ~2,800 from fresh init (v18) is **stable** when resumed
from a 5k-trained checkpoint. This is the direct evidence that the
fragility is fresh-init-specific, and the basis for the v20 recipe.

## 4. Backbone sweep — supporting tables

### 4.1 Triage vs full (proxy quality)

| Backbone | Triage (11) | Full (97) | Triage − Full |
|----------|------------:|----------:|--------------:|
| v11c | 1.388 | 1.292 | **+7.4 %** |
| v13  | 1.514 | 1.451 | +4.4 % |
| v14  | 1.650 | 1.661 | −0.7 % |
| v15  | 1.671 | 1.558 | +7.3 % |
| v16  | 1.428 | 1.335 | +6.9 % |
| v17  | 1.718 | 1.409 | **+21.9 %** |

Triage is pessimistic by ~7% for most arms but is wildly off for v17
(+22%) — it preserves the top rank but compresses mid-pack ordering.

### 4.2 at40k apples-to-apples (equal backbone-step snapshot)

All evaluated from the 40k backbone snapshot (removes the "different
total training" confound across the dropkey arms):

| Backbone | at40k triage GM-MASE |
|----------|---------------------:|
| v11c | 1.388 |
| v13  | 1.514 |
| v14  | 1.485 |
| v15  | 1.774 |
| v16  | 1.439 |
| v17  | 1.692 |

The snapshot effect is arm-dependent (±~10%): v14 improves at the
earlier snapshot (1.650 → 1.485), v15 worsens (1.671 → 1.774), v11c
unchanged (1.388). v11c stays the leader at every snapshot.

v11c snapshot series: earlier-snapshot triage 1.388, **at50k 1.365**
(`post_qhead_chain_v11c_at50k.log`) → v11c keeps improving with more
backbone training.

### 4.3 12L q-head vs 2L q-head

| Backbone | 2L head | 12L head | 12L effect |
|----------|--------:|---------:|------------|
| v11c | 1.388 | 1.519 | **worse** (over-trained backbone) |
| v14  | 1.650 | 1.781 | worse |
| v15  | 1.671 | 1.602 | better (over-constrained) |
| v16  | 1.428 | 1.516 | worse |
| v17  | 1.718 | 1.576 | better (over-constrained) |

A heavier head only helps backbones that are themselves
under-expressive (v15 fcst4, v17 dropkey 0.95); it hurts the good
ones. 2L head is the right default.

### 4.4 v11c reproduction

Fresh q-head retrain on the same frozen v11c backbone:
`post_qhead_chain_v11c_v2.log` → `GM-MASE = 1.3878` — bit-for-bit the
same as the original v11c triage. v11c is a real, reproducible result,
not a seed artifact.

## 5. Key archived artifacts

- Full eval (97 cfg): `results/gift_eval_full_v{11c,13,14,15,16,17}/{all_results.csv,summary.txt}`
- Triage (11 cfg): `results/gift_eval_triage_v{10jepa,11c,11c_at50k,11c_jepa_12L,12_jepa,13_at40k,14_jepa,14_at40k,14_jepa_12L,15_jepa,15_at40k,15_jepa_12L,16_jepa,16_at40k,16_jepa_12L,17_jepa,17_at40k,17_jepa_12L}/`
- Amplitude CSVs (in `/home/jupyter/contrastive-forecasting/checkpoints/`):
  `enc_fcst_v24_DIAGDONE_2800_attn_amplitude.csv`,
  `enc_fcst_v25_DIVERGED_1500_attn_amplitude.csv`,
  `enc_fcst_v23_OLDINSTR_900_attn_amplitude.csv`
- fp16-stability run logs: `results/run_enc_fcst_v{18,19,20,22,23,24,25}_*.log`,
  `results/run_enc_fcst_v20_phaseA_fp32warmup_5k.log`
- Precision-envelope: `results/run_enc_fcst_precenv_*_v11c_5k_15k.log`
- Chain verdict logs: `results/post_qhead_chain_*.log`

## 6. Git branch-divergence note — NEEDS USER DECISION

Two divergent histories carry overlapping-but-not-identical work; do
**not** auto-merge — this is a branch-strategy call for the user.

- **`exp/encoder-forecaster-v9-explore` (authoritative, on origin).**
  Carries the *real* precision API: `09bad2c` (3 dtype flags replace
  the 7-bool mess, always-fp32 loss) + `5fb3628` (forecaster
  bottleneck + attention-amplitude instrumentation). This is the
  branch all v18-v25 / precision-envelope / amplitude work was run on
  and where this session's commits landed (chain wiring + scripts +
  results).

- **`plot/v10-v11-multi-metric` (PR #286, divergent, precision-less).**
  Worktree `/home/jupyter/contrastive-forecasting/.claude/worktrees/v10-v11-curves-plot`.
  Carries the v10/v11 multi-metric **plots** plus a *hand-ported,
  older* copy of the v13-eval/chain + instrumentation work
  (`d34da27` v13 chain, `75a8c6f` v13 eval rollout fix, `8bbec2e`
  diag logger) that does **not** sit on top of the precision-API
  refactor. The same logical changes therefore exist twice, on
  unrelated bases.

The two lines were intentionally **not** reconciled in this session
(merging unrelated histories of the same logical change is a
branch-strategy decision, not a cleanup). Recommended for the user:
pick `exp/encoder-forecaster-v9-explore` as the trunk for the
precision/bottleneck work and cherry-pick *only the plot assets* off
PR #286, rather than merging the divergent eval/instrumentation
duplicates.
