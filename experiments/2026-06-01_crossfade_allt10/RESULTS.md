# #325 — Regime-crossfade synthetic on top of allt·10%

**Verdict.** Adding a third synthetic stream — a **regime crossfade** (10 % of the batch,
each row a monotone blend of two distinct *real* windows that share their past/future with
batch-mates) — on top of #322's best arm **allt·10%** does **not reliably change** downstream
GIFT-Eval accuracy. The point estimates nudge the right way on both q-heads (full-97
GM-Relative MASE: 2L 1.222 → **1.208**, 6L 1.191 → **1.178** — a nominal new best cell), but
**both paired-bootstrap 90 % CIs straddle 0** (2L Δ −0.014, CI [−0.040, +0.012]; 6L Δ −0.013,
CI [−0.039, +0.012]). At a single backbone seed the effect is within config-set noise. The
crossfade *does* measurably change the contrastive training dynamics — it is a genuinely
harder negative (the gap climbs to ~1.18 vs allt·10%'s ~1.03, with a higher residual loss) —
but that extra training signal **does not convert into a reliable forecasting gain**.

## What we asked
#322 found **allt·10%** (90 % real + 10 % forked-ARMA continuations, all-time contrastive
loss, batch 1024) the best cell: full-97 **GM-Relative MASE** — the geometric mean over
GIFT-Eval's 97 configs of (model MASE ÷ seasonal-naive MASE), lower better — of 1.222 (2L
q-head) and **1.191** (6L). The forked-ARMA stream denies the forecaster a content-free
positional code by giving identical pasts divergent futures. #325 keeps that arm fixed and
adds a **regime crossfade** to probe a *different* hard negative: does a blend of two real
windows — sharing one's past and the other's future with its own batch-mates — sharpen the
representation further, or is the all-time fork already saturating what the objective can use?

## The crossfade primitive
For each crossfade row, two distinct real windows A, B from the **same step's real sub-batch**
are z-normalised per series and blended by a monotone weight s(t):

    C(t) = (1 − s(t))·A(t) + s(t)·B(t),   s rises 0→1 across a window [l, l′]

with, per sample (T = window length): midpoint m ~ U(0, T); width w = l′−l ~
LogUniform(T/128, T) (sharp → gradual); l = m − w/2, l′ = m + w/2, clipped to [0, T]; one
s(t) shared across channels. Because A and B remain in the batch as their own rows, C shares
**A's past** and **B's future** with batch-mates — a hard negative position alone cannot
satisfy. (`src/synthetic_crossfade.py`.)

## Result
The single controlled change — batch composition **80 % real / 10 % forked-ARMA / 10 %
crossfade** vs allt·10%'s 90/10/0 — leaves downstream accuracy statistically unchanged.

![Full-97 GM-Relative MASE per q-head: allt·10% (#322) vs allt·10%+crossfade·10% (#325).
Bars near-identical; single-run bootstrap 90 % CIs overlap heavily. v11c and seasonal-naive
marked.](plots/gm_summary.png)

![Paired-bootstrap Δ (crossfade − allt·10%) per head; whisker = 90 % CI over the 97 shared
configs (per-config difficulty cancels). Both Δ are negative (crossfade better) but both
whiskers cross 0 → inconclusive (grey).](plots/delta.png)

| q-head | allt·10% (#322) | + crossfade (#325) | Δ | 90 % CI on Δ | triage-11 base → xf |
|---|--:|--:|--:|:--:|:--:|
| 2L | 1.222 | **1.208** | −0.014 | (−0.040, +0.012) | 1.293 → 1.224 |
| 6L | 1.191 | **1.178** | −0.013 | (−0.039, +0.012) | 1.259 → 1.176 |

Both full-97 Δ are negative but neither CI excludes 0, so neither win is reliable. The
triage-11 subset (the noisy fast 11-config set, kept for continuity with #318/#320/#322)
shows a larger nominal gap — but triage is exactly where single-config noise dominates, and
the authoritative full-97 shrinks it to within the bootstrap band. **6L = 1.178 is nominally
the best cell to date** (below #322's 1.191 and v11c = 1.292), but only nominally.

## Training dynamics — the crossfade is a real hard negative
The backbone converged cleanly (qk-norm + attn-out-norm holding; no directional collapse —
cross-series cosine flat ~3 × 10⁻³). But the crossfade changes the contrastive regime: vs
allt·10%, the **gap** (cos(forecast, future) − cos(forecast, present)) climbs higher (~1.18
vs ~1.03) and the floor-subtracted loss settles higher (~1.0 vs ~0.85). Sharing a real past
and a real future with batch-mates is a harder separation than the fork alone — the model is
pushed to encode the future more strongly. That this larger gap does **not** track a
downstream gain re-confirms #322's finding that the contrastive gap is a training signal, not
a forecasting one.

![Training curves (from step 100), crossfade (solid) vs allt·10% (dashed): loss − InfoNCE
floor (log-log), gap (semilog-x, crossfade ends higher ~1.18), cross-series cosine (flat ~0 =
no collapse).](plots/training_curves_loglog.png)

## Protocol
Exactly one change from #322's allt·10%: **+10 % crossfade** (real fraction 90 % → 80 %;
crossfade rows blended from the real sub-batch, consuming no extra HF rows). Everything else
byte-identical: single-GPU batch 1024 with the GRU patch-encoder gradient-checkpointed,
12.5 k steps, AdamW lr 1e-3 / β1 0.9 / β2 0.98 / wd 0.1, τ 0.10, **qk-norm + attn-out-norm**,
dropkey 0.70, depthwise-conv 3, mixup-p 0.3, ewma span 128, freq/seas-emb 3/3, fp16
attn/ffn/conv + fp32 residual/patch-emb, `--subtract-contrastive-floor`, loss
`cosine_similarity_batch_full_hh_negs_xshh_allt`, seed 20260520. Eval byte-identical to #322:
frozen backbone scored with a fresh 2L and 6L quantile q-head (30 k steps, transformer,
causal, head-ffn-mult 4.0, dropout 0.1, `--head-train-input e_then_f`, `--reconstruction
forecaster`, forecast-len 16, batch 256, cosine LR, `--amp-dtype none`); GIFT-Eval
`--strategy B4`, full-97 + triage-11. Δ is a paired bootstrap (2000 resamples) over the 97
shared configs; **single backbone seed** — the CI captures config-set spread, not seed noise,
so the honest read of the ~0.013 point gaps is "no reliable evidence," not "no effect."

## What we learned & follow-up
A real-window regime crossfade is well-motivated as a hard negative and demonstrably changes
the contrastive geometry, but **on top of the all-time fork at 10 % it does not buy a reliable
downstream gain** — the all-time negative pool is evidently already extracting most of the
available signal. The cheapest way to convert the consistent (if small) positive lean into a
reliable result, if pursued, is **multiple backbone seeds** (2–3) so a paired-over-seeds test
can resolve a ~0.013 effect that a single seed cannot. Absent that, the result is a clean
neutral: the crossfade is not worth adding to the recipe.

*(Operational events — shared-GPU contention, the single-GPU pivot, and a transient HF-CDN
crash + resume during the 6L q-head — are in [`EXECUTION_LOG.md`](EXECUTION_LOG.md); none
affect the result.)*
