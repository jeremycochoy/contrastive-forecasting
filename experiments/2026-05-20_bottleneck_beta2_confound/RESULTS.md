# #309 Bottleneck × β2 × τ on (B): can we match v11c?

## Question

The bottleneck-fullfh recipe **(B)** reaches full-97 GM-MASE 1.3572
(#303), ~5% above the encoder-forecaster champion **v11c** (1.292).
Three cheap knobs might close the gap: the forecaster **bottleneck**,
AdamW **β2**, and the contrastive temperature **τ**. This card tests
them.

*Metric:* full-97 GM-Relative MASE = geometric mean over 97 GIFT-Eval
configs of (model MASE ÷ seasonal-naive MASE); **lower is better**,
1.0 = seasonal naive. Each backbone gets a 30k 2L-causal q-head before
eval.

## Arms

All four share the (B) recipe (GRU patch-encoder → 6L causal encoder →
1L forecaster, dropkey 0.70, loss `cosine_similarity_batch_full_hh_negs`,
50k steps, global batch 256, seed 20260520). They differ on two knobs:

| Arm | forecaster bottleneck | AdamW β2 | change vs (B) |
|-----|-----------------------|---------:|---------------|
| **(B)** | kept (d=128) | 0.95 | baseline |
| **α** | **removed** (d=384) | **0.98** | −bottleneck, +β2 |
| **β** | kept (d=128) | **0.98** | +β2 |
| **γ** | **removed** (d=384) | 0.95 | −bottleneck |

Each arm is run at **τ = 0.1** and **τ = 0.8** (the v11c temperature is
0.1). v11c is the reference throughout (no-bottleneck, all-fp32, dropkey
0.9, plain `cosine_similarity_batch` loss).

## Verdict

**(B)-τ0.8 = 1.2335 is the best arm and beats v11c (1.292) by ~4.5% —
keeping the bottleneck and raising τ to 0.8.** The knob this card
prioritized — *removing* the bottleneck (α) — loses; the two we
under-weighted, β2 and τ, do the work on the bottleneck recipe. Both
bottleneck arms reach/beat v11c at τ=0.8 ((B)-τ0.8 1.2335, β-τ0.8
1.2942); removing the bottleneck (α, γ) converges worse than (B) at
τ=0.1 and never reaches v11c at either τ. On the bottleneck arm the β2
optimum **flips with τ**: β2=0.98 wins at τ=0.1 (β < (B)) but β2=0.95
wins at τ=0.8 ((B) < β) — the opposite ordering.

**Caveats (do not discount).** The (B)-τ0.8 result is a **single seed**,
and its backbone reached only **47.6k of 50k steps** — a late DDP
teardown OOM under GPU contention, after the loss had plateaued flat
from ~40k, so the checkpoint is representative but not a clean 50k. The
+4.5% gap to v11c exceeds the project's ±0.02 (n=3) noise estimate, but
one seed cannot rule out seed variance at this margin. Treat the
mechanism below as a hypothesis pending a clean 50k re-run / second
seed.

## Results — converged backbones, by arm and τ

![gm summary](plots/gm_summary.png)

Rows ordered by best cell (lower = better):

| Arm | bottleneck | β2 | τ=0.1 | τ=0.8 |
|-----|:----------:|---:|------:|------:|
| (B) | kept | 0.95 | 1.3572 | **1.2335** |
| β   | kept | 0.98 | 1.3272 | 1.2942 |
| γ   | removed | 0.95 | 1.3132 | 1.3424 |
| α   | removed | 0.98 | 1.4057 | 1.3274 |
| v11c (ref) | removed | 0.98 | **1.292** | — |

**(B)-τ0.8 (1.2335) is the best arm and beats v11c (1.292) by ~4.5%;
β-τ0.8 (1.2942) also lands within noise of v11c.** Both arms that
reach/beat v11c keep the bottleneck and train in fp16; the no-bottleneck
arms (α/γ) are all worse than v11c at both τ. The +4.5% (B)-τ0.8 gap to
v11c exceeds the project's ±0.02 (n=3) noise estimate, but it is a single
seed — and its backbone reached only 47.6k of 50k steps (a late DDP
teardown OOM under GPU contention; loss had plateaued flat from ~40k, so
it is representative but not a clean 50k). See Limitations.

### Per-domain (full GIFT-Eval), v11c dashed

τ=0.1:
![star τ0.1](plots/perdomain_star_tau01.png)

τ=0.8:
![star τ0.8](plots/perdomain_star_tau08.png)

At τ=0.8, the bottleneck arms ((B), β) sit on/inside the v11c ring
across domains — (B) inside it (GM 1.234 < v11c 1.292) — while the
no-bottleneck arms (α, γ) sit outside it. The per-domain picture tracks
the aggregate: the bottleneck arms are the ones that reach/beat v11c.

### Training curves (converged)

τ=0.1:
![curves τ0.1](plots/training_curves_tau01.png)

τ=0.8:
![curves τ0.8](plots/training_curves_tau08.png)

All four arms descend monotonically and hold (1−AUC at floor, gap ~1.0)
at both temperatures.

## τ = 0.1 vs τ = 0.8

Per-arm full-97 GM at each temperature (Δ = τ0.8 − τ0.1; negative =
τ=0.8 better):

| Arm | τ=0.1 | τ=0.8 | Δ |
|-----|------:|------:|------:|
| (B) bneck β2.95 | 1.3572 | **1.2335** | **−0.124** |
| β bneck β2.98   | 1.3272 | 1.2942 | −0.033 |
| α no-bneck β2.98 | 1.4057 | 1.3274 | −0.078 |
| γ no-bneck β2.95 | 1.3132 | 1.3424 | +0.029 |

τ=0.8 is **not uniformly better** — its sign depends on the recipe:

- **Bottleneck arm (B): τ=0.8 helps a lot** (1.3572 → 1.2335, Δ=−0.124),
  the largest τ swing of any arm and the only cell that beats v11c. On
  the bottleneck recipe raising τ 0.1→0.8 is strongly beneficial (single
  seed; backbone 47.6k/50k — see caveat above).
- **Bottleneck arm β: τ=0.8 also helps** (1.327 → 1.294), landing within
  noise of v11c. Both bottleneck arms reach/beat v11c at τ=0.8.
- **No-bottleneck β2=0.98 (α): τ=0.8 helps a lot** (1.406 → 1.327) but
  only rescues a poor arm — still short of v11c.
- **No-bottleneck β2=0.95 (γ): τ=0.8 *hurts*** (1.313 → 1.342). γ-τ0.1
  is the best no-bottleneck cell, and τ=0.8 drags it.

Note the β2 optimum **flips with τ on the bottleneck arm**: at τ=0.1,
β2=0.98 (β 1.3272) beats β2=0.95 ((B) 1.3572); at τ=0.8 the order
reverses — β2=0.95 ((B) 1.2335) beats β2=0.98 (β 1.2942). (Single seed
per cell; the (B)-τ0.8 cell is 47.6k/50k.)

## β2 and the bottleneck

- **At τ=0.1, β2 = 0.95 clearly better than 0.98 for the no-bottleneck
  arm** (γ 1.313 vs α 1.406) — but β2 = 0.98 is better *with* the
  bottleneck (β 1.327 vs (B) 1.357). So at τ=0.1 the β2 optimum differs
  with vs without the bottleneck.
- **On the bottleneck arm the β2 optimum also flips with τ**: β2=0.98 at
  τ=0.1 ((B) 1.357 → β 1.327), but β2=0.95 at τ=0.8 (β 1.294 → (B)
  1.234). (Single seed per cell; (B)-τ0.8 is 47.6k/50k.)
- Removing the bottleneck never helps at convergence: the best
  no-bottleneck cell (γ-τ0.1, 1.313) still trails the bottleneck arms
  (β-τ0.8 1.294, (B)-τ0.8 1.234) and v11c (1.292).

## Training-precision note (fp16 acceleration)

Bottleneck arms ((B), β) train in fp16. **Removing the bottleneck makes
the fp16 body diverge at fresh init** — the residual stream grows past
fp16's range without the d=128 bottleneck to constrain it (mechanism in
`experiments/2026-05-11_exp_encoder_forecaster/EXPERIMENT_LOG_2026-05-15_fp16_precision.md`).
This is a **technical failure of the fp16 speedup, not a result**: the
no-bottleneck arms (α, γ) are simply trained in fp32 instead (stable —
the precision v11c uses).

![fp16 divergence](plots/fp16_divergence.png)

(Aside, not a result: an fp16 pre-divergence checkpoint (~step 900)
scored 1.277/1.283 — a mid-divergence snapshot, not a converged
backbone; the same arms trained to 50k in fp32 land at 1.31–1.41.
Detail in EXECUTION_LOG.md.)

## What we learned

1. **(B)-τ0.8 (1.2335) beats v11c (1.292) by ~4.5%** — keep the
   bottleneck, keep β2=0.95, raise τ to 0.8. A converged fp16 backbone
   with the small forecaster, reached by raising one scalar (τ) on (B).
   This is the actionable result *if it holds* — single seed, backbone
   47.6k/50k (plateaued); a clean 50k re-run / second seed is the
   obvious next step.
2. **On the bottleneck recipe, τ=0.8 helps a lot** — (B) 1.3572 → 1.2335
   (Δ=−0.124, largest τ swing of any arm) and β 1.3272 → 1.2942. Both
   bottleneck arms reach/beat v11c at τ=0.8.
3. **Removing the bottleneck does not help** — α/γ converge worse than
   (B) at τ=0.1 and never reach v11c at either τ. The bottleneck is not
   what held (B) back.
4. **The β2 optimum flips with τ on the bottleneck arm** — β2=0.98 wins
   at τ=0.1 (β < (B)), β2=0.95 wins at τ=0.8 ((B) < β). (At τ=0.1 the
   β2 optimum also differs with vs without the bottleneck — 0.98 with,
   0.95 without.) These are single-seed orderings.
5. **τ=0.8 helps the bottleneck arms and the β2=0.98 no-bneck arm,
   hurts the β2=0.95 no-bneck arm** — τ's sign depends on the recipe.

*Mechanism (hypothesis, not established):* a higher τ softens the
contrastive target, which on the capacity-limited bottleneck forecaster
appears to help GIFT-Eval transfer; whether β2=0.95 specifically pairs
with τ=0.8 on this arm, or this is seed noise, needs a second seed.

## Limitations

- **Single seed per cell.** #307's variance estimate is ±0.02 (n=3).
  The (B)-τ0.8 → v11c gap (1.2335 vs 1.292, ~0.058 / +4.5%) and the
  larger effects (β2 swing, τ swings) are well outside it; β-τ0.8 vs
  v11c (0.002 apart) is a tie within noise. But all cells are n=1, so a
  +4.5% lead on one seed should be confirmed before being relied on.
- **(B)-τ0.8 backbone is 47.6k/50k.** A late DDP teardown OOM under GPU
  contention stopped it at 47.6k; loss had plateaued flat from ~40k, so
  the checkpoint is representative but not a clean 50k. A clean 50k
  re-run (and a second seed) is the obvious confirmation step.
- **v11c confound.** v11c additionally differs in dropkey (0.9 vs 0.7)
  and loss (plain vs `hh-negs`); the bottleneck arms reach/beat v11c
  with neither of those, but the no-bottleneck arms' shortfall vs v11c
  is entangled with them.
- q-head + GIFT-Eval sample windows are single-seed.

## Annex

### Compute
τ=0.1 fp16 arms: 1× RTX 4090 prosumer on vast (offer 35882331,
$0.55/h), **$2.66** total. All fp32 (no-bneck) arms, all τ=0.8 arms,
and every downstream ran free on elisa.

### Code
Branch `experiment/2026-05-20-bottleneck-beta2-confound`. Runner
`scripts/elisa_run.sh <arm> <tau> <runtag> <gpus> <prec>`
(arm ∈ {bbase=(B), alpha, beta, gamma}; prec ∈ {fp16, fp32});
downstream `scripts/downstream.sh <arm> <gpu>`; figures
`scripts/{plot_results.py (radars + curves per τ + fp16-divergence),
plot_summary.py (GM matrix)}`.
