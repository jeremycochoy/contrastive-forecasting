# qhead-improvements: a transformer head + eval-matched input layout

## Question

The frozen contrastive backbone "backbone-beta"
(`tiny_full4096_moirai_hp_FRESH_RESUME50k_FINAL.pth`; C=1, H=384,
nhead=6, 6 layers, T_RAW=4096) was trained to be competitive with
Moirai. Atop it sits a *forecasting head* — a small network that reads
the backbone's latents and emits the actual multi-step forecast. The
legacy head is a bidirectional GRU inherited from earlier work. **With
the backbone frozen, how far can a better head alone push downstream
GIFT-Eval accuracy, and can it reach Moirai's 0.809?**

> **GM-Rel MASE** (the metric throughout) = geometric mean over the
> benchmark's configs of (model MASE ÷ seasonal-naive MASE), where MASE
> is Mean Absolute Scaled Error. 1.000 = seasonal-naive; lower is
> better; the best leaderboard models reach ~0.67–0.81. The full
> benchmark is **97 configs**; a fast **triage proxy** (defined below)
> uses 11 of them.

## Result

Two head changes carry almost all of the gain. **(1)** Replace the
legacy bidirectional GRU head with a *causal transformer* at the
backbone's width (H=384, nhead=6), stacked to 12 layers — deeper than the
6-layer backbone. **(2)** Match
the head's train-time input layout to what it sees at eval — feed it
`[encoder-latents, forecaster-latents]` (`e_then_f`) under a leak-free
causal mask instead of forecaster-latents only. Together these take
full 97-config GM-Rel MASE from **1.183** (legacy GRU baseline, the #10
backbone) to **1.029** (run R9_E13) — **−13%**, just above
seasonal-naive (1.000), still **+27% vs Moirai** (0.809).

![Full 97-config GIFT-Eval GM-Rel MASE. R9_E13 (solid green, recomputed from this experiment's committed full eval) lands at 1.029 — above seasonal-naive (1.000), still short of Moirai (0.809); the legacy GRU #10 baseline (hatched, carried from the prior #10 report) is 1.183. Lower is better.](plots/headline_full_eval.png)

R9_E13 is recomputed here as the geomean of the 97 per-config Relative
values in
`results/R9_E13_xfmr12L_quant_moirai_cosine_e_then_f_60k_full/summary.txt`
(= 1.0288). The legacy-GRU baseline **1.183** is carried from the prior
#10 RESUME50k report (recorded as GM-MASE 1.1828 in
[notes/CANDIDATES.md](notes/CANDIDATES.md)); its per-config full eval is
not committed in this experiment, so it is drawn hatched and not
recomputed here. None of the head-side axes closed the remaining gap to
Moirai.

### The path to the winner

Each round trains one head recipe and scores it on the 11-config
triage proxy. The triage GM-Rel MASE per round (geomean of the Relative
column in each `results/<run>_triage/summary.txt`):

![Triage GM-Rel MASE per round (11-config proxy; lower better). Grey bars sit at or above seasonal-naive (red dashed, 1.000); the e_then_f winner R9_E13 (dark green, 0.990) is the best round — it and its longer-trained sibling R9_E14 (0.994) are the only two below the line. Moirai's full-eval 0.809 shown for scale (blue dotted). Single triage run per round (eval is deterministic).](plots/round_progression.png)

The thread is monotone where it should be and flat where it plateaus:
the legacy GRU (1.128) → linear probe (R1_E1, 1.066) → causal
transformer (R3_E4, 1.017) → deeper+longer transformer (R5_E7, 1.002),
then a plateau at ~1.00–1.02 that three further axes (longer training,
bidirectional+longer-forecast, Gaussian-NLL loss) do not break. The
`e_then_f` input-layout fix crosses **under** seasonal-naive on triage
(R9_E13 0.990, and R9_E14 0.994 at longer training); R6_E8 (1.089) is the
only large regression (R7_E9 1.020 and R8_E10 1.020 are also small
regressions above the R5_E7 1.002 floor). The full-eval numbers for the
two runs evaluated on all 97 configs:

| run | head + training | triage GM-MASE | full GM-MASE |
|---|---|---|---|
| baseline (legacy GRU) | GRU-q, 30k | 1.128 | 1.183 (carried, #10) |
| **R9_E13** (winner) | **xfmr-q 12L + `e_then_f`, 60k** | **0.990** | **1.029** |

vs the leaderboard on full eval (lower is better):

| Sundial | TimesFM | PatchTST | Chronos | Moirai | Naive | Baseline #10 | **R9_E13** |
|---|---|---|---|---|---|---|---|
| 0.673 | 0.680 | 0.762 | 0.786 | 0.809 | 1.000 | 1.183 | **1.029** |

> **Triage proxy = 11 small-test-set configs** (`bizitobs_*`,
> `ett{1,2}/{15T,H}`, `electricity/H`, `covid_deaths/D`,
> `us_births/D`), kept because each finishes in seconds, turning a ~6 h
> full eval into ~5 min. It is **biased low**: it drops whole
> hard/long-horizon domains (M4, `loop_seattle`, `bitbrains_*`,
> medium/long terms) rather than subsampling within configs. R9_E13 is
> the only run evaluated both ways here: triage 0.990 → full 1.029
> (+0.038). For the baseline, triage 1.128 (this experiment) vs full
> 1.183 (carried from #10, not re-evaluated here) gives +0.055.
> So a triage score below 1.000 (R9_E13's 0.990) does **not** imply
> beating seasonal-naive on the full benchmark — R9_E13 is 1.029 there.
> Full bias rationale in [notes/TRIAGE_NOTE.md](notes/TRIAGE_NOTE.md).

## Protocol

- **Model under test:** the frozen backbone-beta latents + a forecasting
  head trained on top (AdamW; backbone weights never updated). Heads
  vary across rounds along five axes: architecture (GRU / linear /
  causal-transformer 6L / 12L / bidirectional), training length
  (30k / 60k / 100k steps), LR schedule (constant / WSD / cosine), loss
  (pinball-quantile / Gaussian-NLL), and train-time input layout
  (forecaster-latents-only vs `e_then_f`).
- **`e_then_f` layout (the winning change):** at eval the head sees
  `[e_ctx, rolled_f]` — encoder latents for the context window followed
  by rolled forecaster latents — but training used to feed it only
  `f_0..f_{T-1}`. `--head-train-input e_then_f` instead feeds the
  length-2T sequence `[e_0..e_{T-1}, f_0..f_{T-1}]` with a custom
  no-leakage mask: every row is causal, and each f-block row may attend
  to e-columns only up to its own position `p_f`, so it cannot peek at
  `e_{p_f+1}` (which encodes the target patch). Without the mask,
  the head reads leaked target info and training loss collapses; with the
  mask it stays at the no-leak plateau yet eval still improves.
- **Benchmark:** the official GIFT-Eval suite (`gift_eval` + GluonTS),
  deterministic point forecast scored against seasonal-naive, primary
  metric GM-Rel MASE. Every number for our runs is recomputed from the
  per-config outputs in [results/](results/)
  (`<run>_triage/summary.txt`, plus the committed full-eval summary for
  R9_E13 — the only full 97-config eval committed here; R5_E7's full run
  is a 28-config partial, so its full GM-MASE is not quoted); leaderboard
  reference values are published GIFT-Eval figures.

## What worked

1. **Causal transformer head matching the backbone's depth + width.**
   R3_E4 (6L, H=384, nhead=6, ~10.7M params), trained from scratch with
   Moirai HP (β2=0.98, wd=0.1, lr=1e-3), cosine LR + 1k-step warmup:
   triage **1.066 → 1.017** vs the linear probe. The linear probe
   itself already beat the legacy GRU (1.128 → 1.066), so capacity is
   not the binding constraint.
2. **Stack depth + length on the transformer.** 12 layers + 60k steps +
   2k warmup → R5_E7 = **1.002** (−1.4% on top of R3_E4). This is the
   floor for everything that keeps the forecaster-only input layout.
3. **Match the train-time input layout to eval (`e_then_f`).** R9_E13:
   triage **1.002 → 0.990** (R9_E14, the same recipe at 100k, lands at
   0.994 — both e_then_f rounds clear seasonal-naive on triage), and the
   headline full-eval win (1.029).

## What didn't work (informative null results)

1. **Linear probe HP/schedule (R2_E3).** Switching the linear probe to
   Moirai HP + WSD reached the same ~1.066 triage score as the
   constant-LR linear probe (R1_E1) (1.0669 vs 1.0655): the linear head
   is at its representational ceiling regardless of HP.
2. **Bidirectional head + forecast_len=128 (R6_E8).** Triage **1.089**
   vs R3_E4's 1.017 — the only regression. A bidirectional head attends
   to *real* future latents at training but *rolled-out* (error-laden)
   ones at eval. No ablation isolates bidir vs fl128, so the regression
   is consistent with a train/eval mismatch but not pinned to one factor.
3. **Longer training to 100k (R7_E9).** 1.020 ≥ R5_E7's 1.002:
   extending the cosine schedule past 60k did not help.
4. **Gaussian-NLL loss (R8_E10).** Triage 1.020, same as R7_E9 and worse
   than R5_E7's 1.002. The ~1.02 plateau is not a pinball loss-surface
   artifact — swapping to smooth parametric NLL with the same head and
   schedule does not move it.
5. **Longer training under `e_then_f` (R9_E14).** Triage 0.994 vs
   R9_E13's 0.990 at 60k; the +0.004 is not interpreted as a real
   difference — no replicates were run to estimate variance. Longer
   training does not add to the input-layout win either.

## Hypothesis going forward

Of the five head-side axes — architecture, length, schedule, loss,
train/eval input layout — four converge to ~1.00–1.02 triage GM-MASE.
Only matching the train input layout to the eval input layout
(`e_then_f` + leak-free mask) crosses under seasonal-naive on triage
(0.990) and delivers the full-eval headline (1.029). The remaining gap
to Moirai on full eval is **+0.220** (1.029 vs 0.809, +27%), and none
of these head-side changes closed it. The plateau across four
orthogonal head axes — together with the linear probe already beating
the GRU — points (a hypothesis, since no backbone was varied in the
head experiments) to the **frozen backbone's latents** as the binding
constraint rather than head capacity or recipe; scaling or retraining
the backbone is the natural next line, out of scope here.

*A side-investigation of which cheap backbone metric predicts
downstream MASE (AUC ranked best, Spearman ρ≈+0.70, n=5) is in
[notes/BACKBONE_DIAGNOSTICS.md](notes/BACKBONE_DIAGNOSTICS.md) — a
different question that does not bear on the head-improvement thread.
Per-round candidate rationale, the PR/test/artifact inventory, and
operational events (budget, the R7_E9 preemption) live in
[notes/CANDIDATES.md](notes/CANDIDATES.md) and
[notes/PIPELINE.md](notes/PIPELINE.md).*
