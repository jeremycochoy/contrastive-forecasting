«Agent writing»

## Update: full-training eval flips the verdict, and disentanglement attributes the gain to the triplet

The earlier "neutral (CIs straddle 0)" reading was taken at the **best-loss** checkpoint. Evaluating
the same backbone at **full training (last checkpoint)** flips it: **L3+nobn+triplet reliably beats
the base on both heads.** Please update the PR title — it is no longer neutral.

Naming (each arm relative to the base = 6L encoder, 128-wide bottleneck forecaster, no triplet):
L3 = 3-layer encoder; L6+nobn = full-width forecaster; L3+nobn = both; **L3+nobn+triplet = this PR**.

### Headline — L3+nobn+triplet vs base

| head | best-loss GM | Δ (90% CI) | last GM | Δ (90% CI) |
|---|--:|:--:|--:|:--:|
| 2L | 1.220 | +0.007 (−0.014, +0.028) ns | **1.181** | **−0.032 (−0.050, −0.014) better** |
| 6L | 1.211 | +0.013 (−0.012, +0.036) ns | **1.169** | **−0.029 (−0.049, −0.011) better** |

Base GM 1.213 (2L) / 1.198 (6L). Both last-checkpoint intervals sit fully below zero on both heads.

### Disentanglement — the gain tracks the triplet, not bottleneck removal

Running the three changes one at a time (same base, seed, budget):

- **L6+nobn alone is reliably worse** on both heads at both checkpoints.
- **L3 alone is inconsistent** (worse at 2L-best and 6L-last, neutral elsewhere).
- **L3+nobn** (no triplet) is neutral except a marginal 6L-last improvement (−0.015, upper bound −0.002).
- **L3+nobn+triplet** is the only arm reliably better on both heads — and only at full training.

So the full-training gain is attributable to the triplet, not to dropping the bottleneck or shrinking
the encoder.

### Two further notes

- **Train-longer hurts.** Extending 12.5k → 25k reliably worsens both heads (2L +0.037 [+0.018,
  +0.055]; 6L +0.024 [+0.013, +0.035]). Downstream peaks near 12.5k, within the first epoch (25k ≈ 0.6
  epoch; one full pass of `small_v1` ≈ 41,900 steps).
- **Lower contrastive loss ≠ better transfer here.** The triplet raises the pretext loss (1.065 vs
  base 0.895) yet improves downstream; dropping the bottleneck lowers it (0.662, lowest) and raises
  used dimensions, yet transfers worst.

### Pending (do not merge the "neutral" conclusion)

Two arms isolate the triplet further; downstream numbers pending:

- **L6+nobn+triplet** — best-loss landed (2L +0.061 worse, 6L +0.027 ns); last running, ETA midday.
  Tests whether the triplet rescues the otherwise-harmful full-width forecaster at 6 encoder layers.
- **base+triplet** — backbone training, ETA tonight. Cleanest isolation (base vs base+triplet differ
  by the triplet alone). #326 (C-only crossfade on the 10%-fork base) gave ≈−0.013 over its base vs
  this arm's ≈−0.030, so a modest positive effect is expected.

### Report and figures

The report at `experiments/2026-06-03_crossfade_triplet/crossfade_triplet.md` is rewritten around the
best-vs-last flip and the disentanglement. Two figures need a refresh before merge:

- `plots/gm_summary.png` — still shows only the base vs the combined arm at best-loss (the old
  "slightly worse" bars). Regenerate as a disentanglement bar chart (all arms, best + last, both heads).
- `plots/perdomain_delta.png` — computed at the best-loss checkpoint (neutral aggregate). A
  last-checkpoint version would match the new headline; the report flags it as best-loss for now.

`plots/disentangle_metrics.png` (training dynamics, all arms) and `plots/triplet_schematic.png` are
current.

Source of truth for every number above: `/tmp/cf-328-scoreboard.py` (GM + paired-bootstrap Δ + 90% CI
per arm, best and last, 2L and 6L).
