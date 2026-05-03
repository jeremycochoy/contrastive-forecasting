# Head / Rollout Comparison

Compared 6 rollout strategies (2 value-space, 4 latent-space) to determine whether the head/rollout architecture is the bottleneck preventing GIFT-Eval improvement beyond ~1.27 MASE. The backbone's contrastive gap improves from 0.10 to 0.43 with more training, but MASE stays flat -- suggesting the head, not the backbone, is the limiting factor.

## Key Result

Value-space rollout narrowly beat latent-space rollout when using prediction heads (A1=1.275 vs B1=1.258). All B-variants scored similarly (~1.26-1.29), and none significantly outperformed A1. This led to the hypothesis that the head itself was misaligned: it predicted the future instead of reconstructing what each latent represents. That hypothesis was confirmed in the follow-up [reconstruction head experiment](../2026-04-17_reconstruction-head/REPORT.md).

## Documents

| File | Description |
|---|---|
| [DESIGN.md](DESIGN.md) | Detailed design: A1/A2 value-space variants, B1-B4 latent-space variants, implementation, training setup. |
| [EXECUTION_PLAN.md](EXECUTION_PLAN.md) | Infrastructure plan: 3x RTX 4090 on Vast.ai, parallel execution timeline, cost estimate ($6.50). |

## Variants

| ID | Type | Head output | Strategy | GM-Rel MASE |
|---|---|---|---|---|
| A1 | Value-space | 128 values | Slide by 128 | 1.275 |
| A2 | Value-space | W=16 values | Slide by 16 | 1.262 |
| B1 | Latent-space | 128 values | Decode at end | 1.258 |
| B2 | Latent-space | 128, crop to 16 | Decode each step | 1.258 |
| B3 | Latent-space | 128 non-overlap | Decode every 8 | 1.260 |
| B4 | Latent-space | W=16 values | Decode each step | 1.288 |

## Files

| File | Description |
|---|---|
| `DESIGN.md` | Experiment design and rollout strategy definitions |
| `EXECUTION_PLAN.md` | Vast.ai execution plan and cost breakdown |
