# Window size: does halving the patch window (W=32 → W=16) hurt the Tiny backbone?

The Tiny contrastive backbone patches each 4096-step series into windows of W timesteps. Halving W from 32 to 16 doubles the patch count (128 → 256), giving finer temporal resolution but making self-attention cost more (it scales with the square of the sequence length). The question: does the smaller window degrade contrastive learning — and is the extra cost worth it?

> *Contrastive gap = FF − FP: how much more a window's forecast resembles its own future (FF) than its present (FP). It is the margin the contrastive loss grows; higher is better. Here FF/FP are cosine similarities measured on a fixed held-out synthetic (ARMA) validation batch, seed 0.*

## Result

**In this single-run-per-arm test, W=16 did not degrade the gap — it reached a higher one (0.093 vs 0.082 at 10k steps, +13%) while using 37% less VRAM, at the cost of being ~18% slower per step.** Single seed per arm, so treat the ranking as suggestive, not conclusive (see caveat below).

![Contrastive gap vs training step, three arms to 10k steps (gap values parsed from the committed logs in `logs/`). W=16 bs=24 (red) is level with W=32 (blue) early, pulls clear after ~6k steps, and peaks at 0.093 vs W=32's 0.082. W=16 bs=28 (green) tracks bs=24 but was wall-time-capped at step ~6.9k.](plots/gap_vs_step.png)

W=16 is ahead at every matched step from 6k on and the margin widens, so the win is not a single lucky checkpoint. But "step budget" flatters the slower arm, so the fairer axis is wall time:

![Contrastive gap vs wall-clock minutes, same three arms (wall-time axis from the report's per-step timing; the gap-vs-step ranking above is the authoritative comparison). The arms overlap until ~21 min; past that W=16 bs=24 (red) pulls clear, reaching 0.093 by ~32 min. At the W=32 run's own 29-min budget (dashed line), W=16 bs=24 sits at ~0.091 and is still climbing while W=32 has plateaued at 0.082.](plots/gap_vs_walltime.png)

Even after paying the 18% per-step penalty, W=16 bs=24 overtakes W=32 on wall time around 21 min and never gives the lead back. Summary of the three arms:

| Arm | Patches | Batch | VRAM | Speed | Best gap (step) |
|-----|---------|-------|------|-------|-----------------|
| **W=32** | 128 | 32 | 23.4 GB | 5.7 sps | 0.082 (9k) |
| **W=16, bs=24** | 256 | 24 | 14.7 GB | 4.7 sps | **0.093 (9k)** |
| W=16, bs=28 | 256 | 28 | 17.6 GB | 3.9 sps | 0.082 (6k)¹ |

¹ bs=28 was wall-time-capped at 29.2 min (the W=32 run's wall time) and stopped at step 6895; it never ran the 7k–10k steps the other arms did.

**Does the VRAM headroom buy anything if spent on a bigger batch?** No — it backfires. Raising the W=16 batch from 24 to 28 (17.6 GB) slowed training 4.7 → 3.9 sps (17% slower) with no gap gain: in 29 min, bs=28 reaches the same ~0.082 as W=32, whereas bs=24 keeps stepping and reaches 0.093 a few minutes later. **More steps beat a bigger batch at matched wall time; bs=24 is the sweet spot for W=16 on Tiny.**

## Protocol

- **Backbone:** Tiny (C=4, H=512, L=6, GRU encoder, 8 heads, ffn_mult=4, depthwise_conv=3, dropout=0.1). W=32 → 19,960,576 params; W=16 → 19,952,384 (the two differ only in the patch projection, ~8k params).
- **Loss:** contrastive cosine similarity with batch negatives, temperature 0.07, identical across arms (`cosine_similarity_batch_no_time_neg`).
- **Data:** on-the-fly synthetic ARMA batches (T_raw=4096, C=4), AdamW lr=1e-4, no grad clipping. Gap evaluated every 1000 steps on a fixed ARMA validation batch (seed 0).
- **Arms:** W=32 bs=32 and W=16 bs=24 each ran 10k steps; W=16 bs=28 was capped at the W=32 run's wall time (29.2 min). One run per arm, RTX 4090.
- **Why bs differs:** attention memory per sample scales with the square of the sequence length, so W=16's 256-patch sequence needs a smaller batch to fit. bs=24 was W=16's chosen operating point; bs=28 is the follow-up probe.
- **Sources:** gap-vs-step values are parsed directly from `logs/window_test_{w32,w16,w16_bs28}.log` (authoritative). The wall-time axis comes from the per-step timing recorded with the report (the logs carry sps/VRAM but no timestamps). Plot script: [`scripts/plot_window_curves.py`](scripts/plot_window_curves.py).

## What we learned

Halving the patch window to W=16 (256 patches) did not hurt contrastive learning on Tiny in this test — it produced a higher gap at both matched steps and matched wall time, and cut VRAM by more than a third, freeing headroom for larger models. The one cost, ~18% slower steps, was repaid by the per-step learning gain within ~21 minutes. Spending the freed VRAM on a larger batch (bs=28) was counterproductive; bs=24 was best.

**Caveat — single seed.** Each arm is one run on one validation seed, and the W=16-vs-W=32 gap margin (~0.011) is small relative to the run-to-run wobble visible in both curves (e.g. W=32 dips to 0.073 at 8k between two 0.08+ points). This is a directional result — finer windows look at least as good as coarser ones here, not worse — and is not a clean A/B at matched batch size. A confident claim would need multiple seeds and a batch-matched W=32-vs-W=16 pair; this run varies W and batch together. It is not, on this evidence, an architectural conclusion.

*(Notes on the bs=28 wall-time cap and an earlier copy-paste error in the bs=28 step table live in [notes/data-provenance.md](notes/data-provenance.md).)*
