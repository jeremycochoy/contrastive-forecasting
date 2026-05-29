# #322 — execution log

Operational journey for this card (infra, decisions, corrections). The science is
in [`RESULTS.md`](RESULTS.md); the design rationale is in [`PLAN.md`](PLAN.md).

## Base branch
#322 needs three pieces of code, **none merged to `experiments`** yet:
- the 2-GPU DDP cross-rank gather (`src/dist_utils.py`, `--shard-loss-on-batch`) — from #296;
- the forked-ARMA synth (`src/synthetic_forked_arma.py`, `--synth-kind forked-arma`) — from #318;
- the two contrastive losses `cosine_similarity_batch_full_hh_negs[_xshh_allt]` — from #318.

All three are committed on the #320 branch (`experiment/2026-05-26-forked-6Lf`, which
has `experiments`=4c58857 as ancestor + 316 commits). So this card's branch
`experiment/2026-05-29-forked-6Lf-b1024` is reset onto the #320 tip (a8be7f2) — the
standard stacked-experiment-branch pattern (#318→#320→#322). The PR targets
`experiments`; if #318/#320 merge first, rebase shrinks the diff to #322 only.

## Compute layout (elisa, 2× RTX 4090, 24 GB each)
- GPU 1: free (24 GB).
- GPU 0: ~7.9 GB used by a **foreign** project (`/tmp/rnd-symbol-mix-2026-05`,
  `trainline` symbol-mix clamp_sweep — not contrastive-forecasting). Per the
  shared-machine rule, never touched. 2-GPU DDP needs both cards, so heavy DDP runs
  wait for GPU 0 to free; feasibility is measured on the free GPU 1 first.

## HF token
`experiments/hf_token.txt` is gitignored, so the fresh worktree had none; copied from
the #320 worktree (38 B). Required or HF throttles the stream and idles the GPU.

## Feasibility smoke (batch 1024)
Single-process @1024 measures the full-1024 "all-together" loss + full-1024 forward on
one card (zero risk to the foreign GPU-0 job). Outcomes decide the design:
- fits → run each backbone single-GPU @1024 (DDP unnecessary; all-negatives-together native);
- forward OOMs → DDP splits the forward (512/rank) + gathers latents for the full-1024 loss;
- loss OOMs → full pooled 1024 is infeasible on 24 GB for that arm.

(results appended below as measured)

### Smoke results (GPU 1, single-process, byte-identical recipe except batch/steps)
| config | outcome | true max_allocated | sps (steady) | note |
|---|---|---|---|---|
| β @1024 single | **OOM** | — (GRU op alone wanted 17.9 GB) | — | forward can't fit on 1 card |
| β @512 single | ok | **22.55 GB** | ~2.0 (502 ms/step) | forward-bound |
| allt @512 chunk=4 single | ok | **22.55 GB** | ~0.5 (1.98 s/step) | loss stays *under* the GRU peak |

The **identical** 22.55 GB peak for both arms ⇒ the wall is the **GRU patch-encoder
fwd/bwd at 512 seqs/rank**, not the loss. So DDP per-rank = forward(512) [~22.55 GB,
deterministic] + gathered loss(1024); keeping the all-time loss peak below the GRU
peak (small `XSHH_ALLT_CHUNK`) makes **all 5 arms fit at ~22.5–23.5 GB/rank** — but
each rank needs a near-empty 24 GB card ⇒ both GPUs must be free (GPU 0's ~16 GB free
is insufficient). GRU-checkpointing is held in reserve if a rank proves too tight.

### Pivot: 2-GPU DDP → single-GPU @1024 with a checkpointed GRU
GPU 0's occupants turned out to be **5 foreign Jupyter kernels (rnd_dmytro / rnd_kacper)
alive 9–14 days** (~8 GB, idle) plus a rotating clamp_sweep — i.e. GPU 0 is permanently
~8 GB-occupied and will never reach the ~23.6 GB free a DDP rank needs. Evicting another
team's multi-day kernels is off the table. So DDP is abandoned for a **single-GPU @1024
on the near-empty GPU 1** — which pools all 1024 in the negatives *natively* (one batch,
no gather), satisfying the card even more cleanly than DDP.

To fit 1024 on one 24 GB card, the **GRU patch-encoder is gradient-checkpointed + chunked**
(`PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4`, env-gated + training-only in `src/encoders.py`).
The GRU over 65 k sequences was the wall; checkpointing trades its stored bwd activations
for recompute, **byte-identical in the forward**, so the trained backbone equals #320's
recipe at 4× batch. Measured single-GPU @1024 checkpointed (true allocated / step):
- β: **11.90 GB**, 334 ms/step (~3 sps) → ~70 min / 12.5k.
- all-time (chunk=2): **12.43 GB**, 3.53 s/step (~0.28 sps) → ~12 h / 12.5k.
Both well within 24 GB. Total ≈ 39 h on GPU 1 (β first, then the 3 all-time arms),
neighbour-safe. DDP scripts retained but unused.

### Batch-1024 divergence at #320's LR → fixed with LR 5e-4
The first β·0.8% backbone at batch 1024 with #320's LR (1e-3, constant) trained healthily
to ~step 1500 then **diverged**: loss 6→13, gap collapsed 1.2→0.2, R²_rand → negative,
retrieval AUC/Top1 collapsed by ~step 4500 (vs #320 b256 which drops monotonically to
loss 2.25, gap ~1.05, R² 0.997). Diagnosis:
- **Not the GRU checkpointing.** A batch-256 run *with* checkpointing reproduces #320's
  stable batch-256 trajectory (loss 5.54 / gap 0.83 / R² 0.93 / AUC 1.0 at step 120) —
  forward + gradients are byte-correct.
- A genuine **large-batch optimization instability** (4× batch, ~14× pooled negatives).
  Per CLAUDE.md (fix divergence via optimization/normalization, never grad-clip), LR was
  halved 1e-3 → **5e-4**: the collapse is gone (gap rising 0.94→1.05, R² 0.95+, AUC 1.0
  through step 1800; loss rises mildly then plateaus, consistent with the larger
  negative-count InfoNCE floor, not a collapse). 5e-4 is the **minimal** change to make
  batch-1024 trainable — reported as a recipe deviation. The full 12.5k-step β·0.8% run
  is the live stability confirmation through the ~4500 danger zone.

`scripts/plot_collapse.py` → `plots/collapse_handling.png` visualises this: log-log loss
+ gap for b256(stable) / b1024-1e-3(diverges, gap 1.2→0.2 at ~step 4500) /
b1024-5e-4(handled, gap → 1.29). Confirmed: 5e-4 is healthy through step 4300+, past the
1e-3 collapse point.

### Batch-1024 collapse — root cause + fix (QK-norm + attention-output RMSNorm)
b1024 collapsed at every LR/τ (LR 1e-3 ~step 4500, 5e-4 ~6000, τ=0.20 ~5700): gap peaks
then crashes, cross_batch cosine blows up 0.001→0.57, loss rises. Via --log-attn-amplitude:
- It's an **activation-amplitude runaway**. The **ENCODER self-attention output (sa_out)**
  explodes (b256 flat ~35 → b1024 →1300+); the **forecaster** attention stays flat (~36);
  the FFN's residual add is negative — not the FFN. sa_out grows the residual stream
  (38→6400) and, in fp16, corrupts the cosine latents → directional collapse.
- **Two numerical instabilities, both must be fixed:** (1) QK logits explode (→2700, q/k
  weights grow) → **QK-norm**; (2) attention OUTPUT/residual explodes (W_v/out_proj grow) →
  **Gemma2-style sandwich RMSNorm on the attention output** (`--attn-out-norm`, attention
  only). Fixing one leaves the other: qk-norm alone → residual →10573, collapses;
  attn-out-norm alone → residual flat but QK logits →45695 (fp16 overflow → NaN).
- **`--qk-norm --attn-out-norm` @ LR 1e-3 clears the collapse zone.** Through step 6500
  (past every prior collapse point): loss−floor descends 13.3→1.07, gap stable ~1.02,
  cross_batch flat ~0.0004, qk_logit ~17, resid ~50 — the b256-like converging regime.
Both norms are standard, via the verified SDPA path (off = byte-identical, SDPA==MHA
diff 0.0). This is #322's enabling recipe deviation from #320 (unneeded at b256). The
non-standard V-norm was tried and removed. Plots: plots/{collapse_handling,
activation_amplitudes,cosines_through_training,block_split}.png.
