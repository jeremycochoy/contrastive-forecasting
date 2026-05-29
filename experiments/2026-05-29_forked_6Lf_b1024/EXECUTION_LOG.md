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
