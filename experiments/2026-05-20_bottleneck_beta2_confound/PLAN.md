# #309 Bottleneck × β2 confound on (B): can α match v11c?

## Question
(B) closed half the gap to v11c (1.4377→1.3572, still +5% above v11c
1.292), but differed from v11c on ≥6 axes. Two axes are testable now:
forecaster bottleneck and AdamW β2. Test all four cells of {β2 ∈
{0.95, 0.98}} × {fcst-bneck ∈ {kept, removed}}; the (B)=baseline
backbone-of-record cell is at (0.95, kept) with full-97 GM-MASE 1.3572.

## Arms (priority α > β > γ)

| Arm   | β2   | fcst-bneck | Hypothesis                                 |
|-------|-----:|-----------:|--------------------------------------------|
| (B)   | 0.95 | kept       | reference (#303 cl_hh_50k, 1.3572)         |
| **α** | 0.98 | removed    | matches v11c on these two axes → 1.292 |
| **β** | 0.98 | kept       | β2 alone explains some gap                 |
| **γ** | 0.95 | removed    | bottleneck alone explains some gap         |

Recipe = byte-identical to #303 `cl_hh_50k` (box_run.sh) except:
- α : `--adam-beta2 0.98`, drop `--forecaster-d-model/--forecaster-n-heads`
- β : `--adam-beta2 0.98`
- γ : drop `--forecaster-d-model/--forecaster-n-heads`

Loss = `cosine_similarity_batch_full_hh_negs`; 2-GPU DDP, bs128/GPU
(global 256), 50k, dropkey 0.70 shared, fp16 body / fp32 residual+pemb,
seed 20260520.

## Execution
- 1 vast prosumer 2×4090 24GB box, ≥0.95 reliability (currently
  offer **36244527** @ $1.81/h, US, 0.990 reliability).
- Serial α→β→γ via `scripts/box_run_serial.sh`. Each backbone ~2h wall
  (extrapolating #303 cl_hh_50k @ 6.5 sps), total ~6-7h, **~$12-14**.
- Local 15-min `sync_loop` pulls into MAIN checkout (CLAUDE.md rule);
  worktree holds scripts only.
- Q-head 30k 2L-causal + GIFT-Eval triage(11) + full(97) per arm on
  elisa GPU 1 as backbones land (free; q-head is single-GPU).

## Deliverables (per REPORT_STANDARD)
- Per-domain star, 4 lines: (B), α, β, γ, with v11c reference.
- Training curves (loss / dim-usage / loss_tau_ref / 1−AUC) log/log.
- `RESULTS.md` with α-match-v11c verdict.
