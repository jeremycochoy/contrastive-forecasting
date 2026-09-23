# A value-space reference model: the same body, trained on the actual values

*(#415. The run is scheduled but not yet started, so this file holds the
question and the protocol. The result replaces this note.)*

## Question

Does the same architecture reach GM-Relative MASE below 1.0 when it is trained
on the actual values instead of on a latent contrastive objective?

> *GM-Relative MASE = the geometric mean, over the 97 GIFT-Eval configs, of
> (model MASE / seasonal-naive MASE). 1.0 is seasonal-naive. Lower is better.*

Every published reference model clears 1.0: Moirai-Small reads 0.809 and
Sundial 0.673. This project's contrastive line never has. #414's best is
**1.1369**, held by `k3_r100_09_lr56_fix09_dec10k_cos200k` at the end of one
pass over the data. Until a value-space twin of the same body exists, the gap
cannot be attributed to the objective rather than to the architecture.

## What changed, and what did not

The model is #414's cell `arm6_v2_combab_alignT` at `d_model` 384 —
11,431,548 trainable parameters, the size of Moirai-2-Small. The body and the
input head are that cell's, flag for flag.

| | #414 | #415 |
|---|---|---|
| Body, input head, width, patching, normalisation | same | same |
| What the model predicts | the next latent | the next patch's **values** |
| Where the rollout runs | latent space | **value space** |
| What the loss reads | cosine similarity | **the actual values** (pinball, 9 quantiles) |
| Teacher, EMA, `L_rep`, `L_align`, CPC, SIGReg | on | **gone** |
| Optimizer | 5.6e-5, cosine | **Moirai recipe**: 1e-3, wd 0.1, betas (0.9, 0.98), flat |

The rollout is what "value space" means here. At depth j the model re-reads
its own median forecast as the next patch, runs the whole cell on it, and is
supervised against the patch j further out. Depth 3 matches the cell.

## Protocol

- **Data.** `jeremycochoy/gift-pretrain-full-4096`, path `small_v1`,
  `T_raw` 4096, C = 1, batch 64, seed 20260520 — #414's, so 665,000 steps is
  one pass over the data in both runs.
- **Stops.** 40k, 100k, 200k, 300k, 400k, 500k, 600k and 665,000 steps, on one
  trajectory: each leg resumes the previous stop with its optimizer state.
- **Scoring.** #414's path, unchanged, so the numbers compare: a quantile head
  (2-layer transformer, forecast length 16, batch 256, lr 1e-3, head seed
  20260722, 30,000 steps) on the **frozen** backbone, then the 97 GIFT-Eval
  configs under strategy **B4** at forecast length 16. Each stop keeps its own
  `all_results.csv` for the per-dataset comparison against `cos200k`.
- **One run** per stop. GIFT-Eval point scoring is deterministic, so the only
  noise is training init, and this card has no seed sweep.

The one command:

```bash
BB_GPU=0 bash reports/2026-09-23_value_space_reference/run.sh
```

## Result

Pending. The deliverable is the GM-Relative MASE trajectory over the eight
stops and a per-dataset comparison against #414's 1.1369.
