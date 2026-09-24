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
pass over the data. Without a value-space twin of the same body, no number can
say whether the gap comes from the objective or from the architecture.

## What changed, and what did not

The model is #414's cell `arm6_v2_combab_alignT` at `d_model` 384 —
11,431,548 trainable parameters, the size of Moirai-2-Small. The body and the
input head are that cell's, flag for flag. A test diffs the two command lines.

| | #414 `cos200k` | #415 |
|---|---|---|
| Body, input head, width, patching, normalisation | same | same |
| What the model predicts | the next latent | the next patch's **values** |
| Where the rollout runs | latent space | **value space** |
| What the loss reads | cosine similarity | **the actual values** (pinball, 9 quantiles) |
| Teacher, EMA, `L_rep`, `L_align`, CPC, SIGReg | on | **gone** |
| Rate | 5.6e-5 to 1e-6, cosine over 200,000 steps, then 1e-6 | **5e-4 to 1e-6, cosine over 665,000 steps** |
| Batch, weight decay, betas, warmup, grad clip | 64, 0.1, (0.9, 0.98), none, none | same |

The rollout is what "value space" means here. At depth j the model re-reads
its own median forecast as the next patch and runs the whole cell on it. The
loss of depth j reads the patch j further out. Depth 3 matches the cell.

## The rate

The Moirai recipe ran on this corpus in
`experiments/2026-05-03_exp_realonly_full4096_moirai_hp_FINAL/`.
`scripts/run_resume50k.sh` lines 44 and 45 set lr 1e-3 at batch 256, and the
report (line 94) gives a flat rate, no warmup and no grad clip. This card
keeps #414's batch of 64, so 665,000 steps stay one pass. The square-root rule
for Adam then gives 1e-3 × √(64/256) = **5e-4**.

#414 found that every constant rate reaches its best score and then climbs.
So the rate falls by one cosine from 5e-4 to 1e-6 over the 665,000 steps. The
Moirai paper also anneals by cosine.

## Which score answers the question

Each stop trains one head and scores it twice:

- **B4** rolls the forecast out in latent space: it feeds each forecast latent
  back as the next encoder latent. It is #414's protocol, so against 1.1369
  only the backbone differs.
- **A2** rolls the forecast out in value space: it feeds the median forecast
  back as the next 16 values and re-runs the whole cell.

**The comparison with 1.1369 uses A2.** It reads each objective under the
rollout that objective trains. #414's cell trains a latent rollout, and B4
scores it. This model trains a value rollout, and A2 scores it.

Under B4, this model runs a latent operator that its loss never trains. Also,
the head trains on forecaster latents only, and B4 gives it encoder latents
for the context. The contrastive loss makes those two alike, and this loss
does not. So a B4 score of this model measures two mismatches as well as the
objective. B4 stays beside A2 at every stop, as the same-protocol read, and
the card fixes this choice before the run starts.

## Protocol

- **Data.** `jeremycochoy/gift-pretrain-full-4096`, path `small_v1`,
  `T_raw` 4096, C = 1, batch 64, seed 20260520 — #414's, so 665,000 steps is
  one pass over the data in both runs.
- **Stops.** 40k, 100k, 200k, 300k, 400k, 500k, 600k and 665,000 steps, on one
  trajectory: each leg resumes the previous stop with its optimizer state.
- **Scoring.** #414's head, unchanged: a quantile head (2-layer transformer,
  forecast length 16, batch 256, lr 1e-3, head seed 20260722, 30,000 steps)
  on the **frozen** backbone. Then the 97 GIFT-Eval configs at forecast
  length 16, under B4 and under A2. Each stop keeps one `all_results.csv` per
  strategy for the per-dataset comparison against `cos200k`.
- **One run** per stop. GIFT-Eval point scoring is deterministic, so the only
  noise is training init. #414 measured a seed band of 0.008 at 5.6e-5.

The one command, on the vast box. Checkpoints go to `/workspace/ckpt/cf-415`,
which the orchestrator mirrors to elisa:

```bash
BB_GPU=0 bash reports/2026-09-23_value_space_reference/run.sh
```

## Result

Pending. The deliverable is the GM-Relative MASE trajectory over the eight
stops under A2 and B4, and a per-dataset comparison against #414's 1.1369.
