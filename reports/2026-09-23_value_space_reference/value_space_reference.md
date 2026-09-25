# A value-space reference model: the same body, trained on the actual values

*(#415. The first run trained at a flat 1e-3 and stopped at 92,600 steps. The
run with the Moirai schedule trains now. The result replaces this note.)*

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
| Rate | 5.6e-5 to 1e-6, cosine over 200,000 steps, then 1e-6 | **1e-3, a 10,000-step warmup, then cosine to 0 at 166,000 steps** |
| Batch | 64 | **256** |
| Gradient clip | none | **1.0** |
| Weight decay, betas | 0.1, (0.9, 0.98) | same |

The rollout is what "value space" means here. At depth j the model re-reads
its own median forecast as the next patch and runs the whole cell on it. The
loss of depth j reads the patch j further out. Depth 3 matches the cell.

## The rate

The owner asked for the Moirai recipe with its schedule. The Moirai papers
train at batch 256 with AdamW at 1e-3, weight decay 0.1 and betas
(0.9, 0.98). The rate rises linearly over the first 10,000 steps, then falls
by cosine to 0 ([Moirai 2.0](https://arxiv.org/abs/2511.11698)). The uni2ts
pretraining config clips the gradient norm at 1.0. At batch 256, one pass over
the data is 166,000 steps, so the anneal ends there.

The first run of this card trained at a flat 1e-3 with no warmup and no clip,
as `experiments/2026-05-03_exp_realonly_full4096_moirai_hp_FINAL/` did. Its
loss spiked between 25,000 and 50,000 steps, and again at 73,000, 77,000 and
79,000 to 84,000. Its B4 scores were 1.3396 at 10,000 steps, 1.2951 at 25,000
and 1.9036 at 50,000. The owner stopped it at 92,600 steps.

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
  `T_raw` 4096, C = 1, batch 256, seed 20260520. 166,000 steps is one pass
  over the data, the data of #414's 665,000 steps at batch 64.
- **Stops.** 10k, 25k, 50k, 75k, 100k, 125k, 150k and 166,000 steps, on one
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
