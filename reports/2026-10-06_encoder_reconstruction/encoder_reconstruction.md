# The encoder alone through training: the GM-Relative MASE of a reconstruction head

Through training, the encoder alone stays the same in 4 EWMA runs and gets worse in 3 mean/std runs. In the 7 other runs with 2 or more checkpoints, the two heads give no clear answer, and no run gets better with both heads.

R is the GM-Relative MASE of a head that decodes the true horizon from the encoder latents of that horizon. Each checkpoint has two heads with one seed each: a transformer head and a linear head. The head noise is the change of R between two snapshots of one head training (the open markers).

![recon_all](plots/recon_all.png)
*R of each checkpoint, all 19 runs.*

A verdict of "same" or "worse" needs the two heads to agree. For TWN the two heads do not agree, and for 6 mean/std runs the noise of the transformer head is as large as its changes.

![recon_ours_one_patch_size](plots/recon_ours_one_patch_size.png)
*R of the runs with one patch size.*

![recon_ours_patch_sizes](plots/recon_ours_patch_sizes.png)
*R of the runs with patch sizes 8 to 128.*

![recon_selected](plots/recon_selected.png)
*R of the selected runs: all runs but TWN and OCB.*

![overlay_all](plots/overlay_all.png)
*The forecast score (B4, top panel) and R (bottom panel) of each checkpoint, all 19 runs.*

In OBM, BMS, OCF, OAF and OAL, the forecast score gets 22% to 46% better, and R does not get better with both heads.

![overlay_ours_one_patch_size](plots/overlay_ours_one_patch_size.png)
*The forecast score and R of the runs with one patch size.*

![overlay_ours_patch_sizes](plots/overlay_ours_patch_sizes.png)
*The forecast score and R of the runs with patch sizes 8 to 128.*

![overlay_selected](plots/overlay_selected.png)
*The forecast score and R of the selected runs.*

## Verdict of each run

Each cell gives the score at the first and at the last checkpoint of the run, and the ratio of the last score to the first.

| Run | Scaling | Forecast (B4) | R, transformer head | R, linear head | Encoder alone |
|---|---|---|---|---|---|
| CYN | EWMA | 1.1369 → 1.1432, ×1.01 | 0.0293 → 0.0301, ×1.03 | 0.1591 → 0.1422, ×0.89 | same |
| BLK | EWMA | 1.1445 → 1.1423, ×1.00 | 0.0399 → 0.0328, ×0.82 | 0.1965 → 0.1993, ×1.01 | same |
| MIN | EWMA | 1.1435 → 1.1471, ×1.00 | 0.0372 → 0.0462, ×1.24 | 0.1986 → 0.1866, ×0.94 | same |
| OEF | EWMA | 1.2391 → 1.1772, ×0.95 | 0.0877 → 0.0924, ×1.05 | 0.2642 → 0.2765, ×1.05 | same |
| OMB | mean/std | 1.3782 → 1.7952, ×1.30 | 0.3358 → 0.8347, ×2.49 | 0.5347 → 0.9176, ×1.72 | worse |
| OMF | mean/std | 1.5037 → 1.6452, ×1.09 | 0.3429 → 0.9853, ×2.87 | 0.5114 → 1.0914, ×2.13 | worse |
| OBM | mean/std | 2.0703 → 1.6086, ×0.78 | 0.5159 → 0.9944, ×1.93 | 0.6012 → 0.9980, ×1.66 | worse |
| TWN | EWMA | 1.1580 → 1.2211, ×1.05 | 0.0351 → 0.0543, ×1.55 | 0.1961 → 0.2098, ×1.07 | no clear answer |
| BMS | mean/std | 2.2604 → 1.2167, ×0.54 | 0.3397 → 0.4873, ×1.43 | 0.3618 → 0.4258, ×1.18 | no clear answer |
| OBW | mean/std | 1.5006 → 1.4437, ×0.96 | 0.4050 → 0.4281, ×1.06 | 0.5671 → 0.6873, ×1.21 | no clear answer |
| OWR | mean/std | 1.5163 → 1.6190, ×1.07 | 0.4606 → 0.5142, ×1.12 | 0.5551 → 0.5534, ×1.00 | no clear answer |
| OCF | mean/std | 1.8601 → 1.2329, ×0.66 | 0.3988 → 0.2727, ×0.68 | 0.3616 → 0.3648, ×1.01 | no clear answer |
| OAF | mean/std | 1.6538 → 1.2460, ×0.75 | 0.3269 → 0.2967, ×0.91 | 0.3969 → 0.4034, ×1.02 | no clear answer |
| OAL | mean/std | 1.6876 → 1.2173, ×0.72 | 0.3134 → 0.2732, ×0.87 | 0.2927 → 0.3973, ×1.36 | no clear answer |

## Verdict of each group

| Encoder alone | Runs | R, transformer head | R, linear head | Head noise of the transformer head |
|---|---|---|---|---|
| same | CYN, BLK, MIN, OEF | ×0.82 to ×1.24 | ×0.89 to ×1.05 | 16% or less (5 EWMA heads) |
| worse | OMB, OMF, OBM | ×1.93 to ×2.87 | ×1.66 to ×2.13 | 6% or less (2 heads of OMB) |
| no clear answer | TWN | ×1.55 | ×1.07 | no measure for TWN |
| no clear answer | BMS, OBW, OWR, OCF, OAF, OAL | ×0.68 to ×1.43 | ×1.00 to ×1.36 | 15% to 48% (4 heads of BMS and OCF) |

## Head noise

Each row gives the R of one transformer head at two head steps: the snapshot with the best training loss, and the final head. For TWN, the snapshot with the best training loss is the final head. The linear head has no such measure.

| Head | Scaling | Head step of the snapshot | R of the snapshot | R at step 30,000 | Ratio |
|---|---|---|---|---|---|
| CYN 665k | EWMA | 29,500 | 0.0292 | 0.0293 | ×1.00 |
| BLK 100k | EWMA | 28,500 | 0.0398 | 0.0399 | ×1.00 |
| MIN 1,080k | EWMA | 28,500 | 0.0398 | 0.0462 | ×1.16 |
| OEF 40k | EWMA | 28,500 | 0.0836 | 0.0877 | ×1.05 |
| OEF 200k | EWMA | 29,000 | 0.0905 | 0.0924 | ×1.02 |
| OMB 10k | mean/std | 25,000 | 0.3162 | 0.3358 | ×1.06 |
| OMB 166k | mean/std | 25,000 | 0.8652 | 0.8347 | ×0.96 |
| BMS 100k | mean/std | 18,500 | 0.2992 | 0.2097 | ×0.70 |
| BMS 140k | mean/std | 18,500 | 0.3295 | 0.4873 | ×1.48 |
| OCF 40k | mean/std | 25,000 | 0.2931 | 0.3988 | ×1.36 |
| OCF 400k | mean/std | 25,000 | 0.3202 | 0.2727 | ×0.85 |
| TWN 100k | EWMA | 30,000 | 0.0351 | 0.0351 | the same head |
| TWN 420k | EWMA | 30,000 | 0.0543 | 0.0543 | the same head |

## New forecast scores

These 5 checkpoints had no forecast score (B4) before this experiment.

| Checkpoint | Forecast (B4) |
|---|---|
| TWN 420k | 1.2211 |
| LOW 665k | 1.1913 |
| LNG 665k | 1.2706 |
| MIN 1,080k | 1.1471 |
| CYN 1,140k | 1.1452 |

## Protocol

- **Checkpoints.** 56 checkpoints of 19 runs (`scripts/jobs.tsv`). OCB, OWF, OWL, LOW and LNG have one checkpoint each, so they show no change.
- **Heads.** The backbone does not train. A head learns to give the values of patch t from the encoder latent of patch t. The transformer head has the settings of the forecast score (B4): 30,000 steps, batch 256, learning rate 1e-3, seed 20260722, the scaling of the run, and one head per patch size for the runs with patch sizes 8 to 128. The linear head is one linear map with the same settings. Each head trains on the training data of its run.
- **Score.** R uses the 97 GIFT-Eval configs of the forecast score. The encoder reads the context (1,024 values) and the true horizon, and the head decodes the horizon patches. The MASE, the seasonal-naive ratio and the geometric mean are those of the forecast score.
- **Floor.** The floor is the R of a head that gives the normalised value 0: 1.5721 for mean/std, 1.2092 for EWMA on the new data, and 1.1870 for EWMA on the old data.
- **Snapshots.** The head trainer keeps the head of its best training loss. `scripts/snapshot_score.sh` scores it with the eval of R.
- **Files.** `results/job_scores.tsv` gives the forecast score and the R of the two heads for each of the 56 checkpoints. `results/snapshots/scores.tsv` gives each snapshot score.
- **Commands.** `scripts/queue.sh` trains and scores the transformer heads, and `scripts/queue_elisa.sh` the linear heads. Then `python3 scripts/collect.py && python3 scripts/check_scores.py && python3 scripts/plot_recon.py` writes the tables and the figures.
