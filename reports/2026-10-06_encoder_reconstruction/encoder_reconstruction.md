# The encoder alone through training: the GM-Relative MASE of a reconstruction head

Between the kept checkpoints, the encoder alone stays the same in 4 EWMA runs and gets worse in 3 mean/std runs. The 7 other runs with 2 or more kept checkpoints (1 EWMA, 6 mean/std) give no clear answer, and no run gets better with both heads.

R is the GM-Relative MASE of a head that decodes the true horizon from the encoder latents of that horizon. Each checkpoint has two heads with one seed each: a transformer head and a linear head. The head noise is the change of R between two snapshots of one head training (an open marker and its dot).

![recon_all](plots/recon_all.png)
*R of each checkpoint, all 19 runs.*

From the first to the last kept checkpoint, a run is "worse" when each head gets more than 50% worse. It is "same" when each head changes by less than 25% and its measured head noise is under 20%.

![recon_ours_one_patch_size](plots/recon_ours_one_patch_size.png)
*R of the runs with one patch size.*

![recon_ours_patch_sizes](plots/recon_ours_patch_sizes.png)
*R of the runs with patch sizes 8 to 128.*

OMB, OMF and OBM end at an R of 0.83 to 1.09, and the floor of mean/std is 1.57.

![recon_selected](plots/recon_selected.png)
*R of the selected runs: all runs but TWN and OCB.*

![overlay_all](plots/overlay_all.png)
*The forecast score (B4, top panel) and R (bottom panel) of each checkpoint, all 19 runs.*

![overlay_ours_one_patch_size](plots/overlay_ours_one_patch_size.png)
*The forecast score and R of the runs with one patch size.*

![overlay_ours_patch_sizes](plots/overlay_ours_patch_sizes.png)
*The forecast score and R of the runs with patch sizes 8 to 128.*

![overlay_selected](plots/overlay_selected.png)
*The forecast score and R of the selected runs.*

## Verdict of each run

The rule compares the first and the last kept checkpoint of a run.

| Encoder alone | Rule |
|---|---|
| worse | Each of the two heads gets more than 50% worse. |
| same | Each of the two heads changes by less than 25%. One or more heads of the run have a noise measure, and each measure is under 20%. |
| no clear answer | The other runs. |

Each score cell gives the score at the first and at the last kept checkpoint, and the ratio of the last score to the first. The checkpoint cell gives the two steps and the number of kept checkpoints. OMB, OMF, OBM, OBW and OAL count their steps at batch 256. The head noise is that of the transformer head, at the checkpoint in parentheses.

| Run | Scaling | Checkpoints | Forecast (B4) | R, transformer head | R, linear head | Head noise | Encoder alone |
|---|---|---|---|---|---|---|---|
| CYN | EWMA | 665k → 1,330k (3) | 1.1369 → 1.1432, ×1.01 | 0.0293 → 0.0301, ×1.03 | 0.1591 → 0.1422, ×0.89 | 0% (665k) | same |
| BLK | EWMA | 100k → 400k (4) | 1.1445 → 1.1423, ×1.00 | 0.0399 → 0.0328, ×0.82 | 0.1965 → 0.1993, ×1.01 | 0% (100k) | same |
| MIN | EWMA | 1,000k → 1,080k (2) | 1.1435 → 1.1471, ×1.00 | 0.0372 → 0.0462, ×1.24 | 0.1986 → 0.1866, ×0.94 | 16% (1,080k) | same |
| OEF | EWMA | 40k → 200k (3) | 1.2391 → 1.1772, ×0.95 | 0.0877 → 0.0924, ×1.05 | 0.2642 → 0.2765, ×1.05 | 5% (40k), 2% (200k) | same |
| OMB | mean/std | 10k → 166k (8) | 1.3782 → 1.7952, ×1.30 | 0.3358 → 0.8347, ×2.49 | 0.5347 → 0.9176, ×1.72 | 6% (10k), 4% (166k) | worse |
| OMF | mean/std | 10k → 125k (6) | 1.5037 → 1.6452, ×1.09 | 0.3429 → 0.9853, ×2.87 | 0.5114 → 1.0914, ×2.13 | not measured | worse |
| OBM | mean/std | 10k → 50k (3) | 2.0703 → 1.6086, ×0.78 | 0.5159 → 0.9944, ×1.93 | 0.6012 → 0.9980, ×1.66 | not measured | worse |
| TWN | EWMA | 100k → 420k (2) | 1.1580 → 1.2211, ×1.05 | 0.0351 → 0.0543, ×1.55 | 0.1961 → 0.2098, ×1.07 | no earlier snapshot | no clear answer |
| BMS | mean/std | 40k → 140k (3) | 2.2604 → 1.2167, ×0.54 | 0.3397 → 0.4873, ×1.43 | 0.3618 → 0.4258, ×1.18 | 30% (100k), 48% (140k) | no clear answer |
| OBW | mean/std | 10k → 50k (3) | 1.5006 → 1.4437, ×0.96 | 0.4050 → 0.4281, ×1.06 | 0.5671 → 0.6873, ×1.21 | not measured | no clear answer |
| OWR | mean/std | 40k → 200k (3) | 1.5163 → 1.6190, ×1.07 | 0.4606 → 0.5142, ×1.12 | 0.5551 → 0.5534, ×1.00 | not measured | no clear answer |
| OCF | mean/std | 40k → 400k (5) | 1.8601 → 1.2329, ×0.66 | 0.3988 → 0.2727, ×0.68 | 0.3616 → 0.3648, ×1.01 | 36% (40k), 15% (400k) | no clear answer |
| OAF | mean/std | 40k → 200k (3) | 1.6538 → 1.2460, ×0.75 | 0.3269 → 0.2967, ×0.91 | 0.3969 → 0.4034, ×1.02 | not measured | no clear answer |
| OAL | mean/std | 10k → 50k (3) | 1.6876 → 1.2173, ×0.72 | 0.3134 → 0.2732, ×0.87 | 0.2927 → 0.3973, ×1.36 | not measured | no clear answer |

## Head noise

Each row gives the R of one transformer head at two head steps: the snapshot with the best training loss, and the final head. The linear head has no such measure.

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

For 9 of the 56 transformer heads, the snapshot with the best training loss is the final head, so it measures no noise. These heads are BLK 200k, BLK 400k, CYN 1,140k, CYN 1,330k, MIN 1,000k, TWN 100k, TWN 420k, LOW 665k and LNG 665k.

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

- **Checkpoints.** 56 kept checkpoints of 19 runs (`scripts/jobs.tsv`). OCB, OWF, OWL, LOW and LNG have one checkpoint each, so they show no change.
- **Heads.** The backbone does not train. A head learns to give the values of patch t from the encoder latent of patch t. The transformer head has the settings of the forecast score (B4): 30,000 steps, batch 256, learning rate 1e-3, gradient clip 1.0 and seed 20260722. It has the scaling of the run, and one head per patch size for the runs with patch sizes 8 to 128. The linear head is one linear map with the same settings. Each head trains on the training data of its run.
- **Score.** R uses the 97 GIFT-Eval configs of the forecast score. The encoder reads the context (1,024 values) and the true horizon, and the head decodes the horizon patches. The MASE, the seasonal-naive ratio and the geometric mean are those of the forecast score.
- **Floor.** The floor is the R of a head that gives the normalised value 0. It is 1.5721 for mean/std, 1.2092 for EWMA on the new data, and 1.1870 for EWMA on the old data.
- **Snapshots.** The head trainer keeps the head of its best training loss. `scripts/snapshot_score.sh` scores it with the eval of R.
- **Files.** `results/job_scores.tsv` gives the forecast score and the R of the two heads for each of the 56 checkpoints. `results/snapshots/scores.tsv` gives each snapshot score.
- **Commands.** `scripts/queue.sh` trains and scores the transformer heads, and `scripts/queue_elisa.sh` the linear heads. Then `python3 scripts/collect.py && python3 scripts/check_scores.py && python3 scripts/plot_recon.py` writes the tables and the figures.
