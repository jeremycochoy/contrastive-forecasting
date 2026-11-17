# Our copy of Moirai at the width-64 size on GIFT-Eval

One run of our width-64 Moirai copy (MPE, 1 seed, 1 encoder layer, 3 forecaster layers) scores 0.053 to 0.073 higher GM-Relative MASE than MPE at width 384. This run is in the range of the contrastive width-64 runs up to 75k, below each run from 100k, and above only the two 10-L-ALIGN cooldowns (150k and 166k).

![width64_gm_mase](plots/width64_gm_mase.png)

The total loss decreases from 1.12 to 0.87, with almost equal contributions from the 4 rollout depths.

![loss_composition](plots/loss_composition.png)

| Stop | MPE, width 64 | MPE, width 384 | Difference |
|---|---|---|---|
| 10k | 1.0857 | 1.0247 | 0.0610 |
| 25k | 1.0422 | 0.9841 | 0.0581 |
| 50k | 1.0361 | 0.9831 | 0.0529 |
| 75k | 1.0221 | 0.9625 | 0.0596 |
| 100k | 1.0162 | 0.9555 | 0.0607 |
| 125k | 1.0106 | 0.9468 | 0.0638 |
| 150k | 1.0128 | 0.9405 | 0.0723 |
| 166k | 1.0090 | 0.9362 | 0.0728 |
