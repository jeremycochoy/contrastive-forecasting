# The GRN+EMA runs through training: forecast, reconstruction and loss

## Forecast and reconstruction

![Forecast and reconstruction through training](plots/forecast_and_reconstruction.png)

Grey (EMA 0.99995, 6 forecaster layers, stopped at 440k) has the lowest forecast MASE of the GRN runs: 1.0341 at 240k.
Magenta (the same with 3 forecaster layers) has its lowest MASE, 1.0723, at 140k, and teal (EMA 0.9998) has 1.0568 at 100k.
At 40k, yellow (grey with the L_rep weight 0.1 at each step, in training) has 1.1053, grey 1.0864 and magenta 1.0992.

## Learning rate

![Learning rate through training](plots/lr_schedule_099995.png)

The six GRN runs keep the learning rate at 5.6e-5 at each step.
MPE (Moirai) has batch 256 and a peak of 1e-3, and the figure shows its recipe with the peak at 5.6e-5.

## EMA momentum of the teacher

![Half-life of the teacher through training](plots/ema_schedule_099995.png)

The half-life of the teacher of magenta and grey is 13.9k steps at 1k and 33.6k steps at 391k.
The JEPA lines move the momentum to batch 64 with the EMA scaling rule of Busbridge and others (2023): `momentum ** (64 / batch size)`.
At batch 64, the half-life starts at 5.5k steps for I-JEPA and at 16.6k steps for V-JEPA.

## Parts of the total loss

![Parts of the total loss of magenta, grey, BLK and the run with L_rep 0.03 and the depth weights 1/2^j](plots/loss_composition_099995.png)

From 11k, L_align at depths 1 to 3 is 77% to 80% of the total loss of magenta, grey and BLK.
For the run with L_rep 0.03 and the depth weights 1/2^j, this part is 29% to 32% from 11k to 47k.
The total loss of magenta decreases from 1.24 at 11k to 1.05 at 366k, and grey has its maximum, 1.94, at 124k.

## Loss terms

![Each term of the loss of yellow, magenta and grey](plots/loss_curves_099995.png)

Yellow keeps L_rep x its weight between 1.16 and 1.19 from step 100 to 40k.
At 40k, L_align at depths 1 to 3 is 0.92 for yellow and for magenta, and 1.15 for grey.
At 40k, L_align at depth 0 is 0.25 for yellow, 0.24 for magenta and 0.30 for grey.
