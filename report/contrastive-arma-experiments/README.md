# Contrastive ARMA Experiments

This directory contains all documentation and results from the contrastive learning experiments on synthetic ARMA time series, conducted March--April 2026.

## Documents

| File | Description |
|------|-------------|
| [architecture_search_report.md](architecture_search_report.md) | **Main report.** Architecture search, 2M training, recovery search, scaling search, and analysis. Start here. |
| [training_summary.md](training_summary.md) | Quick-reference tables: every run with L/H/heads/LR/params/gap/recovery. Effect analyses by variable. |
| [experiment_log.md](experiment_log.md) | Detailed log of every training run: config, duration, steps, peak gap, failures, compute budget. |
| [ARCHITECTURE_SEARCH_PLAN.md](ARCHITECTURE_SEARCH_PLAN.md) | Original 48-hour autonomous search plan (March 18--20). |

## Visualizations

Images in [`images/`](images/):

| Image | Description |
|-------|-------------|
| `fig_true_vs_predicted.png` | True vs predicted ARMA coefficients (5 sample processes) |
| `fig_scatter_plots.png` | Scatter plots with regression lines (8 coefficients, 300 samples) |
| `fig_error_distributions.png` | MSE error distribution histograms |
| `gap_200k_comparison.png` | Gap trajectories: 12L vs 16L vs 20L at 200k steps |
| `gap_full_training.png` | Gap trajectories: 12L vs 20L up to 2M+ steps |

## Notebooks

Visualization notebooks live in [`../../notebooks/`](../../notebooks/):

| Notebook | Description |
|----------|-------------|
| `gap_trajectories.ipynb` | Gap trajectory plots for scaling comparison (generates `gap_*.png`) |
| `visualize_recovery_gru_h128_l3.ipynb` | 4x2 coefficient recovery visualization (best GRU head on V2 backbone) |
| `visualize_recovery.ipynb` | Original recovery visualization (V1 backbone) |
| `parameter_recovery_experiment.ipynb` | Early recovery experiments |
| `forecast_arma.ipynb` | ARMA forecasting exploration |

## Key Results

- **Best backbone**: 12L H=1024, GRU encoder, FFN 4x, 153.8M params, peak gap **0.203** at 2M steps
- **Best recovery (4x2)**: GRU h=128 l=2 head on 12L backbone, **6.96x** improvement, 92% sign agreement
- **Best recovery (2x2)**: Same head, **8.34x** improvement (surpasses old 7.3x record by 14%)
- **Gap correlates with recovery**: V1 gap=0.105 -> 6.59x, V2 gap=0.203 -> 6.96x
- **Depth scaling**: 20L matches 12L gap (0.203) but at 45% more wall time and slightly worse recovery (6.77x)
- **Total compute**: ~433 GPU-hours on RTX 4090

## Experiment Timeline

| Period | Work |
|--------|------|
| Feb 2026 | Early recovery experiments (H=512, DeepGRU) |
| Mar 18--20 | Autonomous architecture search (5 phases) |
| Mar 21--26 | 2M backbone training, optimizer checkpoint feature |
| Mar 27--30 | Recovery head architecture search (47+ experiments) |
| Mar 30--Apr 1 | Scaling search (12L/16L/20L comparison) |
| Apr 1--12 | 20L full training (2M+ steps), recovery evaluation |
