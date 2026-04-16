# Contrastive Forecasting

A contrastive learning approach to time series representation learning. Trains a transformer-based model to produce latent embeddings that distinguish future from past, then validates these representations by recovering the generative parameters (ARMA coefficients) from frozen embeddings.

## Quick Start

Two example notebooks in [`examples/`](examples/):

| Notebook | Description |
|----------|-------------|
| [`train_contrastive.ipynb`](examples/train_contrastive.ipynb) | Train the contrastive backbone from scratch (~10 min demo) |
| [`parameter_recovery.ipynb`](examples/parameter_recovery.ipynb) | Load a trained backbone and recover ARMA coefficients |

## Repository Structure

```
src/                     Core library
  arma.py                  ARMA process generation
  blocks.py                Transformer blocks (causal attention + depthwise conv)
  encoders.py              Patch encoders (MLP, GRU, Conv, etc.)
  models.py                ConfigurableModel (best architecture)
  recovery.py              Recovery head definitions and factory
  network.py               SimpleModel (original architecture)
  loss.py                  Contrastive loss functions
  checkpoint.py            Checkpoint save/load with optimizer state

scripts/                 Training scripts (best architecture, ready to use)
  train.py                 Contrastive backbone training
  recover.py               Parameter recovery head training

examples/                Getting-started notebooks
tests/                   Unit tests
experiments/             Experiment logs and reports
  contrastive-arma/        Full architecture search (Mar--Apr 2026)
    report/                  Technical report, tables, figures
    scripts/                 Experiment run scripts (all phases)
    notebooks/               Experiment-specific notebooks
    train_*.py               Original training scripts (frozen)
```

## Model Architecture

The best configuration found through architecture search:

- **Encoder**: Bidirectional GRU (reads each 32-step patch as a sequence)
- **Backbone**: 12-layer causal transformer, H=1024, 8 heads, FFN 4x, GELU, depthwise conv k=3
- **Parameters**: 153.8M
- **Key metric**: FF-FP gap = 0.203 at 2M steps (93% higher than MLP encoder baseline)
- **Recovery**: 6.96x improvement over zero-baseline on 4 AR + 4 MA coefficient prediction

See [`experiments/contrastive-arma/report/technical_report.md`](experiments/contrastive-arma/report/technical_report.md) for the full optimization story.

## Training Data

Pretraining uses a mix of real-world and synthetic time series,
prepared by the `training_data_prep` pipeline in the
[`rnd`](https://github.com/jeremycochoy/rnd) repository.

The Wikimedia pageview component is published on HuggingFace:

> **[`jeremycochoy/wikimedia-pageview-timeseries`](https://huggingface.co/datasets/jeremycochoy/wikimedia-pageview-timeseries)**

Five subsets derived from [Wikimedia pageview dumps](https://dumps.wikimedia.org/other/pageview_complete/)
(Dec 2011 -- Oct 2016, CC0 license):

| Subset | Series | Description |
|---|---|---|
| `wiki_hourly` | 3.7M | Hourly pageview counts |
| `wiki_daily` | 2.0M | Daily aggregated counts |
| `wiki_stl_residual` | 531K | STL decomposition -- residual |
| `wiki_stl_seasonal` | 372K | STL decomposition -- seasonal |
| `wiki_stl_trend` | 159K | STL decomposition -- trend |

Each series is a `float32[1025]` window (1024 input + 1 target).
Total ~6.8M series, ~9 GB.

The full pretraining mix also includes
[GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)
and synthetic ARMA/trend/sinusoid series; see the `training_data_prep`
README for mix ratios and pipeline details.

## Training

```bash
# Train contrastive backbone (GPU recommended)
python scripts/train.py --device cuda --total-steps 500000 --save-path model.pth

# Train recovery head on frozen backbone
python scripts/recover.py --device cuda --model-path model.pth --epochs 20000
```

## License

This code is provided for research purposes. While the code can be used freely, **citation is required** when using this work in academic publications or research.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{contrastive_forecasting,
  title={Contrastive Forecasting: A Contrastive Learning Approach to Time Series Forecasting},
  author={Jeremy Cochoy},
  year={2025},
  url={https://github.com/jeremycochoy/contrastive-forecasting},
  note={Research code for contrastive learning in time series forecasting}
}
```

## Contact

For questions about this research code, please contact [jeremy dot cochoy at gmail dot com].
