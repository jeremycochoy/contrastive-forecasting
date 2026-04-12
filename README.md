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
  network.py               Model architectures (SimpleModel)
  loss.py                  Contrastive loss functions
  checkpoint.py            Checkpoint save/load with optimizer state

train_contrastive_v2.py  Contrastive training script (ConfigurableModel)
train_contrastive.py     Original training script (SimpleModel)
train_parameter_recovery_v2.py  Recovery head training (configurable)
train_parameter_recovery.py     Original recovery training

examples/                Getting-started notebooks
tests/                   Unit tests
experiments/             Experiment logs and reports
  contrastive-arma/        Full architecture search results (Mar--Apr 2026)
    report/                  Technical report, tables, figures
    scripts/                 Run scripts for all experiment phases
    notebooks/               Experiment-specific notebooks
```

## Model Architecture

The best configuration found through architecture search:

- **Encoder**: Bidirectional GRU (reads each 32-step patch as a sequence)
- **Backbone**: 12-layer causal transformer, H=1024, 8 heads, FFN 4x, GELU, depthwise conv k=3
- **Parameters**: 153.8M
- **Key metric**: FF-FP gap = 0.203 at 2M steps (93% higher than MLP encoder baseline)
- **Recovery**: 6.96x improvement over zero-baseline on 4 AR + 4 MA coefficient prediction

See [`experiments/contrastive-arma/report/technical_report.md`](experiments/contrastive-arma/report/technical_report.md) for the full optimization story.

## Training

```bash
# Train contrastive backbone (GPU recommended)
python train_contrastive_v2.py --device cuda \
    --encoder-type gru --H 1024 --num-layers 12 --nhead 8 --ffn-mult 4 \
    --activation gelu --depthwise-conv 3 \
    --total-steps 500000 --batch-size 8 --lr 7e-5 \
    --save-path model.pth

# Train recovery head on frozen backbone
python train_parameter_recovery_v2.py --device cuda \
    --model-path model.pth \
    --encoder-type gru --H 1024 --num-layers 12 --nhead 8 --ffn-mult 4 \
    --activation gelu --depthwise-conv 3 \
    --model-type gru --hidden-dim 128 --num-gru-layers 2 \
    --epochs 20000
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
