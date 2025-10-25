# Contrastive Forecasting

Research code for **Contrastive Forecasting**, a contrastive learning approach to time series forecasting. Trains two networks jointly: an encoder that determines feature representations for time series patches and a decoder-only transformer forecaster to predict next patch embeddings.

## Overview

This implementation explores contrastive learning techniques for multivariate time series forecasting using objectives that maximize cosine similarity between forecasted embeddings and ground truth future embeddings while minimizing similarity with past embeddings.

## Code Structure

- `arma.py`: ARMA process generation for synthetic time series data
- `blocks.py`: Transformer blocks with causal attention and depthwise convolutions
- `network.py`: Neural network architectures (SimpleModel, Simple_encoder)
- `loss.py`: Contrastive learning loss functions
- `forecast_arma.ipynb`: Complete training pipeline and experiments

## Usage

The main training pipeline is in `forecast_arma.ipynb`, including:
- Data generation and visualization
- Model initialization and training
- Metrics computation and visualization
- Training state management for resumable training

## License

This code is provided for research purposes. While the code can be used freely, **citation is required** when using this work in academic publications or research.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{contrastive_forecasting,
  title={Contrastive Forecasting: A Contrastive Learning Approach to Time Series Forecasting},
  author={[Jeremy Cochoy]},
  year={2025},
  url={https://github.com/jeremycochoy/contrastive-forecasting},
  note={Research code for contrastive learning in time series forecasting}
}
```

## Research Context

This work explores contrastive learning paradigms for time series forecasting, investigating how forecasted representations can be learned through similarity objectives with future and past representations. The approach combines transformer architectures with contrastive objectives to learn temporal representations for forecasting tasks.

## Contact

For questions about this research code, please contact [jeremy dot cochoy at gmail dot com].
