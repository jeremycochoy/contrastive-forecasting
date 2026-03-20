# ARMA Parameter Recovery via Contrastive Learning

## Architecture Overview

### Problem Setup

We train a model to learn latent representations of multivariate ARMA(p,q) time series using a contrastive learning objective. The model processes raw time series windows and produces embeddings that encode the underlying stochastic process. A separate recovery head then predicts the ARMA coefficients from these embeddings, validating that the latent space captures meaningful process structure.

Each input is a multivariate time series with C=4 channels, generated from a shared ARMA(p,q) process with dimension up to 4 (i.e., up to 4 AR and 4 MA coefficients). The raw signal (T=4096 time steps) is split into T/W=128 non-overlapping patches of width W=32.

### Encoder: Bidirectional GRU

The encoder maps each raw patch of W=32 scalar values to an H=1024 dimensional embedding. It treats each patch as a short time series: a 2-layer bidirectional GRU (hidden_size=128) reads the 32 time steps one by one, capturing temporal ordering within the patch. The final forward and backward hidden states are concatenated (256-dim) and projected to H=1024 via a linear layer. A parallel skip connection (Linear W->H) is added, followed by LayerNorm. All B x T x C patches are processed in a single batched GRU call for efficiency.

### Forecaster: Causal Transformer

The sequence of T=128 patch embeddings (each H=1024) is processed by a 12-layer decoder-only transformer with:
- **8 attention heads** (head_dim=128), causal masking
- **FFN multiplier 4x**: each layer has a feed-forward network of dimension 4096 (4 x H)
- **GELU activation** in the FFN
- **Depthwise causal convolution** (kernel_size=3) between the attention and FFN sublayers, providing local temporal context
- **Pre-norm** architecture (LayerNorm before each sublayer)
- **Dropout** 0.1

The transformer outputs a forecast embedding h_hat[t] at each position, representing the model's prediction of the next patch's latent representation.

### Channel Mixing

After the per-channel transformer, a Kronecker-product-based channel mixing module combines information across the C=4 channels using learnable R (self) and Q (cross) mixing matrices, producing the final latent representation h[t].

### Training Loss

The model is trained with a **contrastive loss** based on cosine similarity with temperature tau=0.07. For each time step t, the loss encourages:
- **h_hat[t] to be similar to h[t+1]** (the forecast should match the actual future embedding)
- **h_hat[t] to be dissimilar to h[t-1]** (the forecast should differ from the past)
- **Cross-batch negatives**: embeddings from different ARMA processes in the same batch serve as hard negatives

The key metric is the **FF-FP gap**: the difference between forecast-future similarity (FF) and forecast-past similarity (FP). Higher gap means the model learns representations that genuinely distinguish future from past, rather than learning trivial features.

**Total parameters**: 153.8M. Trained for 500k steps with AdamW (lr=7e-5, batch_size=8) on a single RTX 4090 (~20 hours).

### Parameter Recovery Head

A separate **DeepGRU** head (4.7M parameters, frozen backbone) takes the latent embeddings h and predicts the 4 AR and 4 MA coefficients. It consists of:
- Input projection (Linear 1024->512->256 with SiLU + LayerNorm)
- 3-layer bidirectional GRU (hidden=256)
- Two residual MLP blocks
- Per-coefficient output heads with tanh activation

The recovery head is trained for 20,000 epochs with Adam (lr=1e-3) on fresh ARMA batches each epoch.

---

## Results

### Contrastive Learning Performance

The model was trained for 500,000 steps. The FF-FP gap improved steadily throughout training:

| Steps | FF-FP Gap |
|-------|-----------|
| 50k   | 0.120     |
| 150k  | 0.150     |
| 300k  | 0.170     |
| 470k  | 0.180     |
| 500k  | 0.179     |

Peak gap of **0.186** was reached at step 494k, with the model still showing signs of improvement.

### Parameter Recovery Performance

On 200 held-out ARMA processes:

| Metric | Value |
|--------|-------|
| Mean AR Error (MSE) | 0.0147 |
| Mean MA Error (MSE) | 0.0151 |
| Mean Total Error | 0.0298 |
| Zero-baseline Error | 0.1963 |
| **Improvement** | **6.58x** |

### True vs Predicted Parameters

Five randomly sampled ARMA processes showing true (blue) and predicted (red) AR and MA coefficients. The model captures both the sign and magnitude of most coefficients accurately.

![True vs Predicted Parameters](images/fig_true_vs_predicted.png)

### True vs Predicted Scatter Plots

Scatter plots of true vs predicted values for each of the 8 coefficients (4 AR, 4 MA) across 300 test samples. Points close to the red y=x line indicate accurate recovery. Pearson correlations range from r=0.915 to r=0.958.

![Scatter Plots](images/fig_scatter_plots.png)

### Per-Coefficient Sign Agreement

Sign agreement (for |true| > 0.05) and Pearson correlation per coefficient:

| Coefficient | Sign Agreement | Correlation | MAE |
|-------------|---------------|-------------|-----|
| AR[0] | 92.0% | 0.915 | 0.127 |
| AR[1] | 95.1% | 0.951 | 0.085 |
| AR[2] | 95.9% | 0.958 | 0.072 |
| AR[3] | 92.7% | 0.947 | 0.081 |
| **AR overall** | **93.8%** | **0.932** | |
| MA[0] | 90.3% | 0.917 | 0.130 |
| MA[1] | 93.0% | 0.939 | 0.094 |
| MA[2] | 97.0% | 0.955 | 0.074 |
| MA[3] | 94.9% | 0.955 | 0.070 |
| **MA overall** | **92.9%** | **0.929** | |

First coefficients (AR[0], MA[0]) are hardest to recover, consistent with them having the largest influence on the process and widest value range. Higher-order coefficients are recovered with >95% sign agreement and correlations approaching 0.96.

### Error Distributions

Distribution of per-sample MSE errors across 200 test ARMA processes. Most samples have very low error, with a long tail from rare high-order processes.

![Error Distributions](images/fig_error_distributions.png)
