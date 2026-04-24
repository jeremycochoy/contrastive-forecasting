# Follow-up Design Notes

Captured design proposals for work that should happen after the initial
periodic-synth-mix experiment lands. Each entry is a design not a plan — the
plan / ablation is tracked as a numbered task in the TaskList.

## 1. Frequency embedding (task #22)

### Motivation

The periodic-synth-mix experiment established that our backbone can *learn*
clean periodics but *cannot transfer* them cleanly to real-world data with
noisy periods and multi-scale structure (daily+weekly). Inspecting the
plots from that experiment revealed two recurring failure modes on periodic
eval data:

- **Amplitude damping** — the model produces a sinusoid of the correct
  frequency but with half the amplitude of ground truth.
- **Phase drift** — the oscillation starts at the wrong phase within a
  daily cycle (e.g. predicting daytime-solar during what should be
  nighttime).

Both symptoms are consistent with a model that has to infer the period
from every context from scratch and produces an averaged / dampened
best-guess. TimesFM, Chronos-2, and Moirai-2 all work around this with
explicit *frequency tokens*.

### Design

Add a learned frequency embedding of small dimension (recommended **E = 3
or 4**).

```python
# In src/freq_embedding.py
class FrequencyEmbedding(nn.Module):
    def __init__(self, num_freqs: int = 9, emb_dim: int = 3):
        super().__init__()
        self.emb = nn.Embedding(num_freqs, emb_dim)

    def forward(self, freq_ids: torch.Tensor) -> torch.Tensor:
        # freq_ids: [B]   -> [B, emb_dim]
        return self.emb(freq_ids)
```

Freq classes (one per dt level in our synth sampling set):

| id | freq   | samples-per-day |
|---:|--------|---:|
| 0  | 10s    | 8640 |
| 1  | 1min   | 1440 |
| 2  | 5min   | 288  |
| 3  | 10min  | 144  |
| 4  | 15min  | 96   |
| 5  | 30min  | 48   |
| 6  | 1h     | 24   |
| 7  | 1d     | 1    |
| 8  | 1w     | 1/7  |

### Injection: concat to patch, not prepend as token

The recommendation is to **concat** the frequency embedding to each patch
along the *feature* axis rather than prepending it as a token. Given input
`X: [B, T, C]` and patch size `W=16`, the per-patch view is
`[B, T/W, C, W]`. With the freq embedding we instead present
`[B, T/W, C, W+E]` to the patch encoder, where the final E values are a
broadcast of the freq embedding.

Rationale:

- **Locality.** Every patch position carries the freq hint directly, so
  the GRU patch encoder processes the two signals together per patch
  rather than relying on long-range attention to broadcast a single
  prepended token.
- **No attention-head cost.** The token approach adds one position per
  input, slightly increasing attention cost and moving the hint "far
  away" from most patches.
- **Closer to TimesFM / Chronos-2 practice.** TimesFM conditions on freq
  per patch token in its patch embedding path; this matches that design.

### Why the embedding dim is small (3 or 4)

An 3-dim embedding carries 12 continuous bits of information (~3 dim × 4
bits/dim). That's enough to disambiguate among 9 frequency classes but
nowhere near enough to carry "the model's full forecast for this freq."
The small dim functions as a **regularizer** — the model is *hinted* at
the frequency class but must still learn period detection from the
actual series.

Larger embedding dims (e.g. 64) would let the model cheat by encoding
frequency-specific forecasting behavior directly in the embedding,
defeating the purpose of pretraining on a mix of frequencies.

### Mixup — the real payoff

The clean architectural win is that embeddings are continuous vectors, so
they support **linear interpolation**. During training we can draw two
series (X1, f1) and (X2, f2) from the batch and mix them:

```python
alpha = Beta(0.2, 0.2).sample()
X_mix   = alpha * X1 + (1-alpha) * X2
emb_mix = alpha * emb(f1) + (1-alpha) * emb(f2)
# feed (X_mix, emb_mix) to the backbone
```

Two concrete benefits:

1. **Continuous latent structure across frequencies.** The contrastive
   loss forces `future(X)` and `past(X)` to be close, and mixup extends
   that property across interpolated frequencies — the backbone learns a
   smooth manifold over freq rather than N discrete clusters. This
   directly addresses the "model locks onto a single discovered period"
   failure we observed.

2. **Implicit multi-period composition.** Real hourly data has both
   daily and weekly cycles superposed. A mixup between
   (synthetic-hourly-daily) and (synthetic-hourly-weekly) produces a
   training sample that has both cycles superposed — exactly the
   structure real data has and our current synth lacks. This is an
   end-run around having to explicitly design a multi-period
   synthesizer.

### Implementation notes

- Dataset pipeline must tag each row with a freq id. base-bundles rows
  have a freq field on disk; synth rows can record the sampled dt when
  generated.
- `src/dataloader.py::MixedPeriodicLoader` would need to propagate both
  the series tensor and the freq id through collation.
- `src/models.py::ConfigurableModel` needs a `freq_embed_dim` constructor
  arg that, when > 0, concats an E-dim broadcast of the embedding to
  each patch.
- Training loop adds the mixup step between `next(data_iter)` and the
  sign-flip augmentation.

### Ablations

- A: baseline = current architecture.
- B: + freq embedding, no mixup.
- C: + freq embedding + mixup.

Should be paired with a periodic-synth-mix run at the same compute budget
as the v3c pair (30k, bs=24) so the comparison is clean.

### Caveat to resolve

Real-world datasets don't always line up with our 9 discrete freq
classes (e.g., '3min' would fall between 1min and 5min). We can either
(a) round to the nearest of the 9, (b) introduce a "continuous freq"
scalar in addition, or (c) use a freq bucket with more classes. Start
with (a) — simplest, minimizes design surface.
