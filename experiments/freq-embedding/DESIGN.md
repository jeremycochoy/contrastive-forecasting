# Freq-Embedding Ablation — Design

User-proposed architectural change addressing the periodic-datasets
failure mode documented in
[`experiments/periodic-synth-mix/REPORT.md`](../periodic-synth-mix/REPORT.md).

## Motivation (recap)

v3c_mix at 30k and 90k both under-predict periodic datasets with:
1. Amplitude damping (correct period, wrong magnitude)
2. Phase drift (wrong offset within each period)

Both symptoms are consistent with a model that infers the period from
every context independently. TS-FM SOTA (TimesFM / Chronos-2 / Moirai-2)
conditions on frequency via a token. We add a similar hint but as a
**small per-patch feature concatenation**, plus **mixup on the
embedding** to teach multi-period composition.

## Architecture

### Frequency classes

Nine discrete classes matching the synthesizer's physical-dt choices,
plus one "unknown" class for HF rows whose freq we don't carry through:

| id | freq    | samples-per-day |
|---:|---------|---:|
| 0  | unknown |   — |
| 1  | 10s     | 8640 |
| 2  | 1min    | 1440 |
| 3  | 5min    | 288 |
| 4  | 10min   | 144 |
| 5  | 15min   | 96 |
| 6  | 30min   | 48 |
| 7  | 1h      | 24 |
| 8  | 1d      | 1 |
| 9  | 1w      | 1/7 |

The synthesizer draws one of classes 1-9; base-bundles HF rows get
class 0 (unknown). This avoids the extra engineering of matching each
HF row to one of the nine canonical frequencies — the ablation is
still well-controlled because the *difference* between arms is
whether the synth half carries its freq hint through.

### Embedding

`FrequencyEmbedding(num_freqs=10, emb_dim=3)`. 10 rows × 3 dims = 30
learned parameters. Initialised small (std=0.02).

### Injection point

Per-patch concat along the feature dim. Given input `X: [B, T, C]`
with patch size `W=16`, the per-patch view for the encoder is
currently `[B, T/W, C, W]`. With the freq embedding, we append the
E-dim embedding as extra features *per channel*:

```
X_patch:   [B, T/W, C, W]         # standard patch view
E_broad:   [B, T/W, C, E]         # emb[freq_id] broadcast
X_aug:     [B, T/W, C, W+E]       # concat along last dim
```

The GRU patch encoder sees both raw values and the freq hint locally
within each patch. Every patch position carries the hint (unlike a
prepended token that only reaches other patches through attention).

## Training

Same Tiny arch, same data pipeline, same contrastive + reconstruction
loss. Two arms:

1. **freqemb_mix** — freq embedding ON, mixup OFF. Tests whether the
   hint alone helps.
2. **freqemb_mixup_mix** — freq embedding ON, mixup ON (p=0.3 per
   step, alpha ~ Beta(0.2, 0.2)). Tests whether interpolation on the
   embedding produces the continuous-manifold benefit we hope for.

Both arms run 30k steps from-scratch, bs=24, lr=1e-4, 50/50 synth mix.
Direct replacement for v3c_mix in the original experiment.

Mixup:

```python
# within each batch, probability p=0.3
alpha = Beta(0.2, 0.2).sample()
idx = torch.randperm(B)
X_mix   = alpha * X + (1-alpha) * X[idx]
emb_mix = alpha * emb(freq) + (1-alpha) * emb(freq[idx])
# feed (X_mix, emb_mix) to the backbone (bypassing the discrete freq lookup)
```

## What we expect

- **If freq embedding helps, we expect freqemb_mix to improve the
  seasonal subset** (73 configs, season>1). The periodic focus subset
  should show the clearest gains. Non-trend / stationary should be
  neutral because those configs map to freq class 0 (unknown) and
  receive the unknown-embedding, which acts as a pass-through.
- **If mixup helps, freqemb_mixup_mix should beat freqemb_mix on
  configs where real data has multi-period structure** — primarily
  hourly datasets where daily+weekly coexist (m4_hourly, solar/H,
  electricity/H). This is the multi-period composition trick
  described in [`FOLLOWUP_DESIGN.md`](../periodic-synth-mix/FOLLOWUP_DESIGN.md).

## What we don't test yet (future ablations)

- Tagging HF rows with their actual freq (would require dataset
  pipeline work). For now they all get class 0.
- Larger embedding dims (4, 8, 16) — small dim is the recommendation
  but a sweep could surface a sweet spot.
- Interaction with the quantile head (queued as task #24).

## Pipeline

```
 scaffold + design  →  freq_embedding module + tests
                             │
                             ▼
       synth generator emits freq_id alongside series
                             │
                             ▼
        MixedPeriodicLoader returns (X, freq_ids) tuples
                             │
                             ▼
       ConfigurableModel accepts freq_ids, concats emb
                             │
                             ▼
           train.py: optional mixup augmentation
                             │
                             ▼
        CPU smoke (both arms, 50 steps each)
                             │
                             ▼
            vast.ai 4090 (sync_loop with the fixed size floors)
                             │
                   ┌─────────┴─────────┐
                   │                   │
            freqemb_mix          freqemb_mixup_mix
              30k                   30k
                   │                   │
              R1 head 30k          R1 head 30k
                   │                   │
              GIFT-Eval B4        GIFT-Eval B4
                   │                   │
                   └─────── compare ────┘
                         (→ pick best for task #24)
```

## Budget

- Two 30k backbone runs at ~5 sps with mix_ratio=0.5 ≈ 3h each = 6h
- Two R1 heads at 30k ≈ 1.7h each = 3.4h
- Two GIFT-Evals ≈ 5h each = 10h
- Total ~20h on one 4090 ≈ $6-7

## Cleanup rules

- Sync loop: use the fixed version with per-pattern size floors.
- Dry-run first tick before leaving unattended.
- Manually confirm backbone, head, losses_csv and log all appear locally.
