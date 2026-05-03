"""
Frequency and seasonality embeddings for patch-level conditioning.

Two parallel categorical axes:

* **Frequency** — wall-clock sample rate ("10s", "1min", ..., "1w"). 10
  classes including "unknown".
* **Seasonality** — dominant period in samples (1, 7, 24, 168, ...) bucketed
  log-style. 10 classes including "unknown".

Each axis has its own small learned embedding table; both get concatenated
to every patch along the feature axis. The pair `(freq_id, seasonality_id)`
is what the GIFT-Eval task descriptor exposes naturally — `freq` is the
pandas freq string, and `seasonality = gluonts.time_feature.get_seasonality(freq)`.

For training-time data we look up `(freq_id, seasonality_id)` from a
static `SOURCE_ID_TO_LABELS` table built off the bundle's `source_id` column.
Synth rows carry their own (freq, seasonality) sampled jointly during
generation. Bundle synth and gift-train rows fall back to (0, 0) because
the build pipeline didn't preserve the metadata.

The small embedding dim (3-4 recommended) acts as a regulariser: just
enough to disambiguate among the classes, not enough to carry forecasting
behaviour directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Public constant — total number of freq classes (including 0=unknown).
NUM_FREQS = 10

# Canonical freq-name ↔ id mapping.
FREQ_NAMES = [
    "unknown", "10s", "1min", "5min", "10min",
    "15min", "30min", "1h", "1d", "1w",
]
FREQ_NAME_TO_ID = {name: i for i, name in enumerate(FREQ_NAMES)}


# ── Seasonality vocab (period in samples) ────────────────────────────────────

NUM_SEASONALITIES = 10

# Human-readable bucket labels. Each non-unknown bucket is "≤ 4·2^(i-1)".
SEASONALITY_NAMES = [
    "unknown", "≤4", "≤8", "≤16", "≤32", "≤64", "≤128", "≤256", "≤512", ">512",
]

# Inclusive upper bounds for buckets 1..8; bucket 9 is everything beyond 512.
_SEASONALITY_BOUNDARIES = (4, 8, 16, 32, 64, 128, 256, 512)


# Per-bucket spp sampling ranges used by the synth's joint (freq, seasonality)
# coverage path. Bucket 0 is the "no period info" sentinel, generated with
# spp >> 1024 so the visible signal in a 1024-window is at most a fraction of
# a cycle (looks aperiodic locally). Buckets 1..9 sample inside their bucket's
# log-range so `seasonality_to_id(sampled_spp)` recovers the bucket id.
SEASONALITY_BUCKET_SPP_RANGES: dict[int, tuple[float, float]] = {
    0: (1024.0, 4096.0),   # "no period" — too long to see a full cycle
    1: (2.0, 4.0),
    2: (5.0, 8.0),
    3: (9.0, 16.0),
    4: (17.0, 32.0),
    5: (33.0, 64.0),
    6: (65.0, 128.0),
    7: (129.0, 256.0),
    8: (257.0, 512.0),
    9: (513.0, 1024.0),
}


def seasonality_to_id(spp: float) -> int:
    """Bucket a period-in-samples (`spp`) into one of NUM_SEASONALITIES classes.

    Doubling buckets. Bucket **0** is the "no information" sentinel and
    is also returned for ``spp <= 1`` because a 1-step seasonality is
    gluonts's default for daily/weekly freqs and carries no useful
    cycle information for the embedding to specialise on. Common
    GIFT-Eval values map as: 1→0, 7→2, 12→3, 24→4, 48→5, 96→6, 168→7,
    288→8.
    """
    if spp is None or spp <= 1:
        return 0
    for i, bound in enumerate(_SEASONALITY_BOUNDARIES, start=1):
        if spp <= bound:
            return i
    return NUM_SEASONALITIES - 1

# Canonical samples-per-day, useful for mapping a sampled spp from the synth
# back to a freq class. 1/7 for weekly is approximate; we never actually
# query it by value, just by id.
SAMPLES_PER_DAY = {
    1: 8640,  # 10s
    2: 1440,  # 1min
    3: 288,   # 5min
    4: 144,   # 10min
    5: 96,    # 15min
    6: 48,    # 30min
    7: 24,    # 1h
    8: 1,     # 1d
    9: 1 / 7, # 1w
}


class FrequencyEmbedding(nn.Module):
    """Small learned embedding table over 10 freq classes.

    Parameters
    ----------
    emb_dim : int
        Embedding dimension. 3 or 4 are the recommended sizes
        (see DESIGN.md — small dim acts as a regulariser).
    num_freqs : int
        Number of freq classes. Defaults to 10 (the module constant
        :data:`NUM_FREQS`, which includes class 0 = unknown).

    Notes
    -----
    Initialised with `torch.nn.init.normal_(std=0.02)` — the same init
    used for token embeddings in transformers. The class 0 ("unknown")
    row is *not* zero-initialised: treating it as another learnable
    class lets the model choose how to handle unlabeled rows instead
    of forcing pass-through.
    """

    def __init__(self, emb_dim: int = 3, num_freqs: int = NUM_FREQS):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_freqs = num_freqs
        self.embedding = nn.Embedding(num_freqs, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, freq_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding for a batch of freq ids.

        Parameters
        ----------
        freq_ids : LongTensor of shape ``[B]``

        Returns
        -------
        Tensor of shape ``[B, emb_dim]``.
        """
        return self.embedding(freq_ids)

    def mix(self, ids_a: torch.Tensor, ids_b: torch.Tensor,
            alpha: torch.Tensor) -> torch.Tensor:
        """Interpolate two freq embeddings with per-sample alpha.

        Used by mixup augmentation: instead of a hard class assignment,
        each sample receives ``alpha * emb(freq_a) + (1 - alpha) * emb(freq_b)``.

        Parameters
        ----------
        ids_a, ids_b : LongTensor of shape ``[B]``
        alpha : FloatTensor of shape ``[B]`` — mixing weights in [0, 1].

        Returns
        -------
        Tensor of shape ``[B, emb_dim]``.
        """
        emb_a = self.embedding(ids_a)                  # [B, E]
        emb_b = self.embedding(ids_b)                  # [B, E]
        a = alpha.to(emb_a.dtype).unsqueeze(-1)        # [B, 1]
        return a * emb_a + (1 - a) * emb_b


def spp_to_freq_id(spp: float) -> int:
    """Best-guess freq class from a samples-per-period value.

    The synthesizer samples spp log-uniformly in [8, 256]. We bucket:

      spp ≤ 16   → 10s / 1min / 5min / 10min (class 1-4 — very sub-daily)
      spp ≤ 48   → 15min / 30min (class 5-6)
      spp ≤ 128  → 1h / 1d (class 7-8)
      else       → 1w (class 9)

    This is *lossy* by design — spp alone doesn't uniquely determine
    the physical dt, but we only need it to tag each synth draw with
    something sensible for mixup to interpolate over.

    Retained for the legacy single-axis path (synth pre-redesign). The
    new dual-axis path uses ``seasonality_to_id`` for the period axis
    and a separately-sampled freq id for the sample-rate axis.
    """
    if spp <= 16:
        # Distribute across 10s, 1min, 5min, 10min roughly uniformly.
        return 1 + int(min(3, int(4 * (spp - 8) / 8.0)))
    elif spp <= 48:
        return 5 if spp <= 24 else 6
    elif spp <= 128:
        return 7 if spp <= 72 else 8
    return 9


# ── gluonts/pandas freq string → id ──────────────────────────────────────────

# Pandas/gluonts use single-letter aliases (T, H, D, W) and number-prefixed
# variants (5T, 1H, 30min). We accept both and any case.
_GLUONTS_FREQ_ALIASES = {
    "10s": "10s",
    "min": "1min", "1min": "1min", "t": "1min", "1t": "1min",
    "5min": "5min", "5t": "5min",
    "10min": "10min", "10t": "10min",
    "15min": "15min", "15t": "15min",
    "30min": "30min", "30t": "30min",
    "h": "1h", "1h": "1h",
    "d": "1d", "1d": "1d",
    "w": "1w", "1w": "1w",
}


def gluonts_freq_to_id(freq) -> int:
    """Map a gluonts/pandas freq string ("1H", "10min", "D", ...) to a freq id.

    Returns 0 (unknown) for anything outside our 10-class vocab — including
    monthly / yearly tasks, which are out-of-distribution for our synth
    training.
    """
    if freq is None:
        return 0
    s = str(freq).strip().lower()
    if not s:
        return 0
    canonical = _GLUONTS_FREQ_ALIASES.get(s)
    if canonical is None:
        return 0
    return FREQ_NAME_TO_ID[canonical]


# ── Source-id → (freq_id, seasonality_id) lookup ─────────────────────────────
#
# Maps the bundle's `source_id` (defined in rnd:training_data_prep/config.py)
# to the dual-axis labels we feed the model. Wiki seasonalities follow the
# same convention as `gluonts.time_feature.get_seasonality`: hourly→24,
# daily→7. STL components keep the underlying hourly freq but the
# seasonality is left unknown (residual+trend have no period; the seasonal
# component's STL period was not preserved by the build pipeline).
# Gift-train and bundle-synth rows lost their per-row metadata and fall
# back to (0, 0).
SOURCE_ID_TO_LABELS: dict[int, tuple[int, int]] = {
    0: (0, 0),                                                # gift (train)
    1: (FREQ_NAME_TO_ID["1h"], seasonality_to_id(24)),        # wiki_hourly
    2: (FREQ_NAME_TO_ID["1d"], seasonality_to_id(7)),         # wiki_daily
    3: (FREQ_NAME_TO_ID["1h"], 0),                            # wiki_stl_residual
    4: (FREQ_NAME_TO_ID["1h"], 0),                            # wiki_stl_seasonal
    5: (FREQ_NAME_TO_ID["1h"], 0),                            # wiki_stl_trend
    6: (0, 0),                                                # synthetic (bundle)
}


# ── SeasonalityEmbedding module ──────────────────────────────────────────────


class SeasonalityEmbedding(nn.Module):
    """Small learned embedding over NUM_SEASONALITIES period buckets.

    Mirrors :class:`FrequencyEmbedding` but indexes the seasonality axis.
    The two embeddings are concatenated to each patch independently so
    the model can learn (freq, seasonality) interactions through the
    encoder rather than collapsing them into a single id.
    """

    def __init__(self, emb_dim: int = 3, num_seasonalities: int = NUM_SEASONALITIES):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_seasonalities = num_seasonalities
        self.embedding = nn.Embedding(num_seasonalities, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, seasonality_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(seasonality_ids)

    def mix(self, ids_a: torch.Tensor, ids_b: torch.Tensor,
            alpha: torch.Tensor) -> torch.Tensor:
        emb_a = self.embedding(ids_a)
        emb_b = self.embedding(ids_b)
        a = alpha.to(emb_a.dtype).unsqueeze(-1)
        return a * emb_a + (1 - a) * emb_b
