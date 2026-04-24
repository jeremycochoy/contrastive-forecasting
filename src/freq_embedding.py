"""
Frequency embedding for patch-level conditioning.

Adds a small learned embedding (default dim 3) per frequency class, which
gets concatenated to each W-step patch's raw values along the feature axis.
The result: the GRU patch encoder sees both the raw values and the freq hint
locally per patch, rather than having to propagate a single prepended token
through the transformer's attention.

Ten classes (see DESIGN.md):

    0 = unknown (default for HF rows we didn't tag)
    1 = 10s, 2 = 1min, 3 = 5min, 4 = 10min, 5 = 15min,
    6 = 30min, 7 = 1h, 8 = 1d, 9 = 1w

The small embedding dim (3-4 recommended) is a regulariser: just enough to
disambiguate among the 10 classes, not enough to carry forecasting behaviour
directly.
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
    """
    if spp <= 16:
        # Distribute across 10s, 1min, 5min, 10min roughly uniformly.
        return 1 + int(min(3, int(4 * (spp - 8) / 8.0)))
    elif spp <= 48:
        return 5 if spp <= 24 else 6
    elif spp <= 128:
        return 7 if spp <= 72 else 8
    return 9
