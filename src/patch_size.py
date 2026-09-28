"""The patch size of a series, from its frequency, as Moirai 1.0 picks it (#417).

A multi-patch model has one patch encoder and one value head per patch size.
The frequency of a series picks the size:

* In training, each sample draws one size uniformly from its frequency's
  range (uni2ts ``DefaultPatchSizeConstraints``).
* At inference, each frequency has one fixed size.

A frequency is a pandas or gluonts string ("H", "15T", "W-SUN", "YE-DEC"),
or an id of the frequency-embedding vocabulary. v2 (#419) keeps the ten v1
ids and adds "4s", "6h", "1M", "1Q" and "1Y", so one table
(``FREQ_NAMES_V2``) reads the ids of both. Id 0 and None mean "no label".
A sample with no label draws from every size. The GiftEvalPretrain stream
(#419) labels each window with its v2 id; the synthetic rows carry no label.
"""

from __future__ import annotations

import numbers
import re

import torch

from .freq_embedding import FREQ_NAMES_V2

PATCH_SIZES = (8, 16, 32, 64, 128)

# uni2ts GetPatchSize / PatchCrop: a sample holds at least two patches.
MIN_TIME_PATCHES = 2

# Frequency class -> (smallest, largest) patch size in training. uni2ts gives
# Q and Y the range 1 to 8; 8 is our smallest size. None is "no label".
TRAIN_RANGE = {
    "S": (64, 128), "T": (32, 128), "H": (32, 64),
    "D": (16, 32), "B": (16, 32), "W": (16, 32),
    "M": (8, 32), "Q": (8, 8), "Y": (8, 8),
    None: (8, 128),
}

# Frequency class -> the one patch size inference reads it with.
INFERENCE_SIZE = {
    "S": 128, "T": 64, "H": 32, "D": 16, "B": 16, "W": 16, "M": 16,
    "Q": 8, "Y": 8,
}

# The base of a frequency string -> its class. "MIN" is pandas "min".
_BASE_CLASS = {
    "S": "S", "T": "T", "MIN": "T", "H": "H", "BH": "H", "D": "D",
    "B": "B", "C": "B", "W": "W", "M": "M", "SM": "M", "Q": "Q",
    "Y": "Y", "A": "Y",
}


def frequency_class(freq) -> str | None:
    """The Moirai frequency class of ``freq``, or None when it has none.

    The multiple ("15" of "15T") and the anchor ("-SUN" of "W-SUN") do not
    change the class. The start and end variants ("MS", "ME", "QE", "YE")
    and the business variants ("BM", "BQ", "BA") take the class of their
    base. Lowercase "ms", "us" and "ns" are sub-second units, which have
    no class.
    """
    if freq is None:
        return None
    if isinstance(freq, numbers.Integral):
        freq = FREQ_NAMES_V2[int(freq)]
    match = re.fullmatch(r"\s*\d*\s*([A-Za-z]+)(?:-\w+)?\s*", str(freq))
    if match is None or match.group(1) in ("ms", "us", "ns"):
        return None
    base = match.group(1).upper()
    for key in (base, base[:-1] if base[-1] in "SE" else None,
                base[1:], base[1:-1] if base[-1] in "SE" else None):
        if key in _BASE_CLASS:
            return _BASE_CLASS[key]
    return None


def patch_size_choices(freq, sizes=PATCH_SIZES, training=True) -> tuple:
    """The patch sizes a series of frequency ``freq`` may take.

    In training: every size in ``sizes`` inside the frequency's range. At
    inference: the frequency's one fixed size, as a 1-tuple. A frequency
    with no class has no inference size, so it raises ValueError there.
    """
    cls = frequency_class(freq)
    if training:
        low, high = TRAIN_RANGE[cls]
        return tuple(p for p in sizes if low <= p <= high)
    if cls is None:
        raise ValueError(f"frequency {freq!r} has no inference patch size")
    return (INFERENCE_SIZE[cls],)


def check_patch_sizes(sizes, base_size=16):
    """Raise ValueError when ``sizes`` leaves a frequency without a size.

    Every frequency needs at least one training size and its inference
    size, and the model's base size must be one of them.
    """
    unknown = sorted(set(sizes) - set(PATCH_SIZES))
    if unknown:
        raise ValueError(f"patch sizes {unknown} are not in {PATCH_SIZES}")
    if base_size not in sizes:
        raise ValueError(f"the base patch size {base_size} is not in {sizes}")
    for cls in TRAIN_RANGE:
        if not patch_size_choices(cls, sizes):
            raise ValueError(f"frequency {cls} has no training size in {sizes}")
    for cls, size in INFERENCE_SIZE.items():
        if size not in sizes:
            raise ValueError(f"frequency {cls} reads at {size}, not in {sizes}")


def draw_patch_sizes(freq_ids, sizes, n, lengths=None) -> torch.Tensor:
    """One patch size per sample: uniform over its frequency's range.

    ``freq_ids`` is the batch's LongTensor of frequency ids, or None for a
    batch with no labels. Returns a CPU LongTensor of ``n`` sizes. The draw
    uses torch's CPU generator, so a seeded run draws the same sizes.

    ``lengths`` (#421) is a LongTensor of the real values of each sample,
    its length after the left padding. A size P must then cut them into at
    least two patches (length >= 2 P), as uni2ts ``GetPatchSize`` asks with
    ``min_time_patches = 2``. A sample that no size of its range fits draws
    from the whole range. None: every size of the range, as #417 draws.
    """
    ids = (torch.zeros(n, dtype=torch.long) if freq_ids is None
           else freq_ids.detach().cpu().long())
    out = torch.empty(n, dtype=torch.long)
    for freq_id in ids.unique().tolist():
        rows = (ids == freq_id).nonzero().squeeze(1)
        choices = torch.tensor(patch_size_choices(int(freq_id), sizes))
        if lengths is None:
            out[rows] = choices[torch.randint(len(choices), (len(rows),))]
        else:
            out[rows] = _draw_fitting(choices, lengths.detach().cpu()[rows])
    return out


def _draw_fitting(choices, lengths):
    """Per row, a uniform draw among the ``choices`` P with ``2 P <= length``,
    or among all of them when none fits."""
    fits = MIN_TIME_PATCHES * choices.view(1, -1) <= lengths.view(-1, 1)
    fits[~fits.any(dim=1)] = True
    pick = (torch.rand(len(lengths)) * fits.sum(dim=1)).long()
    nth = fits.long().cumsum(dim=1) - 1
    return choices[((nth == pick.view(-1, 1)) & fits).long().argmax(dim=1)]
