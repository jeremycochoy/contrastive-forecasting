"""A bank of forecast heads by block size, on the trunk of one transformer
quantile head (``--head-block-sizes`` of the head trainer).

Moirai has one output head per patch size P, and the frequency of a series
picks P. A backbone of ours with one patch size reads patches of W = 16
values, and its forecast head decodes each rolled latent into the next 16
values. This module keeps the backbone and the trunk of the head, and puts
the bank in the output layer only: head P decodes k = P / W consecutive
latents together into the P values of their block, Q quantiles each.

* The trunk (the transformer and the norm of a
  :class:`TransformerQuantileForecastingHead`) is shared by every P.
* The output layer of P reads the trunk outputs at the k positions of one
  block, concatenated (k x H), and gives P x Q values. For P = W (k = 1) it
  is the output layer of the standard head.

The block size of a series is the inference patch size of its Moirai
frequency class (:data:`src.patch_size.INFERENCE_SIZE`): S 128, T 64, H 32,
and D, B, W, M 16. Q and Y read at 8, below one latent of 16 values, so they
use 16. A series with no frequency label uses 16. A class whose size is not
in the bank uses 16 too.

The state dict has its own keys, ``block_body.*`` and ``block_out.<P>.*``.
So the code of a head bank (#412, ``heads.<P>.*``) and of a frequency
family (``freq_*``) does not read it, and the eval reads the bank off the
keys with no flag.

``--head-block-trunks separate`` shares nothing between the sizes: the bank
then holds one full head per block size, a trunk of its own and the output
layer of P.

* Each trunk is a copy of the given trunk, so the trunks start from the
  same weights. The output layers are those of the shared bank.
* In training, a row goes through the trunk of its own P only. The batches,
  the steps and the loss are those of the shared bank.
* The keys of the trunks are ``block_body.<P>.*``. The eval reads the mode
  off them (:func:`block_trunk_mode`), and the member of P uses the trunk
  of P.
"""

from __future__ import annotations

import copy
import re

import torch
import torch.nn as nn

from .forecasting_head import (W, TransformerQuantileForecastingHead,
                               masked_quantile_loss, patch_size_groups,
                               quantile_loss)
from .patch_size import INFERENCE_SIZE, frequency_class

BLOCK_SIZES = (16, 32, 64, 128)
# The trunks of a bank: one for every size, or one per size.
TRUNK_MODES = ("shared", "separate")
SEPARATE_TRUNK_KEY = re.compile(r"block_body\.\d+\.")


def check_block_sizes(sizes) -> tuple:
    """``sizes`` in increasing order. Raises ValueError for a size that is
    no inference patch size of W or more (one latent holds W values), and
    for a set without W: the size of a row with no size of its own."""
    sizes = tuple(sorted(int(p) for p in sizes))
    unknown = sorted(set(sizes) - set(BLOCK_SIZES))
    if unknown:
        raise ValueError(f"block sizes {unknown} are not in {BLOCK_SIZES}")
    if W not in sizes:
        raise ValueError(f"the block sizes {sizes} have no size {W}")
    return sizes


def block_size_of(freq, sizes=BLOCK_SIZES) -> int:
    """The block size of a series of frequency ``freq``: the inference patch
    size of its class when the bank has that size, else W. ``freq`` is a
    frequency string or an id of the stream, as
    :func:`src.patch_size.frequency_class` reads it."""
    size = INFERENCE_SIZE.get(frequency_class(freq))
    return size if size in sizes else W


def row_block_sizes(freq_ids, n, sizes=BLOCK_SIZES) -> torch.Tensor:
    """One block size for each of the ``n`` rows of a batch, from its
    frequency id. ``freq_ids`` is the LongTensor of the batch, or None for a
    batch with no labels: every row then uses W. Returns a CPU LongTensor."""
    ids = (torch.zeros(n, dtype=torch.long) if freq_ids is None
           else freq_ids.detach().cpu().long())
    return torch.tensor([block_size_of(i, sizes) for i in ids.tolist()])


def block_targets(x_norm, size):
    """``(targets, n_blocks)``: the P = ``size`` values of each supervised
    block of a row, ``(B*C, n_blocks, P)``, in the row order of the latents.

    The latents of a row hold T = T_raw / W patches. Block b reads the
    latents of the patches b k to b k + k - 1 (k = P / W) and decodes the
    next k patches: the values W (b k + 1) to W (b k + k + 1) - 1. The block
    whose last target patch is past the window has no target, so n_blocks
    = (T - 1) // k. For k = 1 these are the targets of the standard head.
    Called on a mask (as float), it gives the mask of each target value.
    """
    B, T_raw, C = x_norm.shape
    k, T = size // W, T_raw // W
    n_blocks = (T - 1) // k
    if n_blocks < 1:
        raise ValueError(f"{T_raw} values hold no whole block of {size} "
                         f"values after the first patch of {W}")
    patches = (x_norm[:, :T * W].reshape(B, T, W, C)
               .permute(0, 3, 1, 2).reshape(B * C, T, W))
    blocks = patches[:, 1:1 + n_blocks * k]
    return blocks.reshape(B * C, n_blocks, size), n_blocks


def decode_blocks(layer, features, size, num_quantiles):
    """``(n, T // k, Q, P)``: the blocks of k = P / W consecutive positions
    of ``features`` ``(n, T, H)``, concatenated and decoded by the output
    ``layer`` of P = ``size``. The positions after the last whole block
    decode nothing."""
    n, T, H = features.shape
    k = size // W
    blocks = features[:, :T // k * k].reshape(n, T // k, k * H)
    return layer(blocks).view(n, T // k, num_quantiles, size)


class BlockSizeHead(nn.Module):
    """One trunk and one output layer per block size.

    ``body`` is a :class:`TransformerQuantileForecastingHead` that decodes W
    values: its transformer and its norm are the trunk, and its output layer
    is the head of W. The output layer of a larger P is ``nn.Linear(k H,
    P Q)``. The layers are built in increasing order of P. So with one seed,
    the trunk and the head of W have the weights of the standard head.

    ``trunks`` ``'separate'`` gives each P a trunk of its own: ``body`` for
    W and a copy of it for each larger P. A copy draws no random number, so
    the output layers have the weights of the shared bank of the same seed.
    """

    def __init__(self, body, sizes=BLOCK_SIZES, trunks="shared"):
        super().__init__()
        if trunks not in TRUNK_MODES:
            raise ValueError(f"the trunks of a block head are one of "
                             f"{TRUNK_MODES}, not {trunks!r}")
        if not isinstance(body, TransformerQuantileForecastingHead):
            raise ValueError("the trunk of a block head is the transformer "
                             "quantile head, not a "
                             f"{type(body).__name__}")
        if body.forecast_len != W:
            raise ValueError(f"the trunk decodes {body.forecast_len} values "
                             f"per latent, not {W}")
        self.sizes = check_block_sizes(sizes)
        self.quantile_levels = body.quantile_levels
        self.num_quantiles = body.num_quantiles
        self.trunks = trunks
        first = body.forecast_head
        body.forecast_head = None
        self.block_body = body if trunks == "shared" else nn.ModuleDict({
            str(p): body if p == W else copy.deepcopy(body)
            for p in self.sizes})
        self.block_out = nn.ModuleDict({
            str(p): first if p == W else nn.Linear(
                p // W * first.in_features, p * self.num_quantiles)
            for p in self.sizes})

    def trunk(self, size):
        """The trunk that the rows of the block size ``size`` read."""
        if self.trunks == "shared":
            return self.block_body
        return self.block_body[str(int(size))]

    def features(self, seq, src_mask=None, row_sizes=None):
        """The trunk outputs ``(B*C, T, H)`` of ``seq``, under ``src_mask``
        (:meth:`TransformerQuantileForecastingHead.features`).

        ``row_sizes`` ``(B*C,)`` holds the block size of each row. Separate
        trunks need it: the features of a row come from the trunk of its own
        size. The shared trunk reads every row and ignores it. ``src_mask``
        is one ``(T, T)`` mask over the positions, the same for every row,
        so each group of rows keeps it whole.
        """
        if self.trunks == "shared":
            return self.block_body.features(seq, src_mask)
        if row_sizes is None:
            raise ValueError("separate trunks need the block size of each "
                             "row: pass row_sizes")
        groups = patch_size_groups(row_sizes)
        parts = [self.trunk(size).features(seq[rows.to(seq.device)], src_mask)
                 for size, rows in groups.items()]
        order = torch.cat(list(groups.values())).argsort().to(seq.device)
        return torch.cat(parts)[order]

    def decode_blocks(self, features, size):
        """The blocks of ``features`` decoded by the output layer of
        ``size`` (:func:`decode_blocks`)."""
        return decode_blocks(self.block_out[str(int(size))], features,
                             int(size), self.num_quantiles)

    def forward(self, seq, size, src_mask=None, start=0):
        """``(B*C, (T - start) // k, Q, P)``: the blocks of k latents from
        position ``start`` of ``seq``, each decoded into P = ``size``
        values."""
        features = self.trunk(size).features(seq, src_mask)
        return self.decode_blocks(features[:, start:], size)

    def member(self, size):
        """The head of the block size ``size``, as a standard head."""
        return BlockHeadMember(self, int(size))

    def for_frequency(self, freq):
        """The head that scores a series of frequency ``freq``."""
        return self.member(block_size_of(freq, self.sizes))


class BlockHeadMember(TransformerQuantileForecastingHead):
    """The head of one block size P of a :class:`BlockSizeHead`, for the
    eval: the trunk of P (the one trunk of a shared bank) and the output
    layer of P, as a standard head with
    ``forecast_len`` P that reads ``latents_per_block`` = k latents per
    output. It owns no weight, so the bank makes one for each call.

    :func:`src.forecasting_head.forecast_B4` reads the context in patches
    of W (the member has no ``patch_size``), rolls out whole blocks
    (:func:`src.forecasting_head.rollout_steps`) and decodes them with
    :meth:`decode_from`.
    """

    def __init__(self, bank, size):
        nn.Module.__init__(self)
        body = bank.trunk(size)
        self.block_size = self.forecast_len = int(size)
        self.latents_per_block = self.block_size // W
        self.causal = body.causal
        self.quantile_levels = body.quantile_levels
        self.num_quantiles = body.num_quantiles
        self.transformer, self.norm = body.transformer, body.norm
        self.forecast_head = bank.block_out[str(self.block_size)]
        self.training = body.training

    def forward(self, x, src_mask=None):
        """``(B*C, T // k, Q, P)``: the blocks of ``x`` from position 0."""
        return self.decode_from(x, 0, src_mask)

    def decode_from(self, seq, start, src_mask=None):
        """``(B*C, (T - start) // k, Q, P)``: the trunk runs on the whole
        ``seq``, and the blocks start at position ``start`` (the first
        rolled latent of B4)."""
        return decode_blocks(self.forecast_head,
                             self.features(seq, src_mask)[:, start:],
                             self.block_size, self.num_quantiles)


def block_head_loss(head, features, row_sizes, x_norm, keep=None):
    """The pinball loss of a block head on one batch.

    ``features`` ``(B*C, T, H)`` are the trunk outputs at the forecaster
    positions (:meth:`BlockSizeHead.features`): the shared trunk runs once
    on every row, and separate trunks run each on the rows of their size.
    ``row_sizes`` ``(B*C,)``
    holds the block size of each row. The rows of one size decode their
    blocks with the head of that size, over the targets of
    :func:`block_targets`. ``keep`` (bool, the shape of ``x_norm``) marks
    the values a term counts: no padding (#419). None counts every value.
    The loss of a group counts with its share of the rows, as the head bank
    does (#412).
    """
    loss = 0.0
    for size, rows in patch_size_groups(row_sizes).items():
        rows = rows.to(features.device)
        targets, n_blocks = block_targets(x_norm, size)
        preds = head.decode_blocks(features[rows], size)[:, :n_blocks]
        if keep is None:
            part = quantile_loss(preds, targets[rows])
        else:
            kept, _ = block_targets(keep.float(), size)
            part = masked_quantile_loss(preds, targets[rows],
                                        kept[rows] > 0.5)
        loss = loss + len(rows) / len(row_sizes) * part
    return loss


def block_head_sizes(state_dict) -> tuple:
    """The block sizes of a block-head checkpoint, read off its keys
    ``block_out.<P>.*``, or () for a checkpoint of another head."""
    found = [k.split(".")[1] for k in state_dict if k.startswith("block_out.")]
    return tuple(sorted({int(p) for p in found if p.isdigit()}))


def block_trunk_mode(state_dict) -> str:
    """``'separate'`` for a block-head checkpoint with one trunk per block
    size (keys ``block_body.<P>.*``), else ``'shared'``."""
    separate = any(SEPARATE_TRUNK_KEY.match(k) for k in state_dict)
    return "separate" if separate else "shared"


def block_trunk_state(state_dict) -> dict:
    """The state dict of a standard head from a block-head checkpoint: the
    trunk and the output layer of W, under the keys of
    :class:`TransformerQuantileForecastingHead`. The eval builds the trunk
    from it. With one trunk per block size, it is the trunk of W: every
    trunk has its architecture."""
    separate = block_trunk_mode(state_dict) == "separate"
    body = f"block_body.{W}." if separate else "block_body."
    renames = {body: "", f"block_out.{W}.": "forecast_head."}
    return {new + k[len(old):]: v for k, v in state_dict.items()
            for old, new in renames.items() if k.startswith(old)}
