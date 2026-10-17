"""A family of forecast decoders on a backbone with one patch size, selected
by the frequency of a series (freq_family).

Moirai reads each series at the patch size of its frequency, and has one
value head for each size. A backbone of ours with one patch size always
reads patches of W = 16 values. This module keeps the family in the decoder
of the forecast only: one decoder (a member) for each Moirai frequency
class. Each member decodes the Q quantiles of the W values of the next
patch from the forecaster latents, as the standard head does. The key of a
member is the inference patch size of its class in :mod:`src.patch_size`.
The key selects the decoder and nothing else.

Members: 128 (S), 64 (T), 32 (H) and 16 (D, B, W, M). Moirai reads the
classes Q and Y at the size 8. They are 0.02% of the GiftEvalPretrain
stream, so a member of their own cannot train: they use the base member,
16. A series with no frequency label uses the base member too.

Two bodies:

* ``'shared'``: the transformer and the norm of one standard head, and one
  output layer for each member. The value heads of the Moirai copy have
  this form (``value_heads.<P>``).
* ``'heads'``: one complete standard head for each member.

Two training rules:

* ``'strict'``: a row trains the member that scores its frequency.
* ``'draw'``: a row draws one member from the training range of its
  frequency (``TRAIN_RANGE``), among the members of the family. The score
  always uses the fixed member.

The state dict of a family has its own keys: ``freq_heads.<key>.*``, or
``freq_body.*`` and ``freq_out.<key>.*``. So the code of a head bank (#412,
``heads.<P>.*``) does not read a family as a bank.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .forecasting_head import TransformerQuantileForecastingHead, W
from .patch_size import (INFERENCE_SIZE, PATCH_SIZES, TRAIN_RANGE,
                         frequency_class)

FAMILY_MEMBERS = (16, 32, 64, 128)
FAMILY_BODIES = ("shared", "heads")
FAMILY_RULES = ("strict", "draw")
# The member of a frequency class that has no member of its own.
BASE_MEMBER = W


def check_family_members(members) -> tuple:
    """``members`` in increasing order. Raises ValueError for a key that is
    no Moirai patch size, and for a set with no base member."""
    members = tuple(sorted(int(m) for m in members))
    unknown = sorted(set(members) - set(PATCH_SIZES))
    if unknown:
        raise ValueError(f"family members {unknown} are not in {PATCH_SIZES}")
    if BASE_MEMBER not in members:
        raise ValueError(f"the family {members} has no base member "
                         f"{BASE_MEMBER}")
    return members


def family_member(freq, members=FAMILY_MEMBERS) -> int:
    """The member that scores a series of frequency ``freq``: the inference
    patch size of its class when the family has that member, else the base
    member. ``freq`` is a frequency string or an id of the stream, as
    :func:`src.patch_size.frequency_class` reads it."""
    size = INFERENCE_SIZE.get(frequency_class(freq))
    return size if size in members else BASE_MEMBER


def family_member_choices(freq, members=FAMILY_MEMBERS) -> tuple:
    """The members that a row of frequency ``freq`` can train under the draw
    rule: the members in the training range of its class, or the base member
    when the range holds no member."""
    low, high = TRAIN_RANGE[frequency_class(freq)]
    return tuple(m for m in members if low <= m <= high) or (BASE_MEMBER,)


def family_row_members(freq_ids, n, rule="strict",
                       members=FAMILY_MEMBERS) -> torch.Tensor:
    """One member key for each of the ``n`` rows of a training batch.

    ``freq_ids`` is the LongTensor of the frequency ids of the batch, or
    None for a batch with no labels. Under ``'strict'`` a row gets the
    member that scores its frequency. Under ``'draw'`` it draws uniformly
    among :func:`family_member_choices`, with the CPU generator of torch, so
    a seeded run draws the same members. A row with one choice needs no
    draw. Returns a CPU LongTensor.
    """
    if rule not in FAMILY_RULES:
        raise ValueError(f"unknown family rule {rule!r}. Use one of "
                         f"{FAMILY_RULES}.")
    ids = (torch.zeros(n, dtype=torch.long) if freq_ids is None
           else freq_ids.detach().cpu().long())
    out = torch.empty(n, dtype=torch.long)
    for freq_id in ids.unique().tolist():
        rows = (ids == freq_id).nonzero().squeeze(1)
        choices = torch.tensor(
            (family_member(freq_id, members),) if rule == "strict"
            else family_member_choices(freq_id, members))
        pick = (torch.randint(len(choices), (len(rows),)) if len(choices) > 1
                else torch.zeros(len(rows), dtype=torch.long))
        out[rows] = choices[pick]
    return out


class SharedBodyMember(TransformerQuantileForecastingHead):
    """One member of a shared-body family, as a standard head: the
    transformer and the norm of the body, and the output layer of the
    member. It owns no weight, so the family makes one for each call."""

    def __init__(self, body, forecast_head):
        nn.Module.__init__(self)
        self.forecast_len, self.causal = body.forecast_len, body.causal
        self.quantile_levels = body.quantile_levels
        self.num_quantiles = body.num_quantiles
        self.transformer, self.norm = body.transformer, body.norm
        self.forecast_head = forecast_head
        self.training = body.training


class FrequencyFamilyHead(nn.Module):
    """A family of forecast decoders, one for each member key.

    ``make_head()`` builds one standard quantile head that decodes the W
    values of the next patch. The members are built in increasing order of
    their key. So with one seed, the base member starts from the weights of
    the standard head: with the shared body, the body and the output layer
    of the base member.

    A member (:meth:`member`) has the interface of a standard head, with no
    ``patch_size``: :func:`src.forecasting_head.forecast_B4` reads the
    context in patches of W for each member.
    """

    def __init__(self, make_head, members=FAMILY_MEMBERS, body="heads"):
        super().__init__()
        if body not in FAMILY_BODIES:
            raise ValueError(f"unknown family body {body!r}. Use one of "
                             f"{FAMILY_BODIES}.")
        self.members, self.body = check_family_members(members), body
        if body == "heads":
            self.freq_heads = nn.ModuleDict(
                {str(m): make_head() for m in self.members})
        else:
            self.freq_body, self.freq_out = self.build_shared_body(make_head)

    def build_shared_body(self, make_head):
        """``(body, output layers)``: one standard head with no output
        layer, and one output layer for each member. The first member takes
        the output layer that the head was built with."""
        body = make_head()
        if not isinstance(body, TransformerQuantileForecastingHead):
            raise ValueError("the shared body is the body of the transformer "
                             "quantile head, not of a "
                             f"{type(body).__name__}")
        first = body.forecast_head
        body.forecast_head = None
        layers = [first] + [nn.Linear(first.in_features, first.out_features)
                            for _ in self.members[1:]]
        return body, nn.ModuleDict(dict(zip(map(str, self.members), layers)))

    def member(self, key):
        """The decoder of the member ``key``, as a standard head."""
        if self.body == "heads":
            return self.freq_heads[str(int(key))]
        return SharedBodyMember(self.freq_body, self.freq_out[str(int(key))])

    def for_frequency(self, freq):
        """The decoder that scores a series of frequency ``freq``."""
        return self.member(family_member(freq, self.members))

    def forward(self, latents, members):
        """Each row of ``latents`` through the decoder of its member.

        ``latents`` is ``(B*C, T, H)`` and ``members`` holds one key for
        each row. Returns ``(B*C, T, Q, L)``, in the row order of
        ``latents``.
        """
        members = members.to(latents.device)
        rows = [(members == key).nonzero().squeeze(1)
                for key in members.unique()]
        outs = [self.member(members[r[0]])(latents[r]) for r in rows]
        return torch.cat(outs)[torch.cat(rows).argsort()]


def family_layout_of(state_dict):
    """``(body, members)`` of the family that a head checkpoint holds, read
    off its keys, or None for a checkpoint of another head."""
    for body, prefix in (("heads", "freq_heads."), ("shared", "freq_out.")):
        keys = {k.split(".")[1] for k in state_dict if k.startswith(prefix)}
        if keys:
            return body, tuple(sorted(int(k) for k in keys))
    return None


def family_member_state(state_dict, key) -> dict:
    """The state dict of the member ``key`` of a family checkpoint, with the
    keys of one standard head: its body and its output layer."""
    body, _ = family_layout_of(state_dict)
    renames = ({f"freq_heads.{key}.": ""} if body == "heads"
               else {"freq_body.": "", f"freq_out.{key}.": "forecast_head."})
    return {new + k[len(old):]: v for k, v in state_dict.items()
            for old, new in renames.items() if k.startswith(old)}
