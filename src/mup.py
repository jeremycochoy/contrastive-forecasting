"""Maximal Update Parametrization (muP) with a base width.

Tensor Programs V (Yang and others 2022, arXiv 2203.03466) tunes the learning
rate of a small model and keeps it for a wider model. Its Table 3 gives the
rules for Adam, and its Appendix B.1 writes them with a base width n0. At the
base width every factor is 1, so a model at the base width is the standard
parametrization: the PyTorch default init and one learning rate.

A dimension of a parameter scales when its size differs from its size in the
base model (the same flags at the base width). m_in is the fan_in of the
parameter divided by the fan_in of the base parameter.

| class | dimensions that scale | init | lr | weight decay |
|---|---|---|---|---|
| hidden | fan_out and fan_in | unchanged | / m_in | * m_in |
| output | fan_in only | / sqrt(m_in) | / m_in | * m_in |
| input | fan_out only | unchanged | unchanged | unchanged |
| fixed | none | unchanged | unchanged | unchanged |

The PyTorch default init of a weight has the variance c / fan_in, so the
hidden and input rules need no change. A bias of a layer whose fan_in scales
starts at U(-1/sqrt(fan_in), 1/sqrt(fan_in)) in PyTorch. Its init is
multiplied by sqrt(m_in), so it keeps the bounds of the base layer. The weight
decay factor keeps lr * weight_decay of each tensor, because torch.optim.AdamW
decays a weight by lr * weight_decay at each step (as MuAdam(impl=AdamW) of
the mup package). The attention logits read sqrt(d_head0) / d_head in place of
1 / sqrt(d_head) (Appendix B.1, "Attention Logit Scaling").
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn

HIDDEN, OUTPUT, INPUT, FIXED = "hidden", "output", "input", "fixed"
# Layers whose weight has the layout (fan_out, fan_in, kernel...).
OUT_IN_LAYERS = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
RECURRENT = (nn.GRU, nn.LSTM, nn.RNN)


def fan_in_mult(shape, base_shape, in_dims) -> float:
    """The fan_in of ``shape`` over the fan_in of ``base_shape``."""
    return math.prod(shape[d] for d in in_dims) / math.prod(base_shape[d] for d in in_dims)


def tensor_class(shape, base_shape, out_dims, in_dims) -> str:
    """HIDDEN, OUTPUT, INPUT or FIXED, from the dimensions that scale."""
    out_scales = any(shape[d] != base_shape[d] for d in out_dims)
    in_scales = any(shape[d] != base_shape[d] for d in in_dims)
    if out_scales and in_scales:
        return HIDDEN
    return OUTPUT if in_scales else INPUT if out_scales else FIXED


def weight_dims(module, name, ndim):
    """(fan_out dimensions, fan_in dimensions) of the parameter ``name`` of
    ``module``. An embedding reads its first dimension. Any other weight with
    two or more dimensions has the layout (fan_out, fan_in, ...)."""
    if ndim < 2:
        return tuple(range(ndim)), ()
    if isinstance(module, nn.Embedding):
        return (1,), (0,)
    return (0,), tuple(range(1, ndim))


def layer_weight_name(module, bias_name):
    """The weight that feeds the bias ``bias_name`` of ``module``, or None."""
    if isinstance(module, nn.MultiheadAttention) and bias_name == "in_proj_bias":
        return "in_proj_weight"
    if isinstance(module, OUT_IN_LAYERS) and bias_name == "bias":
        return "weight"
    return None


def parameter_rules(model: nn.Module, base: nn.Module):
    """{parameter name: (class, m_in, bias_of)} for each parameter of
    ``model``. ``base`` has the same flags at the base width (a model on the
    meta device costs no memory). ``bias_of`` names the weight whose fan_in a
    bias follows, or is None."""
    base_shapes = {n: tuple(p.shape) for n, p in base.named_parameters()}
    rules = OrderedDict()
    for mod_name, module in model.named_modules():
        for name, p in module.named_parameters(recurse=False):
            full = f"{mod_name}.{name}" if mod_name else name
            rules[full] = one_rule(module, name, tuple(p.shape), base_shapes, mod_name)
    return rules


def one_rule(module, name, shape, base_shapes, mod_name):
    full = f"{mod_name}.{name}" if mod_name else name
    base_shape = base_shapes[full]
    if isinstance(module, RECURRENT) and shape != base_shape:
        raise NotImplementedError(
            f"muP: the recurrent layer {mod_name or 'model'} changes size with the width "
            "(its init depends on its hidden size). Keep its size fixed.")
    weight = layer_weight_name(module, name)
    if weight is not None:
        w_shape = tuple(getattr(module, weight).shape)
        w_full = f"{mod_name}.{weight}" if mod_name else weight
        in_dims = tuple(range(1, len(w_shape)))
        m_in = fan_in_mult(w_shape, base_shapes[w_full], in_dims)
        return tensor_class(shape, base_shape, (0,), ()), m_in, w_full
    out_dims, in_dims = weight_dims(module, name, len(shape))
    m_in = fan_in_mult(shape, base_shape, in_dims) if in_dims else 1.0
    return tensor_class(shape, base_shape, out_dims, in_dims), m_in, None


@torch.no_grad()
def mup_init_(model: nn.Module, rules) -> None:
    """Turns the PyTorch default init into the muP init: an output weight by
    1 / sqrt(m_in), and a bias whose layer's fan_in scales by sqrt(m_in). Run
    it one time, on a new model. An EMA copy of a module gets the same
    factors, so a teacher stays equal to its student."""
    params = dict(model.named_parameters())
    for name, (kind, m_in, bias_of) in rules.items():
        if kind == OUTPUT and bias_of is None:
            params[name].mul_(1.0 / math.sqrt(m_in))
        elif bias_of is not None and m_in != 1.0:
            params[name].mul_(math.sqrt(m_in))


def lr_wd_mult(kind, m_in, bias_of):
    """(lr factor, weight decay factor) of one tensor."""
    if kind in (HIDDEN, OUTPUT) and bias_of is None:
        return 1.0 / m_in, m_in
    return 1.0, 1.0


def mup_param_groups(model: nn.Module, rules, lr: float, weight_decay: float):
    """AdamW parameter groups: one per (lr factor, weight decay factor), in the
    order of the first parameter of each. A group keeps its lr factor in
    ``lr_mult``, so a schedule sets lr = scheduled lr * lr_mult."""
    groups = OrderedDict()
    for name, p in model.named_parameters():
        lr_m, wd_m = lr_wd_mult(*rules[name])
        group = groups.setdefault((lr_m, wd_m), {
            "params": [], "lr": lr * lr_m, "weight_decay": weight_decay * wd_m,
            "lr_mult": lr_m})
        group["params"].append(p)
    return list(groups.values())


def attn_logit_scale(d_head: int, base_d_head: int) -> float:
    """sqrt(d_head0) / d_head: 1 / sqrt(d_head) at the base width."""
    return math.sqrt(base_d_head) / d_head


def mup_attention_(model: nn.Module, base: nn.Module) -> int:
    """Sets the muP logit scale on each attention layer of ``model`` whose
    d_head differs from the same layer of ``base``. Returns the count of
    layers. A layer at the base d_head keeps 1 / sqrt(d_head)."""
    from .blocks import AttentionChannelMixing, DecoderOnlyTransformerLayer
    base_modules = dict(base.named_modules())
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, AttentionChannelMixing):
            if module.head_dim != base_modules[name].head_dim:
                raise NotImplementedError(
                    f"muP: the channel-mixing attention {name} changes its head size with "
                    "the width, and it has no muP logit scale.")
        elif isinstance(module, DecoderOnlyTransformerLayer):
            d_head = module.self_attn.embed_dim // module.nhead
            b = base_modules[name]
            base_d_head = b.self_attn.embed_dim // b.nhead
            if d_head != base_d_head:
                module.set_attn_logit_scale(attn_logit_scale(d_head, base_d_head))
                count += 1
    return count


def summary(rules) -> str:
    """One line per class: the count of tensors and their lr factors."""
    by_kind = OrderedDict()
    for kind, m_in, bias_of in rules.values():
        lr_m, _ = lr_wd_mult(kind, m_in, bias_of)
        by_kind.setdefault(kind, set()).add(round(lr_m, 6))
    counts = OrderedDict((k, 0) for k in by_kind)
    for kind, _, _ in rules.values():
        counts[kind] += 1
    return "; ".join(f"{k}: {counts[k]} tensors, lr x {sorted(v)}" for k, v in by_kind.items())
