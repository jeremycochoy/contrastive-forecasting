"""The NaN diagnostic mode of the trainer (#421): ``CF_NAN_DEBUG=1``.

Off by default, and inert when off: the trainer registers no hook, and each
probe call below returns at once. On, the trainer records at each step:

* the batch after all transforms (the trainer notes it),
* the max |output| of every module, in call order, and of the tensors inside
  the fp16 attention and of the value-space rollout feedback,
* the max |gradient| of the same tensors, in backward order,
* the loss terms per patch-size group and per rollout depth.

It checks the loss before the backward pass, every gradient before the
gradient clip and the optimizer step, and every weight after the step. On
the first non-finite value it saves a dump with ``torch.save`` and prints a
summary, and the trainer exits with :data:`EXIT_CODE`. The loss and the
gradient checks stop before the step, and their dump holds the weights the
step read. The weight check keeps a copy of the weights from before the
step for its dump.

``CF_NAN_DEBUG_DUMP`` sets the dump path. ``CF_NAN_DEBUG_INJECT`` is a
self-test: ``loss:N`` makes the loss of step N NaN, and ``grad:N`` gives
step N a finite loss with a NaN gradient.
"""

from __future__ import annotations

import math
import os

import torch

EXIT_CODE = 3


def _tensors(value):
    """The tensors of a module output: a tensor, or a tuple or list of them."""
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in _tensors(item)]
    return []


class Probe:
    """Named max |values| of tensors on the compute path, the max |gradient|
    of the same tensors, and scalar notes. Each call is a no-op while the
    probe is inactive."""

    def __init__(self):
        self.active = False
        self.reset()

    def reset(self):
        self.forward, self.backward, self.notes = [], [], []
        # Activation checkpointing runs parts of the forward again inside
        # the backward pass. Their records carry a mark.
        self.in_backward = False

    def record(self, tag, tensor):
        """The max |tensor| now, and the max |gradient| of it in backward."""
        if not (self.active and torch.is_tensor(tensor)
                and tensor.is_floating_point() and tensor.numel() > 0):
            return
        if self.in_backward:
            tag = f"{tag} (again, in backward)"
        self.forward.append((tag, tensor.detach().abs().amax()))
        if tensor.requires_grad and torch.is_grad_enabled():
            tensor.register_hook(self._on_grad(tag))

    def _on_grad(self, tag):
        def hook(grad):
            if self.active:
                self.backward.append((tag, grad.detach().abs().amax()))
        return hook

    def note(self, tag, value):
        """A value to keep as it is, for example a loss term."""
        if self.active:
            self.notes.append((tag, value))


PROBE = Probe()


def _module_hook(tag):
    def hook(module, inputs, output):
        for i, t in enumerate(_tensors(output)):
            PROBE.record(tag if i == 0 else f"{tag}[{i}]", t)
    return hook


def _cpu(value):
    """``value`` with every tensor detached and moved to the CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(v) for v in value)
    return value


def _maxima(records):
    """``[(tag, max |value|)]`` as floats, in the order of the records."""
    if not records:
        return []
    values = torch.stack([v.float() for _, v in records]).cpu().tolist()
    return [(tag, v) for (tag, _), v in zip(records, values)]


def _first_nonfinite(maxima):
    """``(index, tag)`` of the first non-finite max, or None."""
    for i, (tag, v) in enumerate(maxima):
        if not math.isfinite(v):
            return i, tag
    return None


def _nonfinite(named):
    """The names whose tensor holds a non-finite value, with one sync."""
    named = [(n, t) for n, t in named if t is not None]
    if not named:
        return []
    ok = torch.stack([torch.isfinite(t).all() for _, t in named]).cpu()
    return [n for (n, _), good in zip(named, ok.tolist()) if not good]


class NanDebug:
    """The trainer's side of the mode: hooks, the batch notes, the three
    checks and the dump."""

    def __init__(self, model, optimizer, dump_path, inject=None):
        self.model, self.optimizer, self.dump_path = model, optimizer, dump_path
        self.inject = _parse_inject(inject)
        self.handles = [m.register_forward_hook(_module_hook(n or "model"))
                        for n, m in model.named_modules()]
        self.step, self.batch, self.before = None, {}, None

    @classmethod
    def from_env(cls, model, optimizer, default_dump):
        """The mode when ``CF_NAN_DEBUG=1``, else None."""
        if os.environ.get("CF_NAN_DEBUG") != "1":
            return None
        return cls(model, optimizer,
                   os.environ.get("CF_NAN_DEBUG_DUMP") or default_dump,
                   os.environ.get("CF_NAN_DEBUG_INJECT"))

    def start_step(self, step):
        self.step, self.batch, self.before = step, {}, None
        PROBE.reset()
        PROBE.active = True

    def note_batch(self, **items):
        self.batch.update(items)

    def backward_starts(self):
        PROBE.in_backward = True

    def injected(self, loss):
        """``loss``, or at the step of ``CF_NAN_DEBUG_INJECT`` its NaN copy
        (``loss``) or itself plus a term whose gradient is NaN (``grad``):
        0 * sqrt(0 * sum |w|) is 0, and its gradient is 0 * inf."""
        if self.inject is None or self.inject[1] != self.step:
            return loss
        if self.inject[0] == "loss":
            return loss * float("nan")
        w = self._inject_weight()
        return loss + 0.0 * torch.sqrt(0.0 * w.abs().sum())

    def _inject_weight(self):
        named = dict(self.model.named_parameters())
        heads = [n for n in named if n.startswith("value_head")]
        return named[heads[0] if heads else next(iter(named))]

    def loss_is_bad(self, loss):
        """True, after the dump, when the loss is not finite."""
        if math.isfinite(loss):
            return False
        self.dump("loss", loss)
        return True

    def grads_are_bad(self, loss):
        """True, after the dump, when a gradient is not finite. Else it
        keeps a copy of the weights for :meth:`weights_are_bad`."""
        grads = [(n, p.grad) for n, p in self.model.named_parameters()]
        if not _nonfinite(grads):
            self.before = {n: p.detach().clone()
                           for n, p in self.model.named_parameters()}
            PROBE.active = False
            return False
        self.dump("gradient", loss)
        return True

    def weights_are_bad(self, loss, grad_norm=None):
        """True, after the dump, when the step left a weight non-finite."""
        if not _nonfinite(list(self.model.named_parameters())):
            return False
        self.dump("weight after the optimizer step", loss, grad_norm)
        return True

    def dump(self, reason, loss, grad_norm=None):
        PROBE.active = False
        data = self._report(reason, loss, grad_norm)
        folder = os.path.dirname(os.path.abspath(self.dump_path))
        os.makedirs(folder, exist_ok=True)
        torch.save(data, self.dump_path)
        print(summary(data, self.dump_path), flush=True)

    def _report(self, reason, loss, grad_norm):
        params = list(self.model.named_parameters())
        grads = [(n, p.grad) for n, p in params if p.grad is not None]
        weights = self.before or {n: p.detach() for n, p in params}
        forward, backward = _maxima(PROBE.forward), _maxima(PROBE.backward)
        return {
            "step": self.step, "reason": reason, "loss": float(loss),
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": None if grad_norm is None else float(grad_norm),
            "batch": _cpu(self.batch), "notes": _cpu(PROBE.notes),
            "forward": forward, "first_nonfinite_forward":
                _first_nonfinite(forward),
            "backward": backward, "first_nonfinite_backward":
                _first_nonfinite(backward),
            "nonfinite_grads": _nonfinite(grads),
            "nonfinite_weights": _nonfinite(params),
            "grad_max_abs": _maxima([(n, g.abs().amax()) for n, g in grads]),
            "weights_before_step": _cpu(weights),
            "optimizer_state": _cpu(self.optimizer.state_dict()),
        }


def _parse_inject(spec):
    """``"grad:15"`` → ``("grad", 15)``. None when unset."""
    if not spec:
        return None
    kind, step = spec.split(":")
    if kind not in ("loss", "grad"):
        raise ValueError(f"CF_NAN_DEBUG_INJECT takes loss:N or grad:N, "
                         f"not {spec!r}")
    return kind, int(step)


def summary(data, path):
    """The lines the trainer prints when the mode stops a run."""
    lines = [f"*** CF_NAN_DEBUG: non-finite {data['reason']} at step "
             f"{data['step']} (loss {data['loss']:.6g}, lr {data['lr']:.3g}) ***",
             f"  dump: {path}"]
    for key in ("nonfinite_grads", "nonfinite_weights"):
        names = data[key]
        if names:
            lines.append(f"  {key}: {len(names)} (first: "
                         f"{', '.join(names[:5])})")
    for key in ("first_nonfinite_forward", "first_nonfinite_backward"):
        hit = data[key]
        n = len(data[key.replace("first_nonfinite_", "")])
        lines.append(f"  {key}: " + ("none" if hit is None else
                                     f"{hit[1]!r} (record {hit[0]} of {n})"))
    top = sorted(data["forward"], key=lambda r: -_order(r[1]))[:5]
    lines.append("  largest forward values: " + ", ".join(
        f"{t}={v:.3g}" for t, v in top))
    lines += _batch_lines(data["batch"])
    return "\n".join(lines)


def _order(value):
    """A sort key that puts NaN and inf first."""
    return value if math.isfinite(value) else float("inf")


def _batch_lines(batch):
    lines = []
    if "values" in batch:
        lines.append(f"  batch: {tuple(batch['values'].shape)} values, max "
                     f"|x| {batch['values'].abs().max().item():.4g}")
    if "max_z" in batch:
        z = batch["max_z"]
        lines.append(f"  max |z| of the target parts: max {z.max().item():.4g}"
                     f", windows kept {int(batch.get('kept', z >= 0).sum())}"
                     f" of {len(z)}")
    if "patch_size" in batch and batch["patch_size"] is not None:
        sizes, counts = batch["patch_size"].unique(return_counts=True)
        lines.append("  patch sizes: " + ", ".join(
            f"{s}: {c}" for s, c in zip(sizes.tolist(), counts.tolist())))
    return lines
