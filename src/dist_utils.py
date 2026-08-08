"""Distributed-data-parallel helpers for contrastive training.

The contrastive loss (`src.loss.contrastive_latent_loss`) is **batch-coupled**:
every anchor's denominator pools negatives over the *whole* batch (the
`[B,B,T-1,C]` cross-batch term + `logsumexp(..., dim=0)`). Under vanilla DDP
each rank would only see its local shard, so the negative pool would silently
shrink to ``B/world_size`` — a *different, weaker* objective, not just a
slower-but-equivalent one.

To keep multi-GPU training mathematically identical to single-GPU at the same
*global* batch, every rank gathers all ranks' latents with a **differentiable**
all-gather and computes the loss over the full global set. Correctness
(loss value AND gradient) is pinned in tests/test_dist_gather.py against a
single-process full-batch reference.

Gradient bookkeeping (why it is exact, not W× off):
  Every rank computes the *same* global loss L over the *same* gathered
  tensor, so ∂L/∂(gathered) is identical on all ranks. `all_reduce(SUM)` in
  the backward therefore yields ``W·∂L/∂Z`` and each rank keeps its slice
  → ``W·∂L/∂Z_r`` flows into rank r's model params. DDP then *averages*
  param grads across the W ranks (its default), and the ``W`` and ``1/W``
  cancel exactly → the parameter gradient equals the single-process
  full-batch gradient. (At the bare-latent level, with no DDP averaging,
  the reconstructed grad is W× the reference — the tests assert the /W
  relationship explicitly and also assert exact equality once a module is
  DDP-wrapped.)

Single-process / single-GPU is byte-identical to before: when
`torch.distributed` is unavailable, uninitialised, or ``world_size == 1``,
`gather_latents` is a strict no-op (returns its inputs unchanged) and no
process group is ever created.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

__all__ = [
    "is_distributed",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "barrier",
    "setup_distributed",
    "cleanup_distributed",
    "DifferentiableAllGather",
    "gather_latent",
    "gather_latents",
    "broadcast_module",
    "average_gradients",
]


def is_distributed() -> bool:
    """True only when a process group is initialised with world_size > 1."""
    return (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    """Rank 0 (or any non-distributed run). Gate all side effects on this:
    checkpoint saves, CSV/log writes, stdout, the sync_loop."""
    return get_rank() == 0


def barrier() -> None:
    """Safe no-op unless actually distributed."""
    if is_distributed():
        dist.barrier()


def setup_distributed():
    """Initialise the process group from the torchrun-provided env vars.

    Returns ``(rank, world_size, local_rank, distributed)``. When the env
    does not describe a >1 world (no torchrun, or WORLD_SIZE=1), returns
    ``(0, 1, 0, False)`` and does NOT create a process group — the caller's
    single-GPU path stays byte-identical.

    Backend: ``nccl`` if CUDA is available (the GPU path), else ``gloo``
    (CPU correctness tests). The caller is responsible for setting the CUDA
    device to ``local_rank`` before heavy allocation.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0, False

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


class DifferentiableAllGather(torch.autograd.Function):
    """All-gather along dim 0 with gradient flow back to every source rank.

    ``torch.distributed.all_gather`` is not differentiable. This Function
    gathers in the forward and, in the backward, ``all_reduce(SUM)``-s the
    incoming grad and returns this rank's slice (see the module docstring
    for why the resulting W× factor is exactly cancelled by DDP's gradient
    averaging at the parameter level).

    Requires every rank to pass a tensor with the *same* leading (batch)
    size — true for fixed-batch contrastive training.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        ctx.local_bs = x.shape[0]
        x = x.contiguous()
        gathered = [torch.empty_like(x) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, x)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        start = ctx.rank * ctx.local_bs
        return grad_output[start:start + ctx.local_bs]


def broadcast_module(module: torch.nn.Module, src: int = 0) -> None:
    """Broadcast all params + buffers from `src` so every rank starts from
    an identical model (covers resume-on-any-rank and any init
    nondeterminism). No-op when not distributed. This replaces what
    DDP's constructor does for us — we keep the model UNwrapped because
    the trainer calls submodules directly (`model.transformer(...)`,
    `model.tau()`), which DDP's forward-call contract forbids; gradient
    sync is done explicitly by `average_gradients` after backward.
    """
    if not is_distributed():
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


def average_gradients(module: torch.nn.Module) -> None:
    """all_reduce(SUM) every param grad then divide by world_size — exactly
    what DDP does after backward. Combined with the W× from
    `DifferentiableAllGather.backward`, the W and 1/W cancel so the
    parameter gradient equals the single-process full-batch gradient
    (pinned in tests/test_dist_gather.py). No-op when not distributed.
    """
    if not is_distributed():
        return
    world = dist.get_world_size()
    for p in module.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world


def gather_latent(latent: torch.Tensor) -> torch.Tensor:
    """Concatenate ONE latent tensor across ranks along the batch dim.

    One collective per call. Use this for a lone tensor — each rollout depth
    of #373, for instance. `gather_latents` is the pair form; calling it with
    the same tensor twice would issue two all-gathers for one result.

    No-op (returns the input unchanged) when not distributed.
    """
    if not is_distributed():
        return latent
    return DifferentiableAllGather.apply(latent)


def gather_latents(
    forecasted_latent: torch.Tensor, original_latent: torch.Tensor
):
    """Concatenate both latent tensors across ranks along the batch dim.

    No-op (returns the inputs unchanged) when not distributed, so the
    single-GPU objective is byte-identical. When distributed, every rank
    receives the full ``[world_size * B, T, C, H]`` global set so
    `contrastive_latent_loss` pools negatives over the global batch — i.e.
    2-GPU @ B/2 each == single-GPU @ B.
    """
    return gather_latent(forecasted_latent), gather_latent(original_latent)
