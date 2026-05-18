"""Correctness proof for the DDP differentiable latent all-gather.

Spawns 2 **gloo / CPU** processes (no GPU needed — deterministic and
CI-able) and asserts, against a single-process full-batch reference:

  1. The per-rank loss over gathered latents == single-process loss at the
     same global batch (the batch-coupled contrastive objective is
     preserved, NOT silently shrunk to the per-rank shard).

  2. The bare-latent gradient reconstructed across ranks == W × the
     single-process reference (the documented W× that DDP's gradient
     averaging cancels).

  3. With a real module wrapped in DistributedDataParallel, the parameter
     gradient == the single-process full-batch parameter gradient
     EXACTLY (the W and 1/W cancel) — the end-to-end guarantee that
     2-GPU @ B/2 each is identical to 1-GPU @ B.

If any assertion fails in a worker, torch.multiprocessing.spawn re-raises
it in the test process.
"""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from types import SimpleNamespace

from src.loss import contrastive_latent_loss
from src.dist_utils import gather_latents, average_gradients

WORLD = 2
B_LOCAL, T, C, H = 3, 4, 1, 8
B_GLOBAL = WORLD * B_LOCAL
LOSS = "cosine_similarity_batch_full_fh_negs"


def _spec():
    return SimpleNamespace(train_configuration={
        "contrastive_divergence_temperature": 0.1,
        "contrastive_latent_noise": None,
        "loss_shape": LOSS,
        "contrastive_latent_delay": 0,
    })


def _global_latents():
    """Deterministic global latents — identical in every process."""
    g = torch.Generator().manual_seed(20260517)
    f = torch.randn(B_GLOBAL, T, C, H, generator=g, dtype=torch.float64)
    o = torch.randn(B_GLOBAL, T, C, H, generator=g, dtype=torch.float64)
    return f.float(), o.float()


def _reference():
    """Single-process full-batch loss + grads w.r.t. the latents."""
    f, o = _global_latents()
    f = f.clone().requires_grad_(True)
    o = o.clone().requires_grad_(True)
    loss = contrastive_latent_loss((f, o), validation=False, spec=_spec())
    loss.backward()
    return loss.detach(), f.grad.clone(), o.grad.clone()


def _init(rank, fstore):
    dist.init_process_group(
        backend="gloo", init_method=f"file://{fstore}",
        world_size=WORLD, rank=rank,
    )


def _worker_loss_and_latent_grad(rank, fstore):
    _init(rank, fstore)
    try:
        ref_loss, ref_gf, ref_go = _reference()

        f_all, o_all = _global_latents()
        sl = slice(rank * B_LOCAL, (rank + 1) * B_LOCAL)
        f_loc = f_all[sl].clone().requires_grad_(True)
        o_loc = o_all[sl].clone().requires_grad_(True)

        f_g, o_g = gather_latents(f_loc, o_loc)
        assert f_g.shape[0] == B_GLOBAL, f_g.shape
        # Gathered values must match the true global tensor exactly.
        assert torch.allclose(f_g.detach(), f_all), "gather mangled values"

        loss = contrastive_latent_loss((f_g, o_g), validation=False,
                                       spec=_spec())
        assert torch.allclose(loss.detach(), ref_loss, atol=1e-6, rtol=1e-5), (
            f"rank{rank}: loss {loss.item()} != single-proc {ref_loss.item()}")

        loss.backward()
        # Bare-latent grad is W× the reference (no DDP averaging here).
        exp_f = WORLD * ref_gf[sl]
        exp_o = WORLD * ref_go[sl]
        assert torch.allclose(f_loc.grad, exp_f, atol=1e-5, rtol=1e-4), (
            f"rank{rank}: f-grad != W*ref "
            f"(max|Δ|={ (f_loc.grad - exp_f).abs().max().item():.3e})")
        assert torch.allclose(o_loc.grad, exp_o, atol=1e-5, rtol=1e-4), (
            f"rank{rank}: o-grad != W*ref "
            f"(max|Δ|={ (o_loc.grad - exp_o).abs().max().item():.3e})")
    finally:
        dist.destroy_process_group()


class _ToLatents(torch.nn.Module):
    """X[B, Hin] -> latents[B, T, C, H] via one Linear (deterministic init)."""

    def __init__(self, h_in):
        super().__init__()
        torch.manual_seed(777)               # identical θ in every process
        self.lin = torch.nn.Linear(h_in, T * C * H)

    def forward(self, x):
        return self.lin(x).view(-1, T, C, H)


def _ddp_reference(h_in):
    torch.manual_seed(202605)
    x = torch.randn(B_GLOBAL, h_in)
    m = _ToLatents(h_in)
    f = m(x)
    g = torch.Generator().manual_seed(11)
    o = torch.randn(B_GLOBAL, T, C, H, generator=g)
    loss = contrastive_latent_loss((f, o), validation=False, spec=_spec())
    loss.backward()
    return loss.detach(), m.lin.weight.grad.clone(), m.lin.bias.grad.clone()


def _worker_ddp_param_grad(rank, fstore):
    _init(rank, fstore)
    try:
        h_in = 5
        ref_loss, ref_wg, ref_bg = _ddp_reference(h_in)

        torch.manual_seed(202605)
        x_all = torch.randn(B_GLOBAL, h_in)
        g = torch.Generator().manual_seed(11)
        o_all = torch.randn(B_GLOBAL, T, C, H, generator=g)
        sl = slice(rank * B_LOCAL, (rank + 1) * B_LOCAL)

        model = _ToLatents(h_in)
        ddp = torch.nn.parallel.DistributedDataParallel(model)

        f_loc = ddp(x_all[sl])
        o_loc = o_all[sl]
        f_g, o_g = gather_latents(f_loc, o_loc)
        loss = contrastive_latent_loss((f_g, o_g), validation=False,
                                       spec=_spec())
        assert torch.allclose(loss.detach(), ref_loss, atol=1e-6, rtol=1e-5), (
            f"rank{rank}: ddp loss != single-proc")
        loss.backward()      # DDP averages param grads across ranks here

        wg = ddp.module.lin.weight.grad
        bg = ddp.module.lin.bias.grad
        assert torch.allclose(wg, ref_wg, atol=1e-5, rtol=1e-4), (
            f"rank{rank}: DDP weight.grad != single-proc full-batch "
            f"(max|Δ|={(wg - ref_wg).abs().max().item():.3e})")
        assert torch.allclose(bg, ref_bg, atol=1e-5, rtol=1e-4), (
            f"rank{rank}: DDP bias.grad != single-proc full-batch "
            f"(max|Δ|={(bg - ref_bg).abs().max().item():.3e})")
    finally:
        dist.destroy_process_group()


def _sharded_reference(h_in):
    """`--shard-loss-on-batch`: no gather. Objective is the MEAN of the
    per-rank local-shard losses; its grad is what manual DDP averaging
    must reproduce (NOT the global-batch grad — that's the tradeoff)."""
    torch.manual_seed(202605)
    x = torch.randn(B_GLOBAL, h_in)
    m = _ToLatents(h_in)
    g = torch.Generator().manual_seed(11)
    o = torch.randn(B_GLOBAL, T, C, H, generator=g)
    per_shard = []
    for r in range(WORLD):
        sl = slice(r * B_LOCAL, (r + 1) * B_LOCAL)
        per_shard.append(contrastive_latent_loss(
            (m(x[sl]), o[sl]), validation=False, spec=_spec()))
    loss = sum(per_shard) / WORLD
    loss.backward()
    return ([p.detach() for p in per_shard],
            m.lin.weight.grad.clone(), m.lin.bias.grad.clone())


def _worker_sharded_param_grad(rank, fstore):
    _init(rank, fstore)
    try:
        h_in = 5
        ref_shard_losses, ref_wg, ref_bg = _sharded_reference(h_in)

        torch.manual_seed(202605)
        x_all = torch.randn(B_GLOBAL, h_in)
        g = torch.Generator().manual_seed(11)
        o_all = torch.randn(B_GLOBAL, T, C, H, generator=g)
        sl = slice(rank * B_LOCAL, (rank + 1) * B_LOCAL)

        model = _ToLatents(h_in)            # trainer path: UNwrapped model
        f_loc = model(x_all[sl])
        o_loc = o_all[sl]
        # Sharded mode == skip gather_latents: loss over the LOCAL shard.
        loss = contrastive_latent_loss((f_loc, o_loc), validation=False,
                                       spec=_spec())
        assert torch.allclose(loss.detach(), ref_shard_losses[rank],
                              atol=1e-6, rtol=1e-5), (
            f"rank{rank}: sharded loss != single-proc loss on this shard")
        loss.backward()
        average_gradients(model)            # all_reduce(SUM)/W

        wg = model.lin.weight.grad
        bg = model.lin.bias.grad
        assert torch.allclose(wg, ref_wg, atol=1e-5, rtol=1e-4), (
            f"rank{rank}: sharded weight.grad != mean-of-shard-losses grad "
            f"(max|Δ|={(wg - ref_wg).abs().max().item():.3e})")
        assert torch.allclose(bg, ref_bg, atol=1e-5, rtol=1e-4), (
            f"rank{rank}: sharded bias.grad != mean-of-shard-losses grad "
            f"(max|Δ|={(bg - ref_bg).abs().max().item():.3e})")
    finally:
        dist.destroy_process_group()


def _spawn(fn):
    with tempfile.TemporaryDirectory() as d:
        fstore = os.path.join(d, "store")
        mp.spawn(fn, args=(fstore,), nprocs=WORLD, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed N/A")
def test_loss_and_latent_grad_match_single_process():
    """Loss value == single-proc; bare-latent grad == W × reference."""
    _spawn(_worker_loss_and_latent_grad)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed N/A")
def test_ddp_param_grad_exactly_matches_single_process():
    """The end-to-end guarantee: DDP-wrapped param grad == single-GPU
    full-batch param grad (W and 1/W cancel)."""
    _spawn(_worker_ddp_param_grad)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed N/A")
def test_sharded_mode_grad_is_mean_of_per_shard_losses():
    """--shard-loss-on-batch (no gather): each rank's loss == single-proc
    loss on its own shard, and the averaged param grad == grad of the
    MEAN of the per-shard losses. This is a different (weaker) objective
    than the gathered one — by design — so this pins the documented
    tradeoff rather than single-GPU equivalence."""
    _spawn(_worker_sharded_param_grad)
