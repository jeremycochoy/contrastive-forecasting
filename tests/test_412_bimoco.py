"""Tests for #412: the split shape on the contrastive objective by row, for
the bimoco run.

The bimoco objective is L_pred with MoCo negatives plus L_rep with MoCo keys
(``--loss-shape cosine_similarity_batch_split_pred_rep --moco-negatives
--moco-rep-keys``). With the Moirai parts (#421f) the objective trains by
row, so each term must take the padding mask of #419. L_pred builds each
anchor-positive pair on the grid of its own row's patch size. On a step with
several sizes, each pair then fills P / G positions of the grid of the
finest size G (the design of the PR #423 review).

1. One size, no padding: ``pred_term`` equals the unmasked L_pred bit for
   bit.
2. With padding and a dropped row: the loss and the input gradients equal
   those of the batch with the positions removed.
3. Two sizes, a small batch: equal to a reference with explicit indices.
4. Coupling: a size-8 row takes a gradient through the L_pred of a size-128
   anchor.
5. All rows of one size: the multi-size path equals the single-group path.
6. Depth j pairs f^(j) with the teacher 1 + j patches ahead, and drops the
   tail of each group.
7. The run's full command line trains 3 steps, resumes, and drops a
   poisoned row.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.loss import (contrastive_latent_loss, grid_pairs,  # noqa: E402
                      grid_pred_term, pred_anchor_losses, pred_term,
                      rep_only_term)
from src.nan_skip import is_finite_step, restore_rng, rng_state  # noqa: E402
from tests.test_412_ours_moirai import (ALL_SIZES, CPU,  # noqa: E402
                                        MOIRAI_PARTS, MOIRAI_RECIPE, TINY,
                                        batch, corpus, first_pass,
                                        loss_and_grads, losses, model,
                                        poison, poisoned, rows_of, run)

TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")
SPLIT = "cosine_similarity_batch_split_pred_rep"
SIZES = (8, 16, 32, 64, 128)

# The bimoco run the owner asked for on 10-05: OMF's model, data and Moirai
# parts, with L_pred and its MoCo negatives, L_rep and its MoCo keys, tau 1
# on both terms, L_rep at weight 1 for the whole run, and no L_align.
BIMOCO = (
    "--qk-norm", "--attn-out-norm", "--d-model", "384", "--n-heads", "8",
    "--num-encoder-layers", "3", "--num-layers", "3",
    "--encoder-dropkey", "0.70", "--encoder-dropkey-share-heads",
    "--encoder-dropkey-share-layers", "--depthwise-conv", "3",
    "--deprecated-depthwise-conv", "0",
    "--loss-shape", SPLIT, "--moco-negatives", "--moco-rep-keys",
    "--tau", "1.0", "--tau-rep", "1.0", "--align-loss-weight", "0",
    "--ema-embedding", "--ema-encoder", "--ema-tau", "0.9",
    "--cpc-infonce-weight", "0.0", "--sigreg-embedding", "--sigreg-encoding",
    "--sigreg-n-chunk", "2048", "--sigreg-embedding-weight", "1.0",
    "--sigreg-encoding-weight", "1.0", "--encoder-type", "gru",
    "--synth-kind", "forked-arma", "--mix-ratio", "0.0078125",
    "--crossfade-triplets", "1", "--mixup-p", "0.3", "--freq-emb-dim", "3",
    "--seasonality-emb-dim", "3", "--train-rollout-depth", "3",
    "--train-rollout-reduce", "sum", "--rep-loss-weight", "1.0",
    "--gift-pretrain", "--freq-vocab", "v2", "--t-raw", "4096",
    "--n-channels", "1")
FULL_RUN = BIMOCO + MOIRAI_RECIPE + MOIRAI_PARTS + TINY


def spec(**extra):
    """The bimoco loss keys of the trainer's LOSS_SPEC."""
    cfg = {"loss_shape": SPLIT, "contrastive_divergence_temperature": 1.0,
           "contrastive_divergence_temperature_rep": 1.0,
           "moco_negatives": True, "moco_rep_keys": True,
           "train_rollout_reduce": "sum"}
    cfg.update(extra)
    return SimpleNamespace(train_configuration=cfg)


def latents(B=4, T=10, C=1, H=8, seed=0, depth=0, dtype=torch.float32):
    """``(f, h, teacher)`` and f^(1)..f^(depth), with gradients on f, h and
    the depths."""
    g = torch.Generator().manual_seed(seed)

    def make(grad=True):
        return torch.randn(B, T, C, H, generator=g, dtype=dtype,
                           requires_grad=grad)
    return (make(), make(), make(False)), [make() for _ in range(depth)]


def group(n, T, ratio, C=1, H=4, seed=0, depth=0, pad=None):
    """One patch-size group of the grid: ``n`` rows of ``T`` patches, in
    float64."""
    (f, h, teacher), rollout = latents(n, T, C, H, seed, depth,
                                       torch.float64)
    if pad is None:
        pad = torch.zeros(n, T, C, dtype=torch.bool)
    return SimpleNamespace(f=f, o=h, teacher=teacher, rollout=rollout,
                           pad=pad, ratio=ratio)


# ---------------------------------------------------------------------------
# 1. One size, no padding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("C,depth", [(1, 0), (2, 3)])
@pytest.mark.parametrize("moco", [True, False])
def test_one_size_without_padding_gives_the_unmasked_l_pred_bit_for_bit(
        C, depth, moco):
    (f, h, teacher), rollout = latents(C=C, depth=depth)
    s = spec(train_rollout_depth=depth, pred_loss_weight=0.7,
             moco_negatives=moco)
    want_terms, got_terms = {}, {}
    want = contrastive_latent_loss(
        (f, h), False, s, teacher_original_latent=teacher,
        rollout_latents=rollout, rep_loss_weight=0.0, term_out=want_terms)
    no_pad = torch.zeros(f.shape[:3], dtype=torch.bool)
    for pad in (None, no_pad):
        got = pred_term(f, h, s, pad, teacher_original_latent=teacher,
                        rollout_latents=rollout, term_out=got_terms)
        assert torch.equal(got, want)
        assert got_terms["l_pred"] == want_terms["l_pred"]
    leaves = [f, h, *rollout]  # with MoCo negatives L_pred reads no h
    want_grads = torch.autograd.grad(want, leaves, allow_unused=True)
    got_grads = torch.autograd.grad(got, leaves, allow_unused=True)
    for a, b in zip(got_grads, want_grads):
        assert (a is None and b is None) or torch.equal(a, b)


def test_the_split_l_rep_is_the_masked_rep_only_term():
    """L_rep of the split shape, with MoCo keys, is the L_rep of the rep_only
    shape: ``rep_only_term`` gives it on a padded batch."""
    (f, h, teacher), _ = latents(C=2)
    s = spec(pred_loss_weight=0.0, contrastive_divergence_temperature=0.1)
    want = contrastive_latent_loss((f, h), False, s,
                                   teacher_original_latent=teacher)
    no_pad = torch.zeros(f.shape[:3], dtype=torch.bool)
    got = rep_only_term(h, s, no_pad, teacher_original_latent=teacher)
    assert torch.allclose(got, want, rtol=1e-6, atol=0)


def test_pred_term_without_a_teacher_refuses_moco_negatives():
    (f, h, _), _ = latents()
    with pytest.raises(ValueError, match="EMA teacher"):
        pred_term(f, h, spec(), None)
    with pytest.raises(ValueError, match="EMA teacher"):
        rep_only_term(h, spec(), torch.zeros(h.shape[:3], dtype=torch.bool))


# ---------------------------------------------------------------------------
# 2. Padding and a dropped row
# ---------------------------------------------------------------------------

def masked_split(f, h, teacher, rollout, pad, s):
    """The bimoco objective of a padded batch: L_pred + L_rep."""
    return (pred_term(f, h, s, pad, teacher_original_latent=teacher,
                      rollout_latents=rollout)
            + rep_only_term(h, s, pad, teacher_original_latent=teacher))


@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_padding_and_a_dropped_row_give_the_cut_batch(reduce):
    """Every row starts with K padded patches, and row 3 is inert (all
    padding). The loss and its input gradients equal those of the unmasked
    split shape on the batch without row 3 and the first K patches."""
    B, T, C, K, depth = 5, 12, 2, 3, 2
    (f, h, teacher), rollout = latents(B, T, C, depth=depth)
    s = spec(train_rollout_depth=depth, train_rollout_reduce=reduce,
             contrastive_divergence_temperature=0.5)
    pad = torch.zeros(B, T, C, dtype=torch.bool)
    pad[:, :K] = True
    pad[3] = True
    kept = [0, 1, 2, 4]
    got = masked_split(f, h, teacher, rollout, pad, s)
    leaves = [f, h, *rollout]
    got_grads = torch.autograd.grad(got, leaves)
    cut = [t[kept, K:].detach().requires_grad_(True) for t in leaves]
    want = contrastive_latent_loss(
        (cut[0], cut[1]), False, s, teacher_original_latent=teacher[kept, K:],
        rollout_latents=cut[2:])
    want_grads = torch.autograd.grad(want, cut)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)
    removed = torch.ones(B, T, dtype=torch.bool)
    removed[kept, K:] = False
    for got_g, want_g in zip(got_grads, want_grads):
        assert torch.isfinite(got_g).all()
        assert torch.allclose(got_g[kept, K:], want_g, rtol=1e-4, atol=1e-6)
        assert (got_g[removed] == 0).all()


# ---------------------------------------------------------------------------
# 3. Two sizes against a reference with explicit indices
# ---------------------------------------------------------------------------

def unit(v):
    return v / v.norm()


def lse(values):
    return torch.logsumexp(torch.stack(values), dim=0)


def pair_at(g, i, depth, p, c):
    """``(anchor, positive, zy keys)`` of row ``i`` of group ``g`` at grid
    index ``p``, or None when no real pair sits there."""
    t, T = p // g.ratio, g.f.shape[1]
    if t > T - 2 - depth or g.pad[i, t, c] or g.pad[i, t + 1 + depth, c]:
        return None
    f = g.f if depth == 0 else g.rollout[depth - 1]
    zy = [unit(f[i, t + 1, k]) for k in range(f.shape[2])
          if not g.pad[i, t + 1, k]]
    return unit(f[i, t, c]), unit(g.teacher[i, t + 1 + depth, c]), zy


def key_at(g, i, depth, p, c, moco):
    """The cross-batch key of row ``i`` at grid index ``p``: its own
    positive, the key 1 + depth patches ahead of its patch at p."""
    t, T = p // g.ratio, g.f.shape[1]
    if t > T - 2 - depth or g.pad[i, t + 1 + depth, c]:
        return None
    source = g.teacher if moco else g.o
    return unit(source[i, t + 1 + depth, c])


def reference_copy(groups, depth, tau, moco):
    """The masked mean of one depth copy of L_pred, one index at a time."""
    rows = [(g, i) for g in groups for i in range(g.f.shape[0])]
    length = max((g.f.shape[1] - 1 - depth) * g.ratio for g in groups)
    total, count = 0.0, 0
    for p in range(length):
        for c in range(groups[0].f.shape[2]):
            pairs = [pair_at(g, i, depth, p, c) for g, i in rows]
            keys = [key_at(g, i, depth, p, c, moco) for g, i in rows]
            negs = {}
            for r, pair in enumerate(pairs):
                if pair is None:
                    continue
                anchor, _, zy = pair
                sims = [anchor @ z / tau for z in zy]
                sims += [anchor @ k / tau for r2, k in enumerate(keys)
                         if r2 != r and k is not None]
                negs[r] = lse(sims)
            for r, value in negs.items():
                log_pos = pairs[r][0] @ pairs[r][1] / tau
                total = total + lse([log_pos, lse(list(negs.values()))]) \
                    - log_pos
                count += 1
    return total / count


@pytest.mark.parametrize("moco", [True, False])
def test_two_sizes_equal_the_reference_with_explicit_indices(moco):
    """Sizes 8 and 16 (ratio 2), two channels, depth 1. Row 1 has two padded
    patches, row 2 is inert, and row 3 (size 16) has one padded patch."""
    pad8 = torch.zeros(3, 8, 2, dtype=torch.bool)
    pad8[1, :2], pad8[2] = True, True
    pad16 = torch.zeros(2, 4, 2, dtype=torch.bool)
    pad16[0, :1] = True
    groups = [group(3, 8, 1, C=2, seed=1, depth=1, pad=pad8),
              group(2, 4, 2, C=2, seed=2, depth=1, pad=pad16)]
    s = spec(contrastive_divergence_temperature=0.5, pred_loss_weight=0.7)
    terms = {}
    got = grid_pred_term(groups, s, moco_negatives=moco, term_out=terms)
    copies = [reference_copy(groups, j, 0.5, moco) for j in (0, 1)]
    assert torch.allclose(got, 0.7 * (copies[0] + copies[1]), rtol=1e-12)
    assert math.isclose(terms["l_pred"], copies[0].item(), rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 4. Coupling across the sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("moco", [True, False])
def test_a_size_8_row_takes_a_gradient_through_a_size_128_anchor(moco):
    """The L_pred of the size-128 anchor alone moves the size-8 row: its
    denominator pools the negatives of every anchor at its grid index, and
    without MoCo keys its cross-batch keys read the size-8 row's h."""
    fine, coarse = group(1, 32, 1, seed=3), group(1, 2, 16, seed=4)
    values, keep = pred_anchor_losses(
        grid_pairs([fine, coarse], 0, moco, False), 1.0)
    assert keep[1].sum() == 16 and keep[0].sum() == 31
    values[1][keep[1]].sum().backward()

    def moved(t):
        return t.grad is not None and bool(t.grad.abs().sum() > 0)
    assert moved(fine.f) and moved(coarse.f)
    assert moved(fine.o) != moco


@pytest.fixture(scope="module")
def train_py():
    s = importlib.util.spec_from_file_location("train_412_bimoco", TRAIN_PY)
    module = importlib.util.module_from_spec(s)
    s.loader.exec_module(module)
    return module


def bimoco_args(train_py, *extra):
    """The bimoco run's parsed flags at the tiny size, with the loss keys
    set as the trainer sets them."""
    args = train_py.parse_args([*FULL_RUN, *extra])
    train_py.configure_loss_spec(args)
    return args


def row_gradients(train_py, args, sizes, row, scale):
    """The input gradient of each row, with ``row`` scaled by ``scale``,
    and L_pred as the only term."""
    m = model()
    inputs = batch(m, sizes)
    x = inputs["x_norm"].clone()
    x[row] = scale * x[row]
    x.requires_grad_(True)
    loss = train_py.contrastive_objective(m, dict(inputs, x_norm=x), args,
                                          SIZES, rep_w=0.0)[0]
    loss.backward()
    return x.grad


def test_l_pred_alone_couples_the_rows_of_every_size(train_py):
    """With L_rep at weight 0 and no SIGReg, new values in the size-128 row
    move the input gradient of every row of size 8."""
    args = bimoco_args(train_py)
    args.sigreg_embedding = args.sigreg_encoding = False
    sizes = (8, 8, 8, 128, 8, 8, 8, 8)
    one, two = (row_gradients(train_py, args, sizes, 3, s)
                for s in (1.0, 0.5))
    assert torch.isfinite(one).all()
    for row in (0, 1, 2, 4, 5, 6, 7):
        assert not torch.allclose(one[row], two[row]), row


# ---------------------------------------------------------------------------
# 5. One size on the multi-size path
# ---------------------------------------------------------------------------

def step_parts(train_py, m, inputs, args):
    """The groups of a step as ``contrastive_objective`` builds them."""
    x_norm, labels = train_py.inert_inputs(inputs, None)
    groups = train_py.size_groups(inputs["sample_sizes"], m.W)
    lats = {size: train_py.group_forward(
                m, train_py.take_rows(x_norm, rows),
                {k: train_py.take_rows(v, rows) for k, v in labels.items()},
                size, args) for size, rows in groups.items()}
    pads = {size: train_py.group_padding(inputs["pad_mask"], rows, size,
                                         None, x_norm.shape)
            for size, rows in groups.items()}
    return lats, pads, {size: len(lats[size].o) for size in lats}


def terms_and_grads(train_py, terms, m, inputs, args):
    m.zero_grad(set_to_none=True)
    lats, pads, counts = step_parts(train_py, m, inputs, args)
    torch.manual_seed(7)  # SIGReg draws its projections here
    loss, values = terms(m, lats, pads, counts, args, 1.0, None)
    loss.backward()
    return loss, values, {n: p.grad.clone() for n, p in m.named_parameters()
                          if p.grad is not None}


def test_one_size_on_the_multi_size_path_equals_the_single_group_path(
        train_py):
    args = bimoco_args(train_py)
    m = model()
    inputs = batch(m, (32,) * 8)
    a = terms_and_grads(train_py, train_py.grouped_contrastive_terms, m,
                        inputs, args)
    b = terms_and_grads(train_py, train_py.multi_size_contrastive_terms, m,
                        inputs, args)
    assert torch.allclose(a[0], b[0], rtol=1e-6, atol=0)
    assert {"l_pred", "l_rep"} <= set(a[1]["terms"]) == set(b[1]["terms"])
    for name, value in a[1]["terms"].items():
        assert math.isclose(value, b[1]["terms"][name], rel_tol=1e-6), name
    assert set(a[2]) == set(b[2])
    for name, grad in a[2].items():
        assert torch.allclose(grad, b[2][name], rtol=1e-4, atol=1e-7), name


# ---------------------------------------------------------------------------
# 6. Depth j and the tail of each group
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [0, 1, 2])
def test_depth_j_pairs_the_teacher_1_plus_j_patches_ahead(depth):
    """Size 8 (16 patches) and size 32 (4 patches, ratio 4) at depth j: the
    pair of a row at grid index p is f^(j) of its patch t = p // ratio and
    the teacher at t + 1 + j. After the last pair of a group, every index
    is dropped as an anchor and as a key."""
    fine = group(1, 16, 1, seed=5, depth=2)
    coarse = group(1, 4, 4, seed=6, depth=2)
    pairs = grid_pairs([fine, coarse], depth, True, False)
    length = 16 - 1 - depth
    assert pairs["anchor"].shape[1] == length
    for row, g in enumerate((fine, coarse)):
        f = g.f if depth == 0 else g.rollout[depth - 1]
        n_pairs = (g.f.shape[1] - 1 - depth) * g.ratio
        for p in range(length):
            t = p // g.ratio
            dropped = bool(pairs["drop"][row, p, 0])
            assert dropped == (p >= n_pairs), (row, p)
            assert bool(pairs["key_drop"][row, p, 0]) == dropped
            if dropped:
                continue
            assert torch.allclose(pairs["anchor"][row, p, 0],
                                  unit(f[0, t, 0]))
            assert torch.allclose(pairs["pos"][row, p, 0],
                                  unit(g.teacher[0, t + 1 + depth, 0]))
            assert torch.allclose(pairs["zy"][row, p, 0], unit(f[0, t + 1, 0]))


# ---------------------------------------------------------------------------
# The row drop with L_pred
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("culprits", [[2], [0, 5]])
def test_the_step_after_the_drop_equals_the_batch_without_the_rows(
        train_py, culprits):
    """Row 2 shares size 32 with row 7. Rows 0 and 5 are the two rows of
    size 8, so after their drop the grid of the step is that of size 16.
    The search finds the poisoned rows, and the step after the drop equals
    the step on the batch without them: the loss and the gradients."""
    args = bimoco_args(train_py)
    m = model()
    inputs = batch(m, ALL_SIZES)
    for row in culprits:
        inputs = poison(inputs, row)
    objective = train_py.step_objective(args, 1.0)
    rng = rng_state(CPU)
    inputs, loss, grad, faults = first_pass(train_py, m, inputs, args)
    assert not is_finite_step(loss.item(), m)
    found, _, result = train_py.skip_nan_rows(
        m, inputs, args, SIZES, rng, CPU, grad, objective, faults)
    assert sorted(found) == culprits and result is not None
    got = {n: p.grad.clone() for n, p in m.named_parameters()
           if p.grad is not None}
    restore_rng(rng, CPU)
    rest = [r for r in range(len(ALL_SIZES)) if r not in culprits]
    want, want_grads = loss_and_grads(objective, m, rows_of(
        dict(inputs, x_norm=inputs["x_norm"].detach()), rest), args)
    assert torch.allclose(result[0], want, rtol=1e-5, atol=1e-6)
    assert set(got) == set(want_grads)
    for name, grad in want_grads.items():
        assert torch.allclose(got[name], grad, rtol=1e-4, atol=1e-6), name


# ---------------------------------------------------------------------------
# The refusals and the trainer
# ---------------------------------------------------------------------------

def test_the_split_shape_takes_a_row_mask(train_py):
    args = bimoco_args(train_py)
    assert train_py.gift_contrastive_gap(args) is None
    assert train_py.row_mask_refusal("--multi-patch-sizes", args) is None
    args.loss_shape = "cosine_similarity_batch_full_hh_negs_xshh_allt"
    assert "takes no row mask" in train_py.row_mask_refusal("--x", args)


def finite(rows, *columns):
    return all(np.isfinite(float(row[c])) for row in rows for c in columns)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """Three steps of the bimoco command line at the tiny size, then a
    resume to step 4."""
    root = tmp_path_factory.mktemp("bimoco")
    data = corpus(root)
    first = run(REPO_ROOT, root / "save", *FULL_RUN, "--total-steps", "3",
                *data)
    rows = losses(root / "save")
    resumed = run(REPO_ROOT, root / "save", *FULL_RUN, "--total-steps", "4",
                  "--resume", str(root / "save" / "r_final.pth"), *data)
    return root / "save", first, rows, resumed


def test_the_bimoco_command_line_trains_on_the_cpu(full_run):
    save, first, rows, _ = full_run
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    assert "Objective (#412)" in first.stdout and "L_pred" in first.stdout
    assert "NaN/Inf DETECTED" not in first.stdout
    assert [r["step"] for r in rows] == ["1", "2", "3"]
    assert finite(rows, "loss", "loss_tau_ref", "l_pred", "l_rep",
                  "sigreg_e", "sigreg_h", "grad_norm", "cos_err_d3")
    assert all(r["l_align"] == "" for r in rows)
    assert all(float(r["rep_w"]) == 1.0 for r in rows)


def test_the_bimoco_run_resumes(full_run):
    """The resume branches to the run name ``r_r2`` and trains step 4."""
    save, first, _, resumed = full_run
    assert first.returncode == 0, first.stderr[-3000:]
    out = resumed.stdout
    assert resumed.returncode == 0, out[-3000:] + resumed.stderr[-3000:]
    assert "Resumed from" in out and "[      4]" in out
    rows = list(csv.DictReader(open(save / "r_r2_losses.csv")))
    assert [r["step"] for r in rows] == ["4"]
    assert finite(rows, "loss", "l_pred", "l_rep")


def test_the_bimoco_run_drops_a_poisoned_row(tmp_path):
    """The z-filter is off, so the poisoned windows reach the model, and
    through the coupled terms every gradient of the step is NaN. The run
    drops the poisoned rows and goes on."""
    r = run(REPO_ROOT, tmp_path / "save", *FULL_RUN, "--meanstd-z-max", "0",
            "--skip-spike-samples", "0", "--mixup-p", "0",
            "--total-steps", "4", *poisoned(tmp_path))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "NaN/Inf DETECTED" not in r.stdout
    rows = losses(tmp_path / "save")
    assert len(rows) == 4 and finite(rows, "loss", "grad_norm", "l_pred",
                                     "l_rep")
    assert max(int(row["nan_dropped"]) for row in rows) >= 1
