"""Tests for `--train-rollout-reduce hill` (key `train_rollout_reduce`).

`hill` gives the copy of depth j of every f-bearing term the weight
w_j = 1 / (1 + (j / s)^p). Copy 0 is the forecast 1 patch ahead. The scale s
(`--train-rollout-hill-scale`, key `train_rollout_hill_scale`) is the depth
whose copy gets 1/2. The power p (`--train-rollout-hill-power`, key
`train_rollout_hill_power`) sets how fast the weight falls past s. Depth 0
keeps the weight 1, so the terms that carry no `f` and the depth-0 f-side
enter as under `sum`.

What this file pins:

* **The weights.** At s = p = 4: w_0 = 1, w_4 = 1/2, w_8 = 1/17, and the 32
  copies of a run at k = 31.
* **The manual weighted sum of the copies.** At depth k the total is the
  depth-0 total plus each copy j at w_j: on every shape of the #373 sweep
  (the L_align add-on and the split L_pred included), on `grid_pred_term`
  with two patch sizes, on the joint teacher fraction and on the standalone
  terms (`align_loss` in both of its forms, the two CPC InfoNCE terms).
* **The per-depth columns give the total**, in the loss functions and in
  short runs of the real trainer.
* **The four other reductions keep their bits.** The loss functions return
  the values and the gradients of the commit before `hill`, and short runs
  of the trainer write the same losses CSV.
* **The refusals.** A missing scale or power, a value that is not above 0,
  the two flags without `hill`, `hill` with --train-rollout-distances and
  the value-space objective.
* **A run at k = 31 with s = p = 4** writes finite rows.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from src.forecasting_head import value_space_objective
from src.loss import (JOINT_TEACHER_SHAPE, align_loss,
                      contrastive_latent_loss, cpc_infonce_all_loss,
                      cpc_infonce_aux_loss, grid_pred_term, hill_weight,
                      joint_teacher_term, pred_term, reduce_depth_terms,
                      resolve_rollout_hill, resolve_rollout_reduce)

from tests.test_373_train_rollout_depth import (H, REPO_ROOT, TRAIN_PY,
                                                _REP_ONLY, _calls_named,
                                                _keyword, _latents, _spec,
                                                _train_tree,
                                                run_short_training)
from tests.test_401_rollout_reduce_mean import REDUCE_CASES, call
from tests.test_412_bimoco import group
from tests.test_412_ours_moirai import git_tree, run
from tests.test_per_depth_loss_columns import (JOINT_CELL, REP_ONLY_CELL,
                                               SPLIT, SPLIT_CELL, TINY,
                                               latents, losses_csv, names,
                                               padding, spec)

# The pair (s, p) of the run of the card.
RUN_HILL = (4.0, 4.0)
# A pair whose weights at the small depths of these tests are far from 1:
# w_1 = 0.69, w_2 = 0.36, w_3 = 0.2.
STEEP = (1.5, 2.0)
# The last commit before the hill reduction.
COMMIT_BEFORE = "1874a501"
OTHER_REDUCES = ("sum", "mean", "halving", "harmonic")


def hill_keys(hill=STEEP):
    """The config keys of a hill run."""
    return {"train_rollout_reduce": "hill",
            "train_rollout_hill_scale": hill[0],
            "train_rollout_hill_power": hill[1]}


def weighted(copies, hill=STEEP):
    """The manual hill total: copy j divided by 1 + (j / s)^p."""
    s, p = hill
    return sum(c / (1 + (j / s) ** p) for j, c in enumerate(copies))


def steps(totals):
    """The copies from the sums at the depths 0..k: the depth-0 total, then
    the step from depth j - 1 to depth j."""
    return [totals[0]] + [totals[j] - totals[j - 1]
                          for j in range(1, len(totals))]


# ---------------------------------------------------------------------------
# 1. The weights
# ---------------------------------------------------------------------------

class TestTheWeights:

    def test_the_weights_of_the_run(self):
        assert hill_weight(0, *RUN_HILL) == 1.0
        assert hill_weight(4, *RUN_HILL) == 0.5
        assert hill_weight(8, *RUN_HILL) == pytest.approx(1 / 17, rel=1e-15)

    def test_the_weights_of_the_card(self):
        """1, 0.996, 0.941, 0.760, 0.5, ..., 0.0003 at j = 31."""
        got = [hill_weight(j, *RUN_HILL) for j in (0, 1, 2, 3, 4, 31)]
        assert got == pytest.approx([1, 0.996, 0.941, 0.760, 0.5, 0.0003],
                                    abs=5e-4)

    def test_copy_j_takes_its_weight(self):
        terms = [torch.tensor(v, dtype=torch.float64)
                 for v in (1.0, 3.0, 5.0, 7.0)]
        assert float(reduce_depth_terms(terms, "hill", RUN_HILL)) == \
            pytest.approx(1 + 3 * 256 / 257 + 5 * 16 / 17
                          + 7 / (1 + 0.75 ** 4), rel=1e-15)

    def test_the_32_copies_of_the_run(self):
        terms = [torch.tensor(1.0, dtype=torch.float64)] * 32
        assert float(reduce_depth_terms(terms, "hill", RUN_HILL)) == \
            pytest.approx(sum(1 / (1 + (j / 4) ** 4) for j in range(32)),
                          rel=1e-14)

    def test_one_copy_returns_itself(self):
        term = torch.tensor(2.5, dtype=torch.float64)
        assert float(reduce_depth_terms([term], "hill", RUN_HILL)) == 2.5

    def test_a_weight_above_the_float_range_is_zero(self):
        """(31 / 0.01)^400 is above the largest float."""
        assert hill_weight(31, 0.01, 400.0) == 0.0

    def test_the_word_resolves(self):
        assert resolve_rollout_reduce("hill", None) == "hill"
        assert resolve_rollout_reduce(None, "hill") == "hill"
        assert resolve_rollout_hill("hill", (4, 4)) == (4.0, 4.0)

    @pytest.mark.parametrize("reduce", OTHER_REDUCES)
    def test_another_reduction_reads_no_pair(self, reduce):
        assert resolve_rollout_hill(reduce, None) is None
        assert resolve_rollout_hill(reduce, RUN_HILL) is None


# ---------------------------------------------------------------------------
# 2. The manual weighted sum of the copies
# ---------------------------------------------------------------------------

class TestTheIdentity:

    @pytest.mark.parametrize("shape,config_name", REDUCE_CASES)
    @pytest.mark.parametrize("k", [1, 3])
    def test_each_copy_enters_at_its_weight(self, shape, config_name, k):
        lat = _latents(shape, seed=2, depth=k)
        sums = [call(shape, config_name, lat, depth=j, reduce="sum")
                for j in range(k + 1)]
        got = call(shape, config_name, lat, depth=k, **hill_keys())
        assert got == pytest.approx(weighted(steps(sums)), rel=1e-11,
                                    abs=1e-12)

    def test_depth_zero_is_the_sum(self):
        lat = _latents(_REP_ONLY, seed=3)
        cfg = {"align_loss_weight": 1.0}
        assert call(_REP_ONLY, "base", lat, depth=0, **cfg,
                    **hill_keys()) == call(_REP_ONLY, "base", lat, depth=0,
                                           reduce="sum", **cfg)


class TestTheStandaloneTerms:

    @pytest.mark.parametrize("kind", ["cosine", "l2_raw"])
    @pytest.mark.parametrize("k", [1, 3])
    def test_align_loss_weighs_each_depth(self, kind, k):
        f, o, _, rollouts = _latents(_REP_ONLY, seed=11, depth=k)
        sums = [float(align_loss(f, o, 1.0, kind=kind,
                                 rollout_latents=rollouts[:j] or None))
                for j in range(k + 1)]
        assert float(align_loss(f, o, 1.0, rollout_latents=rollouts,
                                depth_reduce="hill", depth_hill=STEEP,
                                kind=kind)) == pytest.approx(
            weighted(steps(sums)), rel=1e-11)

    @pytest.mark.parametrize("fn", [cpc_infonce_aux_loss,
                                    cpc_infonce_all_loss])
    @pytest.mark.parametrize("k", [1, 3])
    def test_cpc_infonce_weighs_each_depth(self, fn, k):
        f, o, _, rollouts = _latents(_REP_ONLY, seed=17, depth=k)
        w1 = torch.nn.Linear(H, H, bias=False).to(torch.float64)
        sums = [float(fn(f, o, w1, rollout_latents=rollouts[:j] or None))
                for j in range(k + 1)]
        assert float(fn(f, o, w1, rollout_latents=rollouts,
                        depth_reduce="hill",
                        depth_hill=STEEP)) == pytest.approx(
            weighted(steps(sums)), rel=1e-11)


def with_depth(groups, depth):
    """The groups with their first ``depth`` rollout copies only."""
    return [SimpleNamespace(**{**vars(g), "rollout": g.rollout[:depth]})
            for g in groups]


def bimoco_spec(**extra):
    cfg = {"loss_shape": SPLIT, "contrastive_divergence_temperature": 0.5,
           "moco_negatives": True, "pred_loss_weight": 0.7}
    cfg.update(extra)
    return SimpleNamespace(train_configuration=cfg)


class TestTheSplitPredAndTheJointFraction:

    def test_grid_pred_term_on_two_patch_sizes(self):
        """Sizes 8 and 16 (ratio 2) on one grid (#412), at depth 2. The
        groups read the word and the pair from the config keys."""
        pad8 = torch.zeros(3, 8, 2, dtype=torch.bool)
        pad8[1, :2] = True
        groups = [group(3, 8, 1, C=2, seed=1, depth=2, pad=pad8),
                  group(2, 4, 2, C=2, seed=2, depth=2)]
        sums = [float(grid_pred_term(with_depth(groups, j), bimoco_spec()))
                for j in range(3)]
        terms = {}
        got = float(grid_pred_term(groups, bimoco_spec(**hill_keys()),
                                   term_out=terms))
        assert got == pytest.approx(weighted(steps(sums)), rel=1e-12)
        assert got == pytest.approx(weighted(
            [0.7 * terms[n] for n in names("l_pred", range(3))]), rel=1e-12)

    @pytest.mark.parametrize("padded", [False, True])
    def test_pred_term_takes_the_word_as_an_argument(self, padded):
        """The trainer passes the word as ``depth_reduce`` and the pair in
        the config keys."""
        f, h, teacher, rollout = latents(seed=4)
        pad = padding(f) if padded else None
        sums = [float(pred_term(f, h, spec(SPLIT, pred_loss_weight=0.7), pad,
                                teacher_original_latent=teacher,
                                rollout_latents=rollout[:j] or None))
                for j in range(len(rollout) + 1)]
        keys = hill_keys()
        del keys["train_rollout_reduce"]
        got = float(pred_term(f, h, spec(SPLIT, pred_loss_weight=0.7, **keys),
                              pad, teacher_original_latent=teacher,
                              rollout_latents=rollout, depth_reduce="hill"))
        assert got == pytest.approx(weighted(steps(sums)), rel=1e-12)

    @pytest.mark.parametrize("padded", [False, True])
    def test_the_joint_fraction_weighs_each_depth(self, padded):
        f, h, teacher, rollout = latents(seed=3)
        pad = padding(f) if padded else None
        sums = [float(joint_teacher_term(
            f, h, spec(JOINT_TEACHER_SHAPE), pad,
            teacher_original_latent=teacher,
            rollout_latents=rollout[:j] or None))
            for j in range(len(rollout) + 1)]
        got = float(joint_teacher_term(
            f, h, spec(JOINT_TEACHER_SHAPE, **hill_keys()), pad,
            teacher_original_latent=teacher, rollout_latents=rollout))
        assert got == pytest.approx(weighted(steps(sums)), rel=1e-12)


# ---------------------------------------------------------------------------
# 3. The per-depth columns give the total
# ---------------------------------------------------------------------------

DEPTHS = range(4)


def test_the_align_copies_give_the_total():
    f, h, teacher, rollout = latents(seed=1)
    terms = {}
    total = align_loss(f, h, 2.0, target_latent=teacher,
                       rollout_latents=rollout, depth_reduce="hill",
                       depth_hill=STEEP, pad_patches=padding(f),
                       term_out=terms)
    assert total.item() == pytest.approx(
        weighted([2.0 * terms[n] for n in names("l_align", DEPTHS)]),
        rel=1e-12)


@pytest.mark.parametrize("shape", [_REP_ONLY, SPLIT])
def test_the_copies_in_the_main_loss_give_the_total(shape):
    """L_rep enters once. The copy of each depth holds L_align, and L_pred
    on the split shape."""
    f, h, _, rollout = latents(seed=2)
    s = spec(shape, align_loss_weight=1.5, pred_loss_weight=0.7,
             train_rollout_depth=3, **hill_keys())
    terms = {}
    total = contrastive_latent_loss((f, h), False, s, rollout_latents=rollout,
                                    rep_loss_weight=0.5, term_out=terms)
    copies = [1.5 * terms[f"l_align_d{j}"]
              + 0.7 * terms.get(f"l_pred_d{j}", 0.0) for j in DEPTHS]
    assert total.item() == pytest.approx(
        0.5 * terms["l_rep"] + weighted(copies), rel=1e-9)


def test_the_joint_copies_give_the_total():
    f, h, teacher, rollout = latents(seed=3)
    terms = {}
    total = joint_teacher_term(
        f, h, spec(JOINT_TEACHER_SHAPE, **hill_keys()), padding(f),
        teacher_original_latent=teacher, rollout_latents=rollout,
        term_out=terms)
    assert total.item() == pytest.approx(
        weighted([terms[n] for n in names("l_joint", DEPTHS)]), rel=1e-12)


@pytest.fixture(scope="module")
def train():
    spec_ = importlib.util.spec_from_file_location("train_hill", TRAIN_PY)
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    return module


def test_l_align_on_two_patch_sizes_gives_the_total(train):
    """Each group adds its copies by its share of the rows (#412)."""
    lats, pads = {}, {}
    for size, steps_, seed in ((8, 8, 7), (16, 4, 8)):
        f, h, _, rollout = latents(seed=seed, T=steps_)
        lats[size] = SimpleNamespace(f=f, o=h, teacher=None,
                                     rollout=rollout[:2])
        pads[size] = padding(f) if size == 8 else None
    args = SimpleNamespace(align_loss_weight=2.0, align_target="student",
                           align_loss_kind="cosine",
                           train_rollout_reduce="hill",
                           train_rollout_hill_scale=STEEP[0],
                           train_rollout_hill_power=STEEP[1])
    values = train.no_contrastive_values()
    total = train.group_align_terms(lats, pads, {8: 3, 16: 2}, args, values)
    assert total.item() == pytest.approx(weighted(
        [2.0 * values["terms"][n] for n in names("l_align", range(3))]),
        rel=1e-12)


HILL_FLAGS = ("--train-rollout-reduce", "hill",
              "--train-rollout-hill-scale", str(STEEP[0]),
              "--train-rollout-hill-power", str(STEEP[1]))


def hill_loss(row, terms, align_weight=1.0, hill=STEEP):
    """The loss of a row from its columns: L_rep once at its weight, and the
    copy of depth j of each term at w_j."""
    rep = float(row["rep_w"]) * float(row["l_rep"]) if row["l_rep"] else 0.0
    depths = range(sum(c.startswith(f"{terms[0]}_d") for c in row))
    copies = [sum((align_weight if t == "l_align" else 1.0)
                  * float(row[f"{t}_d{j}"]) for t in terms) for j in depths]
    return rep + weighted(copies, hill)


@pytest.mark.parametrize("cell,terms", [
    (REP_ONLY_CELL, ("l_align",)),
    (SPLIT_CELL, ("l_pred", "l_align")),
    (JOINT_CELL, ("l_align", "l_joint")),
])
def test_the_columns_give_the_loss_of_a_hill_run(tmp_path, cell, terms):
    """k = 2 on the real trainer. The loss is the hill total of the
    columns, and it is not the sum."""
    r = run(REPO_ROOT, tmp_path / "run", *TINY, *cell, *HILL_FLAGS)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    header, rows = losses_csv(tmp_path / "run")
    assert header[-3 * len(terms):] == [n for t in terms
                                        for n in names(t, range(3))]
    assert len(rows) == 3
    for row in rows:
        loss = float(row["loss"])
        assert loss == pytest.approx(hill_loss(row, terms), rel=1e-5)
        # An infinite scale gives every copy the weight 1: the sum.
        assert loss != pytest.approx(
            hill_loss(row, terms, hill=(math.inf, 1.0)), rel=1e-5)


# ---------------------------------------------------------------------------
# 4. The four other reductions keep their bits
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loss_before(tmp_path_factory):
    """src/loss.py of the commit before the hill reduction, as a module."""
    root = git_tree(COMMIT_BEFORE, tmp_path_factory.mktemp("before"))
    spec_ = importlib.util.spec_from_file_location(
        "loss_before_hill", root / "src" / "loss.py")
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    return module


BIT_TERMS = ("rep_only", "split", "align", "pred", "joint", "cpc_aux",
             "cpc_all")


def value_and_grads(module, name, reduce):
    """The term ``name`` of ``module`` at depth 3 under ``reduce``: its value
    and its gradients on f and on the rollout copies."""
    f, h, teacher, rollout = latents(seed=9)
    leaves = [leaf.requires_grad_(True) for leaf in (f, *rollout)]
    w1 = torch.nn.Linear(6, 6, bias=False).to(torch.float64)
    with torch.no_grad():
        w1.weight.copy_(torch.randn(6, 6, dtype=torch.float64,
                                    generator=torch.Generator().manual_seed(0)))
    main = {shape: spec(shape, align_loss_weight=1.5, pred_loss_weight=0.7,
                        train_rollout_depth=3, train_rollout_reduce=reduce)
            for shape in (_REP_ONLY, SPLIT)}
    calls = {
        "rep_only": lambda: module.contrastive_latent_loss(
            (f, h), False, main[_REP_ONLY], rollout_latents=rollout,
            rep_loss_weight=0.5),
        "split": lambda: module.contrastive_latent_loss(
            (f, h), False, main[SPLIT], rollout_latents=rollout,
            rep_loss_weight=0.5),
        "align": lambda: module.align_loss(
            f, h, 2.0, target_latent=teacher, rollout_latents=rollout,
            depth_reduce=reduce, pad_patches=padding(f)),
        "pred": lambda: module.pred_term(
            f, h, spec(SPLIT, pred_loss_weight=0.7), padding(f),
            teacher_original_latent=teacher, rollout_latents=rollout,
            depth_reduce=reduce),
        "joint": lambda: module.joint_teacher_term(
            f, h, spec(JOINT_TEACHER_SHAPE), padding(f),
            teacher_original_latent=teacher, rollout_latents=rollout,
            depth_reduce=reduce),
        "cpc_aux": lambda: module.cpc_infonce_aux_loss(
            f, h, w1, rollout_latents=rollout, depth_reduce=reduce),
        "cpc_all": lambda: module.cpc_infonce_all_loss(
            f, h, w1, rollout_latents=rollout, depth_reduce=reduce),
    }
    value = calls[name]()
    return value.detach(), torch.autograd.grad(value, leaves,
                                               allow_unused=True)


@pytest.mark.parametrize("reduce", OTHER_REDUCES)
@pytest.mark.parametrize("name", BIT_TERMS)
def test_the_other_reductions_keep_their_bits(loss_before, reduce, name):
    """The value and the gradients of each term, bit for bit."""
    import src.loss as loss_now
    value, grads = value_and_grads(loss_now, name, reduce)
    value_before, grads_before = value_and_grads(loss_before, name, reduce)
    assert torch.equal(value, value_before)
    for now, before in zip(grads, grads_before):
        assert (now is None and before is None) or torch.equal(now, before)


# The split shape with L_align and the CPC auxiliary reaches the main loss,
# the standalone CPC term and the config keys. The joint fraction reaches
# joint_teacher_term and the standalone align_loss.
BYTE_CELLS = {
    "split_cpc": (*SPLIT_CELL, "--cpc-infonce-weight", "0.1"),
    "joint": JOINT_CELL,
}


@pytest.mark.parametrize("reduce", OTHER_REDUCES)
@pytest.mark.parametrize("cell", sorted(BYTE_CELLS))
def test_a_run_of_another_reduction_writes_the_bytes_before(tmp_path,
                                                             reduce, cell):
    """k = 2 on the real trainer: the commit before hill and this one write
    the same losses CSV, byte for byte."""
    before = git_tree(COMMIT_BEFORE, tmp_path)
    flags = (*TINY, *BYTE_CELLS[cell], "--train-rollout-reduce", reduce)
    for leg, root in (("before", before), ("now", REPO_ROOT)):
        r = run(root, tmp_path / leg, *flags)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert len(losses_csv(tmp_path / "now")[1]) == 3
    assert ((tmp_path / "now" / "r_losses.csv").read_bytes()
            == (tmp_path / "before" / "r_losses.csv").read_bytes())


# ---------------------------------------------------------------------------
# 5. The refusals
# ---------------------------------------------------------------------------

class TestTheLibraryRefusals:

    @pytest.mark.parametrize("hill,key", [
        (None, "train_rollout_hill_scale"),
        ((None, 4.0), "train_rollout_hill_scale"),
        ((4.0, None), "train_rollout_hill_power"),
        ((0.0, 4.0), "train_rollout_hill_scale"),
        ((4.0, -1.0), "train_rollout_hill_power"),
        ((float("nan"), 4.0), "train_rollout_hill_scale"),
    ])
    def test_a_missing_or_non_positive_number_raises(self, hill, key):
        with pytest.raises(ValueError, match=key):
            resolve_rollout_hill("hill", hill)

    def test_reduce_depth_terms_without_the_pair_raises(self):
        terms = [torch.tensor(1.0, dtype=torch.float64)] * 3
        with pytest.raises(ValueError, match="train_rollout_hill_scale"):
            reduce_depth_terms(terms, "hill")

    def test_align_loss_without_the_pair_raises(self):
        f, o, _, rollouts = _latents(_REP_ONLY, seed=5, depth=2)
        with pytest.raises(ValueError, match="train_rollout_hill_scale"):
            align_loss(f, o, 1.0, rollout_latents=rollouts,
                       depth_reduce="hill")

    def test_the_config_without_a_key_raises_at_depth_zero(self):
        """The main loss checks the pair before the depths, so a k = 0
        call raises too."""
        f, o, _, _ = _latents(_REP_ONLY, seed=6)
        keys = hill_keys()
        del keys["train_rollout_hill_power"]
        s = _spec(_REP_ONLY, align_loss_weight=1.0, **keys)
        with pytest.raises(ValueError, match="train_rollout_hill_power"):
            contrastive_latent_loss((f, o), validation=False, spec=s)

    def test_a_left_out_depth_refuses_the_word(self):
        f, o, _, roll = _latents(_REP_ONLY, seed=13, depth=3)
        with pytest.raises(ValueError, match="hill"):
            align_loss(f, o, 1.0, rollout_latents=[roll[0], None, roll[2]],
                       depth_reduce="hill", depth_hill=STEEP)

    def test_the_value_space_objective_refuses_the_word(self):
        with pytest.raises(ValueError, match="hill"):
            value_space_objective(None, None, depth=1, reduce="hill")


def parse(train, *flags):
    """The flags of a run, with the one flag that train.py requires."""
    return train.parse_args(["--weight-decay", "0.1", *flags])


class TestTheFlags:

    @pytest.mark.parametrize("flags,message", [
        pytest.param(("--train-rollout-reduce", "hill",
                      "--train-rollout-hill-power", "4"),
                     "hill needs --train-rollout-hill-scale above 0. Got None",
                     id="no_scale"),
        pytest.param(("--train-rollout-reduce", "hill",
                      "--train-rollout-hill-scale", "4"),
                     "hill needs --train-rollout-hill-power above 0. Got None",
                     id="no_power"),
        pytest.param(("--train-rollout-reduce", "hill",
                      "--train-rollout-hill-scale", "0",
                      "--train-rollout-hill-power", "4"),
                     "hill needs --train-rollout-hill-scale above 0. Got 0.0",
                     id="zero_scale"),
        pytest.param(("--train-rollout-reduce", "hill",
                      "--train-rollout-hill-scale", "4",
                      "--train-rollout-hill-power", "-2"),
                     "hill needs --train-rollout-hill-power above 0. Got -2.0",
                     id="negative_power"),
        pytest.param(("--train-rollout-reduce", "hill",
                      "--train-rollout-hill-scale", "nan",
                      "--train-rollout-hill-power", "4"),
                     "hill needs --train-rollout-hill-scale above 0. Got nan",
                     id="nan_scale"),
        pytest.param(("--train-rollout-hill-scale", "4",),
                     "--train-rollout-hill-scale sets a weight of "
                     "--train-rollout-reduce hill", id="scale_with_sum"),
        pytest.param(("--train-rollout-reduce", "harmonic",
                      "--train-rollout-hill-power", "4"),
                     "--train-rollout-hill-power sets a weight of "
                     "--train-rollout-reduce hill", id="power_with_harmonic"),
    ])
    def test_a_bad_pair_is_refused(self, train, flags, message):
        with pytest.raises(SystemExit, match=re.escape(message)):
            train.check_rollout_hill(parse(train, *flags))

    def test_a_full_pair_passes(self, train):
        args = parse(train, *HILL_FLAGS)
        train.check_rollout_hill(args)
        assert train.rollout_hill(args) == STEEP

    def test_another_reduction_without_the_flags_passes(self, train):
        args = parse(train, "--train-rollout-reduce", "harmonic")
        train.check_rollout_hill(args)
        assert train.rollout_hill(args) == (None, None)

    def test_the_distances_refuse_the_word(self, train):
        args = parse(train, "--train-rollout-distances", "1,2,4",
                     *HILL_FLAGS)
        with pytest.raises(SystemExit, match="--train-rollout-reduce hill"):
            train.resolve_rollout_distances(args)

    def test_a_value_space_run_refuses_the_flags(self, train):
        clash = train.value_space_conflicts(
            ["--weight-decay", "0.1", "--value-space-objective", *HILL_FLAGS])
        assert clash == ["--train-rollout-hill-scale",
                         "--train-rollout-hill-power"]

    @pytest.mark.parametrize("flags,message", [
        (("--train-rollout-reduce", "hill", "--train-rollout-hill-scale",
          "4"), "--train-rollout-hill-power above 0"),
        (("--train-rollout-hill-scale", "4"),
         "--train-rollout-hill-scale sets a weight"),
    ])
    def test_the_trainer_refuses_before_step_1(self, tmp_path, flags,
                                               message):
        out = subprocess.run(
            [sys.executable, str(TRAIN_PY), "--weight-decay", "0.1",
             "--device", "cpu", "--hf-repo", "none", "--hf-path", "none",
             "--save-dir", str(tmp_path), *flags],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT),
                 "CUDA_VISIBLE_DEVICES": ""})
        assert out.returncode != 0
        assert message in out.stderr, out.stderr[-2000:]


class TestTheTrainerWiring:

    def test_the_flag_offers_the_word(self):
        choices = {c.args[0].value: _keyword(c, "choices")
                   for c in _calls_named(_train_tree(), "add_argument")
                   if c.args and isinstance(c.args[0], ast.Constant)}
        assert "hill" in ast.literal_eval(choices["--train-rollout-reduce"])

    def test_the_config_keys_carry_the_flags(self):
        src = TRAIN_PY.read_text()
        for key in ("train_rollout_hill_scale", "train_rollout_hill_power"):
            assert (f'LOSS_SPEC.train_configuration["{key}"] = args.{key}'
                    in src)

    @pytest.mark.parametrize("name", ["align_loss", "cpc_infonce_aux_loss",
                                      "cpc_infonce_all_loss"])
    def test_every_standalone_call_carries_the_pair(self, name):
        """These terms read no config, so a call without the pair would
        raise on a hill run."""
        calls = [c for c in _calls_named(_train_tree(), name)
                 if any(kw.arg == "depth_reduce" for kw in c.keywords)]
        assert calls, f"no {name} call carries depth_reduce"
        for c in calls:
            assert any(kw.arg == "depth_hill" for kw in c.keywords), \
                f"{name} takes the reduction but not its pair"


# ---------------------------------------------------------------------------
# 6. A run at k = 31 with the weights of the card
# ---------------------------------------------------------------------------

def test_a_run_at_depth_31_writes_finite_rows(tmp_path):
    """The recipe of the card on a tiny model: L_align at weight 10 over the
    depths 0 to 31, with s = p = 4. 1024 values give 64 patches, and depth
    31 needs 33. The columns give the loss."""
    rows = run_short_training(
        tmp_path, "k31_hill", "--t-raw", "1024", "--loss-shape", _REP_ONLY,
        "--align-loss-weight", "10", "--train-rollout-depth", "31",
        "--train-rollout-reduce", "hill", "--train-rollout-hill-scale", "4",
        "--train-rollout-hill-power", "4")
    assert len(rows) == 2
    for row in rows:
        columns = ["loss", "l_rep", *names("l_align", range(32)),
                   *names("cos_err", range(32))]
        assert all(math.isfinite(float(row[c])) for c in columns)
        assert float(row["loss"]) == pytest.approx(
            hill_loss(row, ("l_align",), align_weight=10.0, hill=RUN_HILL),
            rel=1e-5)
