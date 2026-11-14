"""Tests for `--train-rollout-distances` (config key `train_rollout_distances`).

Distance d is the rollout copy that ties f^(d-1)_t to h_{t+d}. The flag keeps
the copies at the listed distances only. The rollout still composes every
depth up to max(d) - 1. The trainer puts None in place of each depth that the
list leaves out (`drop_unlisted_depths`), and every term skips a None entry
with the true depth of the other entries.

What this file pins:

* **The parse and every refusal.** The flag gives a sorted list of unique
  integers. A distance below 1, a list without 1, a depth other than 0 and
  max(d) - 1, the halving, the harmonic weights and the value-space
  objective each stop the run, and the message names the problem.
* **One function makes the None entries**, for the depths and for the raw
  depths of l2_raw. The chain still composes every depth.
* **The terms read the kept copies only.** align_loss and the L_align add-on
  of contrastive_latent_loss equal the sum of the kept copies, each built one
  by one, and the mean divides by their count. L_pred, the main shapes and
  the CPC terms drop the left-out copy and nothing else. A library call with
  a None entry refuses the halving and the harmonic weights.
* **A copy at depth j averages over its own T - 1 - j pairs**, also with a
  left padding.
* **Without the flag, a k = 3 run is unchanged.** The loss values of
  9e3a0b75, the commit before the flag, and runs of its trainer that write
  the same losses CSV, byte for byte.
* **The CSV writes the columns of the kept depths only.**
* **Short runs of the real trainer** with --train-rollout-distances 1,2,4.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.loss import (align_loss, contrastive_latent_loss,
                      cpc_infonce_all_loss, cpc_infonce_aux_loss, pred_term,
                      rollout_depth_views)
from src.metrics import rollout_cos_error

from tests.test_373_train_rollout_depth import (B, C, H, REPO_ROOT, T,
                                                TRAIN_PY, _REP_ONLY, _SPLIT,
                                                _XSHH_ALLT, _calls_named,
                                                _keyword, _latents, _spec,
                                                _train_tree, load_train_module,
                                                refuse_run, run_short_training)

# The commit before the flag. A run without the flag trains as it did.
BASE_COMMIT = "9e3a0b75"
K = 3
# The distances 1, 2 and 4 keep the depths 0, 1 and 3 of a k = 3 rollout.
KEPT = (1, 3)


@pytest.fixture(scope="module")
def train():
    return load_train_module()


def left_padding(steps=T):
    """[B, T, C] bool: a left zero padding on rows 0 and 1 (#419)."""
    pad = torch.zeros(B, steps, C, dtype=torch.bool)
    pad[0, :2] = True
    pad[1, :1, 0] = True
    return pad


def cpc_w1(seed=5):
    """The bilinear W_1 of the CPC terms, from its own generator."""
    g = torch.Generator().manual_seed(seed)
    w1 = torch.nn.Linear(H, H, bias=False).to(torch.float64)
    with torch.no_grad():
        w1.weight.copy_(torch.randn(H, H, generator=g, dtype=torch.float64)
                        / H ** 0.5)
    return w1


def keep(rollouts, depths):
    """The list f^(1)..f^(k) with None at each depth not in ``depths``."""
    return [f if j in depths else None
            for j, f in enumerate(rollouts, start=1)]


# ---------------------------------------------------------------------------
# The parse and the refusals
# ---------------------------------------------------------------------------

def flag_args(distances, **over):
    """The flags that ``resolve_rollout_distances`` reads."""
    base = dict(train_rollout_distances=distances, train_rollout_depth=0,
                train_rollout_reduce="sum", value_space_objective=False)
    return SimpleNamespace(**{**base, **over})


class TestTheParse:

    @pytest.mark.parametrize("text,want", [
        ("1,2,4,8,16,32", [1, 2, 4, 8, 16, 32]),
        ("4,1,2,2", [1, 2, 4]),
        ("1, 2, 4", [1, 2, 4]),
        ("1", [1]),
    ])
    def test_a_sorted_list_of_unique_ints(self, train, text, want):
        assert train.rollout_distances_arg(text) == want

    @pytest.mark.parametrize("text", ["1,x", "", "1,,2", "1.5,2"])
    def test_a_word_that_is_not_an_integer_is_refused(self, train, text):
        with pytest.raises(argparse.ArgumentTypeError,
                           match="comma-separated list of integers"):
            train.rollout_distances_arg(text)

    def test_the_flag_defaults_to_off(self):
        defaults = {c.args[0].value: _keyword(c, "default")
                    for c in _calls_named(_train_tree(), "add_argument")
                    if c.args and isinstance(c.args[0], ast.Constant)}
        assert ast.literal_eval(defaults["--train-rollout-distances"]) is None

    def test_the_command_line_parses_the_list(self, train):
        args = train.parse_args(["--weight-decay", "0.1",
                                 "--train-rollout-distances", "4,1,2"])
        assert args.train_rollout_distances == [1, 2, 4]
        assert train.parse_args(["--weight-decay", "0.1"]
                                ).train_rollout_distances is None


class TestTheChecks:

    def test_the_depth_becomes_the_largest_distance_minus_one(self, train):
        args = flag_args([1, 2, 4, 8, 16, 32])
        train.resolve_rollout_distances(args)
        assert args.train_rollout_depth == 31
        assert train.kept_rollout_depths(args) == [0, 1, 3, 7, 15, 31]

    def test_the_same_depth_given_twice_is_accepted(self, train):
        args = flag_args([1, 2, 4], train_rollout_depth=3)
        train.resolve_rollout_distances(args)
        assert args.train_rollout_depth == 3

    def test_distance_one_alone_is_depth_zero(self, train):
        args = flag_args([1])
        train.resolve_rollout_distances(args)
        assert args.train_rollout_depth == 0

    def test_the_flag_off_changes_nothing(self, train):
        args = flag_args(None, train_rollout_depth=5,
                         train_rollout_reduce="halving")
        train.resolve_rollout_distances(args)
        assert args.train_rollout_depth == 5
        assert train.kept_rollout_depths(args) is None

    @pytest.mark.parametrize("over,phrase", [
        pytest.param(dict(train_rollout_distances=[0, 1, 2]), "1 or more",
                     id="zero"),
        pytest.param(dict(train_rollout_distances=[-2, 1]), "1 or more",
                     id="negative"),
        pytest.param(dict(train_rollout_distances=[2, 4]), "must hold 1",
                     id="no_distance_one"),
        pytest.param(dict(train_rollout_depth=3), "needs depth 31",
                     id="other_depth"),
        pytest.param(dict(train_rollout_reduce="halving"),
                     "--train-rollout-reduce halving", id="halving"),
        pytest.param(dict(train_rollout_reduce="harmonic"),
                     "--train-rollout-reduce harmonic", id="harmonic"),
        pytest.param(dict(value_space_objective=True),
                     "--value-space-objective", id="value_space"),
    ])
    def test_each_bad_combination_is_refused(self, train, over, phrase):
        args = flag_args([1, 2, 4, 8, 16, 32], **over)
        with pytest.raises(SystemExit, match=phrase):
            train.resolve_rollout_distances(args)

    @pytest.mark.parametrize("flags,phrase", [
        pytest.param(("--train-rollout-distances", "2,4"), "must hold 1",
                     id="no_distance_one"),
        pytest.param(("--train-rollout-distances", "1,2,4",
                      "--train-rollout-depth", "2"), "needs depth 3",
                     id="other_depth"),
        pytest.param(("--train-rollout-distances", "1,2,4",
                      "--train-rollout-reduce", "halving"),
                     "--train-rollout-reduce halving", id="halving"),
        pytest.param(("--train-rollout-distances", "1,2,4",
                      "--train-rollout-reduce", "harmonic"),
                     "--train-rollout-reduce harmonic", id="harmonic"),
        pytest.param(("--train-rollout-distances", "1,2",
                      "--value-space-objective"),
                     "--value-space-objective", id="value_space"),
    ])
    def test_the_cli_refuses_before_step_one(self, tmp_path, flags, phrase):
        stderr = refuse_run(tmp_path, *flags)
        assert "--train-rollout-distances" in stderr
        assert phrase in stderr

    def test_the_cli_names_a_word_that_is_not_an_integer(self, tmp_path):
        out = subprocess.run(
            [sys.executable, str(TRAIN_PY), "--weight-decay", "0.1",
             "--save-dir", str(tmp_path), "--train-rollout-distances", "1,x"],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
        assert out.returncode != 0
        assert "--train-rollout-distances" in out.stderr
        assert "not a comma-separated list of integers" in out.stderr


# ---------------------------------------------------------------------------
# One function makes the None entries
# ---------------------------------------------------------------------------

class TestTheKeptDepths:

    @staticmethod
    def _depths(train, monkeypatch, distances, raw=False):
        """``forecaster_depths`` at k = 3 on stand-in depths. Returns the
        result, the stand-ins and the depth that the chain composed."""
        made = [torch.full((1,), float(j)) for j in (1, 2, 3)]
        made_raw = [torch.full((1,), -float(j)) for j in (1, 2, 3)]
        seen = {}

        def compose(model, f_lat, k, scale=None, return_raw=False):
            seen["k"] = k
            return (made, made_raw) if return_raw else made
        monkeypatch.setattr(train, "rollout_forecaster_latents", compose)
        args = SimpleNamespace(train_rollout_feedback="raw",
                               train_rollout_depth=3,
                               train_rollout_distances=distances)
        out = train.forecaster_depths(None, None, None, args, raw=raw)
        return out, made, made_raw, seen["k"]

    def test_a_left_out_depth_is_none_and_the_chain_composes_all(
            self, train, monkeypatch):
        out, made, _, k = self._depths(train, monkeypatch, [1, 2, 4])
        assert k == 3
        assert out[0] is made[0] and out[1] is None and out[2] is made[2]

    def test_the_raw_depths_leave_out_the_same_depths(self, train,
                                                      monkeypatch):
        (out, out_raw), made, made_raw, _ = self._depths(
            train, monkeypatch, [1, 3], raw=True)
        assert [t is None for t in out] == [True, False, True]
        assert [t is None for t in out_raw] == [True, False, True]
        assert out[1] is made[1] and out_raw[1] is made_raw[1]

    def test_the_flag_off_returns_the_list_as_it_is(self, train,
                                                    monkeypatch):
        out, made, _, _ = self._depths(train, monkeypatch, None)
        assert out is made

    def test_the_trainer_makes_them_in_one_function(self):
        tree = _train_tree()
        owner = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name == "forecaster_depths"][0]
        inside = {id(n) for n in ast.walk(owner)}
        calls = _calls_named(tree, "drop_unlisted_depths")
        assert calls and all(id(c) in inside for c in calls)

    def test_the_config_key_records_the_flag(self):
        assert ('LOSS_SPEC.train_configuration["train_rollout_distances"] = '
                'args.train_rollout_distances') in TRAIN_PY.read_text()


# ---------------------------------------------------------------------------
# The terms read the kept copies only
# ---------------------------------------------------------------------------

def align_copy(f_j, o, j, weight, pad=None):
    """The depth-j copy of L_align, built from its formula: the mean of
    2 - 2 cos(f^(j)_t, o_{t+1+j}) over the anchors t = 0 .. T-2-j that the
    padding does not mark."""
    steps = o.shape[1] - 1 - j
    cos = (F.normalize(f_j[:, :steps], dim=-1)
           * F.normalize(o[:, 1 + j:], dim=-1)).sum(-1)
    pair = 2.0 - 2.0 * cos
    if pad is None:
        return weight * pair.mean()
    real = ~pad[:, :steps]
    return weight * pair[real].sum() / real.sum()


class TestAlignLossKeepsTheListedCopies:

    @pytest.mark.parametrize("padded", [False, True], ids=["no_pad", "pad"])
    def test_the_sum_adds_the_kept_copies(self, padded):
        f, o, _, roll = _latents(_REP_ONLY, seed=11, depth=K)
        pad = left_padding() if padded else None
        got = align_loss(f, o, 0.5, rollout_latents=keep(roll, KEPT),
                         pad_patches=pad)
        want = align_copy(f, o, 0, 0.5, pad) + sum(
            align_copy(roll[j - 1], o, j, 0.5, pad) for j in KEPT)
        assert float(got) == pytest.approx(float(want), rel=0, abs=1e-12)

    @pytest.mark.parametrize("padded", [False, True], ids=["no_pad", "pad"])
    def test_the_mean_divides_by_the_count_of_kept_copies(self, padded):
        f, o, _, roll = _latents(_REP_ONLY, seed=12, depth=K)
        pad = left_padding() if padded else None
        got = align_loss(f, o, 0.5, rollout_latents=keep(roll, KEPT),
                         depth_reduce="mean", pad_patches=pad)
        copies = [align_copy(f, o, 0, 0.5, pad)] + [
            align_copy(roll[j - 1], o, j, 0.5, pad) for j in KEPT]
        assert float(got) == pytest.approx(float(sum(copies)) / 3, rel=0,
                                           abs=1e-12)

    @pytest.mark.parametrize("reduce", ["halving", "harmonic"])
    def test_a_place_weight_with_a_left_out_depth_is_refused(self, reduce):
        f, o, _, roll = _latents(_REP_ONLY, seed=13, depth=K)
        with pytest.raises(ValueError, match=reduce):
            align_loss(f, o, 1.0, rollout_latents=keep(roll, KEPT),
                       depth_reduce=reduce)


def rep_only_add_on(f, o, rollouts, reduce="sum", pad=None, weight=1.0):
    """contrastive_latent_loss on rep_only with the L_align add-on, at the
    depth of ``rollouts``."""
    spec = _spec(_REP_ONLY, align_loss_weight=weight,
                 train_rollout_depth=len(rollouts),
                 train_rollout_reduce=reduce)
    return contrastive_latent_loss((f, o), validation=False, spec=spec,
                                   rollout_latents=rollouts, pad_patches=pad)


class TestTheAddOnKeepsTheListedCopies:
    """The run's own shape: rep_only, where the whole depth part is the
    L_align add-on in the shared tail of contrastive_latent_loss."""

    @pytest.mark.parametrize("padded", [False, True], ids=["no_pad", "pad"])
    def test_the_sum_adds_the_kept_copies(self, padded):
        f, o, _, roll = _latents(_REP_ONLY, seed=21, depth=K)
        pad = left_padding() if padded else None
        got = rep_only_add_on(f, o, keep(roll, KEPT), pad=pad, weight=0.7)
        want = rep_only_add_on(f, o, [], pad=pad, weight=0.7) + sum(
            align_copy(roll[j - 1], o, j, 0.7, pad) for j in KEPT)
        assert float(got) == pytest.approx(float(want), rel=0, abs=1e-12)

    @pytest.mark.parametrize("padded", [False, True], ids=["no_pad", "pad"])
    def test_the_mean_divides_the_f_side_only(self, padded):
        f, o, _, roll = _latents(_REP_ONLY, seed=22, depth=K)
        pad = left_padding() if padded else None
        got = rep_only_add_on(f, o, keep(roll, KEPT), reduce="mean", pad=pad,
                              weight=0.7)
        l_rep = rep_only_add_on(f, o, [], pad=pad, weight=0.0)
        f_side = align_copy(f, o, 0, 0.7, pad) + sum(
            align_copy(roll[j - 1], o, j, 0.7, pad) for j in KEPT)
        assert float(got) == pytest.approx(float(l_rep + f_side / 3), rel=0,
                                           abs=1e-12)

    def test_the_list_keeps_its_length_k(self):
        """resolve_rollout_depth counts the None entries too, so a list
        that drops them is a count error and not a quiet change of k."""
        f, o, _, roll = _latents(_REP_ONLY, seed=23, depth=K)
        spec = _spec(_REP_ONLY, align_loss_weight=1.0, train_rollout_depth=K)
        with pytest.raises(ValueError, match="rollout latent"):
            contrastive_latent_loss((f, o), validation=False, spec=spec,
                                    rollout_latents=[roll[0], roll[2]])

    @pytest.mark.parametrize("reduce", ["halving", "harmonic"])
    def test_a_config_place_weight_with_a_left_out_depth_is_refused(
            self, reduce):
        f, o, _, roll = _latents(_REP_ONLY, seed=24, depth=K)
        with pytest.raises(ValueError, match=reduce):
            rep_only_add_on(f, o, keep(roll, KEPT), reduce=reduce)


def every_f_bearing_term():
    """name -> (latents, rollouts) -> loss, one per consumer of the depths.
    Each takes a depth list of length k = 3."""
    pad, w1 = left_padding(), cpc_w1()

    def main(shape, **cfg):
        def call(lat, rollouts):
            f, o, teacher, _ = lat
            target = teacher if cfg.get("align_target") == "teacher" else None
            spec = _spec(shape, train_rollout_depth=K, **cfg)
            return contrastive_latent_loss(
                (f, o), validation=False, spec=spec,
                teacher_original_latent=target, rollout_latents=rollouts)
        return call

    return {
        "rep_only_add_on": main(_REP_ONLY, align_loss_weight=1.0),
        "rep_only_add_on_teacher": main(_REP_ONLY, align_loss_weight=1.0,
                                        align_target="teacher"),
        "split": main(_SPLIT),
        "xshh_allt": main(_XSHH_ALLT),
        "align_l2_raw": lambda lat, r: align_loss(
            lat[0], lat[1], 0.5, rollout_latents=r, kind="l2_raw"),
        "pred_term_pad": lambda lat, r: pred_term(
            lat[0], lat[1], _spec(_SPLIT), pad, rollout_latents=r),
        "cpc_aux": lambda lat, r: cpc_infonce_aux_loss(
            lat[0], lat[1], w1, rollout_latents=r),
        "cpc_aux_pad": lambda lat, r: cpc_infonce_aux_loss(
            lat[0], lat[1], w1, rollout_latents=r, pad_patches=pad),
        "cpc_all": lambda lat, r: cpc_infonce_all_loss(
            lat[0], lat[1], w1, rollout_latents=r),
    }


TERMS = every_f_bearing_term()


@pytest.mark.parametrize("name", sorted(TERMS))
def test_each_term_drops_the_left_out_copy_and_nothing_else(name):
    """The sum is linear in the copies. So the list (f1, None, f3) gives the
    total of (f1, f2, f3) less the copy at depth 2. That copy is the total
    of (None, f2, None) less the total of (None, None, None), which holds
    depth 0 alone. A term that dropped a kept copy, or kept the left-out
    one, breaks the identity."""
    lat = _latents(_REP_ONLY, seed=31, depth=K)
    roll = lat[3]

    def total(rollouts):
        return float(TERMS[name](lat, rollouts).detach())
    kept, every = total(keep(roll, KEPT)), total(roll)
    only_two, none = total(keep(roll, (2,))), total([None] * K)
    assert kept == pytest.approx(every - only_two + none, rel=0, abs=1e-12)
    assert abs(kept - every) > 1e-6, "the copy at depth 2 entered anyway"


@pytest.mark.parametrize("shape,cfg", [
    pytest.param(_SPLIT, {}, id="split"),
    pytest.param(_XSHH_ALLT, {"align_loss_weight": 0.5}, id="xshh_allt"),
])
def test_the_mean_of_a_main_shape_divides_by_the_kept_count(shape, cfg):
    """The total is the h-side plus the mean of the f-bearing copies that
    enter. The h-side is the depth-0 total less its f-side, which
    ``f_terms_only`` reads. The sum of the kept copies is the summed total
    less the h-side."""
    f, o, _, roll = _latents(_REP_ONLY, seed=32, depth=K)

    def total(rollouts, reduce="sum", f_terms_only=False):
        spec = _spec(shape, train_rollout_depth=len(rollouts),
                     train_rollout_reduce=reduce, **cfg)
        return float(contrastive_latent_loss(
            (f, o), validation=False, spec=spec, rollout_latents=rollouts,
            f_terms_only=f_terms_only))

    h_side = total([None] * K) - total([], f_terms_only=True)
    f_side = total(keep(roll, KEPT)) - h_side
    assert total(keep(roll, KEPT), "mean") == pytest.approx(
        h_side + f_side / 3, rel=0, abs=1e-12)


# ---------------------------------------------------------------------------
# A copy at depth j averages over its own T - 1 - j pairs
# ---------------------------------------------------------------------------

def stepped_latents(steps, start, depth, rows=B, channels=C, width=4):
    """``(f, o, [f^(1)..f^(depth)])`` with a known cos for each pair.

    o_t is e_0 at an even t and e_1 at an odd t. Each f^(j)_t from t =
    ``start`` on equals its target o_{t+1+j}, so its pair loss is 0. Before
    ``start`` it is e_2, at a right angle to each target, so its pair loss
    is 2. A copy that read a target one step off would read 2 everywhere.
    """
    eye = torch.eye(width, dtype=torch.float64)
    t = torch.arange(steps)

    def spread(index):
        return (eye[index].view(1, steps, 1, width)
                .expand(rows, steps, channels, width).contiguous())

    def copy(j):
        return spread(torch.where(t >= start, (t + 1 + j) % 2, 2))
    return copy(0), spread(t % 2), [copy(j) for j in range(1, depth + 1)]


def expected_copy(j, steps, start, padded_rows=0, padded=0, rows=B):
    """2 x the right-angle anchors / the real anchors of the copy at depth
    j. The copy reads the anchors t = 0 .. steps-2-j. ``padded_rows`` rows
    start with ``padded`` padded positions."""
    anchors = steps - 1 - j
    real = padded_rows * (anchors - padded) + (rows - padded_rows) * anchors
    right_angle = (padded_rows * (start - padded)
                   + (rows - padded_rows) * start)
    return 2.0 * right_angle / real


def row_padding(steps, padded_rows, padded):
    pad = torch.zeros(B, steps, C, dtype=torch.bool)
    pad[:padded_rows, :padded] = True
    return pad


class TestEachCopyAveragesItsOwnPairs:

    STEPS, START = 10, 4

    @pytest.mark.parametrize("padded_rows,padded", [(0, 0), (2, 2), (3, 1)])
    @pytest.mark.parametrize("j", [1, 2, 3, 5])
    def test_the_copy_at_depth_j(self, j, padded_rows, padded):
        f, o, roll = stepped_latents(self.STEPS, self.START, j)
        pad = (row_padding(self.STEPS, padded_rows, padded)
               if padded_rows else None)
        depth0 = float(align_loss(f, o, 1.0, pad_patches=pad))
        alone = float(align_loss(f, o, 1.0, rollout_latents=keep(roll, (j,)),
                                 pad_patches=pad))
        args = (self.STEPS, self.START, padded_rows, padded)
        assert depth0 == pytest.approx(expected_copy(0, *args), abs=1e-12)
        assert alone - depth0 == pytest.approx(expected_copy(j, *args),
                                               abs=1e-12)

    @pytest.mark.parametrize("padded_rows,padded", [(0, 0), (2, 2)])
    @pytest.mark.parametrize("reduce", ["sum", "mean"])
    def test_the_distances_1_2_4(self, reduce, padded_rows, padded):
        f, o, roll = stepped_latents(self.STEPS, self.START, K)
        pad = (row_padding(self.STEPS, padded_rows, padded)
               if padded_rows else None)
        got = align_loss(f, o, 1.0, rollout_latents=keep(roll, KEPT),
                         depth_reduce=reduce, pad_patches=pad)
        args = (self.STEPS, self.START, padded_rows, padded)
        want = sum(expected_copy(j, *args) for j in (0, *KEPT))
        if reduce == "mean":
            want /= 3
        assert float(got) == pytest.approx(want, abs=1e-12)

    @pytest.mark.parametrize("padded_rows,padded", [(0, 0), (2, 2)])
    def test_the_add_on_of_rep_only(self, padded_rows, padded):
        f, o, roll = stepped_latents(self.STEPS, self.START, K)
        pad = (row_padding(self.STEPS, padded_rows, padded)
               if padded_rows else None)
        add_on = (rep_only_add_on(f, o, keep(roll, KEPT), pad=pad)
                  - rep_only_add_on(f, o, [None] * K, pad=pad))
        args = (self.STEPS, self.START, padded_rows, padded)
        assert float(add_on) == pytest.approx(
            sum(expected_copy(j, *args) for j in KEPT), abs=1e-12)

    @pytest.mark.parametrize("distance,pairs", [(16, 240), (32, 224)])
    def test_at_256_patches(self, distance, pairs):
        """docs/train_rollout_depth.md: at T = 256 patches, distance 32
        averages 224 pairs and distance 16 averages 240."""
        steps, start = 256, 8
        f, o, roll = stepped_latents(steps, start, distance - 1, rows=1,
                                     channels=1)
        depth0 = float(align_loss(f, o, 1.0))
        alone = float(align_loss(f, o, 1.0, rollout_latents=keep(
            roll, (distance - 1,))))
        assert alone - depth0 == pytest.approx(2.0 * start / pairs,
                                               abs=1e-12)


# ---------------------------------------------------------------------------
# Without the flag, a k = 3 run is unchanged
# ---------------------------------------------------------------------------

# The k = 3 values of 9e3a0b75, the commit before the flag, on the latents of
# ``default_path_cases``.
BASE_VALUES = {
    'rep_only_align_sum': 16.684215561143887,
    'rep_only_align_mean': 10.788057085995293,
    'rep_only_align_halving': 12.678067531003109,
    'rep_only_align_pad_sum': 16.204971257212808,
    'rep_only_align_teacher_sum': 16.7887720292609,
    'split_sum': 34.858555161552125,
    'split_mean': 15.331641986097353,
    'xshh_allt_sum': 34.445138803545554,
    'align_sum': 3.9307723167657302,
    'align_mean': 0.9826930791914326,
    'align_halving': 1.9276983016953404,
    'align_pad_sum': 3.8816698011066326,
    'align_l2_raw_sum': 3.9693869242286226,
    'cpc_aux_sum': 8.983488204871776,
    'cpc_aux_pad_sum': 9.074457590058437,
    'cpc_all_sum': 11.725614010102806,
    'pred_pad_sum': 25.348563579667584,
}
BASE_COS_ERR = [1.0710610969401946, 0.9975181126759947, 1.0008320801876474,
                0.8613610269618934]


def default_path_cases():
    """name -> the loss of a k = 3 call with every depth, as a run without
    the flag makes it."""
    f, o, teacher, roll = _latents(_REP_ONLY, seed=7, depth=K)
    pad, w1 = left_padding(), cpc_w1()

    def main(shape, reduce=None, pad_patches=None, target=None, **cfg):
        if reduce:
            cfg["train_rollout_reduce"] = reduce
        return contrastive_latent_loss(
            (f, o), validation=False,
            spec=_spec(shape, train_rollout_depth=K, **cfg),
            teacher_original_latent=target, rollout_latents=roll,
            pad_patches=pad_patches)

    return {
        'rep_only_align_sum': lambda: main(_REP_ONLY, align_loss_weight=1.0),
        'rep_only_align_mean': lambda: main(_REP_ONLY, 'mean',
                                            align_loss_weight=1.0),
        'rep_only_align_halving': lambda: main(_REP_ONLY, 'halving',
                                               align_loss_weight=1.0),
        'rep_only_align_pad_sum': lambda: main(_REP_ONLY, pad_patches=pad,
                                               align_loss_weight=1.0),
        'rep_only_align_teacher_sum': lambda: main(
            _REP_ONLY, target=teacher, align_loss_weight=1.0,
            align_target='teacher'),
        'split_sum': lambda: main(_SPLIT),
        'split_mean': lambda: main(_SPLIT, 'mean'),
        'xshh_allt_sum': lambda: main(_XSHH_ALLT),
        'align_sum': lambda: align_loss(f, o, 0.5, rollout_latents=roll),
        'align_mean': lambda: align_loss(f, o, 0.5, rollout_latents=roll,
                                         depth_reduce='mean'),
        'align_halving': lambda: align_loss(f, o, 0.5, rollout_latents=roll,
                                            depth_reduce='halving'),
        'align_pad_sum': lambda: align_loss(f, o, 0.5, rollout_latents=roll,
                                            pad_patches=pad),
        'align_l2_raw_sum': lambda: align_loss(f, o, 0.5,
                                               rollout_latents=roll,
                                               kind='l2_raw'),
        'cpc_aux_sum': lambda: cpc_infonce_aux_loss(f, o, w1,
                                                    rollout_latents=roll),
        'cpc_aux_pad_sum': lambda: cpc_infonce_aux_loss(
            f, o, w1, rollout_latents=roll, pad_patches=pad),
        'cpc_all_sum': lambda: cpc_infonce_all_loss(f, o, w1,
                                                    rollout_latents=roll),
        'pred_pad_sum': lambda: pred_term(f, o, _spec(_SPLIT), pad,
                                          rollout_latents=roll),
    }


class TestTheDefaultPathIsUnchanged:

    def test_every_case_is_frozen(self):
        assert set(default_path_cases()) == set(BASE_VALUES)

    @pytest.mark.parametrize("name", sorted(BASE_VALUES))
    def test_the_value_of_the_base_commit(self, name):
        """Tolerance 1e-12 absolute, as #373's frozen values: any change of
        objective fails, and a different BLAS reduction order passes. The
        runs below compare the bytes."""
        value = float(default_path_cases()[name]().detach())
        assert value == pytest.approx(BASE_VALUES[name], rel=0, abs=1e-12)

    def test_the_cos_err_of_the_base_commit(self):
        f, o, _, roll = _latents(_REP_ONLY, seed=7, depth=K)
        assert rollout_cos_error(f, o, roll) == pytest.approx(
            BASE_COS_ERR, rel=0, abs=1e-12)


# Flags of a short CPU run at k = 3, as `run_short_training` gives them, on
# a window of 8 patches: depth 3 needs 5.
SHORT_RUN = (
    "--device", "cpu", "--total-steps", "3", "--batch-size", "2",
    "--lr", "1e-3", "--weight-decay", "0.1", "--mix-ratio", "1.0",
    "--synth-kind", "periodic", "--t-raw", "128", "--n-channels", "1",
    "--d-model", "32", "--n-heads", "2", "--num-layers", "1",
    "--num-encoder-layers", "1", "--log-every", "1", "--seed", "42",
    "--tau", "0.10", "--hf-repo", "none", "--hf-path", "none",
    "--train-rollout-depth", str(K))


def base_tree(tmp_path):
    """The src and the trainer of the base commit, or a skip without that
    history. The loss values above run on every checkout."""
    if subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e",
                       f"{BASE_COMMIT}^{{commit}}"],
                      capture_output=True).returncode:
        pytest.skip(f"{BASE_COMMIT} is not in this checkout's history")
    tree = tmp_path / BASE_COMMIT
    tree.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", BASE_COMMIT, "src",
         "experiments/2026-04-27_freq-embedding/scripts/train.py"],
        capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    return tree


def losses_csv(root, save_dir, *flags):
    """The bytes of the losses CSV of one short CPU run of the trainer in
    ``root``."""
    env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4")
    script = TRAIN_PY.relative_to(REPO_ROOT)
    save_dir.mkdir(parents=True)
    out = subprocess.run(
        [sys.executable, str(root / script), *SHORT_RUN, *flags,
         "--save-dir", str(save_dir), "--run-name", "r"],
        capture_output=True, text=True, env=env, cwd=str(save_dir),
        timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return (save_dir / "r_losses.csv").read_bytes()


@pytest.mark.parametrize("arm", [
    pytest.param(("--loss-shape", _REP_ONLY, "--align-loss-weight", "1.0"),
                 id="rep_only_align"),
    pytest.param(("--loss-shape", _SPLIT, "--align-loss-weight", "0.5",
                  "--skip-nan-samples", "--train-rollout-reduce", "mean"),
                 id="split_by_row_mean"),
    pytest.param(("--loss-shape", _XSHH_ALLT, "--no-main-contrastive-loss",
                  "--align-loss-weight", "1.0", "--cpc-infonce-weight",
                  "0.1"), id="align_standalone_cpc"),
])
def test_a_k3_run_writes_the_losses_csv_of_the_base_commit(tmp_path, arm):
    """The real trainer at k = 3 without the flag, three steps, from the base
    commit and from this tree. Every column of every row is the same, byte
    for byte: the loss, the terms and the cos_err_d0..d3 curves."""
    old = losses_csv(base_tree(tmp_path), tmp_path / "old", *arm)
    new = losses_csv(REPO_ROOT, tmp_path / "new", *arm)
    assert old.decode().count("\n") == 4
    assert new == old


# ---------------------------------------------------------------------------
# The CSV writes the columns of the kept depths only
# ---------------------------------------------------------------------------

def depth_columns(path):
    with open(path) as fh:
        row = next(csv.DictReader(fh))
    return {c: v for c, v in row.items() if c.startswith("cos_err")}


class TestTheCsvColumns:

    @staticmethod
    def _log(logger, cos_err_depths):
        logger.log(1, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 7, 8, False,
                   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                   cos_err_depths=cos_err_depths)
        logger.close()

    def test_the_kept_depths_get_a_column_each(self, train, tmp_path):
        path = str(tmp_path / "losses.csv")
        logger = train.CSVLogger(path, rollout_depth=K, depth_columns=[0, 1, 3])
        self._log(logger, [0.11, 0.22, None, 0.44])
        assert depth_columns(path) == {"cos_err_d0": "0.11",
                                       "cos_err_d1": "0.22",
                                       "cos_err_d3": "0.44"}

    def test_the_card_list_names_its_six_depths(self, train, tmp_path):
        args = flag_args([1, 2, 4, 8, 16, 32])
        train.resolve_rollout_distances(args)
        path = str(tmp_path / "losses.csv")
        logger = train.CSVLogger(path, rollout_depth=args.train_rollout_depth,
                                 depth_columns=train.kept_rollout_depths(args))
        errors = [None] * 32
        for j in (0, 1, 3, 7, 15, 31):
            errors[j] = j / 100
        self._log(logger, errors)
        assert depth_columns(path) == {f"cos_err_d{j}": str(j / 100)
                                       for j in (0, 1, 3, 7, 15, 31)}

    def test_without_the_flag_every_depth_keeps_its_column(self, train,
                                                           tmp_path):
        path = str(tmp_path / "losses.csv")
        logger = train.CSVLogger(
            path, rollout_depth=K,
            depth_columns=train.kept_rollout_depths(flag_args(None)))
        self._log(logger, [0.11, 0.22, 0.33, 0.44])
        assert list(depth_columns(path)) == [f"cos_err_d{j}"
                                             for j in range(K + 1)]

    def test_the_metric_gives_none_for_a_left_out_depth(self):
        f, o, _, roll = _latents(_REP_ONLY, seed=41, depth=K)
        every = rollout_cos_error(f, o, roll)
        kept = rollout_cos_error(f, o, keep(roll, KEPT))
        assert kept == [every[0], every[1], None, every[3]]


# ---------------------------------------------------------------------------
# Short runs of the real trainer
# ---------------------------------------------------------------------------

def first_row(tmp_path, name, *flags):
    return run_short_training(tmp_path, name, "--t-raw", "128", *flags)[0]


@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_a_run_with_the_distances_1_2_4_drops_the_copy_at_depth_2(
        tmp_path, reduce):
    """The run's own cell, rep_only with L_align, at k = 3 and at the
    distances 1, 2 and 4. Step 0 reads the same weights and the same batch
    (equal ``loss_tau_ref``). L_align at depth j is 2 x cos_err_dj, so:

    * sum: the k = 3 loss less the distances loss is 2 x cos_err_d2.
    * mean: the distances loss is l_rep + 2 x (d0 + d1 + d3) / 3.
    """
    arm = ("--loss-shape", _REP_ONLY, "--align-loss-weight", "1.0",
           "--train-rollout-reduce", reduce)
    full = first_row(tmp_path, f"k3_{reduce}", *arm,
                     "--train-rollout-depth", "3")
    kept = first_row(tmp_path, f"d124_{reduce}", *arm,
                     "--train-rollout-distances", "1,2,4")
    assert float(kept["loss_tau_ref"]) == float(full["loss_tau_ref"]), \
        "the two runs did not see the same batch"
    assert list(depth_columns_of(kept)) == ["cos_err_d0", "cos_err_d1",
                                            "cos_err_d3"]
    errors = {j: float(full[f"cos_err_d{j}"]) for j in range(K + 1)}
    for j in (0, 1, 3):
        assert float(kept[f"cos_err_d{j}"]) == errors[j]
    loss = float(kept["loss"])
    assert loss == loss and abs(loss) < float("inf")
    if reduce == "sum":
        assert float(full["loss"]) - loss == pytest.approx(2 * errors[2],
                                                           rel=1e-5)
    else:
        assert loss == pytest.approx(
            float(kept["l_rep"]) + 2 * (errors[0] + errors[1] + errors[3]) / 3,
            rel=1e-5)


def depth_columns_of(row):
    return {c: v for c, v in row.items() if c.startswith("cos_err")}


@pytest.mark.parametrize("arm", [
    pytest.param(("--loss-shape", _SPLIT, "--align-loss-weight", "0.5",
                  "--skip-nan-samples"), id="split_by_row"),
    pytest.param(("--loss-shape", _SPLIT, "--align-loss-weight", "0.5",
                  "--multi-patch-sizes", "8,16,32,64,128", "--t-raw", "1024",
                  "--rev-norm-kind", "meanstd"), id="patch_size_groups"),
])
def test_the_objective_by_row_trains_at_the_distances(tmp_path, arm):
    """The objective by row (#412): its forward, its fault search and the
    terms of each patch-size group read the same kept depths."""
    rows = run_short_training(tmp_path, "rows", *arm,
                              "--train-rollout-distances", "1,2,4")
    for row in rows:
        loss = float(row["loss"])
        assert loss == loss and abs(loss) < float("inf")
        assert list(depth_columns_of(row)) == ["cos_err_d0", "cos_err_d1",
                                               "cos_err_d3"]
