"""Tests for #373 — `--train-rollout-depth k` (config key `train_rollout_depth`).

The flag duplicates every loss term that ties the forecaster output `f` to the
encoder latent `h`. Copy `j` ties `f^(j)_t` to `h_{t+1+j}`, where `f^(j)` is the
forecaster applied to its own output `j` more times. `k = 0` is today's
objective.

What this file pins:

* **k = 0 is a no-op for every shape.** Frozen values (`K0_REFERENCE`),
  captured from the pre-#373 code, for every shape the dispatch accepts and
  for the modifier combinations the 14 cells of #373 use.
* **The shape list is read from the dispatch**, not hardcoded, so a shape
  added later cannot be missed by the sweep.
* **k = 1 runs on every shape**, returns a finite scalar, and equals the k = 0
  value plus a depth-1 term built the same way from `f^(1)`.
* **A depth-`j` copy holds no depth-0 tensor**: `L_pred^(1)` matches an
  independent formula built only from `f^(1)` and `h_{·+2}`, and does NOT
  match the same formula with either negative family left at depth 0.
* **Terms that carry no `f`** — `L_rep`, the h-only half of `mse` — enter the
  total once, at any k.
* **`cpc_multistep` / `cpc_multistep_cpcnegs` raise** and name themselves at
  any k > 0. No other shape raises.
* **The re-entry composes the operator the eval composes**: one depth equals
  one token of `rollout_latent` on the same input.
* **The eval numerics do not change.** #373 changes the training objective
  only, so `rollout_latent` keeps the ambient precision it ran at before the
  forecaster stack moved into `TransformerBlock.forecaster_forward`, and the
  cached causal mask still follows the input's device.
* **The second positive of `add_pos_htft` shifts too** (#373 rule 3), against
  an independent formula.
* **`--subtract-contrastive-floor` at k > 0**, on the split shape (one floor
  per depth on the f-bearing side only) and on the h-only `rep_only` shape.
* **`rollout_cos_error`** — depth 0 reproduces the `ff` column, each deeper
  entry reads its own shifted window.
* **The trainer wiring**: the re-entry, the per-depth all-gather, the CSV
  header/row alignment, the depth-0 `loss_tau_ref`, and a 4-step end-to-end
  run at k = 2.
* **No path of the loss block trains at k = 0.** All three reach the depths,
  statically and through a run of the real trainer:
  `--no-main-contrastive-loss` (standalone `align_loss`), the main term, and
  the main term on a `rep_only` shape, where the whole depth contribution is
  the `L_align` add-on in the shared tail. `--shard-loss-on-batch` crosses
  all three. A run that trains at k = 0 still writes k + 1 plausible
  `cos_err_dj` curves, because the diagnostic reads the depth tensors and
  not the loss.
* **The guards**, each against its own message: a negative depth, a
  non-transformer forecaster, and a depth no term can consume — that last
  one on each of its three routes.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import src.loss as loss_module
from src.blocks import TransformerBlock
from src.forecasting_head import rollout_forecaster_latents, rollout_latent
from src.loss import (align_loss, contrastive_latent_loss,
                      cpc_infonce_all_loss, cpc_infonce_aux_loss,
                      rollout_depth_views)
from src.metrics import rollout_cos_error
from src.models import ConfigurableModel, compute_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = (REPO_ROOT / "experiments" / "2026-04-27_freq-embedding"
            / "scripts" / "train.py")

CPC_SHAPES = ('cpc_multistep', 'cpc_multistep_cpcnegs')

B, T, C, H, K_HEADS = 4, 6, 2, 8, 3

_NORMALIZED = loss_module._NORMALIZED_FORM_SHAPES
_SPLIT = 'cosine_similarity_batch_split_pred_rep'
_REP_ONLY = 'cosine_similarity_batch_rep_only'
_XSHH_ALLT = 'cosine_similarity_batch_full_hh_negs_xshh_allt'


def dispatch_loss_shapes():
    """Every `loss_shape` the dispatch in `contrastive_latent_loss` accepts.

    Read from the source of the dispatch itself — a shape added later shows
    up in this sweep without editing the test (#373 rule 4).
    """
    src = inspect.getsource(loss_module.contrastive_latent_loss)
    pattern = re.compile(
        r"train_config\.get\('loss_shape'\)[\s\\]*(?:==|in)[\s\\]*"
        r"(\([^)]*\)|'[^']*')", re.S)
    shapes = []
    for match in pattern.finditer(src):
        shapes += re.findall(r"'([^']+)'", match.group(1))
    return sorted(set(shapes))


ALL_SHAPES = dispatch_loss_shapes()
FOUR_D_SHAPES = [s for s in ALL_SHAPES if s not in CPC_SHAPES]


def _spec(shape, **overrides):
    tc = {
        "contrastive_divergence_temperature": 0.1,
        "contrastive_latent_noise": None,
        "loss_shape": shape,
        "contrastive_latent_delay": 0,
    }
    tc.update(overrides)
    return SimpleNamespace(train_configuration=tc)


def _latents(shape, seed=0, depth=0, requires_grad=False):
    """(f, o, teacher_o, [f^(1)..f^(depth)]) — independent random latents.

    The rollout copies are independent tensors: the loss must treat whatever
    it is handed as `f^(j)`, and independence makes a depth-`j` term that
    leaked a depth-0 tensor visible.
    """
    g = torch.Generator().manual_seed(seed)

    def _draw(*dims):
        x = torch.randn(*dims, generator=g, dtype=torch.float64)
        return x.requires_grad_(requires_grad)

    if shape in CPC_SHAPES:
        f = _draw(B, T, C, K_HEADS, H)
    else:
        f = _draw(B, T, C, H)
    o = _draw(B, T, C, H)
    teacher_o = _draw(B, T, C, H)
    rollouts = [_draw(B, T, C, H) for _ in range(depth)]
    return f, o, teacher_o, rollouts


# Modifier combinations. `base` and `pos_in_denom` sweep the whole dispatch.
# the rest are the combinations #373's 14 cells train with.
K0_CONFIGS = {
    'base': dict(cfg={}, teacher=False),
    'pos_in_denom': dict(
        cfg={'include_positive_in_denominator': True}, teacher=False),
    # arm4 / cell B5: xshh_allt + pos-in-denominator + floor + moco-negatives.
    'arm4': dict(
        cfg={'include_positive_in_denominator': True,
             'subtract_contrastive_floor': True,
             'moco_negatives': True}, teacher=True),
    # arm5 / arm6_v2 cells: L_align on the teacher target.
    'align_teacher': dict(
        cfg={'align_loss_weight': 1.0, 'align_target': 'teacher'}, teacher=True),
    # arm6_v2: L_align (student target) + MoCo keys on L_rep.
    'align_moco_rep': dict(
        cfg={'align_loss_weight': 1.0, 'moco_rep_keys': True}, teacher=True),
    # The floor on the SPLIT shape: two terms, two floors, and only the
    # f-bearing one repeats per depth.
    'split_floor': dict(cfg={'subtract_contrastive_floor': True},
                        teacher=False),
    # #336 / #379 knobs that move where f and h sit.
    'stopgrad_pos': dict(cfg={'stopgrad_positive_h': True}, teacher=False),
    'tau_rep': dict(
        cfg={'contrastive_divergence_temperature_rep': 0.5,
             'pred_loss_weight': 0.7, 'rep_loss_weight': 0.3}, teacher=False),
}


def k0_cases():
    """(shape, config name) pairs the loss accepts — the sweep of this file."""
    allowed = {
        'base': ALL_SHAPES,
        'pos_in_denom': _NORMALIZED,
        'arm4': (_XSHH_ALLT,),
        'align_teacher': (_SPLIT, _REP_ONLY, _XSHH_ALLT),
        'align_moco_rep': (_SPLIT, _REP_ONLY),
        'split_floor': (_SPLIT,),
        'stopgrad_pos': (_SPLIT, _REP_ONLY, _XSHH_ALLT),
        'tau_rep': (_SPLIT, _REP_ONLY),
    }
    return [(shape, name) for shape in ALL_SHAPES for name in K0_CONFIGS
            if shape in allowed[name]]


def _call(shape, config_name, depth=0, latents=None, depth_index=0, **extra):
    """Run `contrastive_latent_loss` for one (shape, config) case at depth k.

    `depth_index=j` enters the function the way a depth copy does: the depth
    loop re-enters with `train_rollout_depth=0, depth_index=j` on shifted
    views, so this is the ONLY way a test reaches the copy's branches. A
    `depth=k` call cannot: the depth-0 body runs first, so a shape that
    raises there never reaches its own copies.
    """
    case = K0_CONFIGS[config_name]
    f, o, teacher_o, rollouts = latents
    cfg = dict(case['cfg'])
    cfg.update(extra)
    if depth:
        cfg['train_rollout_depth'] = depth
    return contrastive_latent_loss(
        (f, o), validation=False, spec=_spec(shape, **cfg),
        teacher_original_latent=teacher_o if case['teacher'] else None,
        rollout_latents=rollouts[:depth] if depth else None,
        depth_index=depth_index)


# k = 0 values captured from the pre-#373 code with the latents above. The
# published k = 0 numbers of #373's 14 cells stay a valid baseline only while
# these hold.
K0_REFERENCE = {
    ('cosine_similarity', 'base'): 7.503063996374145,
    ('cosine_similarity_batch', 'base'): 7.750364922149427,
    ('cosine_similarity_batch', 'pos_in_denom'): 7.779780710748372,
    ('cosine_similarity_batch_add_f_cross_negs', 'base'): 8.630952338083508,
    ('cosine_similarity_batch_add_neg_htft', 'base'): 7.821465996253089,
    ('cosine_similarity_batch_add_pos_htft', 'base'): 5.3276494884796195,
    ('cosine_similarity_batch_add_pos_htft_add_f_cross_negs', 'base'): 6.2082369044137,
    ('cosine_similarity_batch_add_skip_f_negs', 'base'): 7.810566788837844,
    ('cosine_similarity_batch_full_ff_negs', 'base'): 7.847115248149441,
    ('cosine_similarity_batch_full_ff_negs', 'pos_in_denom'): 7.87143925647994,
    ('cosine_similarity_batch_full_fh_hh_ff_negs', 'base'): 8.63826962004852,
    ('cosine_similarity_batch_full_fh_hh_ff_negs', 'pos_in_denom'): 8.647815359774409,
    ('cosine_similarity_batch_full_fh_hh_negs', 'base'): 8.525172529070025,
    ('cosine_similarity_batch_full_fh_hh_negs', 'pos_in_denom'): 8.535828073244787,
    ('cosine_similarity_batch_full_fh_negs', 'base'): 8.177210941132893,
    ('cosine_similarity_batch_full_fh_negs', 'pos_in_denom'): 8.192938934331874,
    ('cosine_similarity_batch_full_hh_ff_negs', 'base'): 8.270429949551733,
    ('cosine_similarity_batch_full_hh_ff_negs', 'pos_in_denom'): 8.284335243079703,
    ('cosine_similarity_batch_full_hh_negs', 'base'): 8.07401288800395,
    ('cosine_similarity_batch_full_hh_negs', 'pos_in_denom'): 8.091294887922343,
    ('cosine_similarity_batch_full_hh_negs_xbfree', 'base'): 8.22800887332544,
    ('cosine_similarity_batch_full_hh_negs_xbfree', 'pos_in_denom'): 8.238587204174518,
    ('cosine_similarity_batch_full_hh_negs_xshh', 'base'): 7.906023867903142,
    ('cosine_similarity_batch_full_hh_negs_xshh', 'pos_in_denom'): 7.929531269731005,
    ('cosine_similarity_batch_full_hh_negs_xshh_allt', 'base'): 9.111158849872979,
    ('cosine_similarity_batch_full_hh_negs_xshh_allt', 'pos_in_denom'): 9.120692816171422,
    ('cosine_similarity_batch_full_hh_negs_xshh_allt', 'arm4'): 9.35239321298506,
    ('cosine_similarity_batch_full_hh_negs_xshh_allt', 'align_teacher'): 11.269468193057037,
    ('cosine_similarity_batch_full_hh_negs_xshh_allt', 'stopgrad_pos'): 9.111158849872979,
    ('cosine_similarity_batch_no_time_neg', 'base'): 6.537327286188794,
    ('cosine_similarity_batch_no_time_neg', 'pos_in_denom'): 6.639973139337556,
    ('cosine_similarity_batch_rep_only', 'base'): 8.989816132750237,
    ('cosine_similarity_batch_rep_only', 'align_teacher'): 11.017597477176613,
    ('cosine_similarity_batch_rep_only', 'align_moco_rep'): 9.97496984116677,
    ('cosine_similarity_batch_rep_only', 'stopgrad_pos'): 8.989816132750237,
    # #409 scaled this entry by the `rep_loss_weight` of the `tau_rep`
    # case, 0.3. Before #409 the `rep_only` shape read no rep weight,
    # so the entry held the unweighted 4.865525677822549. No published
    # run moves: every `rep_only` command line takes the default 1.0.
    ('cosine_similarity_batch_rep_only', 'tau_rep'): 1.4596577033467646,
    ('cosine_similarity_batch_split_pred_rep', 'base'): 15.73789835105975,
    ('cosine_similarity_batch_split_pred_rep', 'align_teacher'): 17.880922522184363,
    ('cosine_similarity_batch_split_pred_rep', 'align_moco_rep'): 16.83829488617452,
    ('cosine_similarity_batch_split_pred_rep', 'split_floor'): 11.172642572978022,
    ('cosine_similarity_batch_split_pred_rep', 'stopgrad_pos'): 15.73789835105975,
    ('cosine_similarity_batch_split_pred_rep', 'tau_rep'): 6.183315256163423,
    ('cosine_similarity_batch_square', 'base'): 8.243637551311775,
    ('cosine_similarity_batch_square', 'pos_in_denom'): 8.255603572073548,
    ('cosine_similarity_old', 'base'): 5.3032521654212035,
    ('cpc_multistep', 'base'): 8.016385700118915,
    ('cpc_multistep_cpcnegs', 'base'): 5.246147048327489,
    ('mse', 'base'): -0.11020670711760516,
}


class TestDispatchCoverage:

    def test_every_documented_shape_is_swept(self):
        """The sweep reads the dispatch, so #373's shape list is covered."""
        assert len(ALL_SHAPES) >= 24
        for shape in ('cosine_similarity_old', 'mse', 'cpc_multistep',
                      _SPLIT, _REP_ONLY, _XSHH_ALLT):
            assert shape in ALL_SHAPES

    def test_every_swept_case_is_frozen(self):
        assert set(k0_cases()) == set(K0_REFERENCE)


class TestDepthZeroIsUnchanged:

    @pytest.mark.parametrize('shape,config_name', k0_cases())
    def test_frozen_value(self, shape, config_name):
        """k = 0 reproduces the pre-#373 value.

        Tolerance 1e-12 absolute on values of order 10 — tight enough that
        any change of objective fails, loose enough to survive a different
        BLAS reduction order. The in-process comparisons below are exact.
        """
        latents = _latents(shape)
        assert float(_call(shape, config_name, latents=latents)) == \
            pytest.approx(K0_REFERENCE[(shape, config_name)], rel=0, abs=1e-12)

    @pytest.mark.parametrize('shape', ALL_SHAPES)
    def test_zero_depth_key_is_a_no_op(self, shape):
        """An explicit `train_rollout_depth = 0` changes nothing."""
        f, o, _teacher, _roll = _latents(shape, seed=3)
        plain = contrastive_latent_loss((f, o), False, _spec(shape))
        keyed = contrastive_latent_loss(
            (f, o), False, _spec(shape, train_rollout_depth=0))
        assert float(plain) == float(keyed)


def _expected_depth_term(shape, config_name, latents, depth):
    """The depth-`j` term, rebuilt independently of the depth loop.

    Same call on the depth-`j` views (`f^(j)` against `h` shifted by `j`),
    minus the terms that carry no `f` — `L_rep` in the split shape, the whole
    `rep_only` shape, and `mse`'s `− mse(h_t, h_{t+1})` half — which enter the
    total once and so must NOT reappear here.
    """
    case = K0_CONFIGS[config_name]
    f, o, teacher_o, rollouts = latents
    teacher = teacher_o if case['teacher'] else None
    f_view, o_view, teacher_view = rollout_depth_views(
        rollouts[depth - 1], o, depth, teacher)
    if shape == _REP_ONLY:
        # h-anchored end to end: only the L_align add-on this arm pairs it
        # with is f-bearing, so only that repeats.
        weight = case['cfg'].get('align_loss_weight', 0.0)
        if not weight:
            return torch.zeros((), dtype=torch.float64)
        target = (teacher_view
                  if case['cfg'].get('align_target') == 'teacher' else None)
        return align_loss(f_view, o_view, weight, target_latent=target)
    if shape == 'mse':
        return F.mse_loss(o_view[:, 1:], f_view[:, :-1])
    cfg = dict(case['cfg'])
    if shape == _SPLIT:
        cfg['rep_loss_weight'] = 0.0
    term = contrastive_latent_loss(
        (f_view, o_view), validation=False, spec=_spec(shape, **cfg),
        teacher_original_latent=teacher_view)
    if shape == _SPLIT and case['cfg'].get('subtract_contrastive_floor'):
        # The depth-0 call re-based L_rep by its own floor once. This
        # rebuild is a plain depth-0 call, so it subtracted that constant a
        # second time — add it back, because a depth copy carries L_pred
        # only. (L_pred's floor has no T in it, so it is the same constant
        # on the shorter depth-j window.)
        _f_pred, f_rep = loss_module._split_pred_rep_floors(
            0.1, B, o_view.shape[1], C)
        term = term + float(f_rep)
    return term


class TestDepthSumsFBearingTermsOnly:

    @pytest.mark.parametrize('shape,config_name',
                             [c for c in k0_cases() if c[0] not in CPC_SHAPES])
    def test_depth_one_is_k0_plus_one_copy(self, shape, config_name):
        """L(k=1) = L(k=0) + L^(1), and L^(1) holds no f-free term."""
        latents = _latents(shape, depth=1)
        k0 = _call(shape, config_name, latents=latents)
        k1 = _call(shape, config_name, depth=1, latents=latents)
        expected = k0 + _expected_depth_term(shape, config_name, latents, 1)
        assert torch.isfinite(k1)
        assert float(k1) == pytest.approx(float(expected), rel=0, abs=1e-12)

    @pytest.mark.parametrize('shape,config_name',
                             [c for c in k0_cases() if c[0] not in CPC_SHAPES])
    def test_depth_three_sums_every_copy(self, shape, config_name):
        """L(k=3) = L(k=0) + Σ_{j=1..3} L^(j) — a sum, not a mean."""
        latents = _latents(shape, depth=3)
        expected = _call(shape, config_name, latents=latents)
        for depth in (1, 2, 3):
            expected = expected + _expected_depth_term(
                shape, config_name, latents, depth)
        got = _call(shape, config_name, depth=3, latents=latents)
        assert float(got) == pytest.approx(float(expected), rel=0, abs=1e-12)

    def test_f_free_shape_is_a_no_op_at_depth(self):
        """`rep_only` carries no f: every depth adds exactly nothing."""
        latents = _latents(_REP_ONLY, depth=3)
        k0 = _call(_REP_ONLY, 'base', latents=latents)
        k3 = _call(_REP_ONLY, 'base', depth=3, latents=latents)
        assert float(k3) == float(k0)

    def test_l_rep_is_counted_once(self):
        """The split shape's L_rep must not scale with k."""
        latents = _latents(_SPLIT, depth=2)
        pred_only = dict(rep_loss_weight=0.0)
        rep_only = dict(pred_loss_weight=0.0)
        rep_k0 = _call(_SPLIT, 'base', latents=latents, **rep_only)
        rep_k2 = _call(_SPLIT, 'base', depth=2, latents=latents, **rep_only)
        assert float(rep_k2) == float(rep_k0)
        # ... while the f-bearing L_pred does grow with the depth.
        pred_k0 = _call(_SPLIT, 'base', latents=latents, **pred_only)
        pred_k2 = _call(_SPLIT, 'base', depth=2, latents=latents, **pred_only)
        assert float(pred_k2) > float(pred_k0)

    def test_mse_h_only_half_is_counted_once(self):
        """`mse` duplicates mse(h_{t+1+j}, f^(j)_t), not − mse(h_t, h_{t+1})."""
        latents = _latents('mse', depth=1)
        f, o, _teacher, rollouts = latents
        k0 = _call('mse', 'base', latents=latents)
        k1 = _call('mse', 'base', depth=1, latents=latents)
        f_view, o_view, _ = rollout_depth_views(rollouts[0], o, 1)
        assert float(k1 - k0) == pytest.approx(
            float(F.mse_loss(o_view[:, 1:], f_view[:, :-1])), rel=0, abs=1e-12)


class TestAnchorRange:

    def test_depth_views_shift_every_h_index(self):
        """f^(j)_t against h_{t+1+j}, anchors t = 0..T-2-j."""
        f, o, teacher_o, rollouts = _latents(_SPLIT, depth=2)
        f_view, o_view, t_view = rollout_depth_views(
            rollouts[1], o, 2, teacher_o)
        assert f_view.shape[1] == T - 2 and o_view.shape[1] == T - 2
        assert torch.equal(f_view, rollouts[1][:, :T - 2])   # f^(2)_t
        assert torch.equal(o_view[:, 1:], o[:, 3:])          # h_{t+1+2}
        assert torch.equal(o_view[:, :-1], o[:, 2:T - 1])    # h_{t+2}
        assert torch.equal(t_view, teacher_o[:, 2:])


class TestDepthCopyHoldsNoDepthZeroTensor:
    """#373 rule 2, on the split shape's L_pred — the term where `f` sits in
    the numerator AND in every denominator family."""

    @staticmethod
    def _l_pred_reference(f_j, o, depth, tau=0.1, teacher_o=None,
                          zy_from=None, xb_from=None):
        """L_pred^(j) written out from the issue's worked example.

        `zy_from` / `xb_from` override the f used by one negative family, so a
        depth-0 leak in either can be detected by construction.
        """
        neg_inf = float('-inf')
        f_view, o_view, t_view = rollout_depth_views(f_j, o, depth, teacher_o)
        fore = F.normalize(f_view, p=2, dim=-1)
        orig = F.normalize(o_view, p=2, dim=-1)
        pos_src = orig if t_view is None else F.normalize(t_view, p=2, dim=-1)
        zy_src = fore if zy_from is None else F.normalize(
            zy_from[:, :o.shape[1] - depth], p=2, dim=-1)
        xb_src = fore if xb_from is None else F.normalize(
            xb_from[:, :o.shape[1] - depth], p=2, dim=-1)
        f_t, h_next = fore[:, :-1], pos_src[:, 1:]           # f^(j)_t, h_{t+1+j}
        log_pos = (f_t * h_next).sum(-1) / tau
        sims_zy = (zy_src[:, 1:].unsqueeze(3) * zy_src[:, :-1].unsqueeze(2)
                   ).sum(-1)                                  # f^(j)_{t+1} ↔ f^(j)_t
        log_neg_zy = torch.logsumexp(sims_zy / tau, dim=2)
        h_key = orig[:, 1:]                                   # h'_{t+1+j}
        sims_xb = torch.matmul(
            xb_src[:, :-1].permute(1, 2, 0, 3),
            h_key.permute(1, 2, 0, 3).transpose(-2, -1)).permute(2, 3, 0, 1)
        batch_size = o.shape[0]
        mask = ~torch.eye(batch_size, dtype=torch.bool).view(
            batch_size, batch_size, 1, 1)
        log_neg_xb = torch.logsumexp(
            (sims_xb / tau).masked_fill(~mask, neg_inf), dim=1)
        per_anchor = torch.logsumexp(
            torch.stack([log_neg_zy, log_neg_xb], dim=0), dim=0)
        log_neg_total = torch.logsumexp(per_anchor, dim=0, keepdim=True)
        log_denom = torch.logsumexp(
            torch.stack([log_pos, log_neg_total.expand_as(log_pos)], dim=0),
            dim=0)
        return (log_denom - log_pos).mean()

    def test_depth_one_l_pred_matches_the_issue_formula(self):
        latents = _latents(_SPLIT, depth=1)
        f, o, _teacher, rollouts = latents
        rep_off = dict(rep_loss_weight=0.0)
        k0 = _call(_SPLIT, 'base', latents=latents, **rep_off)
        k1 = _call(_SPLIT, 'base', depth=1, latents=latents, **rep_off)
        want = self._l_pred_reference(rollouts[0], o, 1)
        assert float(k1 - k0) == pytest.approx(float(want), rel=0, abs=1e-12)

    @pytest.mark.parametrize('leak', ['zy_from', 'xb_from'])
    def test_a_depth_zero_negative_family_would_be_caught(self, leak):
        """Both f-anchored families rebuild at depth 1: keeping either at
        depth 0 changes the value, so the formula above pins both."""
        latents = _latents(_SPLIT, depth=1)
        f, o, _teacher, rollouts = latents
        leaked = self._l_pred_reference(rollouts[0], o, 1, **{leak: f})
        clean = self._l_pred_reference(rollouts[0], o, 1)
        assert abs(float(leaked) - float(clean)) > 1e-6
        rep_off = dict(rep_loss_weight=0.0)
        k0 = _call(_SPLIT, 'base', latents=latents, **rep_off)
        k1 = _call(_SPLIT, 'base', depth=1, latents=latents, **rep_off)
        assert abs(float(k1 - k0) - float(leaked)) > 1e-6

    def test_perturbing_f1_alone_moves_the_depth_one_negatives(self):
        """f^(1) at the last window step feeds ONLY the adjacent f↔f family.
        perturbing it must move the loss."""
        latents = _latents(_SPLIT, depth=1)
        _f, _o, _teacher, rollouts = latents
        base = _call(_SPLIT, 'base', depth=1, latents=latents,
                     rep_loss_weight=0.0)
        bumped = rollouts[0].clone()
        bumped[:, T - 2] += 1.0
        moved = _call(_SPLIT, 'base', depth=1, rep_loss_weight=0.0,
                      latents=(_f, _o, _teacher, [bumped]))
        assert abs(float(moved) - float(base)) > 1e-6

    def test_moco_keys_shift_with_the_depth(self):
        """With --moco-negatives the cross-batch key is the teacher's
        h^T_{t+1+j}, not h^T_{t+1}."""
        latents = _latents(_XSHH_ALLT, depth=1)
        f, o, teacher_o, rollouts = latents
        k0 = _call(_XSHH_ALLT, 'arm4', latents=latents)
        k1 = _call(_XSHH_ALLT, 'arm4', depth=1, latents=latents)
        shifted = _expected_depth_term(_XSHH_ALLT, 'arm4', latents, 1)
        assert float(k1 - k0) == pytest.approx(float(shifted), rel=0, abs=1e-12)
        # The same term with the teacher key left at t+1 differs.
        unshifted_teacher = torch.cat(
            [teacher_o[:, :1], teacher_o[:, :-1]], dim=1)
        f_view, o_view, _ = rollout_depth_views(rollouts[0], o, 1)
        stale = contrastive_latent_loss(
            (f_view, o_view), False,
            _spec(_XSHH_ALLT, **K0_CONFIGS['arm4']['cfg']),
            teacher_original_latent=unshifted_teacher[:, 1:])
        assert abs(float(stale) - float(shifted)) > 1e-6


class TestFloorStaysGradientNeutral:

    def test_floor_shifts_the_value_but_not_the_gradient(self):
        """--subtract-contrastive-floor at k > 0 stays a constant offset."""
        latents = _latents(_XSHH_ALLT, depth=3, requires_grad=True)
        f, o, teacher_o, rollouts = latents
        with_floor = _call(_XSHH_ALLT, 'arm4', depth=3, latents=latents)
        no_floor = _call(_XSHH_ALLT, 'arm4', depth=3, latents=latents,
                         subtract_contrastive_floor=False)
        g_with = torch.autograd.grad(with_floor, [f] + rollouts,
                                     retain_graph=True)
        g_without = torch.autograd.grad(no_floor, [f] + rollouts)
        for a, b in zip(g_with, g_without):
            assert torch.equal(a, b)
        assert float(with_floor) != float(no_floor)


class TestUnsupportedShapesRaise:

    @pytest.mark.parametrize('shape', CPC_SHAPES)
    def test_cpc_multistep_raises_and_names_itself(self, shape):
        latents = _latents(shape, depth=1)
        f, o, _teacher, rollouts = latents
        with pytest.raises(NotImplementedError) as excinfo:
            contrastive_latent_loss(
                (f, o), False, _spec(shape, train_rollout_depth=1),
                rollout_latents=rollouts)
        assert shape in str(excinfo.value)
        assert 'train_rollout_depth' in str(excinfo.value)

    @pytest.mark.parametrize('shape', FOUR_D_SHAPES)
    def test_no_other_shape_raises(self, shape):
        latents = _latents(shape, depth=1)
        assert torch.isfinite(_call(shape, 'base', depth=1, latents=latents))


class TestDepthArgumentGuards:

    def test_depth_without_rollout_latents_raises(self):
        f, o, _teacher, _roll = _latents(_SPLIT)
        with pytest.raises(ValueError, match='train_rollout_depth'):
            contrastive_latent_loss(
                (f, o), False, _spec(_SPLIT, train_rollout_depth=2))

    def test_length_mismatch_raises(self):
        f, o, _teacher, rollouts = _latents(_SPLIT, depth=1)
        with pytest.raises(ValueError, match='train_rollout_depth'):
            contrastive_latent_loss(
                (f, o), False, _spec(_SPLIT, train_rollout_depth=2),
                rollout_latents=rollouts)

    def test_rollout_latents_without_depth_raises(self):
        f, o, _teacher, rollouts = _latents(_SPLIT, depth=1)
        with pytest.raises(ValueError, match='train_rollout_depth'):
            contrastive_latent_loss(
                (f, o), False, _spec(_SPLIT), rollout_latents=rollouts)

    def test_explicit_override_beats_the_config_key(self):
        """The `loss_tau_ref` diagnostic forces depth 0 on a k > 0 run."""
        latents = _latents(_SPLIT, depth=2)
        f, o, _teacher, _roll = latents
        ref = contrastive_latent_loss(
            (f, o), False, _spec(_SPLIT, train_rollout_depth=2),
            train_rollout_depth=0)
        assert float(ref) == float(
            contrastive_latent_loss((f, o), False, _spec(_SPLIT)))

    def test_depth_past_the_sequence_raises(self):
        f, o, _teacher, rollouts = _latents(_SPLIT, depth=1)
        with pytest.raises(ValueError, match='train_rollout_depth'):
            contrastive_latent_loss(
                (f[:, :2], o[:, :2]), False,
                _spec(_SPLIT, train_rollout_depth=1),
                rollout_latents=[rollouts[0][:, :2]])


class TestAuxiliaryFBearingTerms:

    def test_align_loss_sums_the_depths(self):
        _f, o, teacher_o, rollouts = _latents(_SPLIT, depth=2)
        f = _f
        total = align_loss(f, o, weight=0.5, rollout_latents=rollouts)
        want = align_loss(f, o, weight=0.5)
        for depth in (1, 2):
            f_view, o_view, _ = rollout_depth_views(
                rollouts[depth - 1], o, depth)
            want = want + align_loss(f_view, o_view, weight=0.5)
        assert float(total) == pytest.approx(float(want), rel=0, abs=1e-12)

    def test_align_loss_teacher_target_shifts_too(self):
        f, o, teacher_o, rollouts = _latents(_SPLIT, depth=1)
        total = align_loss(f, o, target_latent=teacher_o,
                           rollout_latents=rollouts)
        f_view, o_view, t_view = rollout_depth_views(
            rollouts[0], o, 1, teacher_o)
        want = (align_loss(f, o, target_latent=teacher_o)
                + align_loss(f_view, o_view, target_latent=t_view))
        assert float(total) == pytest.approx(float(want), rel=0, abs=1e-12)

    @pytest.mark.parametrize('fn', [cpc_infonce_aux_loss, cpc_infonce_all_loss])
    def test_cpc_infonce_sums_the_depths(self, fn):
        f, o, _teacher, rollouts = _latents(_SPLIT, depth=2)
        w1 = torch.nn.Linear(H, H, bias=False).double()
        total = fn(f, o, w1, rollout_latents=rollouts)
        want = fn(f, o, w1)
        for depth in (1, 2):
            f_view, o_view, _ = rollout_depth_views(
                rollouts[depth - 1], o, depth)
            want = want + fn(f_view, o_view, w1)
        assert float(total) == pytest.approx(float(want), rel=0, abs=1e-12)


def _tiny_backbone(seed=0, num_layers=2, num_encoder_layers=1):
    """fp32 throughout: the forecaster's last layer runs in fp32 by design."""
    torch.manual_seed(seed)
    model = ConfigurableModel(
        C=1, H=8, W=4, encoder_type='mlp', intermediate_dim=8,
        num_layers=num_layers, nhead=2, ffn_mult=1, dropout=0.0,
        num_encoder_layers=num_encoder_layers, rev_norm_span=None,
        rev_norm_kind='none', freq_emb_dim=0, forecaster_d_model=4,
        forecaster_n_heads=2)
    return model.eval()


class TestReEntryIsTheEvalOperator:
    """#373 rule 5 — one depth of re-entry equals one token of the eval
    rollout on the same input."""

    def test_one_depth_equals_one_rollout_token(self):
        model = _tiny_backbone()
        seq = torch.randn(3, 7, 8)                          # (B*C, T, H)
        with torch.no_grad():
            re_entry = model.transformer.forecaster_forward(seq)
        token = rollout_latent(model, seq, 1)
        assert torch.equal(re_entry[:, -1], token[:, 0])

    def test_rollout_forecaster_latents_uses_that_operator(self):
        model = _tiny_backbone()
        f0 = torch.randn(3, 7, 1, 8)                        # (B, T, C, H)
        depths = rollout_forecaster_latents(model, f0, 2)
        assert len(depths) == 2
        assert [d.shape for d in depths] == [f0.shape, f0.shape]
        f0_bc = f0.permute(0, 2, 1, 3).reshape(3, 7, 8)
        assert torch.equal(depths[0][:, -1, 0], rollout_latent(model, f0_bc, 1)[:, 0])
        f1_bc = depths[0].detach().permute(0, 2, 1, 3).reshape(3, 7, 8)
        assert torch.equal(depths[1][:, -1, 0], rollout_latent(model, f1_bc, 1)[:, 0])

    def test_gradient_flows_through_the_chain(self):
        """No detach between passes — f is shaped as an input as well as an
        output (#373 decision)."""
        model = _tiny_backbone()
        f0 = torch.randn(3, 7, 1, 8, requires_grad=True)
        depths = rollout_forecaster_latents(model, f0, 2)
        grad, = torch.autograd.grad(depths[1].sum(), f0)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def _forecaster_reference(model, seq):
    """The forecaster stack written out, at the ambient precision.

    The body `rollout_latent` ran before the stack moved into
    `TransformerBlock.forecaster_forward`: down proj → causal layers → up
    proj, with no cast and no fp32 tail. An independent reference for "the
    eval numerics did not change".
    """
    block = model.transformer
    n = seq.size(1)
    mask = torch.triu(torch.ones(n, n, device=seq.device), diagonal=1).bool()
    mask = mask.float().masked_fill(mask, float('-inf'))
    x = block.fcst_down_proj(seq)
    for layer in block.layers:
        x = layer(x, tgt_mask=mask, tgt_is_causal=True)
    return block.fcst_up_proj(x)


class TestEvalRolloutNumericsAreUnchanged:
    """#373 changes the TRAINING objective only.

    The forecaster stack moved into `TransformerBlock.forecaster_forward`
    so the training forward, the eval rollout and the rollout depth compose
    one operator. The training forward's precision policy — last layer and
    output forced to fp32 — must NOT travel to the eval rollout, which has
    always run every layer at the ambient precision. In fp32 the two are the
    same arithmetic, so a bf16 model is what tells them apart.
    """

    def test_rollout_latent_keeps_the_ambient_precision(self):
        model = _tiny_backbone().to(torch.bfloat16)
        seq = torch.randn(3, 7, 8, dtype=torch.bfloat16)
        token = rollout_latent(model, seq, 1)
        assert token.dtype == torch.bfloat16
        with torch.no_grad():
            want = _forecaster_reference(model, seq)[:, -1:]
        assert torch.equal(token, want)

    def test_a_multi_token_rollout_matches_the_reference(self):
        model = _tiny_backbone().to(torch.bfloat16)
        seq = torch.randn(3, 7, 8, dtype=torch.bfloat16)
        got = rollout_latent(model, seq, 3)
        with torch.no_grad():
            cur, want = seq, []
            for _ in range(3):
                token = _forecaster_reference(model, cur)[:, -1:]
                want.append(token)
                cur = torch.cat([cur, token], dim=1)
        assert torch.equal(got, torch.cat(want, dim=1))

    def test_each_caller_composes_its_own_precision_policy(self, monkeypatch):
        """One operator, two precision policies — pinned so they cannot swap.

        The eval rollout runs the stack at the ambient precision
        (`fp32_tail=False`); the training forward and the rollout depth force
        the fp32 tail, which is what the loss has always read.

        The spy binds against the REAL signature, so a caller that passes
        nothing records `forecaster_forward`'s own default. A spy that
        hardcoded `True` would keep passing if the signature default flipped
        to False, while the training path silently lost its fp32 tail.
        """
        seen = []
        original = TransformerBlock.forecaster_forward
        signature = inspect.signature(original)

        def spy(self, x, *args, **kwargs):
            bound = signature.bind(self, x, *args, **kwargs)
            bound.apply_defaults()
            seen.append(bound.arguments['fp32_tail'])
            return original(self, x, *args, **kwargs)

        monkeypatch.setattr(TransformerBlock, 'forecaster_forward', spy)
        model = _tiny_backbone()
        rollout_latent(model, torch.randn(3, 7, 8), 2)
        assert seen == [False, False]
        seen.clear()
        rollout_forecaster_latents(model, torch.randn(3, 7, 1, 8), 2)
        assert seen == [True, True]
        seen.clear()
        model(torch.randn(3, 28, 1))
        assert seen == [True]

    def test_the_two_policies_agree_in_fp32(self):
        """Same operator: the policies differ only in where a cast lands, so
        an fp32 model gives one answer either way."""
        model = _tiny_backbone()
        x = torch.randn(3, 7, 8)
        with torch.no_grad():
            assert torch.equal(model.transformer.forecaster_forward(x),
                               model.transformer.forecaster_forward(
                                   x, fp32_tail=False))


class TestCausalMaskFollowsTheInput:
    """The cached mask is keyed on the sequence length. The eval rollout used
    to build its own mask on the input's device every call, so the cache must
    also rebuild when the device changes — otherwise a mask cached by a
    forward on one device is handed to a rollout on another."""

    def test_a_cached_mask_on_another_device_is_rebuilt(self):
        model = _tiny_backbone()
        block = model.transformer
        x = torch.randn(2, 5, 8)
        block.causal_mask = block._generate_square_subsequent_mask(5).to('meta')
        assert block.causal_mask_for(x).device == x.device

    def test_the_rollout_survives_a_mask_cached_elsewhere(self):
        model = _tiny_backbone()
        seq = torch.randn(3, 7, 8)
        want = rollout_latent(model, seq, 2)
        model.transformer.causal_mask = \
            model.transformer._generate_square_subsequent_mask(7).to('meta')
        assert torch.equal(rollout_latent(model, seq, 2), want)


class TestTheEvalRolloutLeavesNoModuleState:
    """The rollout grows the sequence one token at a time, so it never hits
    the length-keyed cache. Writing it anyway would overwrite the backbone's
    mask k times per rollout and leave it at T + n — churn on one thread, a
    race on a backbone an eval harness shares between threads or streams.
    The body this function ran before #373 built a local mask and touched
    nothing, so `cache=False` restores that.
    """

    def test_a_rollout_leaves_the_cached_mask_alone(self):
        model = _tiny_backbone()
        model(torch.randn(3, 28, 1))                    # caches T = 7
        before = model.transformer.causal_mask
        assert before is not None and before.size(0) == 7
        rollout_latent(model, torch.randn(3, 7, 8), 3)
        assert model.transformer.causal_mask is before

    def test_a_rollout_on_a_fresh_backbone_caches_nothing(self):
        model = _tiny_backbone()
        rollout_latent(model, torch.randn(3, 7, 8), 3)
        assert model.transformer.causal_mask is None

    def test_the_uncached_mask_equals_the_cached_one(self):
        block = _tiny_backbone().transformer
        x = torch.randn(2, 5, 8)
        assert torch.equal(block.causal_mask_for(x, cache=False),
                           block.causal_mask_for(x))

    def test_the_rollout_output_is_unchanged(self):
        """`cache=False` picks where the mask is stored, not what it is."""
        model = _tiny_backbone()
        seq = torch.randn(3, 7, 8)
        want = rollout_latent(model, seq, 3)
        model.transformer.causal_mask = None
        assert torch.equal(rollout_latent(model, seq, 3), want)

    def test_the_training_depth_still_caches(self):
        """The rollout depth runs at the forward's fixed T, so the cache hits
        every call and stays worth writing."""
        model = _tiny_backbone()
        rollout_forecaster_latents(model, torch.randn(3, 7, 1, 8), 2)
        assert model.transformer.causal_mask is not None
        assert model.transformer.causal_mask.size(0) == 7


def checkpoint_counter(monkeypatch):
    """Count every `torch.utils.checkpoint.checkpoint` call from here on."""
    calls = []
    real = torch.utils.checkpoint.checkpoint

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, 'checkpoint', counting)
    return calls


class TestBackboneCheckpointGateIsOneDefinition:
    """#327's `BACKBONE_CKPT` gate — env-gated, training only, and blind to
    `x.requires_grad` (unlike the neighbouring `FCST_GRAD_CKPT`).

    Before #373 the encoder loop and the forecaster loop shared one local
    inside `forward`. The forecaster loop moved to `forecaster_forward`, so
    both now read `backbone_ckpt_enabled()` — one definition, and the two
    stacks cannot drift apart.
    """

    def _count(self, monkeypatch, model, x, no_grad=False):
        calls = checkpoint_counter(monkeypatch)
        if no_grad:
            with torch.no_grad():
                model(x)
        else:
            model(x)
        return len(calls)

    def test_both_stacks_checkpoint_their_non_last_layers(self, monkeypatch):
        monkeypatch.setenv('BACKBONE_CKPT', '1')
        monkeypatch.delenv('FCST_GRAD_CKPT', raising=False)
        model = _tiny_backbone(num_layers=3, num_encoder_layers=2).train()
        n = self._count(monkeypatch, model, torch.randn(3, 28, 1))
        assert n == (2 - 1) + (3 - 1)

    def test_the_forecaster_stack_does_not_gate_on_requires_grad(
            self, monkeypatch):
        """`FCST_GRAD_CKPT` gates on `x.requires_grad`; `BACKBONE_CKPT` never
        did. Under `no_grad` the forecaster input carries no grad, so a gate
        that copied its neighbour would drop the forecaster's 2 calls."""
        monkeypatch.setenv('BACKBONE_CKPT', '1')
        monkeypatch.delenv('FCST_GRAD_CKPT', raising=False)
        model = _tiny_backbone(num_layers=3, num_encoder_layers=2).train()
        n = self._count(monkeypatch, model, torch.randn(3, 28, 1),
                        no_grad=True)
        assert n == (2 - 1) + (3 - 1)

    @pytest.mark.parametrize('training,env,want', [
        (True, '1', True), (False, '1', False),
        (True, '0', False), (True, None, False)])
    def test_the_gate_reads_the_env_and_the_mode(self, monkeypatch, training,
                                                 env, want):
        block = _tiny_backbone().transformer
        if env is None:
            monkeypatch.delenv('BACKBONE_CKPT', raising=False)
        else:
            monkeypatch.setenv('BACKBONE_CKPT', env)
        block.train(training)
        assert block.backbone_ckpt_enabled() is want

    def test_the_gate_is_off_by_default(self, monkeypatch):
        monkeypatch.delenv('BACKBONE_CKPT', raising=False)
        model = _tiny_backbone(num_layers=3, num_encoder_layers=2).train()
        assert self._count(monkeypatch, model, torch.randn(3, 28, 1)) == 0

    def test_the_eval_rollout_never_checkpoints(self, monkeypatch):
        """Checkpointing is a training-memory knob: it trades stored
        activations for a backward recompute. The rollout runs under
        `no_grad` and its pre-#373 body ran every layer flat, so a backbone
        a caller left in `train()` must not start checkpointing an eval."""
        monkeypatch.setenv('BACKBONE_CKPT', '1')
        monkeypatch.delenv('FCST_GRAD_CKPT', raising=False)
        model = _tiny_backbone(num_layers=3, num_encoder_layers=2).train()
        calls = checkpoint_counter(monkeypatch)
        rollout_latent(model, torch.randn(3, 7, 8), 2)
        assert len(calls) == 0

    def test_the_training_depth_still_checkpoints(self, monkeypatch):
        """The other `forecaster_forward` caller keeps the knob: one call
        per non-last forecaster layer, per depth."""
        monkeypatch.setenv('BACKBONE_CKPT', '1')
        monkeypatch.delenv('FCST_GRAD_CKPT', raising=False)
        model = _tiny_backbone(num_layers=3, num_encoder_layers=2).train()
        calls = checkpoint_counter(monkeypatch)
        rollout_forecaster_latents(model, torch.randn(3, 7, 1, 8), 2)
        assert len(calls) == 2 * (3 - 1)

    def test_the_rollout_output_survives_the_gate(self, monkeypatch):
        """The gate picks where activations are stored, not what comes
        out."""
        model = _tiny_backbone(num_layers=3, num_encoder_layers=2).train()
        seq = torch.randn(3, 7, 8)
        monkeypatch.delenv('BACKBONE_CKPT', raising=False)
        want = rollout_latent(model, seq, 2)
        monkeypatch.setenv('BACKBONE_CKPT', '1')
        assert torch.equal(rollout_latent(model, seq, 2), want)


class TestBothCheckpointGatesFollowTheEvalPolicy:
    """`fp32_tail=False` is the eval policy: every layer flat, no cast, no
    gradient-checkpointing. BOTH gates follow it, so neither `BACKBONE_CKPT`
    nor `FCST_GRAD_CKPT` can put a backward recompute in an eval.

    `rollout_latent` runs under `no_grad`, where `FCST_GRAD_CKPT`'s
    `x.requires_grad` term already holds it off. That term is not the
    policy: a caller that composes `forecaster_forward(fp32_tail=False)`
    outside `no_grad` would checkpoint an eval pass. The gate is pinned at
    the policy, not at the caller.
    """

    @staticmethod
    def _count(monkeypatch, fp32_tail):
        calls = checkpoint_counter(monkeypatch)
        model = _tiny_backbone(num_layers=3, num_encoder_layers=2).train()
        seq = torch.randn(3, 7, 8, requires_grad=True)
        model.transformer.forecaster_forward(seq, fp32_tail=fp32_tail)
        return len(calls)

    def test_the_eval_policy_checkpoints_nothing(self, monkeypatch):
        monkeypatch.setenv('FCST_GRAD_CKPT', '1')
        monkeypatch.delenv('BACKBONE_CKPT', raising=False)
        assert self._count(monkeypatch, fp32_tail=False) == 0

    def test_the_training_policy_still_checkpoints(self, monkeypatch):
        """One call per non-last forecaster layer — the memory knob the 14
        cells run with."""
        monkeypatch.setenv('FCST_GRAD_CKPT', '1')
        monkeypatch.delenv('BACKBONE_CKPT', raising=False)
        assert self._count(monkeypatch, fp32_tail=True) == 3 - 1

    def test_the_output_survives_the_gate(self, monkeypatch):
        """The gate picks where activations are stored, not what comes
        out."""
        model = _tiny_backbone(num_layers=3, num_encoder_layers=2).train()
        seq = torch.randn(3, 7, 8, requires_grad=True)
        monkeypatch.delenv('FCST_GRAD_CKPT', raising=False)
        want = model.transformer.forecaster_forward(seq)
        monkeypatch.setenv('FCST_GRAD_CKPT', '1')
        got = model.transformer.forecaster_forward(seq)
        assert torch.equal(got, want)


class TestSecondPositiveOfAddPosHtftShifts:
    """#373 rule 3 on the one shape that names it: `add_pos_htft` carries a
    second positive cos(h_t, f_t), which at depth j is cos(h_{t+j}, f^(j)_t).
    Checked against the shape written out longhand, not against the loss."""

    SHAPE = 'cosine_similarity_batch_add_pos_htft'

    @staticmethod
    def _reference(f_j, o, depth, tau=0.1, shift_second_positive=True):
        """`cosine_similarity_batch_add_pos_htft` at depth `depth`.

        `shift_second_positive=False` leaves the second positive's h index at
        h_t — the depth-0 tensor rule 3 forbids — so the test can show the
        two differ.
        """
        f_view, o_view, _ = rollout_depth_views(f_j, o, depth)
        fore = F.normalize(f_view, p=2, dim=-1)
        orig = F.normalize(o_view, p=2, dim=-1)
        hy_hat, hy, hx = fore[:, :-1], orig[:, 1:], orig[:, :-1]
        if not shift_second_positive:
            hx = F.normalize(o, p=2, dim=-1)[:, :o_view.shape[1] - 1]
        positives = (torch.exp((hy_hat * hy).sum(-1) / tau)
                     + torch.exp((hy_hat * hx).sum(-1) / tau))
        def _pair(a, b):
            return torch.exp(
                (a.unsqueeze(3) * b.unsqueeze(2)).sum(-1) / tau)
        neg_xy = _pair(hx, hy).sum(dim=2)
        neg_xy_hat = _pair(hx, hy_hat).sum(dim=2)
        sims_xx = _pair(hx, hx)
        eye_c = torch.eye(C, dtype=torch.bool).view(1, 1, C, C)
        neg_xx = sims_xx.masked_fill(eye_c, 0).sum(dim=2)
        neg_zy = _pair(fore[:, 1:], hy_hat).sum(dim=2)
        sims_xb = (hy.unsqueeze(0) * hy_hat.unsqueeze(1)).sum(-1)
        eye_b = torch.eye(B, dtype=torch.bool).view(B, B, 1, 1)
        neg_xb = sims_xb.div(tau).exp().masked_fill(eye_b, 0).sum(dim=1)
        negatives = neg_xy + neg_xx + neg_zy + neg_xy_hat + neg_xb
        return -torch.log(
            positives / negatives.sum(dim=0, keepdim=True)).mean()

    def test_depth_zero_matches_the_longhand_shape(self):
        """The reference is the shape itself — pin it at depth 0 first."""
        f, o, _teacher, _roll = _latents(self.SHAPE)
        assert float(self._reference(f, o, 0)) == pytest.approx(
            K0_REFERENCE[(self.SHAPE, 'base')], rel=0, abs=1e-12)

    def test_the_depth_term_shifts_both_positives(self):
        latents = _latents(self.SHAPE, depth=1)
        _f, o, _teacher, rollouts = latents
        k0 = _call(self.SHAPE, 'base', latents=latents)
        k1 = _call(self.SHAPE, 'base', depth=1, latents=latents)
        want = self._reference(rollouts[0], o, 1)
        assert float(k1 - k0) == pytest.approx(float(want), rel=0, abs=1e-12)

    def test_leaving_the_second_positive_at_h_t_would_be_caught(self):
        latents = _latents(self.SHAPE, depth=1)
        _f, o, _teacher, rollouts = latents
        stale = self._reference(rollouts[0], o, 1,
                                shift_second_positive=False)
        shifted = self._reference(rollouts[0], o, 1)
        assert abs(float(stale) - float(shifted)) > 1e-6
        k0 = _call(self.SHAPE, 'base', latents=latents)
        k1 = _call(self.SHAPE, 'base', depth=1, latents=latents)
        assert abs(float(k1 - k0) - float(stale)) > 1e-6


class TestFloorAtDepthOverEveryShape:
    """`--subtract-contrastive-floor` re-bases by a constant, so it must stay
    gradient-neutral at every k, and a shape that cannot re-base at k = 0
    must not start re-basing at k > 0.

    `include_positive_in_denominator` follows the shape here. Its own guard
    fires BEFORE the floor block, so a shape swept with the wrong value
    raises on that guard and never reaches the floor at all — the raise
    would look like cover and be none.
    """

    @staticmethod
    def _floor_flags(shape):
        """The floor needs the normalized form on the pooled shapes. The
        split shape carries its own two-term floor and its own form."""
        return dict(include_positive_in_denominator=shape in _NORMALIZED,
                    subtract_contrastive_floor=True)

    @pytest.mark.parametrize('shape', FOUR_D_SHAPES)
    def test_the_floor_is_a_constant_offset_at_depth_three(self, shape):
        latents = _latents(shape, depth=3, requires_grad=True)
        f, _o, _teacher, rollouts = latents
        floor_on = self._floor_flags(shape)
        try:
            with_floor = _call(shape, 'base', depth=3, latents=latents,
                               **floor_on)
        except NotImplementedError as first:
            # A shape the floor cannot re-base raises the SAME way at every
            # depth — the depth loop must not turn that into a per-depth
            # constant quietly added to the loss. Three entries, and the
            # third is the only one that reaches a copy: the depth-0 body
            # runs before the loop, so `depth=3` raises there.
            assert 'subtract_contrastive_floor' in str(first)
            with pytest.raises(NotImplementedError) as at_zero:
                _call(shape, 'base', latents=latents, **floor_on)
            with pytest.raises(NotImplementedError) as in_copy:
                _call(shape, 'base', latents=latents, depth_index=3,
                      **floor_on)
            assert str(at_zero.value) == str(first)
            assert str(in_copy.value) == str(first)
            return
        no_floor = _call(shape, 'base', depth=3, latents=latents,
                         **dict(floor_on, subtract_contrastive_floor=False))
        g_with = torch.autograd.grad(with_floor, [f] + rollouts,
                                     retain_graph=True)
        g_without = torch.autograd.grad(no_floor, [f] + rollouts)
        for a, b in zip(g_with, g_without):
            assert torch.equal(a, b)

    def test_the_sweep_reaches_the_floor_on_every_shape(self):
        """Both arms above are floor arms — no shape drops out on the
        `include_positive_in_denominator` guard, which sits earlier.

        Every shape asserts, including the ones that raise nothing: a shape
        that neither refuses the floor by name nor moves the value reached
        no floor branch at all, and raising nothing is not a pass.
        """
        reached = []
        for shape in FOUR_D_SHAPES:
            latents = _latents(shape, depth=3)
            floor_on = self._floor_flags(shape)
            try:
                on = _call(shape, 'base', latents=latents, **floor_on)
            except NotImplementedError as exc:
                assert 'subtract_contrastive_floor' in str(exc), shape
                reached.append(shape)
                continue
            off = _call(shape, 'base', latents=latents,
                        **dict(floor_on, subtract_contrastive_floor=False))
            assert float(off - on) > 0.0, f'{shape}: the floor changed nothing'
            reached.append(shape)
        assert reached == FOUR_D_SHAPES

    def test_the_split_shape_re_bases_l_pred_once_per_depth(self):
        """L_pred's floor repeats with the depth. L_rep's does not."""
        latents = _latents(_SPLIT, depth=3)
        f_pred, f_rep = loss_module._split_pred_rep_floors(0.1, B, T, C)
        offsets = []
        for depth in (0, 3):
            on = _call(_SPLIT, 'split_floor', depth=depth, latents=latents)
            off = _call(_SPLIT, 'split_floor', depth=depth, latents=latents,
                        subtract_contrastive_floor=False)
            offsets.append(float(off - on))
        assert offsets[0] == pytest.approx(float(f_pred) + float(f_rep),
                                           rel=0, abs=1e-12)
        assert offsets[1] == pytest.approx(4 * float(f_pred) + float(f_rep),
                                           rel=0, abs=1e-12)

    def test_an_h_only_shape_refuses_the_floor_at_every_depth(self):
        """`rep_only` has no positive, so the floor refuses it. It raised at
        k = 0 and must keep raising — a depth copy of an h-only shape adds
        exactly zero, so a floor there would be a constant with no term.

        Three entries, and only the third reaches a copy: a `depth=3` call
        raises in the depth-0 body before the loop starts, so `depth_index=3`
        is what enters the copy's own floor branch.
        """
        latents = _latents(_REP_ONLY, depth=3)
        floor_on = self._floor_flags(_REP_ONLY)
        messages = []
        for depth, index in ((0, 0), (3, 0), (0, 3)):
            with pytest.raises(NotImplementedError) as excinfo:
                _call(_REP_ONLY, 'base', depth=depth, latents=latents,
                      depth_index=index, **floor_on)
            messages.append(str(excinfo.value))
        assert 'subtract_contrastive_floor' in messages[0]
        assert messages[0] == messages[1] == messages[2]


class TestRolloutCosError:
    """`cos_err_d0..dk` — the per-depth diagnostic #373 asks for."""

    def test_depth_zero_reproduces_the_ff_column(self):
        f, o, _teacher, _roll = _latents(_SPLIT)
        f, o = f.float(), o.float()
        ff, _fp, _tp, _cb = compute_metrics(f, o, 1)
        assert rollout_cos_error(f, o)[0] == pytest.approx(1.0 - ff, abs=1e-6)

    def test_one_entry_per_depth_plus_depth_zero(self):
        f, o, _teacher, rollouts = _latents(_SPLIT, depth=3)
        assert len(rollout_cos_error(f, o, rollouts)) == 4
        assert len(rollout_cos_error(f, o)) == 1

    def test_each_depth_reads_its_own_shifted_window(self):
        f, o, _teacher, rollouts = _latents(_SPLIT, depth=3)
        errors = rollout_cos_error(f, o, rollouts)
        for depth, f_depth in enumerate(rollouts, start=1):
            want = 1.0 - F.cosine_similarity(
                f_depth[:, :T - 1 - depth], o[:, 1 + depth:],
                dim=-1, eps=1e-8).mean()
            assert errors[depth] == pytest.approx(float(want), rel=0,
                                                  abs=1e-12)

    def test_a_perturbed_depth_moves_only_its_own_entry(self):
        f, o, _teacher, rollouts = _latents(_SPLIT, depth=3)
        before = rollout_cos_error(f, o, rollouts)
        bumped = list(rollouts)
        bumped[1] = bumped[1] + 1.0
        after = rollout_cos_error(f, o, bumped)
        assert after[2] != before[2]
        assert [after[j] for j in (0, 1, 3)] == [before[j] for j in (0, 1, 3)]

    def test_the_metric_carries_no_gradient(self):
        f, o, _teacher, rollouts = _latents(_SPLIT, depth=1,
                                            requires_grad=True)
        errors = rollout_cos_error(f, o, rollouts)
        assert all(isinstance(e, float) for e in errors)


def load_train_module():
    """Import train.py as a module so its classes can be exercised."""
    spec = importlib.util.spec_from_file_location("train_py_373", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _train_tree():
    return ast.parse(TRAIN_PY.read_text())


def _calls_named(tree, name):
    """Every ast.Call in `tree` whose callee is `name`."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = (func.id if isinstance(func, ast.Name)
                  else getattr(func, 'attr', None))
        if called == name:
            out.append(node)
    return out


def _keyword(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


# Every loss in train.py that ties `f` to `h`. A call to one of these that
# carries no depth trains its term at k = 0, whatever the flag says.
F_BEARING_LOSSES = ('contrastive_latent_loss', 'align_loss',
                    'cpc_infonce_aux_loss', 'cpc_infonce_all_loss')


def is_none_literal(node):
    """Whether an AST node is the literal `None`."""
    return isinstance(node, ast.Constant) and node.value is None


def f_bearing_calls_missing_depths(tree):
    """The f-bearing loss calls in `tree` that carry no depth.

    A call passes on either of two grounds: it takes `rollout_latents` and
    the value is not the literal `None`, or it pins `train_rollout_depth=0`
    — the depth-0 reference diagnostic, which must stay comparable across k.

    `None` is the argument's own default, so a call that spells it out reads
    as wired and still trains at k = 0. The check counts it as missing.
    """
    missing = []
    for name in F_BEARING_LOSSES:
        for call in _calls_named(tree, name):
            depths = _keyword(call, 'rollout_latents')
            if depths is not None and not is_none_literal(depths):
                continue
            pin = _keyword(call, 'train_rollout_depth')
            if pin is not None and ast.literal_eval(pin) == 0:
                continue
            missing.append(f'{name}:{call.lineno}')
    return missing


def unwire_one_call(tree, name, to_none=False):
    """Unwire the first wired `name` call, one of the two ways.

    Default: strip `rollout_latents` off it. `to_none`: keep the keyword and
    pass the literal `None`, the argument's own default. Both train that
    term at k = 0, so the check above must catch both. Raises when no call
    of that name carries the depths — so every f-bearing loss in train.py is
    reached by at least one wired call.
    """
    for call in _calls_named(tree, name):
        depths = _keyword(call, 'rollout_latents')
        if depths is None or is_none_literal(depths):
            continue
        if to_none:
            for kw in call.keywords:
                if kw.arg == 'rollout_latents':
                    kw.value = ast.Constant(value=None)
        else:
            call.keywords = [kw for kw in call.keywords
                             if kw.arg != 'rollout_latents']
        return tree
    raise AssertionError(f'no {name} call in train.py carries rollout_latents')


def refuse_run(tmp_path, *flags):
    """Run train.py expecting one of its guards, and return its stderr.

    `--weight-decay` is required, so a command without it dies inside
    argparse with a usage dump that names every flag of the parser. A guard
    test that asserts on a flag name alone passes on that dump and never
    reaches the guard, so this helper supplies the required flag and refuses
    the usage dump outright.
    """
    out = subprocess.run(
        [sys.executable, str(TRAIN_PY), '--weight-decay', '0.1',
         '--hf-repo', 'none', '--hf-path', 'none',
         '--save-dir', str(tmp_path), *flags],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, 'PYTHONPATH': str(REPO_ROOT)})
    assert out.returncode != 0
    assert 'usage:' not in out.stderr, \
        f'died in argparse, never reached a guard:\n{out.stderr[-2000:]}'
    return out.stderr


def shard_branch(tree):
    """The `if not args.shard_loss_on_batch:` statement of train.py."""
    branches = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                and 'shard_loss_on_batch' in ast.unparse(n.test)]
    assert len(branches) == 1, 'expected exactly one shard-loss branch'
    return branches[0]


class TestTrainerWiring:
    """The trainer is where the depth is built, gathered, logged and kept out
    of the reference diagnostics. None of that is reachable from the loss
    tests, so it is pinned here."""

    def test_the_flag_defaults_to_the_historical_objective(self):
        defaults = {}
        for call in _calls_named(_train_tree(), 'add_argument'):
            if call.args and isinstance(call.args[0], ast.Constant):
                defaults[call.args[0].value] = _keyword(call, 'default')
        assert '--train-rollout-depth' in defaults
        assert ast.literal_eval(defaults['--train-rollout-depth']) == 0

    def test_the_depths_are_built_from_the_forward_output(self):
        """Which tensor feeds the re-entry. Static, on the parse tree, so it
        survives a reflow. No run can show it, because a depth built from the
        wrong tensor still trains and still logs. The runtime guard that the
        depth reaches the objective at all is the 4-step run at the bottom of
        this file."""
        calls = _calls_named(_train_tree(), 'rollout_forecaster_latents')
        assert len(calls) == 1
        args = [ast.unparse(a) for a in calls[0].args]
        assert args == ['model', 'f_lat', 'args.train_rollout_depth']

    def test_every_depth_is_gathered_with_one_all_gather(self):
        """`gather_latents` takes a pair and issues TWO all-gathers. A depth
        is one tensor, so it goes through the single-tensor `gather_latent`
        — one collective per depth, not two.

        Static, on the parse tree. `gather_latent` counts its collectives at
        runtime in `tests/test_dist_gather.py`, but only under DDP: off DDP
        both forms are no-ops, so the 4-step single-process run cannot tell
        them apart and no runtime check here can either.
        """
        tree = _train_tree()
        depth_gathers = [c for c in _calls_named(tree, 'gather_latent')
                         if len(c.args) == 1]
        assert len(depth_gathers) == 1
        comps = [n for n in ast.walk(tree) if isinstance(n, ast.ListComp)
                 and n.elt is depth_gathers[0]]
        assert len(comps) == 1, "the depth gather is not a per-depth loop"
        assert ast.unparse(comps[0].generators[0].iter) == 'rollout_lats'
        for call in _calls_named(tree, 'gather_latents'):
            unparsed = [ast.unparse(a) for a in call.args]
            assert unparsed != ['f_j', 'f_j'], \
                "a depth must not go through the two-collective pair form"

    def test_loss_tau_ref_stays_at_depth_zero(self):
        """The τ-reference curve must be comparable across k, so the
        diagnostic call forces depth 0 and passes no rollout latents."""
        tree = _train_tree()
        ref = [c for c in _calls_named(tree, 'contrastive_latent_loss')
               if _keyword(c, 'train_rollout_depth') is not None]
        assert len(ref) == 1
        assert ast.literal_eval(_keyword(ref[0], 'train_rollout_depth')) == 0
        assert _keyword(ref[0], 'rollout_latents') is None

    def test_the_kind_guard_names_the_kind_it_rejects(self):
        """The guard rejects every non-transformer forecaster, so its message
        must name the kind at hand, not only the CPC families."""
        train = load_train_module()
        src = inspect.getsource(train.main)
        guard = src.split('args.train_rollout_depth > 0 and '
                          'args.forecaster_kind != "transformer"')[1]
        message = guard.split('SystemExit')[1].split(')')[0]
        assert 'args.forecaster_kind' in message

    @pytest.mark.parametrize('kind', ['cpc', 'linear_cpc'])
    def test_a_non_transformer_forecaster_is_refused(self, kind, tmp_path):
        stderr = refuse_run(tmp_path, '--train-rollout-depth', '3',
                            '--forecaster-kind', kind)
        assert kind in stderr
        assert 'applies the forecaster to its own output' in stderr

    def test_a_negative_depth_is_refused(self, tmp_path):
        stderr = refuse_run(tmp_path, '--train-rollout-depth', '-1')
        assert '--train-rollout-depth must be ≥ 0' in stderr


class TestTheNoConsumerGuardCoversEveryRoute:
    """A k > 0 run whose every f-bearing term is off adds exactly zero per
    depth. It trains at k = 0 and still writes k + 1 plausible `cos_err_dj`
    curves, because the diagnostic reads the depth tensors, not the loss.

    Three routes reach that state through the MAIN term, and the guard has
    to cover all three: `--no-main-contrastive-loss` drops the term,
    `rep_only` is h-anchored end to end so its depth copy is zero, and
    `--pred-loss-weight 0` zeroes `L_pred`, the f-bearing half of the split
    shape. `L_align` or the CPC auxiliary rescues any of them.
    """

    @staticmethod
    def _args(**over):
        base = dict(no_main_contrastive_loss=False, loss_shape=_XSHH_ALLT,
                    pred_loss_weight=1.0)
        return SimpleNamespace(**{**base, **over})

    @pytest.mark.parametrize('over,phrase', [
        pytest.param(dict(no_main_contrastive_loss=True),
                     '--no-main-contrastive-loss', id='no_main'),
        pytest.param(dict(loss_shape=_REP_ONLY), _REP_ONLY, id='rep_only'),
        pytest.param(dict(loss_shape=_SPLIT, pred_loss_weight=0.0),
                     '--pred-loss-weight', id='split_pred_weight_zero'),
    ])
    def test_the_gap_names_the_route(self, over, phrase):
        train = load_train_module()
        gap = train.main_term_depth_gap(self._args(**over))
        assert gap and phrase in gap

    @pytest.mark.parametrize('over', [
        pytest.param({}, id='pooled_shape'),
        pytest.param(dict(loss_shape=_SPLIT), id='split_with_its_weight'),
        pytest.param(dict(loss_shape=_SPLIT, pred_loss_weight=0.5),
                     id='split_at_half_weight'),
    ])
    def test_a_main_term_that_takes_the_depths_reports_no_gap(self, over):
        train = load_train_module()
        assert train.main_term_depth_gap(self._args(**over)) is None

    def test_the_split_shape_ignores_the_rep_side_weight(self):
        """`--rep-loss-weight 0` drops the h-anchored half. `L_pred` still
        carries the depths, so it is not a route to a silent k = 0."""
        train = load_train_module()
        assert train.main_term_depth_gap(
            self._args(loss_shape=_SPLIT, rep_loss_weight=0.0)) is None

    @pytest.mark.parametrize('flags,phrase', [
        pytest.param(('--no-main-contrastive-loss', '--sigreg-encoding'),
                     '--no-main-contrastive-loss', id='no_main'),
        pytest.param(('--loss-shape', _REP_ONLY), _REP_ONLY, id='rep_only'),
        pytest.param(('--loss-shape', _SPLIT, '--pred-loss-weight', '0.0'),
                     '--pred-loss-weight', id='split_pred_weight_zero'),
    ])
    def test_each_route_is_refused_at_the_cli(self, tmp_path, flags, phrase):
        stderr = refuse_run(tmp_path, '--train-rollout-depth', '3', *flags)
        assert '--train-rollout-depth' in stderr
        assert 'no term to enter' in stderr
        assert phrase in stderr

    @pytest.mark.parametrize('rescue', ['align_loss_weight',
                                        'cpc_infonce_weight'])
    def test_a_rescued_route_has_a_consumer(self, rescue):
        """The guard fires on the missing consumer, not on the shape: add a
        term that ties f to h and the same run is legal."""
        train = load_train_module()
        weights = dict(align_loss_weight=0.0, cpc_infonce_weight=0.0)
        args = self._args(loss_shape=_REP_ONLY, **weights)
        assert train.rollout_depth_has_no_consumer(args) is True
        setattr(args, rescue, 1.0)
        assert train.rollout_depth_has_no_consumer(args) is False

    def test_the_same_pair_runs_once_a_term_can_consume_the_depths(
            self, tmp_path):
        """End to end on one of the three routes: add `--align-loss-weight`
        and the run the guard refused above trains."""
        rows = run_short_training(
            tmp_path, 'nomain_align_k2', '--no-main-contrastive-loss',
            '--sigreg-encoding', '--align-loss-weight', '1.0',
            '--train-rollout-depth', '2')
        errs = [float(rows[0][f'cos_err_d{j}']) for j in range(3)]
        assert len(set(errs)) == 3


class TestNoBranchOfTheLossBlockTrainsAtDepthZero:
    """A run labelled k must train at k on EVERY branch of the loss block.

    Two branches split it. `--shard-loss-on-batch` picks whether the
    latents are all-gathered. `--no-main-contrastive-loss` picks the main
    contrastive term or the standalone `align_loss`. A branch that dropped
    the depths would train its term at k = 0 while the CSV still wrote
    k + 1 plausible `cos_err_dj` curves, because the diagnostic reads
    `rollout_lats`, not the loss.
    """

    def test_every_f_bearing_loss_call_carries_the_depths(self):
        assert f_bearing_calls_missing_depths(_train_tree()) == []

    @pytest.mark.parametrize('to_none', [False, True],
                             ids=['dropped', 'passed_none'])
    @pytest.mark.parametrize('name', F_BEARING_LOSSES)
    def test_the_check_catches_an_unwired_call(self, name, to_none):
        """Teeth: strip the keyword off one call — or leave it and pass the
        argument's own default, `None` — and the check reports it."""
        assert f_bearing_calls_missing_depths(
            unwire_one_call(_train_tree(), name, to_none=to_none))

    def test_the_depths_are_built_outside_the_shard_branch(self):
        """`--shard-loss-on-batch` skips the GATHER and nothing else. A
        depth built inside that branch would leave every sharded run at
        k = 0."""
        tree = _train_tree()
        inside = {id(n) for n in ast.walk(shard_branch(tree))}
        builds = _calls_named(tree, 'rollout_forecaster_latents')
        assert builds
        assert not [c for c in builds if id(c) in inside]

    def test_no_f_bearing_loss_call_sits_inside_the_shard_branch(self):
        """The loss runs on both sides of the branch, so a call inside it
        would be a call the sharded path never makes."""
        tree = _train_tree()
        inside = {id(n) for n in ast.walk(shard_branch(tree))}
        for name in F_BEARING_LOSSES:
            assert not [c for c in _calls_named(tree, name)
                        if id(c) in inside], name

    def test_only_the_gather_reads_the_depths_in_the_shard_branch(self):
        """What the branch may hold: gathers, and nothing else that reads
        the depths. A call that read `rollout_lats` in here would run on the
        gathered path alone, so a sharded run would train that term at
        k = 0. Calls that never touch the depths are the branch's own
        business — the two tests above cover the f-bearing ones."""
        branch = shard_branch(_train_tree())
        gathers = {id(c) for name in ('gather_latent', 'gather_latents')
                   for c in _calls_named(branch, name)}
        readers = [ast.unparse(n) for n in ast.walk(branch)
                   if isinstance(n, ast.Call) and id(n) not in gathers
                   and 'rollout_lats' in ast.unparse(n)]
        assert readers == []


def run_short_training(tmp_path, run_name, *extra, steps=2):
    """A short CPU run of the real trainer. Returns its CSV rows."""
    _assert_train_deps_available()
    save_dir = tmp_path / "runs"
    save_dir.mkdir(exist_ok=True)
    env = {**os.environ,
           "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get(
               "PYTHONPATH", "")}
    result = subprocess.run(
        [sys.executable, "-u", str(TRAIN_PY),
         "--device", "cpu", "--total-steps", str(steps), "--batch-size", "2",
         "--lr", "1e-3", "--weight-decay", "0.1",
         "--save-dir", str(save_dir), "--run-name", run_name,
         "--mix-ratio", "1.0", "--synth-kind", "periodic",
         "--t-raw", "64", "--n-channels", "1", "--d-model", "32",
         "--n-heads", "2", "--num-layers", "1", "--num-encoder-layers", "1",
         "--log-every", "1", "--seed", "42", "--tau", "0.10",
         "--hf-repo", "none", "--hf-path", "none", *extra],
        env=env, capture_output=True, text=True, cwd=str(tmp_path),
        timeout=900)
    if result.returncode != 0:
        pytest.fail(f"train.py rc={result.returncode}\n"
                    f"stdout:\n{result.stdout[-3000:]}\n"
                    f"stderr:\n{result.stderr[-3000:]}")
    with open(save_dir / f"{run_name}_losses.csv") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize('arm', [
    pytest.param(('--loss-shape', _XSHH_ALLT), id='main_contrastive'),
    pytest.param(('--loss-shape', _XSHH_ALLT, '--no-main-contrastive-loss',
                  '--align-loss-weight', '1.0'), id='align_standalone'),
    pytest.param(('--loss-shape', _REP_ONLY, '--align-loss-weight', '1.0'),
                 id='rep_only_align_add_on'),
])
def test_a_sharded_run_trains_at_the_depth_it_is_given(tmp_path, arm):
    """`--shard-loss-on-batch` at k = 2, on all three paths of the loss block.

    Two arms split it — `--no-main-contrastive-loss` picks the standalone
    `align_loss`, its absence picks the main term. The main arm splits again
    on the shape: `rep_only` is h-anchored end to end, so its depth copy
    returns zeros and the whole depth contribution is the `L_align` add-on
    in the shared tail of `contrastive_latent_loss`. Twelve of the 14 cells
    run that third path, and it is the one the first two arms never enter.

    Step 0 is the discriminating row. The depth pass runs AFTER the
    forward, so the k = 0 and k = 2 runs reach their first loss on the same
    weights and the same batch. `loss_tau_ref` is pinned to depth 0, so it
    stays equal and proves the two runs saw the same inputs. `loss` moves
    only if the depths entered the objective. Unwire the branch and the two
    losses are the same number.
    """
    rows = {}
    for depth in (0, 2):
        rows[depth] = run_short_training(
            tmp_path, f'shard_k{depth}', *arm, '--shard-loss-on-batch',
            '--train-rollout-depth', str(depth))[0]
    assert float(rows[0]['loss_tau_ref']) == pytest.approx(
        float(rows[2]['loss_tau_ref']), rel=0, abs=1e-9), \
        "the two runs did not see the same batch"
    assert abs(float(rows[0]['loss']) - float(rows[2]['loss'])) > 1e-6, \
        "the sharded path trains at k = 0"
    errs = [float(rows[2][f'cos_err_d{j}']) for j in range(3)]
    assert len(set(errs)) == 3


class TestLossesCsvCarriesThePerDepthColumns:

    @staticmethod
    def _row(logger, cos_err_depths):
        logger.log(1, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 7, 8, False,
                   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                   cos_err_depths=cos_err_depths)
        logger.close()

    @staticmethod
    def _read(path):
        with open(path) as fh:
            return list(csv.DictReader(fh))

    def test_a_k0_run_writes_no_cos_err_column(self, tmp_path):
        train = load_train_module()
        path = str(tmp_path / "k0_losses.csv")
        logger = train.CSVLogger(path, rollout_depth=0)
        self._row(logger, None)
        assert not [c for c in self._read(path)[0] if c.startswith('cos_err')]

    def test_a_k3_run_appends_one_column_per_depth(self, tmp_path):
        train = load_train_module()
        path = str(tmp_path / "k3_losses.csv")
        logger = train.CSVLogger(path, rollout_depth=3)
        self._row(logger, [0.11, 0.22, 0.33, 0.44])
        row = self._read(path)[0]
        assert [c for c in row if c.startswith('cos_err')] == \
            ['cos_err_d0', 'cos_err_d1', 'cos_err_d2', 'cos_err_d3']
        assert [row[f'cos_err_d{j}'] for j in range(4)] == \
            ['0.11', '0.22', '0.33', '0.44']
        assert row['ema_tau'] == ''
        assert row['loss'] == '0.5'

    def test_a_short_depth_list_leaves_blanks_not_a_shifted_row(self, tmp_path):
        train = load_train_module()
        path = str(tmp_path / "short_losses.csv")
        logger = train.CSVLogger(path, rollout_depth=3)
        self._row(logger, [0.11])
        row = self._read(path)[0]
        assert row['cos_err_d0'] == '0.11'
        assert [row[f'cos_err_d{j}'] for j in (1, 2, 3)] == ['', '', '']

    def test_resuming_a_k0_csv_at_k3_is_refused(self, tmp_path):
        """The extra columns change the schema, so appending would shift
        every value of every later row."""
        train = load_train_module()
        path = str(tmp_path / "resume_losses.csv")
        logger = train.CSVLogger(path, rollout_depth=0)
        self._row(logger, None)
        with pytest.raises(SystemExit, match="schema mismatch"):
            train.CSVLogger(path, rollout_depth=3)


def _assert_train_deps_available():
    for mod in ("torch", "numpy", "datasets"):
        try:
            __import__(mod)
        except ImportError as exc:                       # pragma: no cover
            pytest.fail(f"train.py dep {mod!r} not importable: {exc}. "
                        "Do NOT skip — the depth wiring would go untested.")


def test_a_four_step_run_at_depth_two_trains_and_logs(tmp_path):
    """End-to-end: the flag reaches the loss, the forecaster re-enters, and
    the CSV carries three finite `cos_err_dj` curves.

    Everything above tests one piece. This is the only test that runs the
    depth through the real trainer: forward → re-entry → gather → loss →
    backward → CSV.
    """
    rows = run_short_training(tmp_path, "depth2", "--loss-shape", _XSHH_ALLT,
                              "--train-rollout-depth", "2", steps=4)
    assert rows, "no CSV rows — the step loop never ran"
    for row in rows:
        assert float(row["loss"]) == float(row["loss"])          # not NaN
        for depth in range(3):
            assert 0.0 <= float(row[f"cos_err_d{depth}"]) <= 2.0
        # Each depth re-enters the forecaster, so f^(1) and f^(2) are their
        # own tensors read against their own shifted window. Equal values
        # would mean the loop handed depth 0 back k times.
        errs = [float(row[f"cos_err_d{d}"]) for d in range(3)]
        assert len(set(errs)) == 3
    assert "cos_err_d3" not in rows[0]
