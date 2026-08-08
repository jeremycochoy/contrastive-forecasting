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
  any k > 0; no other shape raises.
* **The re-entry composes the operator the eval composes**: one depth equals
  one token of `rollout_latent` on the same input.
"""

import inspect
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import src.loss as loss_module
from src.forecasting_head import rollout_forecaster_latents, rollout_latent
from src.loss import (align_loss, contrastive_latent_loss,
                      cpc_infonce_all_loss, cpc_infonce_aux_loss,
                      rollout_depth_views)
from src.models import ConfigurableModel

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


# Modifier combinations. `base` and `pos_in_denom` sweep the whole dispatch;
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
        'stopgrad_pos': (_SPLIT, _REP_ONLY, _XSHH_ALLT),
        'tau_rep': (_SPLIT, _REP_ONLY),
    }
    return [(shape, name) for shape in ALL_SHAPES for name in K0_CONFIGS
            if shape in allowed[name]]


def _call(shape, config_name, depth=0, latents=None, **extra):
    """Run `contrastive_latent_loss` for one (shape, config) case at depth k."""
    case = K0_CONFIGS[config_name]
    f, o, teacher_o, rollouts = latents
    cfg = dict(case['cfg'])
    cfg.update(extra)
    if depth:
        cfg['train_rollout_depth'] = depth
    return contrastive_latent_loss(
        (f, o), validation=False, spec=_spec(shape, **cfg),
        teacher_original_latent=teacher_o if case['teacher'] else None,
        rollout_latents=rollouts[:depth] if depth else None)


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
    ('cosine_similarity_batch_rep_only', 'tau_rep'): 4.865525677822549,
    ('cosine_similarity_batch_split_pred_rep', 'base'): 15.73789835105975,
    ('cosine_similarity_batch_split_pred_rep', 'align_teacher'): 17.880922522184363,
    ('cosine_similarity_batch_split_pred_rep', 'align_moco_rep'): 16.83829488617452,
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
    return contrastive_latent_loss(
        (f_view, o_view), validation=False, spec=_spec(shape, **cfg),
        teacher_original_latent=teacher_view)


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
        """f^(1) at the last window step feeds ONLY the adjacent f↔f family;
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


def _tiny_backbone(seed=0):
    """fp32 throughout: the forecaster's last layer runs in fp32 by design."""
    torch.manual_seed(seed)
    model = ConfigurableModel(
        C=1, H=8, W=4, encoder_type='mlp', intermediate_dim=8,
        num_layers=2, nhead=2, ffn_mult=1, dropout=0.0,
        num_encoder_layers=1, rev_norm_span=None, rev_norm_kind='none',
        freq_emb_dim=0, forecaster_d_model=4, forecaster_n_heads=2)
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
