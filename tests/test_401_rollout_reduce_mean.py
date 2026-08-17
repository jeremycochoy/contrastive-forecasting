"""Tests for #401 — `--train-rollout-reduce {sum,mean}` (key `train_rollout_reduce`).

#373's `--train-rollout-depth k` SUMS the k + 1 copies of every loss term that
ties the forecaster output `f` to the encoder latent `h`. The f-side then
carries k + 1 times its k = 0 weight against the terms that carry no `f` —
`L_rep`, `L_rep_moco`, `align_moco`, SIGReg and `mse`'s h-only half — which
enter the total once at any k. `docs/train_rollout_depth.md` states that risk.

#401 adds the other reduction. `mean` divides the k + 1 copies by k + 1, so the
f-side holds its k = 0 weight at every depth. The depth then changes WHAT the
model trains on, and not HOW MUCH the f-side outweighs the rest.

What this file pins:

* **`sum` is the default, and #373 is byte-identical.** The whole #373 sweep —
  every shape the dispatch accepts, every modifier combination its 14 cells
  train with — reproduces its frozen k = 0 value under the new key, unset and
  set to `sum`, and reproduces its summed k > 0 value.
* **The mean of one copy equals the sum of one copy.** k = 0 holds one copy,
  and the two reductions return the same number on every case.
* **The mean does not scale with k.** When every depth copy is a copy of the
  depth-0 f-term, the mean at k = 1, 4 and 8 equals the k = 0 total, while the
  sum grows by one copy per depth.
* **The h-side stays out of the division.** The terms that carry no `f` enter
  once at their configured weight under both reductions.
* **The reduction identity holds on every shape**, against the f-side the
  depth copies carry, which `f_terms_only` reads at depth 0.
* **Every h-only term enters once**, one case per term, each with an h-side
  defined WITHOUT the probe: `L_rep`, `L_rep_moco`, the `f_rep` floor
  constant, `mse`'s h-only half, `align_moco` and SIGReg.
* **The auxiliary f-bearing terms reduce too**: the standalone `align_loss`
  and the two CPC InfoNCE terms.
* **The guards**: a reduction that is not `sum` or `mean` raises and names
  itself, and the CLI accepts the two words only.
* **The trainer wiring**: the flag, its default, the config key, the three
  standalone call sites, and runs of the real trainer that show k = 0
  unchanged and k > 0 moved.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys

import pytest
import torch

import src.loss as loss_module
from src.loss import (_split_pred_rep_floors, align_loss, align_moco_loss,
                      contrastive_latent_loss, cpc_infonce_all_loss,
                      cpc_infonce_aux_loss, resolve_rollout_reduce,
                      sigreg_loss)

from tests.test_373_train_rollout_depth import (B, C, CPC_SHAPES, H, K0_CONFIGS,
                                                K0_REFERENCE, REPO_ROOT,
                                                TRAIN_PY, _REP_ONLY, _SPLIT,
                                                _calls_named, _keyword,
                                                _latents, _spec, _train_tree,
                                                k0_cases, run_short_training)

# The #373 sweep, minus the two CPC families: their forecaster is K parallel
# linear heads, so they raise at any k > 0 and carry no depth to reduce.
REDUCE_CASES = [(shape, name) for shape, name in k0_cases()
                if shape not in CPC_SHAPES]


def call(shape, config_name, latents, depth=0, reduce=None,
         f_terms_only=False, **extra):
    """`contrastive_latent_loss` for one (shape, config) case at depth k.

    `reduce=None` leaves the key unset, which is the pre-#401 call. A string
    sets `train_rollout_reduce`. `f_terms_only` asks for the f-side alone —
    the part the depth copies carry, and the part the mean divides.
    """
    case = K0_CONFIGS[config_name]
    f, o, teacher_o, rollouts = latents
    cfg = dict(case['cfg'])
    cfg.update(extra)
    if depth:
        cfg['train_rollout_depth'] = depth
    if reduce is not None:
        cfg['train_rollout_reduce'] = reduce
    return float(contrastive_latent_loss(
        (f, o), validation=False, spec=_spec(shape, **cfg),
        teacher_original_latent=teacher_o if case['teacher'] else None,
        rollout_latents=rollouts[:depth] if depth else None,
        f_terms_only=f_terms_only))


# The deepest k this file reaches. A depth needs k + 2 time steps to keep one
# anchor, and #373's T = 6 stops at k = 4.
T_FLAT = 12


def flat_latents(depth=0, seed=0):
    """(f, o, teacher_o, [f^(1)..f^(k)]) — every latent CONSTANT along time.

    A copy at depth j reads f^(j)_0..f^(j)_{T-1-j} against h_j..h_{T-1}. When
    both tensors repeat one vector along time, every window holds the same
    pairs. A term whose value does not depend on the anchor COUNT then returns
    the same number at every depth, which makes "the mean does not scale with
    k" an exact equality and not a bound.

    Every rollout copy is a clone of `f`, so the k + 1 copies are k + 1 copies
    of one term.
    """
    g = torch.Generator().manual_seed(seed)

    def draw():
        x = torch.randn(B, C, H, generator=g, dtype=torch.float64)
        return x.unsqueeze(1).expand(B, T_FLAT, C, H).contiguous()

    f, o, teacher_o = draw(), draw(), draw()
    return f, o, teacher_o, [f.clone() for _ in range(depth)]


# The two shapes whose depth copy is one term with no time pool, so the copies
# above are identical: `rep_only` pairs L_rep with the f-bearing L_align add-on
# (this study's own cell), and `mse` pairs its h-only half with mse(h, f).
IDENTICAL_COPY_CASES = [
    pytest.param(_REP_ONLY, {'align_loss_weight': 1.0}, id='rep_only_align'),
    pytest.param('mse', {}, id='mse'),
]


class TestTheDefaultIsUnchanged:
    """#373's objective is what an unset key and `sum` both return."""

    @pytest.mark.parametrize('shape,config_name', k0_cases())
    @pytest.mark.parametrize('reduce', [None, 'sum', 'mean'])
    def test_depth_zero_reproduces_the_frozen_value(self, shape, config_name,
                                                    reduce):
        """The mean of ONE copy is the sum of one copy, on every case."""
        lat = _latents(shape, seed=0)
        assert call(shape, config_name, lat, reduce=reduce) == pytest.approx(
            K0_REFERENCE[(shape, config_name)], rel=0, abs=1e-12)

    @pytest.mark.parametrize('shape,config_name', REDUCE_CASES)
    @pytest.mark.parametrize('k', [1, 3])
    def test_an_unset_key_still_sums(self, shape, config_name, k):
        lat = _latents(shape, seed=1, depth=k)
        assert call(shape, config_name, lat, depth=k) == pytest.approx(
            call(shape, config_name, lat, depth=k, reduce='sum'),
            rel=0, abs=1e-12)


class TestTheMeanDoesNotScaleWithK:
    """k + 1 copies of one term reduce to that term, at every k."""

    @pytest.mark.parametrize('shape,cfg', IDENTICAL_COPY_CASES)
    @pytest.mark.parametrize('k', [1, 4, 8])
    def test_the_mean_returns_the_depth_zero_total(self, shape, cfg, k):
        lat = flat_latents(depth=k)
        base = call(shape, 'base', lat, depth=0, **cfg)
        assert call(shape, 'base', lat, depth=k, reduce='mean',
                    **cfg) == pytest.approx(base, rel=1e-11, abs=1e-12)

    @pytest.mark.parametrize('shape,cfg', IDENTICAL_COPY_CASES)
    @pytest.mark.parametrize('k', [1, 4, 8])
    def test_the_sum_grows_by_one_copy_per_depth(self, shape, cfg, k):
        """The control: the same latents under `sum` do scale with k."""
        lat = flat_latents(depth=k)
        base = call(shape, 'base', lat, depth=0, **cfg)
        f_side = call(shape, 'base', lat, depth=0, f_terms_only=True, **cfg)
        assert abs(f_side) > 1e-6, 'the case carries no f-side to scale'
        assert call(shape, 'base', lat, depth=k, reduce='sum',
                    **cfg) == pytest.approx(base + k * f_side, rel=1e-11)


class TestTheHSideStaysOutOfTheDivision:
    """`L_rep` enters once at its configured weight under both reductions."""

    @pytest.mark.parametrize('k', [1, 3])
    def test_only_the_f_side_is_divided(self, k):
        lat = _latents(_REP_ONLY, seed=5, depth=k)
        # `rep_only` at align weight 0 is h-anchored end to end: L_rep alone.
        h_side = call(_REP_ONLY, 'base', lat, depth=0, align_loss_weight=0.0)
        cfg = {'align_loss_weight': 1.0}
        summed = call(_REP_ONLY, 'base', lat, depth=k, reduce='sum', **cfg)
        meaned = call(_REP_ONLY, 'base', lat, depth=k, reduce='mean', **cfg)
        # Σ L_align^(j) is what the two reductions share. Divide the WHOLE
        # total instead and this equality fails by k·L_rep.
        assert (meaned - h_side) * (k + 1) == pytest.approx(
            summed - h_side, rel=1e-11)

    @pytest.mark.parametrize('shape,config_name', REDUCE_CASES)
    @pytest.mark.parametrize('k', [1, 3])
    def test_the_reduction_identity_holds_on_every_shape(self, shape,
                                                         config_name, k):
        lat = _latents(shape, seed=2, depth=k)
        h_side = (call(shape, config_name, lat, depth=0)
                  - call(shape, config_name, lat, depth=0, f_terms_only=True))
        summed = call(shape, config_name, lat, depth=k, reduce='sum')
        meaned = call(shape, config_name, lat, depth=k, reduce='mean')
        assert meaned == pytest.approx(
            h_side + (summed - h_side) / (k + 1), rel=1e-11, abs=1e-12)

    def test_the_f_side_probe_drops_the_h_only_term(self):
        """`f_terms_only` reads the part a depth copy carries, and no more."""
        lat = _latents(_SPLIT, seed=7)
        full = call(_SPLIT, 'base', lat, depth=0)
        f_side = call(_SPLIT, 'base', lat, depth=0, f_terms_only=True)
        # L_pred is the f-bearing half of the split shape; --rep-loss-weight 0
        # leaves it alone.
        assert f_side == pytest.approx(
            call(_SPLIT, 'base', lat, depth=0, rep_loss_weight=0.0),
            rel=0, abs=1e-12)
        assert abs(full - f_side) > 1e-6


def reduction_identity(h_side, summed, meaned, k):
    """`mean` divides the f-side of the total, and nothing else.

    `summed - h_side` is the f-side at depth k, k + 1 copies of it. The mean
    holds the same h-side and divides that part by k + 1.
    """
    assert meaned == pytest.approx(h_side + (summed - h_side) / (k + 1),
                                   rel=1e-11, abs=1e-12)


class TestEveryHOnlyTermEntersOnce:
    """One case per term that carries no `f`.

    `test_the_reduction_identity_holds_on_every_shape` reads its h-side out of
    `f_terms_only`, so it holds whatever the probe returns: a probe that
    dropped a term it should keep would move both sides of that equality by
    the same amount. Every case here defines the h-side another way — a weight
    set to zero, a constant taken from the floor helper, a term computed here
    with torch, or a term the trainer adds outside the reduced total.
    """

    @pytest.mark.parametrize('k', [1, 3])
    def test_l_rep_enters_once(self, k):
        """`--pred-loss-weight 0` leaves `w_rep · L_rep` — the h-side of the
        split shape, by a weight and not by the probe."""
        lat = _latents(_SPLIT, seed=31, depth=k)
        reduction_identity(
            call(_SPLIT, 'base', lat, depth=0, pred_loss_weight=0.0),
            call(_SPLIT, 'base', lat, depth=k, reduce='sum'),
            call(_SPLIT, 'base', lat, depth=k, reduce='mean'), k)

    @pytest.mark.parametrize('shape', [_SPLIT, _REP_ONLY])
    @pytest.mark.parametrize('k', [1, 3])
    def test_l_rep_moco_enters_once(self, shape, k):
        """The teacher-keyed form of the same term, on both shapes that carry
        it. `align_moco_rep` is this study's own cell: L_align on the student
        plus MoCo keys on L_rep. L_align is the f-side, so the h-side drops it
        and, on the split shape, L_pred with it."""
        lat = _latents(shape, seed=33, depth=k)
        off = {'align_loss_weight': 0.0}
        if shape == _SPLIT:
            off['pred_loss_weight'] = 0.0
        reduction_identity(
            call(shape, 'align_moco_rep', lat, depth=0, **off),
            call(shape, 'align_moco_rep', lat, depth=k, reduce='sum'),
            call(shape, 'align_moco_rep', lat, depth=k, reduce='mean'), k)

    @pytest.mark.parametrize('k', [1, 3])
    def test_the_f_rep_floor_constant_enters_once(self, k):
        """`subtract_contrastive_floor` on the split shape subtracts two
        constants. `f_pred` re-bases the f-side, so it repeats per depth;
        `f_rep` re-bases L_rep, so it enters once. The h-side here takes
        `f_pred` back from the floor helper, which is where the loss reads
        it too — a decomposition by weights alone cannot separate the two."""
        lat = _latents(_SPLIT, seed=35, depth=k)
        f, o = lat[0], lat[1]
        B, T, C = o.shape[0], o.shape[1], o.shape[2]
        f_pred, _ = _split_pred_rep_floors(0.1, B, T, C)
        h_side = call(_SPLIT, 'split_floor', lat, depth=0,
                      pred_loss_weight=0.0) + float(f_pred)
        reduction_identity(
            h_side,
            call(_SPLIT, 'split_floor', lat, depth=k, reduce='sum'),
            call(_SPLIT, 'split_floor', lat, depth=k, reduce='mean'), k)

    @pytest.mark.parametrize('k', [1, 3])
    def test_the_mse_h_only_half_enters_once(self, k):
        """`mse` is `mse(h_{t+1}, f_t) − mse(h_t, h_{t+1})`. The second half
        carries no f. It is computed here with torch, so nothing in the loss
        defines the h-side of this case."""
        lat = _latents('mse', seed=37, depth=k)
        o = lat[1]
        h_side = -float(torch.nn.functional.mse_loss(o[:, :-1], o[:, 1:]))
        reduction_identity(
            h_side,
            call('mse', 'base', lat, depth=k, reduce='sum'),
            call('mse', 'base', lat, depth=k, reduce='mean'), k)

    # ---- the two h-only terms the trainer adds outside the reduced total ----

    @pytest.mark.parametrize('fn', [align_moco_loss, sigreg_loss])
    def test_a_standalone_h_only_term_takes_no_depth(self, fn):
        """`align_loss` and the two CPC terms carry `rollout_latents` and
        `depth_reduce`, because they hold f. These two hold h alone, so a
        depth copy of either would be the h-side counted k + 1 times."""
        params = inspect.signature(fn).parameters
        for name in ('rollout_latents', 'depth_reduce'):
            assert name not in params, f'{fn.__name__} takes {name}'

    @pytest.mark.parametrize('name', ['align_moco_loss', 'sigreg_loss'])
    def test_the_trainer_passes_no_depth_to_it(self, name):
        calls = _calls_named(_train_tree(), name)
        assert calls, f'no {name} call in the trainer'
        for c in calls:
            for kw in c.keywords:
                assert kw.arg not in ('rollout_latents', 'depth_reduce'), \
                    f'{name} is given the depths'

    @pytest.mark.parametrize('k', [1, 3])
    def test_align_moco_enters_once(self, k):
        """The total the trainer builds: the reduced contrastive term plus
        `align_moco`. The reduction reaches the first and not the second."""
        lat = _latents(_REP_ONLY, seed=39, depth=k)
        moco = float(align_moco_loss(lat[1], lat[2], tau=0.1, weight=1.0))
        assert abs(moco) > 1e-6, 'the case carries no align_moco to hold out'
        cfg = {'align_loss_weight': 1.0}
        reduction_identity(
            call(_REP_ONLY, 'base', lat, depth=0, align_loss_weight=0.0)
            + moco,
            call(_REP_ONLY, 'base', lat, depth=k, reduce='sum', **cfg) + moco,
            call(_REP_ONLY, 'base', lat, depth=k, reduce='mean', **cfg) + moco,
            k)

    @pytest.mark.parametrize('k', [1, 3])
    def test_sigreg_enters_once(self, k):
        """SIGReg reads the encoder latent alone. Its projections are drawn
        fresh on every call, so this one passes a fixed set: a term that
        changed between the three calls would hide what they measure."""
        lat = _latents(_REP_ONLY, seed=41, depth=k)
        o = lat[1]
        g = torch.Generator().manual_seed(43)
        proj = torch.randn(o.shape[-1], 8, generator=g, dtype=o.dtype)
        proj = proj / proj.norm(dim=0, keepdim=True)
        sig = float(sigreg_loss(o, projections=proj))
        assert abs(sig) > 1e-9, 'the case carries no SIGReg to hold out'
        cfg = {'align_loss_weight': 1.0}
        reduction_identity(
            call(_REP_ONLY, 'base', lat, depth=0, align_loss_weight=0.0) + sig,
            call(_REP_ONLY, 'base', lat, depth=k, reduce='sum', **cfg) + sig,
            call(_REP_ONLY, 'base', lat, depth=k, reduce='mean', **cfg) + sig,
            k)


class TestAuxiliaryFBearingTerms:
    """The three standalone terms carry no h-only half, so the mean is the
    plain mean of their k + 1 copies."""

    @pytest.mark.parametrize('k', [1, 3])
    def test_align_loss_reduces(self, k):
        f, o, teacher_o, rollouts = _latents(_REP_ONLY, seed=11, depth=k)
        summed = float(align_loss(f, o, 1.0, rollout_latents=rollouts))
        meaned = float(align_loss(f, o, 1.0, rollout_latents=rollouts,
                                  depth_reduce='mean'))
        assert meaned == pytest.approx(summed / (k + 1), rel=1e-11)

    @pytest.mark.parametrize('k', [1, 4])
    def test_align_loss_of_identical_copies_does_not_scale(self, k):
        f, o, _, rollouts = flat_latents(depth=k, seed=13)
        base = float(align_loss(f, o, 1.0))
        assert float(align_loss(f, o, 1.0, rollout_latents=rollouts,
                                depth_reduce='mean')) == pytest.approx(
            base, rel=1e-11)

    @pytest.mark.parametrize('fn', [cpc_infonce_aux_loss, cpc_infonce_all_loss])
    @pytest.mark.parametrize('k', [1, 3])
    def test_cpc_infonce_reduces(self, fn, k):
        f, o, _, rollouts = _latents(_REP_ONLY, seed=17, depth=k)
        w1 = torch.nn.Linear(H, H, bias=False).to(torch.float64)
        summed = float(fn(f, o, w1, rollout_latents=rollouts))
        meaned = float(fn(f, o, w1, rollout_latents=rollouts,
                          depth_reduce='mean'))
        assert meaned == pytest.approx(summed / (k + 1), rel=1e-11)

    @pytest.mark.parametrize('fn', [align_loss, cpc_infonce_aux_loss,
                                    cpc_infonce_all_loss])
    def test_one_copy_reduces_to_itself(self, fn):
        f, o, _, _ = _latents(_REP_ONLY, seed=19)
        rest = (1.0,) if fn is align_loss else (
            torch.nn.Linear(H, H, bias=False).to(torch.float64),)
        assert float(fn(f, o, *rest, depth_reduce='mean')) == pytest.approx(
            float(fn(f, o, *rest)), rel=0, abs=1e-12)


class TestGuards:

    @pytest.mark.parametrize('bad', ['median', 'SUM', 'avg', 0])
    def test_an_unknown_reduction_raises_and_names_itself(self, bad):
        with pytest.raises(ValueError, match='train_rollout_reduce'):
            resolve_rollout_reduce(bad, None)

    def test_the_default_is_the_sum(self):
        assert resolve_rollout_reduce(None, None) == 'sum'

    def test_the_function_argument_wins_over_the_config_key(self):
        assert resolve_rollout_reduce('mean', 'sum') == 'mean'
        assert resolve_rollout_reduce(None, 'mean') == 'mean'

    def test_the_key_is_refused_on_the_shapes_that_cannot_compose(self):
        lat = _latents('cpc_multistep', seed=23, depth=1)
        with pytest.raises(NotImplementedError, match='train_rollout_depth'):
            call('cpc_multistep', 'base', lat, depth=1, reduce='mean')


class TestTrainerWiring:
    """The flag, its default, and the four calls it has to reach."""

    def test_the_flag_defaults_to_the_sum(self):
        defaults, choices = {}, {}
        for c in _calls_named(_train_tree(), 'add_argument'):
            if c.args and isinstance(c.args[0], ast.Constant):
                defaults[c.args[0].value] = _keyword(c, 'default')
                choices[c.args[0].value] = _keyword(c, 'choices')
        assert '--train-rollout-reduce' in defaults
        assert ast.literal_eval(defaults['--train-rollout-reduce']) == 'sum'
        assert ast.literal_eval(
            choices['--train-rollout-reduce']) == ['sum', 'mean']

    def test_the_config_key_carries_the_flag(self):
        src = TRAIN_PY.read_text()
        assert ('LOSS_SPEC.train_configuration["train_rollout_reduce"] = '
                'args.train_rollout_reduce') in src

    @pytest.mark.parametrize('name', ['align_loss', 'cpc_infonce_aux_loss',
                                      'cpc_infonce_all_loss'])
    def test_every_standalone_f_bearing_call_carries_the_reduction(self, name):
        """The main term reads the config key; these three take an argument,
        so an unwired call would sum while the run asked for the mean."""
        calls = [c for c in _calls_named(_train_tree(), name)
                 if any(kw.arg == 'rollout_latents' for kw in c.keywords)]
        assert calls, f'no {name} call carries rollout_latents'
        for c in calls:
            assert any(kw.arg == 'depth_reduce' for kw in c.keywords), \
                f'{name} takes the depths but not the reduction'

    def test_the_cli_refuses_a_third_word(self, tmp_path):
        out = subprocess.run(
            [sys.executable, str(TRAIN_PY), '--weight-decay', '0.1',
             '--hf-repo', 'none', '--hf-path', 'none',
             '--save-dir', str(tmp_path),
             '--train-rollout-reduce', 'median'],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, 'PYTHONPATH': str(REPO_ROOT)})
        assert out.returncode != 0
        assert '--train-rollout-reduce' in out.stderr


@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_a_depth_zero_run_is_unchanged_by_the_flag(tmp_path, reduce):
    """End to end: the reduction is a no-op at k = 0, to the last digit."""
    arm = ('--loss-shape', _REP_ONLY, '--align-loss-weight', '1.0')
    rows = run_short_training(tmp_path, f'k0_{reduce}', *arm,
                              '--train-rollout-depth', '0',
                              '--train-rollout-reduce', reduce)
    ref = run_short_training(tmp_path, 'k0_ref', *arm,
                             '--train-rollout-depth', '0')
    assert float(rows[0]['loss']) == pytest.approx(
        float(ref[0]['loss']), rel=0, abs=0)


def test_a_run_at_depth_two_trains_on_the_mean(tmp_path):
    """The study's own cell, k = 2, both reductions, on the real trainer.

    `loss_tau_ref` is pinned to depth 0, so an equal value says the two runs
    reached their first loss on the same weights and the same batch. `loss`
    then moves only because the reduction moved it.
    """
    arm = ('--loss-shape', _REP_ONLY, '--align-loss-weight', '1.0',
           '--train-rollout-depth', '2')
    rows = {r: run_short_training(tmp_path, f'k2_{r}', *arm,
                                  '--train-rollout-reduce', r)[0]
            for r in ('sum', 'mean')}
    assert float(rows['sum']['loss_tau_ref']) == pytest.approx(
        float(rows['mean']['loss_tau_ref']), rel=0, abs=1e-9), \
        'the two runs did not see the same batch'
    assert abs(float(rows['sum']['loss'])
               - float(rows['mean']['loss'])) > 1e-6, \
        'the run trains on the sum whatever the flag says'
    # Same objective, so the per-depth diagnostic is untouched by the flag.
    for j in range(3):
        assert float(rows['sum'][f'cos_err_d{j}']) == pytest.approx(
            float(rows['mean'][f'cos_err_d{j}']), rel=0, abs=1e-9)


def test_the_module_comment_states_both_reductions():
    """`src/loss.py`'s rollout-depth header is where a reader learns what the
    total is. It said SUM only, which is now one of two."""
    with open(loss_module.__file__) as fh:
        marker = fh.read().split('def rollout_depth_views')[0]
    assert 'train_rollout_reduce' in marker
    assert 'mean' in marker.lower()
