"""Tests for `--train-rollout-reduce harmonic` (key `train_rollout_reduce`).

`harmonic` gives the depth-j copy of every f-bearing term the weight
1 / (j + 1), so the copy at the distance d = j + 1 gets 1/d: 1, 1/2, 1/3 and
1/4 at k = 3. Depth 0 keeps its weight 1, so the terms that carry no `f`
(L_rep, SIGReg, ...) and the depth-0 f-side enter as under `sum`. Only the
copies of depths 1..k shrink. At k = 0 the four reductions agree.

What this file pins:

* **The weights.** `reduce_depth_terms` adds copy j at 1 / (j + 1), also at
  the run's depth k = 31, and one copy returns itself.
  `resolve_rollout_reduce` accepts the word.
* **The identity on every shape of the #373 sweep.** At depth k the total is
  the depth-0 total plus each copy j at 1 / (j + 1). Copy j is the step of
  the sum from depth j - 1 to depth j.
* **The standalone f-bearing terms** (`align_loss` in both of its forms and
  the two CPC InfoNCE terms) reduce the same way.
* **The value-space objective refuses the word**, and names it.
* **The trainer wiring.** The CLI offers the word, and a short run of the real
  trainer at k = 2 trains on a total above the halving and below the sum, on
  the same batch.
"""

from __future__ import annotations

import ast

import pytest
import torch

import src.loss as loss_module
from src.forecasting_head import value_space_objective
from src.loss import (align_loss, cpc_infonce_all_loss, cpc_infonce_aux_loss,
                      reduce_depth_terms, resolve_rollout_reduce)

from tests.test_373_train_rollout_depth import (H, _REP_ONLY, _calls_named,
                                                _keyword, _latents,
                                                _train_tree, run_short_training)
from tests.test_401_rollout_reduce_mean import REDUCE_CASES, call


def harmonic(totals):
    """The harmonic total from the sums at depths 0..k: the depth-0 total
    plus each copy j, the step from depth j - 1 to depth j, at 1 / (j + 1)."""
    return totals[0] + sum((totals[j] - totals[j - 1]) / (j + 1)
                           for j in range(1, len(totals)))


class TestTheWeights:

    def test_copy_j_takes_one_over_j_plus_one(self):
        terms = [torch.tensor(v, dtype=torch.float64) for v in (1.0, 3.0, 5.0, 7.0)]
        assert float(reduce_depth_terms(terms, 'harmonic')) == pytest.approx(
            1.0 + 3.0 / 2 + 5.0 / 3 + 7.0 / 4, rel=0, abs=1e-15)

    def test_the_32_distances_take_one_over_d(self):
        """k = 31, the depth of the run: 32 copies of 1 add up to
        1 + 1/2 + ... + 1/32."""
        terms = [torch.tensor(1.0, dtype=torch.float64)] * 32
        assert float(reduce_depth_terms(terms, 'harmonic')) == pytest.approx(
            sum(1.0 / d for d in range(1, 33)), rel=0, abs=1e-13)

    def test_one_copy_returns_itself(self):
        term = torch.tensor(2.5, dtype=torch.float64)
        assert float(reduce_depth_terms([term], 'harmonic')) == 2.5

    def test_the_word_resolves(self):
        assert resolve_rollout_reduce('harmonic', None) == 'harmonic'
        assert resolve_rollout_reduce(None, 'harmonic') == 'harmonic'


class TestTheIdentity:

    @pytest.mark.parametrize('shape,config_name', REDUCE_CASES)
    @pytest.mark.parametrize('k', [1, 3])
    def test_each_copy_enters_at_one_over_j_plus_one(self, shape,
                                                     config_name, k):
        lat = _latents(shape, seed=2, depth=k)
        sums = [call(shape, config_name, lat, depth=j, reduce='sum')
                for j in range(k + 1)]
        assert call(shape, config_name, lat, depth=k,
                    reduce='harmonic') == pytest.approx(
            harmonic(sums), rel=1e-11, abs=1e-12)

    def test_depth_zero_is_the_sum(self):
        lat = _latents(_REP_ONLY, seed=3)
        cfg = {'align_loss_weight': 1.0}
        assert call(_REP_ONLY, 'base', lat, depth=0, reduce='harmonic',
                    **cfg) == call(_REP_ONLY, 'base', lat, depth=0,
                                   reduce='sum', **cfg)


class TestTheStandaloneTerms:

    @pytest.mark.parametrize('kind', ['cosine', 'l2_raw'])
    @pytest.mark.parametrize('k', [1, 3])
    def test_align_loss_weighs_each_depth(self, kind, k):
        f, o, _, rollouts = _latents(_REP_ONLY, seed=11, depth=k)
        sums = [float(align_loss(f, o, 1.0, kind=kind,
                                 rollout_latents=rollouts[:j] or None))
                for j in range(k + 1)]
        assert float(align_loss(f, o, 1.0, rollout_latents=rollouts,
                                depth_reduce='harmonic',
                                kind=kind)) == pytest.approx(
            harmonic(sums), rel=1e-11)

    @pytest.mark.parametrize('fn', [cpc_infonce_aux_loss, cpc_infonce_all_loss])
    @pytest.mark.parametrize('k', [1, 3])
    def test_cpc_infonce_weighs_each_depth(self, fn, k):
        f, o, _, rollouts = _latents(_REP_ONLY, seed=17, depth=k)
        w1 = torch.nn.Linear(H, H, bias=False).to(torch.float64)
        sums = [float(fn(f, o, w1, rollout_latents=rollouts[:j] or None))
                for j in range(k + 1)]
        assert float(fn(f, o, w1, rollout_latents=rollouts,
                        depth_reduce='harmonic')) == pytest.approx(
            harmonic(sums), rel=1e-11)


def test_the_value_space_objective_refuses_the_word():
    with pytest.raises(ValueError, match='harmonic'):
        value_space_objective(None, None, depth=1, reduce='harmonic')


def test_the_flag_offers_the_word():
    choices = {c.args[0].value: _keyword(c, 'choices')
               for c in _calls_named(_train_tree(), 'add_argument')
               if c.args and isinstance(c.args[0], ast.Constant)}
    assert 'harmonic' in ast.literal_eval(choices['--train-rollout-reduce'])


def test_a_run_at_depth_two_trains_on_the_harmonic_weights(tmp_path):
    """k = 2 on the real trainer. The same batch (equal `loss_tau_ref`)
    gives halving < harmonic < sum: the copy of depth 2 enters at 1/4, 1/3
    and 1, and the copy of depth 1 at 1/2, 1/2 and 1."""
    arm = ('--loss-shape', _REP_ONLY, '--align-loss-weight', '1.0',
           '--train-rollout-depth', '2')
    rows = {r: run_short_training(tmp_path, f'k2_{r}', *arm,
                                  '--train-rollout-reduce', r)[0]
            for r in ('sum', 'halving', 'harmonic')}
    for reduce in ('halving', 'harmonic'):
        assert float(rows['sum']['loss_tau_ref']) == pytest.approx(
            float(rows[reduce]['loss_tau_ref']), rel=0, abs=1e-9), \
            'the runs did not see the same batch'
    loss = {r: float(row['loss']) for r, row in rows.items()}
    assert loss['halving'] < loss['harmonic'] < loss['sum']


def test_the_module_comment_states_the_harmonic_weights():
    with open(loss_module.__file__) as fh:
        marker = fh.read().split('def rollout_depth_views')[0]
    assert 'harmonic' in marker
