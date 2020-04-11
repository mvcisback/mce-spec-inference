import pytest

import aiger_bv as BV
import aiger_coins as C
import funcy as fn
import networkx as nx
import numpy as np
import scipy as sp
from hypothesis import given, settings
from networkx.drawing.nx_pydot import write_dot

from mce.test_scenarios import scenario1, scenario_reactive
from mce.test_scenarios import DET_SCENARIOS, SCENARIOS
from mce.spec import concretize
from mce.policy3 import policy, fit


def test_policy():
    spec, sys = scenario_reactive()
    cspec = concretize(spec, sys, 3)

    ctrls = [policy(cspec)(3), policy(cspec, 3)]
    for i, ctrl in enumerate(ctrls):
        assert 0 <= ctrl.psat <= 1
        assert len(ctrl.ref2action_dist) == 5
        assert all(len(v) == 2 for v in ctrl.ref2action_dist.values())
        assert all(
            sum(v.values()) == pytest.approx(1) 
            for v in ctrl.ref2action_dist.values()
        )

    pctrl = policy(cspec)
    # Agent gets monotonically more optimal
    psats = [pctrl(x).psat for x in range(10)]
    assert all(x >= y for x, y in fn.with_prev(psats, 0))


def test_policy_markov_chain():
    spec, sys = scenario_reactive()
    cspec = concretize(spec, sys, 3)

    ctrl = policy(cspec, 3)
    adj, _ = ctrl.stochastic_matrix()

    assert adj[0, 0] == 1
    assert adj[1, 1] == 1

    row_sums = adj.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums))


def test_policy_markov_chain_psat():
    spec, sys = scenario_reactive()
    cspec = concretize(spec, sys, 3)

    adj, _ = fit(cspec, 0.7).stochastic_matrix()

    root_vec = sp.sparse.csr_matrix((adj.shape[0], 1))
    root_vec[2] = 1

    true_vec = sp.sparse.csr_matrix((adj.shape[0], 1))
    true_vec[1] = 1

    vec = root_vec.T
    for _ in range(cspec.order.horizon * cspec.order.total_bits):
        vec = vec @ adj

    assert (vec @ true_vec).todense() == pytest.approx(0.7)

    vec = true_vec
    for _ in range(cspec.order.horizon * cspec.order.total_bits):
        vec = adj @ vec

    assert (root_vec.T @ vec).todense() == pytest.approx(0.7)


def test_fit():
    spec, sys = scenario_reactive()
    cspec = concretize(spec, sys, 3)

    assert fit(cspec, 0.7).psat == pytest.approx(0.7)


def test_sample_smoke():
    spec, sys = scenario_reactive()
    cspec = concretize(spec, sys, 3)
    ctrl = policy(cspec, 3)
    actions = ctrl.simulate()
    assert len(actions) == 3
    assert isinstance(cspec.accepts(actions), bool)


def test_long_horizon():
    # TODO: test that more scenarios work with long horizons.
    for scenario in [scenario1, scenario_reactive]:
        spec, mdp = scenario()
        cspec = concretize(spec, mdp, 20)
        ctrl = fit(cspec, 0.96)

        assert ctrl.psat == pytest.approx(0.96)


@given(SCENARIOS)
def test_psat_monotonicity(scenario):
    spec, mdp = scenario()
    cspec = concretize(spec, mdp, 3)

    prob = 0
    for i in range(10):
        ctrl = policy(cspec, i)
        prev, prob = prob, ctrl.psat
        assert prev <= prob
