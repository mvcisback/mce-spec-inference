from itertools import product

import hypothesis.strategies as st
import pytest

from hypothesis import given, settings
from numpy import logaddexp, exp, log, ceil

from mce.policy2 import policy as policy
from mce.test_scenarios import scenario1, scenario_reactive, SCENARIOS
from mce.test_scenarios import DET_SCENARIOS


V2 = logaddexp(-1, 1)
V1 = logaddexp(log(2) - 1, V2)
V0 = logaddexp(2*log(2) - 1, V1)


@settings(deadline=None)
@given(SCENARIOS)
def test_psat_monotonicity(scenario):
    spec, mdp = scenario()
    ctrl = policy(mdp, spec, horizon=3, coeff=1)

    prob = 0
    for i in range(10):
        ctrl.coeff = i
        prev, prob = prob, ctrl.psat
        assert prev <= prob


@settings(deadline=None)
@given(DET_SCENARIOS)
def test_coeff_zero(scenario):
    # Lower bound on negated policy.
    spec, mdp = scenario()
    spec2, mdp2 = scenario1(negate=True)

    ctrl = policy(mdp, spec, horizon=3, coeff=0)
    ctrl2 = policy(mdp2, spec2, horizon=3, coeff=0)

    psat, psat2 = ctrl.psat, ctrl2.psat
    if psat not in (0, 1):
        assert psat2 == pytest.approx(1 - psat)

    
def test_psat_mock():
    spec, mdp = scenario1()
    ctrl = policy(mdp, spec, horizon=3, coeff=1)

    psat_expected = exp(1 - V0)
    assert 0 <= psat_expected <= 1
    assert ctrl.psat == pytest.approx(psat_expected)


def test_trc_likelihood():
    spec, mdp = scenario1()
    ctrl = policy(mdp, spec, horizon=3, coeff=1)

    sys_actions = states = 3*[{'a': (True,)}]
    trc = ctrl.encode_trc(sys_actions, states)
    assert trc == [True, True, True]
    
    l_sat = 1 - log(exp(1) + 7*exp(-1))
    assert ctrl.log_likelihood_ratio(trc) == pytest.approx(l_sat)

    l_fail = -1 - log(exp(1) + 7*exp(-1))
    for trc in product(*(3*[[False, True]])):
        llr = ctrl.log_likelihood_ratio(trc)
        expected = l_sat if all(trc) else l_fail - (2 - trc.index(False))*log(2)
        assert llr == pytest.approx(expected)


def test_fit():
    spec, mdp = scenario1()
    ctrl = policy(mdp, spec, horizon=3, psat=0.8)
    assert ctrl.coeff > 1
    assert ctrl.psat == pytest.approx(0.8)


def test_reactive_psat(coeff=2, horizon=3):
    spec, mdp = scenario_reactive()
    ctrl = policy(mdp, spec, horizon=horizon, coeff=coeff)
    tbl = ctrl.tbl

    assert tbl[tbl.bdd.bdd.true, False] == exp(coeff)
    assert tbl[tbl.bdd.bdd.false, True] == exp(coeff)

    assert tbl[tbl.bdd.bdd.true, True] == exp(-coeff)
    assert tbl[tbl.bdd.bdd.false, False] == exp(-coeff)

    lvl_map = {}
    for (node, negated), val in ctrl.tbl.items():
        if node in (node.bdd.false, node.bdd.true):
            continue

        lvl_map.setdefault(node.level, val)
        assert lvl_map[node.level] == val

    assert set(lvl_map.keys()) == set(range(2*horizon - 1))

    for lvl, val in lvl_map.items():
        if lvl % 2:
            assert val == lvl_map[lvl + 1]
        else: 
            paths = 2**(horizon - (lvl/2)) - 1
            expected = exp(coeff) + paths*exp(-coeff)
            assert val == pytest.approx(expected)

    sat_prob = ctrl.psat
    log_prob = pytest.approx(ctrl.lsat)
    x = exp(2*coeff)
    expect_sat_prob = 1 / (1 + (2**horizon - 1)/ x)
    assert sat_prob == pytest.approx(expect_sat_prob)
