from itertools import product

import hypothesis.strategies as st
import pytest
import theano
import theano.tensor as T
from hypothesis import given, settings
from numpy import logaddexp, exp, log, ceil

from mce.policy import policy
from mce.test_scenarios import scenario1, scenario_reactive, SCENARIOS
from mce.test_scenarios import DET_SCENARIOS


V2 = logaddexp(-1, 1)
V1 = logaddexp(log(2) - 1, V2)
V0 = logaddexp(2*log(2) - 1, V1)


def function(*args, **kwargs):
    kwargs['on_unused_input'] = 'ignore'
    kwargs['mode'] = 'FAST_COMPILE'
    return theano.function(*args, **kwargs)

def test_smoke_policy():
    spec, mdp = scenario1()

    ctrl = policy(mdp, spec, horizon=3)
    assert len(ctrl.tbl2) == 5

    for (node, _), val in ctrl.tbl2.items():
        f = function([ctrl.coeff], T.log(val))

        if node.var is None:
            expected = 2*int(node == ctrl.bdd.bdd.true) - 1
            assert f(3) == 3*expected
            assert f(0) == 0
            continue

        action = ctrl.relabels.inv[node.var]
        if action == 'a[0]##time_2':
            assert f(1) == pytest.approx(V2)
        elif action == 'a[0]##time_1':
            assert f(1) == pytest.approx(V1)
        elif action == 'a[0]##time_0':
            assert f(1) == pytest.approx(V0)

    psat = function([ctrl.coeff], ctrl.psat())

    psat_expected = exp(1 - V0)
    assert 0 <= psat_expected <= 1
    assert psat(1) == pytest.approx(psat_expected)

    prand = ctrl.bdd.count(3) / 2**3
    assert psat(0) == pytest.approx(prand)


@settings(deadline=None)
@given(SCENARIOS)
def test_psat_monotonicity(scenario):
    spec, mdp = scenario()
    ctrl = policy(mdp, spec, horizon=3)
    psat = function([ctrl.coeff], ctrl.psat())

    prob = 0
    for i in range(10):
        prev, prob = prob, psat(i)
        assert prev <= prob


@settings(deadline=None)
@given(DET_SCENARIOS)
def test_coeff_zero(scenario):
    # Lower bound on negated policy.
    spec, mdp = scenario()
    ctrl = policy(mdp, spec, horizon=3)
    psat = function([ctrl.coeff], ctrl.psat())

    spec2, mdp2 = scenario1(negate=True)
    ctrl2 = policy(mdp2, spec2, horizon=3)
    psat2 = function([ctrl2.coeff], ctrl2.psat())

    assert psat(0) in (0, 1) or psat2(0) == pytest.approx(1 - psat(0))

    
def test_psat_mock():
    spec, mdp = scenario1()
    ctrl = policy(mdp, spec, horizon=3)
    ctrl.fix_coeff(1)
    assert ctrl._fitted

    psat_expected = exp(1 - V0)
    assert 0 <= psat_expected <= 1
    assert ctrl.psat() == pytest.approx(psat_expected)


def test_trc_likelihood():
    spec, mdp = scenario1()
    ctrl = policy(mdp, spec, horizon=3)
    ctrl.fix_coeff(1)

    sys_actions = states = 3*[{'a': (True,)}]
    trc = ctrl.encode_trc(sys_actions, states)
    assert trc == [True, True, True]
    
    l_sat = 1 - log(exp(1) + 7*exp(-1))
    assert ctrl._log_likelihood(trc) == pytest.approx(l_sat)

    l_fail = -1 - log(exp(1) + 7*exp(-1))
    for trc in product(*(3*[[False, True]])):
        ll = ctrl._log_likelihood(trc)
        expected = l_sat if all(trc) else l_fail
        assert ll == pytest.approx(expected)

    sys_actions2 = states2 = 3*[{'a': (False,)}]
    demos = [(sys_actions, states), (sys_actions2, states2)]
    l_demo = ctrl.log_likelihood(demos)
    assert l_demo == pytest.approx(l_fail + l_sat)


def test_empierical_sat_prob():
    spec, mdp = scenario1()
    ctrl = policy(mdp, spec, horizon=3)

    sys_actions = states = 3*[{'a': (True,)}]
    sys_actions2 = states2 = 3*[{'a': (False,)}]
    demos = [(sys_actions, states), (sys_actions2, states2)]

    assert ctrl.empirical_sat_prob(demos) == 1/2


def test_fit():
    spec, mdp = scenario1()
    ctrl = policy(mdp, spec, horizon=3)
    ctrl.fit(0.8)
    assert ctrl.coeff > 1
    assert ctrl.psat() == pytest.approx(0.8)


def test_reactive_psat(coeff=2, horizon=3):
    spec, mdp = scenario_reactive()
    ctrl = policy(mdp, spec, horizon=horizon)
    ctrl.fix_coeff(coeff)

    assert ctrl.tbl2[ctrl.bdd.bdd.true, False] == exp(coeff)
    assert ctrl.tbl2[ctrl.bdd.bdd.true, True] == exp(-coeff)
    assert ctrl.tbl2[ctrl.bdd.bdd.false, False] == exp(-coeff)
    assert ctrl.tbl2[ctrl.bdd.bdd.false, True] == exp(coeff)

    lvl_map = {}
    for (node, negated), val in ctrl.tbl2.items():
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

    sat_prob = ctrl.psat()
    log_prob = pytest.approx(ctrl.psat(return_log=True))
    x = exp(2*coeff)
    expect_sat_prob = 1 / (1 + (2**horizon - 1)/ x)
    assert sat_prob == pytest.approx(expect_sat_prob)
