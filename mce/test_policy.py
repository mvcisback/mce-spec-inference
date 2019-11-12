from itertools import combinations_with_replacement as combinations

import hypothesis.strategies as st
import pytest
import theano
from hypothesis import given
from numpy import logaddexp, exp, log

from mce.policy import policy
from mce.test_scenarios import scenario1, SCENARIOS


V2 = logaddexp(-1, 1)
V1 = logaddexp(log(2) - 1, V2)
V0 = logaddexp(2*log(2) - 1, V1)


def function(*args, **kwargs):
    kwargs['on_unused_input'] = 'ignore'
    return theano.function(*args, **kwargs)

def test_smoke_policy():
    spec, mdp = scenario1()

    ctrl = policy(mdp, spec, horizon=3)
    assert len(ctrl.tbl) == 5


    for node, val in ctrl.tbl.items():
        f = function([ctrl.coeff], val)

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


@given(SCENARIOS)
def test_psat_monotonicity(scenario):
    spec, mdp = scenario()
    ctrl = policy(mdp, spec, horizon=3)
    psat = function([ctrl.coeff], ctrl.psat())

    prev = 0
    for i in range(10):
        assert prev <= psat(i)


@given(SCENARIOS)
def test_coeff_zero(scenario):
    # Lower bound on negated policy.
    spec, mdp = scenario()
    ctrl = policy(mdp, spec, horizon=3)
    psat = function([ctrl.coeff], ctrl.psat())

    

    spec2, mdp2 = scenario1(negate=True)
    ctrl2 = policy(mdp2, spec2, horizon=3)
    psat2 = function([ctrl2.coeff], ctrl2.psat())

    assert psat(0) in (0, 1) or psat2(0) == psat(0)


@given(SCENARIOS)
def test_negated_policy_lowbound(scenario):
    spec, mdp = scenario()
    ctrl = policy(mdp, spec, horizon=3)
    psat = function([ctrl.coeff], ctrl.psat())

    spec2, mdp2 = scenario(negate=True)
    ctrl2 = policy(mdp2, spec2, horizon=3)
    psat2 = function([ctrl2.coeff], ctrl2.psat())

    for i in range(1, 10):
        assert psat2(i) >= 1 - psat(i)
        assert psat(i) >= 1 - psat2(i)

    
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
    
    l_sat = 1 - V0
    assert ctrl._log_likelihood(trc) == pytest.approx(l_sat)

    l_fail = log((1 - exp(l_sat)) / 7)
    for trc in combinations([False, True], 3):
        ll = ctrl._log_likelihood(trc)
        if all(trc):
            assert ll == pytest.approx(l_sat)
        else:
            assert ll == pytest.approx(l_fail)

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
