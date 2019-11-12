import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as PLTL
import pytest
from theano import function
from numpy import logaddexp, exp, log

from mce.policy import policy


def sys1():
    spec = PLTL.atom('a').historically()
    spec = BV.aig2aigbv(spec.aig)
    spec = C.circ2mdp(spec)

    mdp = C.circ2mdp(BV.identity_gate(1, 'a'))    
    return spec, mdp


V2 = logaddexp(-1, 1)
V1 = logaddexp(log(2) - 1, V2)
V0 = logaddexp(2*log(2) - 1, V1)


def test_smoke_policy():
    spec, mdp = sys1()

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

    prev = 0
    for i in range(10):
        assert prev <= psat(i)


def test_psat_mock():
    spec, mdp = sys1()
    ctrl = policy(mdp, spec, horizon=3)
    ctrl.fix_coeff(1)
    assert ctrl._fitted

    psat_expected = exp(1 - V0)
    assert 0 <= psat_expected <= 1
    assert ctrl.psat() == pytest.approx(psat_expected)
