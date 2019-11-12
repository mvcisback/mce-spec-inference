import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as PLTL
import pytest
from theano import function
from numpy import logaddexp, log

from mce.policy import policy


def test_smoke():
    spec = PLTL.atom('a').historically()
    spec = BV.aig2aigbv(spec.aig)
    spec = C.circ2mdp(spec)

    mdp = C.circ2mdp(BV.identity_gate(1, 'a'))
    ctrl = policy(mdp, spec, horizon=3)
    assert len(ctrl.tbl) == 5

    V2 = logaddexp(-1, 1)
    V1 = logaddexp(log(2) - 1, V2)
    V0 = logaddexp(2*log(2) - 1, V1)

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
