import pytest
import theano

from mce.policy import policy
from mce.policy2 import policy_tbl
from mce.test_scenarios import scenario1, scenario_reactive


def function(*args, **kwargs):
    kwargs['on_unused_input'] = 'ignore'
    kwargs['mode'] = 'FAST_COMPILE'
    return theano.function(*args, **kwargs)


def test_smoke():
    for scenario in [scenario1, scenario_reactive]:
        spec, mdp = scenario()
        ctrl = policy(mdp, spec, horizon=3)

        ptbl = policy_tbl(ctrl.bdd, ctrl.order, 1)
        assert ptbl.horizon == 3

        assert set(ctrl.tbl2.keys()) == set(ptbl.keys())

        for key in ptbl.keys():
            assert ctrl.tbl2[key].eval({ctrl.coeff: 1}) == ptbl[key]

        lsat = ctrl.psat(return_log=True).eval({ctrl.coeff: 1})
        assert lsat == pytest.approx(ptbl.lsat)
