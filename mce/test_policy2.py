from itertools import product

from pytest import approx

from mce.policy import policy as _policy
from mce.policy2 import policy_tbl, policy
from mce.test_scenarios import scenario1, scenario_reactive


def test_policy_tbl():
    for scenario in [scenario1, scenario_reactive]:
        spec, mdp = scenario()
        ctrl = _policy(mdp, spec, horizon=3)
        ctrl.fix_coeff(1)

        ctrl2 = policy(mdp, spec, horizon=3, coeff=1, manager=ctrl.bdd.bdd)
        ptbl = ctrl2.tbl
        assert ptbl.horizon == 3

        assert set(ctrl.tbl2.keys()) == set(ptbl.keys())

        for key in ptbl.keys():
            assert ctrl.tbl2[key] == approx(ptbl[key])

        lsat = ctrl.psat(return_log=True)
        assert lsat == approx(ptbl.lsat)

        n_bits = ptbl.order.total_bits * ptbl.horizon
        for trc in product(*(n_bits*[[True, False]])):
            llr1 = ptbl.log_likelihood_ratio(trc)
            llr2 = ctrl._log_likelihood(trc)
            assert llr2 == approx(llr1)

        ctrl = _policy(mdp, spec, horizon=3)
        ctrl.fix_coeff(2)
        ctrl2.coeff = 2
        assert ctrl2.coeff == 2


def test_long_horizon():
    for scenario in [scenario1, scenario_reactive]:
        spec, mdp = scenario()
        ctrl = policy(mdp, spec, horizon=20, coeff=1)
        ctrl.fit(0.96)
        assert ctrl.coeff > 0
        assert ctrl.psat == approx(0.96)
