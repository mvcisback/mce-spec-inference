import aiger_ptltl as PLTL
import funcy as fn
import hypothesis.strategies as st
from hypothesis import given

from mce import infer
from mce.test_scenarios import sys1, sys2, SPEC1, SPEC2
from mce.utils import empirical_sat_prob


def create_demos(n):
    actions1 = states1 = 3*[{'a': (True, )}]
    actions2 = states2 = 3*[{'a': (False, )}]
    actions3 = states3 = [
        {'a': (True, )}, 
        {'a': (False, )}, 
        {'a': (False, )},
    ]

    return [
        (actions2, states2),
        (actions3, states3),
    ] + n*[(actions1, states1)]


SPECS = [
    SPEC1,   # Historically a
    ~SPEC1,  # Once ~a
    SPEC2,   # True
    ~SPEC2   # False
]


def test_monotonic_empirical_sat_prob():
    mdp = sys1()
    psat = 0
    for i in range(5):
        demos = create_demos(i)
        prev, psat = psat, empirical_sat_prob(mdp, demos)
        assert prev <= psat


def test_infer_log_likelihood():
    mdp = sys1()
    demos = create_demos(10)

    spec, spec2score = infer.spec_mle(mdp, demos, SPECS)
    assert max(spec2score.values()) == spec2score[spec]
    assert max(spec2score.values()) == spec2score[SPEC1]
