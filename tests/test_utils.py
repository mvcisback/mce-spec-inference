from mce.utils import empirical_sat_prob
from mce.test_scenarios import SPEC1, SPEC2, SPEC3


ACTION_SEQS = 4*[3*[{'a': (True,), 'c': (True,)}]] + [
    3*[{'a': (True,), 'c': (False,)}],
    [{'a': (True,), 'c': (False,)}] + 2*[{'a': (False,), 'c': (True,)}],
]

TRCS = [(None, actions) for actions in ACTION_SEQS]


def test_empirical_sat_prob_smoke():
    specs = [SPEC1, SPEC2, SPEC3]
    psats = [5/6, 1, 4/6]

    for spec, psat in zip(specs, psats):
        assert psat == empirical_sat_prob(spec, TRCS)