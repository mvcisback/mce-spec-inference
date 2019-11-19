import aiger_bv as BV

from mce.utils import empirical_sat_prob, ltl2monitor
from mce.test_scenarios import SPEC1, SPEC2, SPEC3 


ACTION_SEQS = 4*[3*[{'a': (True,), 'c': (True,)}]] + [
    3*[{'a': (True,), 'c': (False,)}],
    [{'a': (True,), 'c': (False,)}] + 2*[{'a': (False,), 'c': (True,)}],
]

TRCS = [(None, actions) for actions in ACTION_SEQS]


def test_empirical_sat_prob_smoke():
    specs = [SPEC1, SPEC2, SPEC3]
    psats = [7/8, 1, ]

    for spec in zip(specs, psats):
        monitor = ltl2monitor(SPEC1)
        empirical_sat_prob(monitor, TRCS)
