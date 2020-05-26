import pytest

import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as PLTL
import networkx as nx 
import numpy as np
import funcy as fn

from mce.test_scenarios import scenario_reactive
from mce.spec import concretize
from mce.policy3 import fit
from mce.demos import prefix_tree


def act(action, coin):
    return {'a': (action,), 'c': (coin,),}


def test_prefix_tree():
    _, sys = scenario_reactive()
    spec, sys = scenario_reactive()
    
    encoded = [
        [act(True, True), act(True, True), act(True, True)],
        [act(True, True), act(False, True), act(False, True)],
    ]

    def to_demo(encoded_trc):
        io_seq = zip(encoded_trc, sys.aigbv.simulate(encoded_trc))
        for inputs, (outputs, _) in io_seq:
            outputs = fn.project(outputs, sys.outputs)
            inputs = fn.project(inputs, sys.inputs)
            yield inputs, outputs

    # Technical debt where sys_actions and env_actions 
    # are two different lists.
    demos = [list(zip(*to_demo(etrc))) for etrc in encoded]

    tree = prefix_tree(sys, demos)

    cspec = concretize(spec, sys, 3)
    ctrl = fit(cspec, 0.7, bv=True)
    lprob = tree.log_likelihood(ctrl, actions_only=True)
    assert lprob < 0

    assert tree.psat(cspec) == 1/2
