import aiger_bv as BV
import aiger_coins as C
import networkx as nx 

from mce.test_scenarios import scenario_reactive
from mce.spec import concretize
from mce.policy3 import fit
from mce.demos import annotate_surprise


def act(action, coin):
    return {'a': (action,), 'c': (coin,),}


def test_bdd_trace():
    spec, sys = scenario_reactive()
    monitor = C.MDP(BV.aig2aigbv(spec.aig) | BV.sink(1, ['c_next']))
    cspec = concretize(monitor, sys, 3)

    actions = [act(True, True), act(True, False), act(True, True)]
    ctrl = fit(cspec, 3)
    annotate_surprise(ctrl, [actions]*3)
    adj = nx.adjacency_matrix(ctrl.graph, weight="rel_entr")

    actions = [act(True, True), act(True, False), act(False, True)]
    ctrl = fit(cspec, 3)
    adj = nx.adjacency_matrix(ctrl.graph, weight="rel_entr")
    annotate_surprise(ctrl, [actions]*3)


