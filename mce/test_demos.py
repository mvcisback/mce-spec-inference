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
    assert not cspec.accepts(actions)
    ctrl = fit(cspec, 0.7)
    graph, root, sinks = annotate_surprise(ctrl, [actions]*3)
    adj1 = nx.adjacency_matrix(graph, weight="rel_entr")

    actions = [act(True, True), act(True, False), act(False, True)]
    assert cspec.accepts(actions)
    ctrl = fit(cspec, 0.9)
    graph, root, sinks = annotate_surprise(ctrl, [actions]*3)
    adj2 = nx.adjacency_matrix(graph, weight="rel_entr")

    assert adj1.sum() > adj2.sum()

