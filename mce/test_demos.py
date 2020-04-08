import pytest

import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as PLTL
import networkx as nx 
import numpy as np

from mce.test_scenarios import scenario_reactive
from mce.spec import concretize
from mce.policy3 import fit
from mce.demos import surprise_graph, log_likelihoods


def act(action, coin):
    return {'a': (action,), 'c': (coin,),}


def test_surprise_increase():
    spec, sys = scenario_reactive()
    cspec = concretize(spec, sys, 3)

    actions = [act(True, True), act(True, False), act(True, True)]
    assert not cspec.accepts(actions)
    ctrl = fit(cspec, 0.7)
    graph, root, sinks = surprise_graph(ctrl, [actions]*3)
    adj1 = nx.adjacency_matrix(graph, weight="rel_entr")

    actions = [act(True, True), act(True, False), act(False, True)]
    assert cspec.accepts(actions)
    ctrl = fit(cspec, 0.9)
    graph, root, sinks = surprise_graph(ctrl, [actions]*3)
    adj2 = nx.adjacency_matrix(graph, weight="rel_entr")

    assert adj1.sum() > adj2.sum()


def test_trivially_true():
    _, sys = scenario_reactive()
    spec = PLTL.atom(True)
    cspec = concretize(spec, sys, 3)

    actions = [act(True, True), act(True, True), act(True, True)]
    assert cspec.accepts(actions)
    ctrl = fit(cspec, 0.7)
    assert ctrl.psat == 1
    graph, root, sinks = surprise_graph(ctrl, [actions]*3)
    adj1 = nx.adjacency_matrix(graph, weight="rel_entr")

    assert adj1.sum() > 0


def test_trivially_true_ll():
    _, sys = scenario_reactive()
    spec = PLTL.atom(True)
    cspec = concretize(spec, sys, 3)

    actions1 = [act(True, True), act(True, True), act(True, True)]
    actions2 = [act(True, True), act(False, True), act(False, True)]
    assert cspec.accepts(actions1)
    ctrl = fit(cspec, 0.7)
    assert ctrl.psat == 1
    lprob1 = log_likelihoods(ctrl, [actions1])
    lprob2 = log_likelihoods(ctrl, [actions2])
    assert lprob1 == lprob2
    assert lprob1 == pytest.approx(-6*np.log(2))  # Uniform.
