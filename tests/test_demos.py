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
from mce.demos import visitation_graph, log_likelihoods, prefix_tree


def act(action, coin):
    return {'a': (action,), 'c': (coin,),}


def test_visitation():
    spec, sys = scenario_reactive()
    cspec = concretize(spec, sys, 3)

    actions = [act(True, True), act(True, False), act(True, True)]
    assert not cspec.accepts(actions)
    ctrl = fit(cspec, 0.7)
    graph, root, sinks = visitation_graph(ctrl, [actions]*3)

    adj1 = nx.adjacency_matrix(graph, weight="visitation")
    assert adj1.sum() == 3 * 2 + 1  # horizon + bits per round + DUMMY

    actions2 = [act(False, False), act(True, False), act(True, True)]
    ctrl = fit(cspec, 0.7)
    graph, root, sinks = visitation_graph(ctrl, [actions, actions2]*3)

    from mce.draw import draw
    draw(graph, 'foo.dot')

    adj1 = nx.adjacency_matrix(graph, weight="visitation")
    assert adj1.sum() == 3 * 2 + 1


def test_trivially_true():
    _, sys = scenario_reactive()
    spec = PLTL.atom(True)
    cspec = concretize(spec, sys, 3)

    actions = [act(True, True), act(True, True), act(True, True)]
    assert cspec.accepts(actions)
    ctrl = fit(cspec, 0.7)
    assert ctrl.psat == 1
    graph, root, sinks = visitation_graph(ctrl, [actions]*3)
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

    for node in tree.tree.nodes():
        data = tree.tree.nodes()[node]
        data['label'] = f'{data["source"]}\nvisits={data.get("visits", 0)}'

    nx.drawing.nx_pydot.write_dot(tree.tree, 'prefix_tree.dot')

    cspec = concretize(spec, sys, 3)
    ctrl = fit(cspec, 0.7, bv=True)
    lprob = tree.log_likelihood(ctrl, actions_only=True)
    assert lprob < 0

    assert tree.psat(cspec) == 1/2
