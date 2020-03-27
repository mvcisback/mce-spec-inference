import pytest

import aiger_bv as BV
import aiger_coins as C
from networkx.drawing.nx_pydot import write_dot

from mce.test_scenarios import scenario_reactive
from mce.spec import concretize
from mce.policy3 import policy


def test_policy():
    spec, sys = scenario_reactive()
    monitor = C.MDP(BV.aig2aigbv(spec.aig) | BV.sink(1, ['c_next']))
    cspec = concretize(monitor, sys, 3)

    graph = policy(cspec, 3)

    assert len(graph.nodes) == 10
    assert len(graph.edges) == 16

    for node in graph.nodes:
        assert graph.out_degree[node] <= 2
        
        prob = sum(graph[x][y]['label'] for x, y in graph.out_edges(node))
        assert pytest.approx(prob, 1)

    write_dot(graph, 'foo4.dot')
