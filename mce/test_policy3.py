import pytest

import aiger_bv as BV
import aiger_coins as C
import funcy as fn
from networkx.drawing.nx_pydot import write_dot

from mce.test_scenarios import scenario_reactive
from mce.spec import concretize
from mce.policy3 import policy, fit


def test_policy():
    spec, sys = scenario_reactive()
    monitor = C.MDP(BV.aig2aigbv(spec.aig) | BV.sink(1, ['c_next']))
    cspec = concretize(monitor, sys, 3)

    ctrls = [policy(cspec)(3), policy(cspec, 3)]
    for i, ctrl in enumerate(ctrls):
        assert 0 <= ctrl.psat <= 1
        graph = ctrl.graph
        assert len(graph.nodes) == 10
        assert len(graph.edges) == 16

        for node in graph.nodes:
            assert graph.out_degree[node] <= 2

            prob = sum(graph[x][y]['label'] for x, y in graph.out_edges(node))
            assert pytest.approx(prob, 1)

    pctrl = policy(cspec)
    # Agent gets monotonically more optimal
    psats = [pctrl(x).psat for x in range(10)]
    assert all(x >= y for x, y in fn.with_prev(psats, 0))


def test_fit():
    spec, sys = scenario_reactive()
    monitor = C.MDP(BV.aig2aigbv(spec.aig) | BV.sink(1, ['c_next']))
    cspec = concretize(monitor, sys, 3)

    assert fit(cspec, 0.7).psat == pytest.approx(0.7)
