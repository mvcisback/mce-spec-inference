import aiger_bv as BV
import aiger_coins as C
from networkx.drawing import nx_pydot


from mce.test_scenarios import scenario_reactive
from mce.spec import concretize

from mce.qdd import spec2graph



def test_spec2graph():
    spec, sys = scenario_reactive()
    monitor = C.MDP(BV.aig2aigbv(spec.aig) | BV.sink(1, ['c_next']))
    cspec = concretize(monitor, sys, 3)

    graph, root = spec2graph(cspec)
    nx_pydot.write_dot(graph, "foo2.dot")

    assert len(graph.nodes) == 9
    assert len(graph.edges) == 14

    for node in graph.nodes:
        assert graph.out_degree[node] <= 2