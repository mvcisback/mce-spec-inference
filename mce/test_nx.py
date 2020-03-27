import aiger_bv as BV
import aiger_coins as C
from networkx.drawing import nx_pydot

from mce.test_scenarios import scenario_reactive
from mce.spec import concretize

from mce.nx import spec2graph


def test_spec2graph():
    spec, sys = scenario_reactive()
    monitor = C.MDP(BV.aig2aigbv(spec.aig) | BV.sink(1, ['c_next']))
    cspec = concretize(monitor, sys, 3)

    graph, root, sink = spec2graph(cspec)
    nx_pydot.write_dot(graph, "foo2.dot")

    assert len(graph.nodes) == 10
    assert len(graph.edges) == 16

    for node in graph.nodes:
        assert graph.out_degree[node] <= 2


    # Hack too re-weight coinl
    sys2 = C.coin((1, 4), 'c') >> C.MDP(sys.aigbv >> BV.sink(1, ['##valid']) )
    cspec2 = concretize(monitor, sys2, 3)

    graph, root, sink = spec2graph(cspec2)
    nx_pydot.write_dot(graph, "foo3.dot")

    assert len(graph.nodes) == 10
    assert len(graph.edges) == 16

    for node in graph.nodes:
        assert graph.out_degree[node] <= 2
