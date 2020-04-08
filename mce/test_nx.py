import aiger_bv as BV
import aiger_coins as C
import networkx as nx

from mce.test_scenarios import scenario_reactive
from mce.spec import concretize

from mce.nx import spec2graph


def test_spec2graph():
    spec, sys = scenario_reactive()
    monitor = C.MDP(BV.aig2aigbv(spec.aig) | BV.sink(1, ['c_next']))
    cspec = concretize(monitor, sys, 3)

    graph, root, _ = spec2graph(cspec)


    assert nx.is_directed_acyclic_graph(graph)

    # BDD size
    assert len(graph.nodes) == 10
    assert len(graph.edges) == 16

    for node in graph.nodes:
        assert graph.out_degree[node] <= 2


def test_reweighted():
    spec, sys = scenario_reactive()
    monitor = C.MDP(BV.aig2aigbv(spec.aig) | BV.sink(1, ['c_next']))

    # Hack too re-weight coinl
    sys2 = C.coin((1, 4), 'c') >> C.MDP(sys.aigbv >> BV.sink(1, ['##valid']) )
    cspec2 = concretize(monitor, sys2, 3)

    graph, root, _ = spec2graph(cspec2)

    assert nx.is_directed_acyclic_graph(graph)
    assert len(graph.nodes) == 12
    assert len(graph.edges) == 20

    for node in graph.nodes:
        assert graph.out_degree[node] <= 2


def test_nx2qdd():
    spec, sys = scenario_reactive()
    monitor = C.MDP(BV.aig2aigbv(spec.aig) | BV.sink(1, ['c_next']))
    cspec = concretize(monitor, sys, 3)

    graph, root, _ = spec2graph(cspec, qdd=True)

    assert nx.is_directed_acyclic_graph(graph)
    assert len(graph.nodes) == 12 + 4
    assert len(graph.edges) == 22

    for node in graph.nodes:
        assert graph.out_degree[node] <= 2
