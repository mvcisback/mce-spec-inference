from collections import defaultdict
from typing import Tuple, List, Sequence, Mapping

import funcy as fn
import networkx as nx
import numpy as np
from bdd2dfa.b2d import BNode
from scipy.special import logsumexp

from mce.policy3 import BitPolicy


Demo = List[Mapping[str, bool]]
Demos = List[Demo]

AbstractDemo = Sequence[Mapping[str, bool]]
AbstractDemos = Sequence[AbstractDemo]

Edge = Tuple[int, int]


def edge2bits(dfa, ctrl, bits):
    refs = [x.ref for x in dfa.trace(bits)]
    assert len(refs) == len(bits) + 1
    assert refs[0] in ctrl.graph.nodes

    curr, buff = refs[0], []
    for i, (ref, bit) in enumerate(zip(refs[1:], bits)):
        # First bit always counts as one.
        # Thus firs element of array represents visitation count.
        buff.append(int(bit or (len(buff) == 0)))

        if curr == ref:
            break  # At sink

        if ref not in ctrl.graph.nodes:
            continue

        yield (i, (curr, ref)), np.array(buff)
        curr, buff = ref, []


def demos2edge_dist(ctrl: BitPolicy, demos: Demos):
    mapping = {}
    dfa = ctrl.spec._as_dfa(qdd=True)

    for bits in map(ctrl.spec.flatten, demos):
        # Update edge counts
        mapping = fn.merge_with(sum, mapping, edge2bits(dfa, ctrl, bits))

    @fn.memoize
    def node_count(time, node) -> int:
        visited_edges = (
            e for e in ctrl.graph.out_edges(node) if (time, e) in mapping
        )
        return sum(mapping[time, e][0] for e in visited_edges)

    def to_freq(timed_edge_counts):
        (time, edge), counts = timed_edge_counts
        freq = counts / node_count(time, edge[0])
        assert 0 <= freq <= 1
        return (time, edge), freq 

    timed_edge_freqs = fn.walk(to_freq, mapping)

    # Reindex in terms of edges.
    return fn.group_values(
        (e, (t, f)) for (t, e), f in timed_edge_freqs.items()
    )


def annotate_surprise(ctrl: BitPolicy, demos: Demos):
    edge_dists = demos2edge_dist(ctrl, demos)

    def expected(edge):
        """Probability of bits on edges perscribed by the policy."""
        return fn.chain([ctrl.prob(*edge)], fn.repeat(0.5))

    for node in nx.topological_sort(ctrl.graph):
        out_edges = ctrl.graph.out_edges(node)

        for edge in ctrl.graph.out_edges(node):
            # Compute rel entr between empirical edge distribution
            # and policy edge distribution.

            dkl = 0
            for t, observed in edge_dists.get(edge, []):
                for p_policy, p_data in zip(expected(edge), observed):
                    if p_data == 0:
                        continue

                    dkl += p_data * (np.log(p_data) - np.log(p_policy))

            ctrl.graph[edge[0]][edge[1]]['rel_entr'] = dkl
