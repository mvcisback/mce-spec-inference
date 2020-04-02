from collections import defaultdict
from typing import Tuple, List, Sequence, Mapping

import funcy as fn
import networkx as nx
import numpy as np
from bdd2dfa.b2d import BNode
from scipy.special import rel_entr, logsumexp

from mce.nx import Node
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
    for ref, bit in zip(refs[1:], bits):
        buff.append(int(bit))

        if ref not in ctrl.graph.nodes:
            continue
            
        # Note first element of array  represents visitation count.
        yield (curr, ref), np.array([1] + buff)
        curr, buff = ref, []


def demos2edge_dist(ctrl: BitPolicy, demos: Demos):
    mapping = {}
    dfa = ctrl.spec._as_dfa(qdd=True)

    for bits in map(ctrl.spec.flatten, demos):
        # Update edge counts
        mapping = fn.merge_with(sum, mapping, edge2bits(dfa, ctrl, bits))

    def to_dist(x):
        return x[1:] / x[0]

    return fn.walk_values(to_dist, mapping)


def annotate_surprise(ctrl: BitPolicy, demos: Demos):
    edge_dists = demos2edge_dist(ctrl, demos)

    for node in nx.topological_sort(ctrl.graph):
        out_edges = ctrl.graph.out_edges(node)

        for edge in ctrl.graph.out_edges(node):
            # Compute rel entr between empirical edge distribution
            # and policy edge distribution.

            if edge not in edge_dists:
                dkl = 0 
            else:
                p_data = edge_dists[edge]
                p_policy = [ctrl.prob(*edge)] + (len(p_data) - 1)*[0.5]
                dkl = rel_entr(p_data, p_policy)[0]

            ctrl.graph[edge[0]][edge[1]]['rel_entr'] = dkl
