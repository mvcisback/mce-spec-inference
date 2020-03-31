from collections import Counter, defaultdict
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

Edge = Tuple[Node, Node]
Edges = Sequence[Tuple]


def demo_to_decision_edges(ctrl: BitPolicy, demo: AbstractDemo) -> Edges:
    is_decision = ctrl.spec.order.is_decision

    def bnode2node(bnode: BNode) -> Node:
        lvl = bnode.node.level
        decision = is_decision(lvl) and bnode.node.var is not None
        return Node(decision=decision, **bnode.__dict__)

    demo = map(bnode2node, demo)

    for curr, prev in fn.rest(fn.with_prev(demo)):
        edge = (prev, curr)
        if edge in ctrl.graph.edges:
            yield edge


def decision_edge_histograph(ctrl: BitPolicy, demos: Demos) -> Counter:
    """Returns non-zero visitation counts of decision edges by demos."""
    demos: AbstractDemos = map(ctrl.spec.abstract_trace, demos)
    edges: Edges = [list(demo_to_decision_edges(ctrl, demo)) for demo in demos]
    return Counter(fn.cat(edges))


def annotate_surprise(ctrl: BitPolicy, demos: Demos):
    mat, node2idx = ctrl.stochastic_matrix()
    hist = decision_edge_histograph(ctrl, demos)

    for node in nx.topological_sort(ctrl.graph):
        out_edges = ctrl.graph.out_edges(node)
        demos_at_node = sum(hist[e] for e  in out_edges)

        for edge in ctrl.graph.out_edges(node):
            # Compute rel entr between empirical edge distribution
            # and policy edge distribution.
            if demos_at_node == 0:
                dkl = 0
            else:
                p_data = hist[edge] / demos_at_node
                p_policy = ctrl.prob(*edge)
                dkl = rel_entr([p_data], [p_policy])[0]

            ctrl.graph[edge[0]][edge[1]]['rel_entr'] = dkl
