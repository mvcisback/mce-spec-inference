__all__ = ['visitation_graph', 'encode_trcs', 'log_likelihoods', 'prefix_tree']

from collections import Counter
from typing import Tuple, List, Sequence, Mapping

import attr
import funcy as fn
import networkx as nx
import numpy as np
from bdd2dfa.b2d import BNode
from scipy.special import logsumexp

from mce.policy3 import BitPolicy, BVPolicy
from mce.spec import ConcreteSpec
from mce.qbvnode import QBVNode


Demo = List[Mapping[str, bool]]
Demos = List[Demo]

AbstractDemo = Sequence[Mapping[str, bool]]
AbstractDemos = Sequence[AbstractDemo]

Edge = Tuple[int, int]


@attr.s(frozen=True, auto_attribs=True)
class PrefixTree:
    tree: nx.DiGraph
    root: str

    def action(self, tnode: str):
        """Returns the action that lead to this tree node."""
        return self.tree.nodes[tnode]['source']

    def visits(self, tnode: str) -> int:
        """Returns the number of times this tree node was
        visited in the demonstrations."""
        return self.tree.nodes[tnode]['visits']

    @property
    def ndemos(self) -> int:
        """Number of demonstrations this prefix tree represents."""
        return self.vists(self.root)

    def log_likelihood(self, ctrl: BVPolicy, relative=False) -> float:
        if not relative:
            raise NotImplementedError

        logp = 0
        for edge in self._generate_product_edges(ctrl.spec):
            (tnode, qbv_node), (tnode2, qbv_node2) = edge
            action, visits = self.action(tnode2), self.visits(tnode2)

            if qbv_node.is_decision:
                logp += visits * ctrl.prob(qbv_node, action, log=True)

        return logp

    def _generate_product_edges(self, spec: ConcreteSpec):
        """Walk edges of the synchronous composition of the prefix
        tree and a concrete spec's bit-vector qdd."""
        # TODO: consider letting spec produce qbvnodes.
        order = spec.order
        qnode = spec._as_dfa(qdd=True).start

        qbv_root = QBVNode(qnode, order)
        stack = [(self.root, qbv_root)]
        while stack:
            tnode, qbv_node = stack.pop()
            for tnode2 in self.tree.neighbors(tnode):
                qbv_node2 = qbv_node.transition(self.action(tnode2))
                yield (tnode, qbv_node), (tnode2, qbv_node2)
                stack.append((tnode2, qbv_node2))

    
    def psat(self, spec: ConcreteSpec):
        order = spec.order
        qnode = spec._as_dfa(qdd=True).start

        qbv_root = QBVNode(qnode, order)
        stack = [(self.root, qbv_root)]
        while stack:
            tnode, qbv_node = stack.pop()

            for tnode2 in self.tree.neighbors(tnode):
                action = self.tree.nodes[tnode2]['source']
                visits = self.tree.nodes[tnode2]['visits']

                if qbv_node.is_decision:
                    logp += visits * ctrl.prob(qbv_node, action, log=True)

                qbv_node2 = qbv_node.transition(action)
                stack.append((tnode2, qbv_node2))

        return logp
        # TODO 
        pass
            


def interleave(dyn, etrc):
    """
    1. Interleaves system and environment inputs.
    2. Makes dictionaries hashable for prefix tree.
    """
    for inputs in etrc:
        yield tuple(fn.project(inputs, dyn.inputs).items())
        yield tuple(fn.project(inputs, dyn.env_inputs).items())


def annotate_visits(tree, root, encoded):
    """Returns counter indicating how often each edge was 
       visited in the demonstrations.
    """

    for node in tree.nodes:
        tree.nodes[node]['visits'] = 0

    def advance(node, sym):
        for node2 in tree.neighbors(node):
            if sym == tree.nodes[node2]['source']:
                return node2

        raise RuntimeError

    for etrc in encoded:
        curr = root
        for sym in etrc:
            tree.nodes[curr]['visits'] += 1
            curr = advance(curr, sym)
        tree.nodes[curr]['visits'] += 1


def prefix_tree(dyn, trcs) -> PrefixTree:
    """Encodes i/o traces as sequences of (sys, env) actions."""
    encoded = encode_trcs(dyn, trcs)
    encoded = [list(interleave(dyn, etrc)) for etrc in encoded]

    tree, root = nx.prefix_tree(encoded)       # This tree has a dummy sink
    tree.remove_node(nx.generators.trees.NIL)  # which we don't want.

    annotate_visits(tree, root, encoded)

    return PrefixTree(tree, root)


def encode_trcs(dyn, trcs):
    """Encodes i/o traces as sequences of (sys, env) actions."""
    return [_encode_trc(dyn, *v) for v in trcs]


def _encode_trc(dyn, sys_actions, states):
    return dyn.encode_trc(sys_actions, states)


def log_likelihoods(ctrl: BitPolicy, demos: Demos) -> float:
    dfa = ctrl.spec._as_dfa(qdd=True)

    def trace(actions):
        bits = ctrl.spec.flatten(actions)
        trc = ((s.ref, s.debt) for s in dfa.trace(bits))
        return zip(trc, bits)

    # Convert to sequence of bdd, action pairs.
    io_seq = fn.chain(fn.mapcat(trace, demos))
    return sum(ctrl.prob(n, a, log=True, qdd=True) for n, a in io_seq)


def node2action_hist(spec: ConcreteSpec, demos: Demos):
    dfa = spec._as_dfa(qdd=True)

    def trace(actions):
        bits = spec.flatten(actions)
        trc = ((s.ref, s.debt) for s in dfa.trace(bits))
        return zip(trc, bits)

    traces = fn.mapcat(trace, demos)
    node2actions = fn.group_values(traces)
    return fn.walk_values(Counter, node2actions)


def visitation_graph(ctrl: BitPolicy, demos: Demos):
    observed_bias = node2action_hist(ctrl.spec, demos)

    graph, root, real_sinks = ctrl.markov_chain()
    for node in nx.topological_sort(graph):
        for node2 in graph.neighbors(node):
            action = graph.edges[node, node2]['action']

            if (node not in observed_bias):
                visitation = 0
            else:
                if action is not None:
                    visitation = observed_bias[node][action]
                else:
                    visitation = observed_bias[node][False]
                    visitation += observed_bias[node][True]

            graph.edges[node, node2]['visitation'] = visitation / len(demos)

    for sink in real_sinks:
        incoming = graph.in_edges(sink)
        visitation = sum(graph.edges[e]['visitation'] for e in incoming)
        graph.edges[sink, "DUMMY"]['visitation'] = visitation

    return graph, root, real_sinks
