__all__ = ['surprise_graph', 'encode_trcs', 'log_likelihoods']

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


def node2observed_bias(ctrl: BitPolicy, demos: Demos):
    dfa = ctrl.spec._as_dfa(qdd=True)

    def trace(actions):
        bits = ctrl.spec.flatten(actions)
        trc = ((s.ref, s.debt) for s in dfa.trace(bits))
        return zip(trc, bits)

    traces = fn.mapcat(trace, demos)
    node2actions = fn.group_values(traces)

    return fn.walk_values(np.mean, node2actions)


def surprise_graph(ctrl: BitPolicy, demos: Demos):
    observed_bias = node2observed_bias(ctrl, demos)

    graph, root, real_sinks = ctrl.markov_chain()
    for node in nx.topological_sort(graph):
        for node2 in graph.neighbors(node):
            action = graph.edges[node, node2]['action']

            if (node not in observed_bias):
                surprise = 0
            else:
                observed = observed_bias[node]
                if action is False:           # Could be {True, False, None}.
                    observed = 1 - observed   # Bias is prob of action = True.

                if observed == 0:
                    surprise = 0
                else:
                    expected = ctrl.prob(node, action, qdd=True)
                    surprise = observed * (np.log(observed) - np.log(expected))

            graph.edges[node, node2]['rel_entr'] = surprise

    return graph, root, real_sinks
