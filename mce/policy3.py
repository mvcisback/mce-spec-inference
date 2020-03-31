__all__ = ["Policy", "policy", "fit"]

from functools import partial
from typing import Tuple, List, Optional, Hashable

import attr
import funcy as fn
import networkx as nx
import numpy as np
import scipy as sp
from bidict import bidict
from scipy.special import logsumexp
from scipy.optimize import brentq

from mce.spec import ConcreteSpec
from mce.nx import spec2graph


@attr.s(frozen=True, auto_attribs=True)
class BitPolicy:
    graph: nx.DiGraph
    root: Hashable
    sinks: Tuple[Hashable] = attr.ib(converter=tuple)

    @property
    def psat(self) -> float:
        """
        Probability of satisfying the underlying concrete
        specification using this policy.
        """
        return np.exp(self.graph.nodes[self.root]['lsat'])

    def __getitem__(self, node_pair):
        node, node2 = node_pair
        assert node2 in self.graph[node], "node2 must be a child of node."
        return self.graph[node][node2]
        
    def prob(self, node, node2, log=False) -> float:
        """Probability of transition from node to node2."""
        prob = self[node, node2]['label']
        return np.log(prob) if log else prob

    def action(self, node, node2) -> Optional[bool]:
        """Action required to transition node to node2."""
        return self[node, node2]['action']

    def simulate(self, seed: int = None):
        """Generates tuples of (state, action, next_state) and the
        probability transitioning from state to next_state.
        """
        np.random.seed(seed)

        node = self.root
        while self.graph.out_degree(node) > 0:
            kids = list(self.graph.neighbors(node))
            assert len(kids) == self.graph.out_degree(node)

            probs = [self.prob(node, k) for k in kids]
            kid = np.random.choice(kids, p=probs)

            # Pr(s' | s, a)
            yield (node, self.action(node, kid), kid), self.prob(node, kid)
            node = kid

    def stochastic_matrix(self):
        """
        Returns underlying markov chain as stochastic_matrix and a map
        nodes to index.

        If the matrix more than one node, e.g., the concrete spec is
        not constant, then:

        - The false node is index 0.
        - The true node is index 1.
        - The root (start of episode) node is index 2.

        Note that true and false have self loops added to make the
        matrix stochastic.
        """
        false, true = sorted(self.sinks, key=lambda x: int(x.var))

        # Get nodes in correct order.
        nodes = [false, true, self.root]
        nodes += list(set(self.graph.nodes) - set(nodes))

        mat = nx.adjacency_matrix(self.graph, nodes, weight="label")

        # Make sink node self loop in stochastic matrix.
        mat[0, 0] = 1
        mat[1, 1] = 1

        return mat, bidict(enumerate(nodes))


def policy(spec: ConcreteSpec, coeff: Optional[float] = None):
    reference_graph, root, sinks = spec2graph(spec)

    @fn.cache(5)  # Cache results for 5 seconds.
    def ppolicy(coeff):
        graph = reference_graph.reverse(copy=True)

        # Iterate in reverse topological order (ignoring dummy sink).
        for node in nx.topological_sort(graph):
            if isinstance(node.var, bool):
                graph.nodes[node]['val'] = coeff*int(node.var)                
                graph.nodes[node]['lsat'] = 0 if node.var else -float('inf')
                continue
                
            kids = [c for (c, _) in graph.in_edges(node)]
            vals = np.array([graph.nodes[c]['val'] for c in kids])
            lsats = np.array([graph.nodes[c]['lsat'] for c in kids])

            if not node.decision:
                probs = np.array([graph[c][node]['label'] for c in kids])
                graph.nodes[node]['val'] = (probs * vals).sum()
            else:
                # Account for skipped decisions.
                # Note: This is arguably a bug in the model.
                skipped = np.array([
                    spec.order.skipped_decisions(node.level, c.level) for c in kids
                ])
                vals = vals + skipped * np.log(2)

                state_val = graph.nodes[node]['val'] = logsumexp(vals)
                probs = np.exp(vals - state_val)

                # Add action_probs to edges. 
                for prob, child in zip(probs, kids):
                    graph[child][node]['label'] = prob

            graph.nodes[node]['lsat'] = logsumexp(lsats, b=probs)
        
        return BitPolicy(graph.reverse(copy=False), root=root, sinks=sinks)

    if coeff is None:
        return ppolicy

    return ppolicy(coeff)


def fit(cspec: ConcreteSpec, psat: float, top: float=100) -> BitPolicy:
    pctrl = policy(cspec)

    def f(coeff):
        return pctrl(coeff).psat - psat

    if f(-top) > 0:
        coeff = 0
    elif f(top) < 0:
        coeff = top
    else:
        coeff = brentq(f, -top, top)

    if coeff < 0:
        # More likely the negated spec than this one.
        coeff = 0  

    return pctrl(coeff)
