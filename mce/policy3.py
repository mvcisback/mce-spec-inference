from __future__ import annotations

__all__ = ["BitPolicy", "policy", "fit", "BVPolicy"]

import random
from functools import lru_cache
from typing import Optional, Mapping, Union

import attr
import funcy as fn
import networkx as nx
import numpy as np
import scipy as sp
from bidict import bidict
from scipy.special import logsumexp
from scipy.optimize import root_scalar
from bdd2dfa.b2d import QNode

from mce.spec import ConcreteSpec as Spec
from mce.nx import spec2graph
from mce.qbvnode import QBVNode


@attr.s(frozen=True, auto_attribs=True)
class BitPolicy:
    spec: Spec
    ref2action_dist: Mapping[int, Mapping[bool, float]]
    lsat: float

    @property
    def psat(self) -> float:
        """
        Probability of satisfying the underlying concrete
        specification using this policy.
        """
        return np.exp(self.lsat)

    @property
    def size(self) -> int:
        """
        Number of BDD nodes with non-uniform defined action
        distributions.
        """
        return len(self.ref2action_dist)

    def prob(self, node, action, log=False, qdd=False) -> float:
        """Probability of agent applying action given bdd node."""
        if isinstance(node, QNode):
            node = (node.ref, node.debt)

        if action is None:
            prob = 1
        else:
            if qdd:
                node, debt = node
            else:
                debt = 0

            if debt > 0:
                prob = 0.5
            else:
                uniform = {True: 0.5, False: 0.5}
                prob = self.ref2action_dist.get(node, uniform)[action]
        return np.log(prob) if log else prob

    def markov_chain(self):
        def dprob(ref_debt, action):
            ref, debt = ref_debt
            return 1 if debt > 0 else self.prob(ref, action)

        return spec2graph(self.spec, qdd=True, dprob=dprob)

    def _simulate(self, seed: int = None, copolicy=False):
        np.random.seed(seed)
        graph, node, _ = self.markov_chain()

        while graph.out_degree(node) > 0:
            kids = list(graph.neighbors(node))
            assert len(kids) == graph.out_degree(node)

            probs = np.array([graph.edges[node, k]['prob'] for k in kids])

            if copolicy:
                probs = 1 - probs

            kid, *_ = random.choices(kids, weights=probs)

            # Pr(s' | s, a)
            action = graph.edges[node, kid]['action']

            node = kid
            if node == "DUMMY":
                continue
            elif action is None:
                action = random.choice([True, False])

            yield action

    def simulate(self, seed: int = None, copolicy=False):
        """
        Generates tuples of (state, action, next_state) and the
        probability transitioning from state to next_state.
        """
        bits = self._simulate(seed, copolicy)
        chunks = fn.chunks(self.spec.order.total_bits, bits)
        return [self.spec.unflatten(c)[0] for c in chunks]

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
        graph, root, real_sinks = self.markov_chain()
        assert len(real_sinks) > 0

        if len(real_sinks) == 1:
            mat = sp.sparse.csr_matrix((1, 1))
            mat[0, 0] = 1
            return mat, bidict(enumerate(real_sinks))

        var = lambda x: graph.nodes(data=True)[x]['var']  # noqa: E731
        false, true = sorted(real_sinks, key=var)

        # Get nodes in correct order.
        nodes = ["DUMMY", false, true, root]
        nodes += list(set(graph.nodes) - set(nodes))

        mat = nx.adjacency_matrix(graph, nodes, weight="prob")
        mat = mat[1:, 1:]

        # Make sink node self loop in stochastic matrix.
        mat[0, 0] = 1
        mat[1, 1] = 1

        return mat, bidict(enumerate(nodes[1:]))


def policy(spec: Spec, coeff: Optional[float] = None) -> BitPolicy:
    reference_graph, root, real_sinks = spec2graph(spec)

    @lru_cache(maxsize=3)
    def ppolicy(coeff):
        graph = reference_graph.reverse(copy=True)

        var = lambda x: graph.nodes(data=True)[x]['var']  # noqa: E731
        lvl = lambda x: graph.nodes(data=True)[x]['lvl']  # noqa: E731
        decision = lambda x: graph.nodes(data=True)[x]['decision']  # noqa: E731, E501

        # Iterate in reverse topological order (ignoring dummy sink).
        for node in fn.rest(nx.topological_sort(graph)):
            if isinstance(var(node), bool):
                graph.nodes[node]['val'] = coeff*int(var(node))
                graph.nodes[node]['lsat'] = 0 if var(node) else -float('inf')
                continue

            kids = [c for (c, _) in graph.in_edges(node)]
            vals = np.array([graph.nodes[c]['val'] for c in kids])
            lsats = np.array([graph.nodes[c]['lsat'] for c in kids])

            if not decision(node):
                probs = np.array([graph[c][node]['prob'] for c in kids])
                graph.nodes[node]['val'] = (probs * vals).sum()
            else:
                # Account for skipped decisions.
                # Note: This is arguably a bug in the model.
                order = spec.order
                skipped = np.array([
                    order.skipped_decisions(lvl(node), lvl(c)) for c in kids
                ])
                vals = vals + skipped * np.log(2)

                state_val = graph.nodes[node]['val'] = logsumexp(vals)
                probs = np.exp(vals - state_val)

                # Add action_probs to edges.
                for prob, child in zip(probs, kids):
                    graph[child][node]['prob'] = prob

            graph.nodes[node]['lsat'] = logsumexp(lsats, b=probs)

        def decision_nodes():
            return (n for n in graph.nodes if graph.nodes[n]['decision'])

        def action_dist(node):
            assert graph.nodes[node]['decision']
            out_edges = (graph.edges[e] for e in graph.in_edges(node))

            return {
                data['action']: data['prob'] for data in out_edges
            }

        ref2adist = {n: action_dist(n) for n in decision_nodes()}

        return BitPolicy(
            spec=spec,
            lsat=graph.nodes[root]['lsat'],
            ref2action_dist=ref2adist
        )

    if coeff is None:
        return ppolicy

    return ppolicy(coeff)


@attr.s(frozen=True, auto_attribs=True)
class BVPolicy:
    bitpolicy: BitPolicy

    @property
    def psat(self) -> float:
        return self.bitpolicy.psat

    def prob(self, node, action, log=False) -> float:
        if isinstance(node, QNode):
            node = QBVNode(node, self.bitpolicy.spec.order)
        assert isinstance(node, QBVNode)

        logp = 0
        _, trc = node.trace(action)
        for qnode, bit in trc:
            logp += self.bitpolicy.prob(qnode, bit, log=True, qdd=True)

        return logp if log else np.exp(logp)

    @property
    def size(self):
        return self.bitpolicy.size

    @property
    def spec(self):
        return self.bitpolicy.spec


Policy = Union[BitPolicy, BVPolicy]


def fit(spec: Spec, psat: float, top: float = 100, bv: bool = False) -> Policy:
    """Fit a max causal ent policy with satisfaction probability psat."""
    assert 0 <= psat <= 1
    pctrl = policy(spec)

    def f(coeff):
        return pctrl(coeff).psat - psat

    if f(-top) > 0:
        coeff = 0
    elif f(top) < 0:
        coeff = top
    else:
        res = root_scalar(f, bracket=(-top, top), method="brentq")
        coeff = res.root

    if coeff < 0:
        # More likely the negated spec than this one.
        coeff = 0

    ctrl = pctrl(coeff)
    return BVPolicy(ctrl) if bv else ctrl
