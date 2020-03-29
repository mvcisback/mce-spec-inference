__all__ = ["Policy", "policy", "fit"]

from functools import partial
from typing import Optional, Hashable

import attr
import funcy as fn
import networkx as nx
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import brentq

from mce.spec import ConcreteSpec
from mce.nx import spec2graph


@attr.s(frozen=True, auto_attribs=True)
class Policy:
    graph: nx.DiGraph
    root: Hashable

    @property
    def psat(self) -> float:
        return self.graph.nodes[self.root]['psat']        


def policy(spec: ConcreteSpec, coeff: Optional[float] = None):
    reference_graph, root, sink = spec2graph(spec)

    @fn.cache(5)  # Cache results for 5 seconds.
    def ppolicy(coeff):
        graph = reference_graph.reverse(copy=True)

        # Iterate in reverse topological order (ignoring dummy sink).
        for node in fn.rest(nx.topological_sort(graph)):
            if isinstance(node.var, bool):
                graph.nodes[node]['val'] = coeff*int(node.var)                
                graph.nodes[node]['psat'] = int(node.var)
                continue
                
            kids = [c for (c, _) in graph.in_edges(node)]
            vals = np.array([graph.nodes[c]['val'] for c in kids])
            psats = np.array([graph.nodes[c]['psat'] for c in kids])

            skipped = np.array([
                spec.order.skipped_decisions(node.level, c.level) for c in kids
            ])


            if not node.decision:
                probs = np.array([graph[c][node]['label'] for c in kids])

                # Note: This is arguably a bug in the model.
                probs *= 2**skipped  # Account for compressed paths.

                graph.nodes[node]['val'] = (probs * vals).sum()
                graph.nodes[node]['psat'] = (probs * psats).sum()
                continue

            state_val = graph.nodes[node]['val'] = logsumexp(vals)
            probs = []
            for k, child in zip(skipped, kids):
                action_val = graph.nodes[child]['val']

                # Note: This is arguably a bug in the model.
                action_val += k * np.log(2)  # Account for skipped paths.

                prob_action = np.exp(action_val - state_val)
                graph[child][node]['label'] = prob_action
                probs.append(prob_action)

            graph.nodes[node]['psat'] = (probs * psats).sum()
        
        return Policy(graph.reverse(copy=False), root=root)

    if coeff is None:
        return ppolicy

    return ppolicy(coeff)


def fit(cspec: ConcreteSpec, psat: float, top: float=100) -> Policy:
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
