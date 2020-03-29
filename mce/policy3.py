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
        return np.exp(self.graph.nodes[self.root]['lsat'])


def policy(spec: ConcreteSpec, coeff: Optional[float] = None):
    reference_graph, root, sink = spec2graph(spec)

    @fn.cache(5)  # Cache results for 5 seconds.
    def ppolicy(coeff):
        graph = reference_graph.reverse(copy=True)

        # Iterate in reverse topological order (ignoring dummy sink).
        for node in fn.rest(nx.topological_sort(graph)):
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
