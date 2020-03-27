from functools import partial
from typing import Optional

import attr
import funcy as fn
import networkx as nx
import numpy as np
from scipy.special import logsumexp

from mce.spec import ConcreteSpec
from mce.nx import spec2graph


def policy(spec: ConcreteSpec, coeff: Optional[float] = None) -> nx.DiGraph:
    reference_graph, root, sink = spec2graph(spec)
    
    def ppolicy(coeff):
        graph = reference_graph.reverse(copy=True)


        # Iterate in reverse topological order (ignoring dummy sink).
        for node in fn.rest(nx.topological_sort(graph)):
            if isinstance(node.var, bool):
                graph.nodes[node]['val'] = coeff*int(node.var)
                continue
                
            children = [c for (c, _) in graph.in_edges(node)]
            vals = np.array([graph.nodes[c]['val'] for c in children])

            if not node.decision:
                probs = np.array([graph[c][node]['label'] for c in children])
                graph.nodes[node]['val'] = (probs * vals).sum()
                continue

            graph.nodes[node]['val'] = logsumexp(vals)
            for child in children:
                delta = graph.nodes[child]['val'] - graph.nodes[node]['val']
                graph[child][node]['label'] = np.exp(delta)
        
        return graph.reverse(copy=False)

    if coeff is None:
        return ppolicy

    return ppolicy(coeff)
