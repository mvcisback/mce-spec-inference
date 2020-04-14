__all__ = ['spec2graph']

from typing import Optional

import attr
import funcy as fn
import networkx as nx
from bdd2dfa.b2d import to_dfa, BNode

from mce.spec import ConcreteSpec


def spec2graph(spec: ConcreteSpec, qdd=False, dprob=None) -> nx.DiGraph:
    dfa = spec._as_dfa(qdd=qdd)

    if dprob is None:
        dprob = lambda *_: None

    def is_sink(state) -> bool:
        return state.node.var is None

    def is_decision(state) -> bool:
        lvl = state.node.level
        if qdd:
            lvl -= state.debt
        return spec.order.is_decision(lvl)

    def key(state):
        return (state.ref, state.debt) if qdd else state.ref

    sinks = set()
    g = nx.DiGraph()

    @fn.memoize
    def _node(state):
        decision = is_decision(state) and not is_sink(state)
        lvl = state.node.level
        var = state.label() if state.node.var is None else state.node.var
        node = key(state)
        g.add_node(node, lvl=lvl, var=var, decision=decision)
        return node

    stack = [dfa.start]
    while len(stack) > 0:
        state = stack.pop()
        action2succ = {a: dfa._transition(state, a) for a in dfa.inputs}

        for action, succ in action2succ.items():
            if key(succ) not in g.nodes:
                stack.append(succ)

            if succ == state:  # Sink
                sinks.add(_node(succ))
                continue                

            if qdd and state.debt > 0:
                g.add_edge(_node(state), _node(succ), action=None, prob=1)
            else:
                if is_decision(state):
                    prob = dprob(key(state), action)
                else:
                    prob = 1/2

                g.add_edge(_node(state), _node(succ), action=action, prob=prob)

    g.add_node("DUMMY", lvl=None, var=None, decision=False)
    for sink in sinks:
        g.add_edge(sink, "DUMMY", action=None, prob=1)

    g = nx.freeze(g)    
    return g, _node(dfa.start), list(sinks)
