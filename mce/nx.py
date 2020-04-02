from typing import Optional

import attr
import funcy as fn
import networkx as nx
from bdd2dfa.b2d import to_dfa, BNode

from mce.spec import ConcreteSpec


@attr.s(frozen=True, auto_attribs=True, repr=False)
class Node(BNode):
    decision: bool = False

    def __repr__(self):
        template = "{}\n----\nlevel={}\nid={}\nis_decision={}"
        return template.format(self.var, self.level, self.id, self.decision)

    @property
    def id(self) -> int:
        return self.node.ref

    @property
    def level(self) -> int:
        return self.node.level

    @property
    def var(self) -> Optional[str]:
        return self.label() if self.node.var is None else self.node.var

    @property
    def is_leaf(self):
        return self.node.var is None


def spec2graph(spec: ConcreteSpec) -> nx.DiGraph:
    dfa = spec._as_dfa()

    def is_sink(state) -> bool:
        return state.node.var is None

    def is_decision(state) -> bool:
        lvl = state.node.level
        return spec.order.is_decision(lvl)

    def merge(dists):
        return fn.merge_with(lambda xs: sum(xs)/2, *dists)

    sinks = []
    g = nx.DiGraph()

    @fn.memoize
    def _node(state):
        decision = is_decision(state) and not is_sink(state)
        node = Node(decision=decision, **state.__dict__)

        g.add_node(state.ref, lvl=node.level, var=node.var, decision=decision)

        if node.is_leaf:
            sinks.append(state.ref)

        return state.ref

    @fn.memoize
    def build(state):
        if is_sink(state):            
            return {state: 1}

        action2succ = {a: dfa._transition(state, a) for a in dfa.inputs}
        action2dist = {a: build(s) for a, s in action2succ.items()}

        if not is_decision(state):
            return merge(action2dist.values())

        for action, succ in action2succ.items():
            g.add_edge(_node(state), _node(succ), action=action, prob=None)

            if is_decision(succ):
                continue

            for state2, prob in action2dist[action].items():
                g.add_edge(
                    _node(succ), _node(state2), action=None, prob=prob
                )

        return {state: 1}    

    build(dfa.start)

    g.add_node("DUMMY", lvl=None, var=None, decision=False)
    for sink in sinks:
        g.add_edge(sink, "DUMMY", action=None, prob=1)

    g = nx.freeze(g)    

    return g, _node(dfa.start), sinks
