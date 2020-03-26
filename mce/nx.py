from fractions import Fraction

import attr
import funcy as fn
import networkx as nx
from bdd2dfa.b2d import to_dfa

from mce.spec import ConcreteSpec


@attr.s(frozen=True, auto_attribs=True, repr=False)
class Node:
    var: str
    level: int
    is_decision: bool
    id: int

    def __repr__(self):
        template = "{}\n----\nlevel={}\nid={}\ndecision={}"
        return template.format(self.var, self.level, self.id, self.is_decision)


def spec2graph(spec: ConcreteSpec) -> nx.DiGraph:
    dfa = to_dfa(spec.bexpr, qdd=False)

    count = 0

    @fn.memoize
    def _node(state):
        nonlocal count
        count += 1
        return Node(
            var=state.node.var,
            is_decision=spec.order.is_decision(state.node.level), 
            level=state.node.level,
            id=count,
        )

    g = nx.DiGraph()

    root = _node(dfa.start)
    stack = [(dfa.start, root)]
    while len(stack) > 0:
        state, node = stack.pop()

        if node.var is None:  # Leaf in BDD.
            continue

        for action in dfa.inputs:
            state2 = dfa._transition(state, action)
            node2 = _node(state2)
            stack.append((state2, node2))
            prob = None if node.is_decision else Fraction(1, 2)
            g.add_edge(node, node2, action=action, prob=prob)

    return g, root
