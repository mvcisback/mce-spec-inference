from typing import Union

import attr
import funcy as fn
import networkx as nx
from bdd2dfa.b2d import to_dfa

from mce.spec import ConcreteSpec


@attr.s(frozen=True, auto_attribs=True, repr=False)
class Node:
    var: Union[str, bool]
    level: int
    id: int
    decision: int

    def __repr__(self):
        template = "{}\n----\nlevel={}\nid={}\nis_decision={}"
        return template.format(self.var, self.level, self.id, self.decision)


def spec2graph(spec: ConcreteSpec) -> nx.DiGraph:
    dfa = to_dfa(spec.bexpr, qdd=False)

    def is_sink(state) -> bool:
        return state.node.var is None

    def is_decision(state) -> bool:
        lvl = state.node.level
        return spec.order.is_decision(lvl)


    def merge(dists):
        return fn.merge_with(lambda xs: sum(xs)/2, *dists)

    count = 0
    sinks = []

    @fn.memoize
    def _node(state):
        nonlocal count
        count += 1

        var = state.node.var
        if state.node.var is None:
            var = dfa._label(state)

        node = Node(
            var=var,
            id=count, 
            level=state.node.level,
            decision=is_decision(state) and not is_sink(state)
        )

        if state.node.var is None:
            sinks.append(node)

        return node

    g = nx.DiGraph()

    @fn.memoize
    def build(state):
        if is_sink(state):            
            return {state: 1}

        action2succ = {a: dfa._transition(state, a) for a in dfa.inputs}
        action2dist = {a: build(s) for a, s in action2succ.items()}

        if not is_decision(state):
            return merge(action2dist.values())

        for action, succ in action2succ.items():
            g.add_edge(_node(state), _node(succ), action=action, label=None)

            if is_decision(succ):
                continue

            for state2, weight in action2dist[action].items():
                g.add_edge(
                    _node(succ), _node(state2), action=None, label=weight
                )

        return {state: 1}    

    build(dfa.start)

    # Add dummy sink node so there is a unique start and end to the graph.
    assert len(sinks) == 2
    true, false = sinks
    assert true.level == false.level
    
    dummy_sink = Node(
        var=None,
        id=count + 1, 
        level=true.level + 1,
        decision=False,
    )    

    g.add_edge(true, dummy_sink, action=None, label=1)
    g.add_edge(false, dummy_sink, action=None, label=1)

    return g, _node(dfa.start), dummy_sink
