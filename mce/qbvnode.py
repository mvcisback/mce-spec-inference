"""
Machinery for using word-level / bit-vector transitions on QDD rather
than bit-level.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple, List

import attr
import funcy
from pyrsistent import pmap
from bdd2dfa.b2d import QNode

from mce.order import BitOrder
from mce.bdd import TIMED_INPUT_MATCHER


Trace = List[Tuple[QNode, bool]]


@attr.s(frozen=True, auto_attribs=True)
class QBVNode:
    """Bit Vector wrapper around QDD Node."""
    qnode: QNode
    order: BitOrder

    def trace(self, action) -> Tuple[QBVNode, Trace]:
        """
        Returns:
          1. QBVNode transitioned to by action.
          2. The sequence of bits/qdd nodes along the path from
             self.qnode to the final qnode. Doesn't include
             final qnode.
        """
        trc = list(self._trace(pmap(action)))
        return attr.evolve(self, qnode=trc[-1][0]), trc

    @lru_cache
    def _trace(self, action):
        qnode, order = self.qnode, self.order

        round_idx = order.round_index(qnode.node.level)
        assert round_idx in (0, order.decision_bits)
        is_decision = round_idx == 0
        size = order.decision_bits if is_decision else order.chance_bits

        for i in range(size):
            name, _, idx = TIMED_INPUT_MATCHER.match(qnode.node.var).groups()
            bit = action[name][int(idx)]
            yield (qnode, bit)
            qnode = qnode.transition(bit)

        round_idx = order.round_index(qnode.node.level) == 0
        assert round_idx in (0, order.decision_bits)


    def transition(self, action) -> QBVNode:
        """QBVNode transitioned to by following bits in action."""
        return self.trace(action)[0]


__all__ = ['QBVNode']
