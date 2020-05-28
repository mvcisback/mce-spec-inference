"""
Machinery for using word-level / bit-vector transitions on QDD rather
than bit-level.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple, List

import attr
from pyrsistent import pmap
from bdd2dfa.b2d import QNode

from mce.order import BitOrder
from mce.utils import TIMED_INPUT_MATCHER


Trace = List[Tuple[QNode, bool]]


@attr.s(frozen=True)
class QBVNode:
    """Bit Vector wrapper around QDD Node."""
    qnode: QNode = attr.ib()
    order: BitOrder = attr.ib()

    @qnode.validator
    def check_round_idx(self, *_):
        round_idx = self.order.round_index(self.level) == 0
        return round_idx in (0, self.order.decision_bits)

    def trace(self, action) -> Tuple[QBVNode, Trace]:
        """
        Returns:
          1. QBVNode transitioned to by action.
          2. The sequence of bits/qdd nodes along the path from
             self.qnode to the final qnode. Doesn't include
             final qnode.
        """
        *trc, (qnext, _) = list(self._trace(pmap(action)))

        bv_next = attr.evolve(self, qnode=qnext)
        return bv_next, trc

    @lru_cache
    def _trace(self, action):
        return list(self.__trace(action))

    def __trace(self, action):
        qnode, order = self.qnode, self.order
        size = order.decision_bits if self.is_decision else order.chance_bits

        for i in range(size):
            var = qnode.node.bdd.var_at_level(qnode.node.level - qnode.debt)
            name, _, idx = TIMED_INPUT_MATCHER.match(var).groups()
            bit = action[name][int(idx)]
            yield (qnode, bit)
            qnode = qnode.transition(bit)

        yield (qnode, None)

    def transition(self, action) -> QBVNode:
        """QBVNode transitioned to by following bits in action."""
        return self.trace(action)[0]

    @property
    def is_decision(self):
        return self.order.is_decision(self.level)

    @property
    def level(self):
        return self.qnode.node.level - self.qnode.debt


__all__ = ['QBVNode']
