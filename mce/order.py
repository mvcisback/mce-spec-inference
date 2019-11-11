from math import log
from typing import Tuple

import attr


def skipped_decisions_naive(order, lvl1, lvl2):
    if lvl2 < lvl1:
        return -order.skipped_decisions(lvl2, lvl1)

    return sum(order.is_decision(lvl) for lvl in range(lvl1 + 1, lvl2))


@attr.s(frozen=True, auto_attribs=True)
class BitOrder:
    decision_bits: int
    chance_bits: int
    horizon: int

    @property
    def total_bits(self) -> int:
        return self.decision_bits + self.chance_bits

    def round_index(self, lvl: int) -> int:
        return lvl % self.total_bits

    def is_decision(self, lvl: int) -> bool:
        return self.round_index(lvl) < self.decision_bits

    def delta(self, lvl1, lvl2, val):
        return val + self.skipped_decisions(lvl1, lvl2)*log(2)

    def time_interval(self, lvl: int) -> Tuple[int]:
        idx = self.round_index(lvl)
        return lvl - idx, lvl + (self.total_bits - 1 - idx)
    
    def interval(self, lvl: int) -> Tuple[int]:
        bot, top = self.time_interval(lvl)
        boundary = bot + self.decision_bits - 1

        if self.is_decision(lvl):
            return bot, boundary
        return boundary + 1, top

    def time_step(self, lvl):
        return lvl // self.total_bits

    def skipped_time_steps(self, lvl1, lvl2):
        if lvl2 < lvl1:
            return -self.skipped_time_steps(lvl2, lvl1)

        t1_prev, t2_prev = map(self.time_step, (lvl1, lvl2))
        return max(0, t2_prev - t1_prev - 1)

    def skipped_decisions(self, lvl1, lvl2) -> int:
        if lvl2 < lvl1:
            return -self.skipped_decisions(lvl2, lvl1)
        return skipped_decisions_naive(self, lvl1, lvl2)
