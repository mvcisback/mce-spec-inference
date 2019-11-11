from math import log
from typing import Tuple

import attr


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
        elif lvl2 - lvl1 <= 1:
            return 0

        skipped_rounds = self.skipped_time_steps(lvl1, lvl2)

        lvl1, lvl2 = lvl1 + 1, lvl2 - 1
        levels_between = lvl2 - lvl1 + 1

        itvl1, itvl2 = map(self.interval, (lvl1, lvl2))
        if itvl1 == itvl2 and self.is_decision(lvl1):
            return levels_between
        
        ri1, ri2 = self.round_index(lvl1), self.round_index(lvl2)
        total = self.decision_bits * skipped_rounds
        if itvl1 != itvl2:
            total += self.is_decision(lvl2) * (ri2 + 1)
            total += self.is_decision(lvl1) * (self.decision_bits - ri1)

        assert 0 <= total <= levels_between
        return total
