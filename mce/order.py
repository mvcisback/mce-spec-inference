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

    def is_decision(self, lvl_ctx) -> bool:
        lvl = lvl_ctx if isinstance(lvl_ctx, int) else lvl_ctx.curr_lvl
        return self.round_index(lvl) < self.decision_bits

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
        elif self.total_bits == 1:
            assert self.decision_bits == 1
            return lvl2 - lvl1 - 1

        itvl1, itvl2 = map(self.time_interval, (lvl1 + 1, lvl2 - 1))
        if itvl1 == itvl2:
            return self._rrd(lvl1 + 1, lvl2 - 1)

        skipped_rounds = self.skipped_time_steps(lvl1, lvl2)

        start = (lvl1 >= itvl1[0])*self._rrd(lvl1 + 1, itvl1[1])
        middle = self.decision_bits * skipped_rounds
        end = (lvl2 <= itvl2[1])*self._used_decisions(lvl2 - 1)
        return start + middle + end

    def _rrd(self, lvl1, lvl2):
        """Remaining Round Decisions"""
        if not self.is_decision(lvl1):
            return 0

        idx1, idx2 = map(self.round_index, (lvl1, lvl2))
        idx2 = min(idx2, self.decision_bits - 1)
        return idx2 - idx1 + 1

    def _used_decisions(self, lvl):
        """Remaining Round Decisions"""
        return min(self.decision_bits, self.round_index(lvl) + 1)

    def on_boundary(self, ctx):
        if self.first_real_decision(ctx):
            return True
        return self.interval(ctx.curr_lvl) !=  self.interval(ctx.prev_lvl)

    def first_real_decision(self, ctx):
        return ctx.prev_lvl is None

    def prev_was_decision(self, ctx):
        if self.first_real_decision(ctx):
            return self.decisions_on_edge(ctx) > 0
        return self.is_decision(ctx.prev_lvl)

    def decisions_on_edge(self, ctx):
        prev_lvl = 0 if self.first_real_decision(ctx) else ctx.prev_lvl
        return self.skipped_decisions(prev_lvl, ctx.curr_lvl)
