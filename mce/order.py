from math import log

import attr


@attr.s(frozen=True, auto_attribs=True)
class BitOrder:
    decision_bits: int
    chance_bits: int
    horizon: int

    @property
    def total_bits(self) -> int:
        return self.decision_bits + self.chance_bits

    def round_index(self, lvl) -> int:
        return lvl % self.total_bits

    def is_decision(self, lvl) -> bool:
        return self.round_index(level) >= self.decision_bits

    def skipped_decisions(self, lvl1, lvl2) -> int:
        ri1, ri2 = self.round_index(lvl1), self.round_index(lvl2)
        tb = self.total_bits

        skipped_rounds = ((lvl2 - ri2) - (lvl1 - (tb - ri1)))
        assert skipped_rounds % tb == 0
        skipped_rounds //= tb

        start = self.is_decision(lvl1) * (self.decision_bits - ri1)
        middle = self.decision_bits * skipped_rounds
        end = self.is_decision(lvl2) * ri2

        total = start + middle + end
        assert total < (lvl2 - lvl1)
        return total

    def delta(self, lvl1, lvl2, val):
        return val + self.skipped_decisions(lvl1, lvl2)*log(2)

    def interval(self, lvl):
        # TODO
        raise NotImplementedError
