import hypothesis.strategies as st
from hypothesis import given

from mce.order import BitOrder


def skipped_decisions_naive(order, lvl1, lvl2):
    if lvl2 < lvl1:
        return -order.skipped_decisions(lvl2, lvl1)

    return sum(order.is_decision(lvl) for lvl in range(lvl1 + 1, lvl2))


def test_smoke():
    order = BitOrder(decision_bits=2, chance_bits=3, horizon=4)

    assert order.is_decision(0)
    assert order.is_decision(1)
    assert not order.is_decision(2)
    assert not order.is_decision(3)
    assert not order.is_decision(4)
    assert order.is_decision(5)

    assert order.time_step(2) == 0
    assert order.time_step(7) == 1
    assert order.time_step(12) == 2
    assert order.time_step(16) == 3

    assert order.interval(0) == (0, 1)
    assert order.interval(1) == (0, 1)
    assert order.interval(2) == (2, 4)
    assert order.interval(3) == (2, 4)
    assert order.interval(4) == (2, 4)
    assert order.interval(5) == (5, 6)

    assert order.skipped_time_steps(0, 6) == 0
    assert order.skipped_time_steps(0, 9) == 0
    assert order.skipped_time_steps(4, 9) == 0
    assert order.skipped_time_steps(0, 11) == 1
    assert order.skipped_time_steps(0, 16) == 2
    assert order.skipped_time_steps(0, 21) == 3
    assert order.skipped_time_steps(5, 14) == 0
    assert order.skipped_time_steps(4, 10) == 1
    assert order.skipped_time_steps(4, 15) == 2
    assert order.skipped_time_steps(5, 15) == 1
    assert order.skipped_time_steps(4, 14) == 1

    assert order.skipped_decisions(0, 2) == 1
    assert order.skipped_decisions(0, 6) == 2
    assert order.skipped_decisions(0, 7) == 3
    assert order.skipped_decisions(1, 2) == 0
    assert order.skipped_decisions(1, 3) == 0
    assert order.skipped_decisions(0, 16) == 6
    assert order.skipped_decisions(1, 16) == 5
    assert order.skipped_decisions(2, 16) == 5


NAT = st.integers(min_value=0, max_value=10)
POS_NAT = st.integers(min_value=1, max_value=10)


@given(POS_NAT, NAT)
def test_deterministic(db, lvl):
    order = BitOrder(decision_bits=db, chance_bits=0, horizon=4)
    assert order.is_decision(lvl)
    assert order.time_interval(lvl) == order.interval(lvl)


@given(POS_NAT, NAT, NAT, NAT)
def test_skipped_decisions(db, cb, lvl, offset):
    order = BitOrder(decision_bits=db, chance_bits=cb, horizon=4)
    assert order.skipped_decisions(lvl, lvl + offset) == \
        skipped_decisions_naive(order, lvl, lvl + offset)


@given(POS_NAT, NAT)
def test_monotonicity(db, cb):
    order = BitOrder(decision_bits=db, chance_bits=cb, horizon=4)

    for i in range(1, 15):
        assert order.skipped_time_steps(i, i) == 0
        for j in range(4):
            k = i + (order.total_bits)*(j + 1)
            assert order.skipped_time_steps(i, k) == j
            assert order.skipped_time_steps(i, k) == \
                -order.skipped_time_steps(k, i)

    for i in range(15):
        prev = order.skipped_decisions(i, i)
        assert prev == 0
        for j in range(i+1, 15):
            curr = order.skipped_decisions(i, j)
            assert prev <= curr


def test_smoke2():
    order = BitOrder(decision_bits=1, chance_bits=1, horizon=4)
    for x in range(15):
        assert order.time_step(2*x) == x
        assert order.time_step(2*x + 1) == x
        assert order.interval(x) == (x, x)

        assert order.skipped_time_steps(x, x) == 0
        assert order.skipped_time_steps(x, x + 1) == 0
        assert order.skipped_time_steps(x, x + 2) == 0
        assert order.skipped_time_steps(x, x + 3) <= 1
        assert order.skipped_time_steps(x, x + 4) == 1

        assert order.skipped_decisions(x, x) == 0
        assert order.skipped_decisions(x, x + 1) == 0
        assert order.skipped_decisions(x, x + 2) <= 1
        assert order.skipped_decisions(x, x + 3) == 1
        assert order.skipped_decisions(x, x + 4) <= 2
