from mce.order import BitOrder

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

    for i in range(1, 15):
        assert order.skipped_time_steps(i, i) == 0
        for j in range(4):
            k = i + 5*(j + 1)
            assert order.skipped_time_steps(i, k) == j
            assert order.skipped_time_steps(i, k) == \
                -order.skipped_time_steps(k, i)

    for i in range(15):
        prev = order.skipped_decisions(i, i)
        assert prev == 0
        for j in range(i+1, 15):
            curr = order.skipped_decisions(i, j)
            assert prev <= curr
