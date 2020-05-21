from mce.test_scenarios import scenario_reactive
from mce.equiv import equiv


def test_equiv_smoke():
    _, sys = scenario_reactive()

    state = sys._aigbv.latch2init

    for t in range(3):
        assert equiv(sys, 3, t, state1=state, state2=state)
