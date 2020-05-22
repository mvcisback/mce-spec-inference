import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as LTL

from mce.test_scenarios import scenario_reactive
from mce.equiv import equiv_states


X = LTL.atom('x')
Y = LTL.atom('y')
SYS = C.circ2mdp(BV.aig2aigbv((X.once() | Y.once()).aig))


def test_equiv_states_smoke():
    state = SYS._aigbv.latch2init

    for t in range(3):
        assert equiv_states(SYS, 3, t, state1=state, state2=state)

    state1 = SYS.aigbv({'x': (True,), 'y': (False, )})[1]
    state2 = SYS.aigbv({'x': (False,), 'y': (True, )})[1]
    assert state1 != state2

    for t in range(3):
        assert equiv_states(SYS, 3, t, state1=state1, state2=state2)
