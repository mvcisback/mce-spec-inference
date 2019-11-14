from itertools import product

import aiger
import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as PLTL
import funcy as fn
import hypothesis.strategies as st


def sys1():
    return C.circ2mdp(BV.identity_gate(1, 'a'))


def sys2():
    mdp = sys1()
    mdp <<= C.coin((1, 8), name='c')
    mdp >>= C.circ2mdp(BV.sink(1, ['c']))  # HACK
    return mdp


def sys3():
    mdp = sys1()
    mdp |= C.circ2mdp(BV.tee(1, {'c': ['c', 'c_next']}))
    coin = (~C.coin((1, 2), name='c')).with_output('c')
    mdp <<= coin

    delay = BV.aig2aigbv(aiger.delay(['c'], [True]))
    delay = C.circ2mdp(delay)

    return mdp >> delay


SPEC1 = PLTL.atom('a').historically()
SPEC2 = PLTL.atom(True)
SPEC3 = (PLTL.atom('a') == PLTL.atom('c')).historically()


def gen_scenario(spec, sys):
    def scenario(negate=False):
        spec2 = ~spec if negate else spec
        return spec2, sys

    return scenario


scenario1 = gen_scenario(
    spec=SPEC1, sys=sys1(),
)


scenario2 = gen_scenario(
    spec=SPEC2, sys=sys1(),
)

scenario_reactive = gen_scenario(
    spec=SPEC3, sys=sys3(),
)


def make_strategy(spec_sys):
    spec, sys = spec_sys
    return st.just(gen_scenario(spec=spec, sys=sys()))


DET_SCENARIOS = st.one_of(
    fn.lmap(make_strategy, product([SPEC1, SPEC2], [sys1, sys2]))
)

NOT_DET_SCENARIOS = st.one_of(
    fn.lmap(make_strategy, product([SPEC3, SPEC2], [sys3]))
)

SCENARIOS = st.one_of(DET_SCENARIOS, NOT_DET_SCENARIOS)
