import aiger
import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as PLTL
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
    mdp |= C.circ2mdp(BV.identity_gate(1, 'c'))
    coin = C.coin((1, 2), name='c')
    mdp <<= coin

    delay = BV.aig2aigbv(aiger.delay('c', [False]))
    delay = C.circ2mdp(delay)

    c = BV.atom(1, 'c', signed=False)
    a = BV.atom(1, 'a', signed=False)
    test = C.circ2mdp((c == a).with_output('a'))


    return mdp >> delay >> test


SPEC1 = PLTL.atom('a').historically()
SPEC2 = PLTL.atom(True)


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
    spec=SPEC1, sys=sys3(),
)


SCENARIOS = st.one_of(
    st.just(gen_scenario(spec=SPEC1, sys=sys1())),
    st.just(gen_scenario(spec=SPEC2, sys=sys1())),
    st.just(gen_scenario(spec=SPEC1, sys=sys2())),
    st.just(gen_scenario(spec=SPEC2, sys=sys2())),
    st.just(gen_scenario(spec=SPEC1, sys=sys3())),
    st.just(gen_scenario(spec=SPEC2, sys=sys3())),
)
