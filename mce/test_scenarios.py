import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as PLTL
import hypothesis.strategies as st

def sys1():
    return C.circ2mdp(BV.identity_gate(1, 'a'))


def sys2():
    mdp = sys1()
    mdp <<= C.coin((1, 8), name='c')
    mdp >>= C.circ2mdp(BV.sink(3, ['c']))  # HACK
    return mdp


def sys3():
    mdp = sys2()
    mdp >>= C.circ2mdp(BV.aig2aigbvsink(3, ['c']))  # HACK
    return mdp


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


SCENARIOS = st.one_of(
    st.just(gen_scenario(spec=SPEC1, sys=sys1())),
    st.just(gen_scenario(spec=SPEC2, sys=sys1())),
    st.just(gen_scenario(spec=SPEC1, sys=sys2())),
    st.just(gen_scenario(spec=SPEC2, sys=sys2())),
)
