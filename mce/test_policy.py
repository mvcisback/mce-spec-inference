import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as PLTL

from mce.policy import policy


def test_smoke():
    spec = PLTL.atom('a').historically()
    spec = BV.aig2aigbv(spec.aig)
    spec = C.circ2mdp(spec)

    mdp = C.circ2mdp(BV.identity_gate(1, 'a'))
