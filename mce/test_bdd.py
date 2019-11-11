import re

import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as PLTL

from mce.bdd import to_bdd


TIMED_INPUT_MATCHER = re.compile(r'(.*)\[\d+\]##time_(\d+)')


def test_smoke():
    spec = PLTL.atom('a').historically()
    spec = BV.aig2aigbv(spec.aig)
    spec = C.circ2mdp(spec)
    spec <<= C.coin((1, 8), name='c')
    spec >>= C.circ2mdp(BV.sink(3, ['c']))  # HACK

    bdd, manager, input2var, order = to_bdd(spec, horizon=3)    
    for i in range(order.total_bits*order.horizon):
        t = order.time_step(i)
        var = input2var.inv[manager.var_at_level(i)]
        action, t2 = TIMED_INPUT_MATCHER.match(var).groups()
        assert t == int(t2)
        decision = action in spec.inputs
        assert decision == order.is_decision(i)
