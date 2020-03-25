import aiger_bv as BV
import aiger_bdd
import funcy as fn

from mce.test_scenarios import scenario_reactive
from mce.preimage import preimage
from mce.bdd import to_bdd

def xor(x, y):
    return (x | y) & ~(x & y)


def test_preimage():
    spec, mdp = scenario_reactive()

    sys1 = mdp.aigbv >> BV.sink(1, ['c_next', '##valid'])
    sys2 = sys1 >> BV.aig2aigbv(spec.aig)

    def act(action, coin):
        return {'a': (action,), 'c': (coin,),}

    actions = [act(True, True), act(True, False), act(True, True)]
    observations = fn.lpluck(0, sys1.simulate(actions))

    expr = preimage(observations, sys1)
    assert expr.inputs == {
        'c##time_0', 'c##time_1', 'c##time_2',
        'a##time_0', 'a##time_1', 'a##time_2',
    }

    bexpr1, manager, _, order = to_bdd(sys2, horizon=3)

    bexpr2, _, input2var = aiger_bdd.to_bdd(
        expr, manager=manager, renamer=lambda _, x: x
    )

    # TODO: check bexpr2 accepts actions.

    bexpr3 = xor(bexpr1, bexpr2)
    # TODO: check bexpr3 toggles value of actions.

