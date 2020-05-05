import aiger_bv as BV
import aiger_bdd
import funcy as fn

from mce.test_scenarios import scenario_reactive
from mce.preimage import preimage
from mce.bdd import to_bdd2


def xor(x, y):
    return (x | y) & ~(x & y)


def test_preimage():
    spec, mdp = scenario_reactive()

    sys1 = mdp.aigbv >> BV.sink(1, ['c_next', '##valid'])
    sys2 = sys1 >> BV.aig2aigbv(spec.aig)

    def act(action, coin):
        return {'a': (action,), 'c': (coin,)}

    actions = [act(True, True), act(True, False), act(True, True)]
    observations = fn.lpluck(0, sys1.simulate(actions))

    expr = preimage(observations, sys1)
    assert expr.inputs == {
        'c##time_0', 'c##time_1', 'c##time_2',
        'a##time_0', 'a##time_1', 'a##time_2',
    }

    bexpr1, manager, order = to_bdd2(sys2, horizon=3)

    def accepts(bexpr, actions):
        """Check if bexpr accepts action sequence."""
        timed_actions = {}
        for t, action in enumerate(actions):
            c, a = action['c'], action['a']
            timed_actions.update(
                {f'c##time_{t}[0]': c[0], f'a##time_{t}[0]': a[0]}
            )

        assert timed_actions.keys() == manager.vars.keys()
        tmp = manager.let(timed_actions, bexpr)
        assert tmp in (manager.true, manager.false)
        return tmp == manager.true

    assert not accepts(bexpr1, actions)

    bexpr2, _, input2var = aiger_bdd.to_bdd(
        expr, manager=manager, renamer=lambda _, x: x
    )

    assert accepts(bexpr2, actions)
    assert not accepts(~bexpr2, actions)

    bexpr3 = xor(bexpr1, bexpr2)

    assert accepts(bexpr3, actions)
