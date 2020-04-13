import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as PLTL
import funcy as fn

from mce.bdd import TIMED_INPUT_MATCHER, to_bdd2
from mce.test_scenarios import scenario_reactive


def test_smoke():
    spec = PLTL.atom('a').historically()
    spec = BV.aig2aigbv(spec.aig)
    spec = C.circ2mdp(spec)
    spec <<= C.coin((1, 8), name='c')
    spec >>= C.circ2mdp(BV.sink(1, ['c']))  # HACK

    bdd, manager, order = to_bdd2(spec, horizon=3)

    assert bdd.dag_size == 4

    for i in range(order.total_bits*order.horizon):
        t = order.time_step(i)
        var = manager.var_at_level(i)
        action, t2, _ = TIMED_INPUT_MATCHER.match(var).groups()
        assert t == int(t2)
        decision = action in spec.inputs
        assert decision == order.is_decision(i)


def test_smoke2():
    spec, mdp = scenario_reactive()

    spec_circ = BV.aig2aigbv(spec.aig)
    mdp >>= C.circ2mdp(spec_circ)
    output = spec.output

    bdd, manager, order = to_bdd2(mdp, horizon=3, output=output)

    assert bdd.dag_size == 7

    def translate(mapping):
        return mapping

    assert bdd.count(6) == 8

    node = bdd

    assert bdd.bdd.let(translate({
        'a##time_0[0]': True,
        'c##time_0[0]': False,
        'a##time_1[0]': True,
    }), bdd) == bdd.bdd.false

    assert bdd.bdd.let(translate({
        'a##time_0[0]': True,
        'c##time_0[0]': True,
        'a##time_1[0]': False,
    }), bdd) == bdd.bdd.false

    assert node.low == bdd.bdd.false

    assert bdd.bdd.let(translate({
        'a##time_0[0]': True,
        'c##time_0[0]': True,
        'a##time_1[0]': True,
        'c##time_1[0]': True,
        'a##time_2[0]': True,
        'c##time_2[0]': True,
    }), bdd) == bdd.bdd.true

    assert bdd.bdd.let(translate({
        'a##time_0[0]': True,
        'c##time_0[0]': False,
        'a##time_1[0]': False,
        'c##time_1[0]': False,
        'a##time_2[0]': False,
        'c##time_2[0]': False,
    }), bdd) == bdd.bdd.true

    assert bdd.bdd.let(translate({
        'c##time_0[0]': False,
        'a##time_1[0]': True,
    }), bdd) == bdd.bdd.false

    assert bdd.bdd.let(translate({
        'c##time_0[0]': True,
        'a##time_1[0]': False,
    }), bdd) == bdd.bdd.false

    assert bdd.bdd.let(translate({
        'c##time_1[0]': False,
        'a##time_2[0]': True,
    }), bdd) == bdd.bdd.false

    assert bdd.bdd.let(translate({
        'c##time_1[0]': True,
        'a##time_2[0]': False,
    }), bdd) == bdd.bdd.false

    assert bdd.bdd.let(translate({
        'c##time_2[0]': False,
    }), bdd) == bdd

    assert bdd.bdd.let(translate({
        'c##time_2[0]': True,
    }), bdd) == bdd
