import re

import aiger_bv as BV
import aiger_coins as C
import aiger_ptltl as PLTL
import funcy as fn

from mce.bdd import to_bdd, TIMED_INPUT_MATCHER
from mce.test_scenarios import scenario_reactive


def test_smoke():
    spec = PLTL.atom('a').historically()
    spec = BV.aig2aigbv(spec.aig)
    spec = C.circ2mdp(spec)
    spec <<= C.coin((1, 8), name='c')
    spec >>= C.circ2mdp(BV.sink(1, ['c']))  # HACK

    bdd, manager, input2var, order = to_bdd(spec, horizon=3)

    assert bdd.dag_size == 4

    for i in range(order.total_bits*order.horizon):
        t = order.time_step(i)
        var = input2var.inv[manager.var_at_level(i)]
        action, _, t2 = TIMED_INPUT_MATCHER.match(var).groups()
        assert t == int(t2)
        decision = action in spec.inputs
        assert decision == order.is_decision(i)


def test_smoke2():
    spec, mdp = scenario_reactive()
    orig_mdp = mdp

    spec_circ = BV.aig2aigbv(spec.aig)
    mdp >>= C.circ2mdp(spec_circ)
    output = spec_circ.omap[spec.output][0]

    bdd, manager, input2var, order = to_bdd(mdp, horizon=3, output=output)

    assert bdd.dag_size == 7

    levels = {
        'a[0]##time_0': 0,
        'c[0]##time_0': 1,
        'a[0]##time_1': 2,
        'c[0]##time_1': 3,
        'a[0]##time_2': 4,
        'c[0]##time_2': 5,
    }

    def translate(mapping):
        return fn.walk_keys(input2var.get, mapping)

    #assert bdd.bdd.var_levels == translate(levels)
    assert bdd.count(6) == 8

    node = bdd

    assert bdd.bdd.let(translate({
        'a[0]##time_0': True,
        'c[0]##time_0': False,
        'a[0]##time_1': True,
    }), bdd) == bdd.bdd.false

    assert bdd.bdd.let(translate({
        'a[0]##time_0': True,
        'c[0]##time_0': True,
        'a[0]##time_1': False,
    }), bdd) == bdd.bdd.false

    assert node.low == bdd.bdd.false

    assert bdd.bdd.let(translate({
        'a[0]##time_0': True,
        'c[0]##time_0': True,
        'a[0]##time_1': True,
        'c[0]##time_1': True,
        'a[0]##time_2': True,
        'c[0]##time_2': True,
    }), bdd) == bdd.bdd.true

    assert bdd.bdd.let(translate({
        'a[0]##time_0': True,
        'c[0]##time_0': False,
        'a[0]##time_1': False,
        'c[0]##time_1': False,
        'a[0]##time_2': False,
        'c[0]##time_2': False,
    }), bdd) == bdd.bdd.true


    assert bdd.bdd.let(translate({
        'c[0]##time_0': False,
        'a[0]##time_1': True,
    }), bdd) == bdd.bdd.false

    assert bdd.bdd.let(translate({
        'c[0]##time_0': True,
        'a[0]##time_1': False,
    }), bdd) == bdd.bdd.false

    assert bdd.bdd.let(translate({
        'c[0]##time_1': False,
        'a[0]##time_2': True,
    }), bdd) == bdd.bdd.false

    assert bdd.bdd.let(translate({
        'c[0]##time_1': True,
        'a[0]##time_2': False,
    }), bdd) == bdd.bdd.false

    assert bdd.bdd.let(translate({
        'c[0]##time_2': False,
    }), bdd) == bdd

    assert bdd.bdd.let(translate({
        'c[0]##time_2': True,
    }), bdd) == bdd
