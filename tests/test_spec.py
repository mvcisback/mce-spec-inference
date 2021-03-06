import funcy as fn

from mce.test_scenarios import scenario_reactive
from mce.spec import concretize


def act(action, coin):
    return {'a': (action,), 'c': (coin,)}


def test_concretize():
    spec, sys = scenario_reactive()
    cspec = concretize(spec, sys, 3)

    actions = [act(True, True), act(True, False), act(True, True)]

    assert not cspec.accepts(actions)

    cspec2 = cspec.toggle(actions)
    assert cspec2.accepts(actions)

    assert cspec2.imap == cspec.imap
    assert set(cspec.imap.keys()) == {'a'}
    assert set(cspec.emap.keys()) == {'c'}


def test_flatten():
    spec, sys = scenario_reactive()
    cspec = concretize(spec, sys, 3)

    actions = [act(True, True), act(True, False), act(True, True)]
    bits = cspec.flatten(actions)
    assert bits == [True, True, True, False, True, True]

    assert cspec.unflatten(bits) == actions


def test_abstract_trace():
    spec, sys = scenario_reactive()
    cspec = concretize(spec, sys, 3)

    actions = [act(True, True), act(True, False), act(True, True)]
    trc = list(cspec.abstract_trace(actions))
    for prev, curr in fn.rest(fn.with_prev(trc)):
        if prev == curr:
            assert prev.node.level == 6
            assert prev.node == cspec.manager.false
        else:
            clvl, cdebt = curr.node.level, curr.debt
            plvl, pdebt = prev.node.level, prev.debt
            assert (clvl, -cdebt) < (plvl, -pdebt)
