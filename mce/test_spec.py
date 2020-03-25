import aiger_bv as BV
import aiger_coins as C

from mce.test_scenarios import scenario_reactive
from mce.spec import concretize


def act(action, coin):
    return {'a': (action,), 'c': (coin,),}


def test_concretize():
    spec, sys = scenario_reactive()
    monitor = C.MDP(BV.aig2aigbv(spec.aig) | BV.sink(1, ['c_next']))

    cspec = concretize(monitor, sys, 3)

    actions = [act(True, True), act(True, False), act(True, True)]

    assert not cspec.accepts(actions)

    cspec2 = cspec.toggle(actions)
    assert cspec2.accepts(actions)

    assert cspec2.imap == cspec.imap
    assert set(cspec.imap.keys()) == {'a'}
    assert set(cspec.emap.keys()) == {'c'}
