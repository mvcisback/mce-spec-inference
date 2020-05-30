import funcy as fn

from mce.test_scenarios import scenario_reactive
from mce.spec import concretize
from mce.policy3 import fit
from mce.demos import prefix_tree


def act(action, coin):
    return {'a': (action,), 'c': (coin,)}


def test_prefix_tree():
    spec, sys = scenario_reactive()

    encoded = [
        [act(True, True), act(True, True), act(True, True)],
        [act(True, True), act(False, True), act(False, False)],
    ]

    def to_demo(encoded_trc):
        io_seq = zip(encoded_trc, sys.aigbv.simulate(encoded_trc))
        for inputs, (_, state) in io_seq:
            inputs = fn.project(inputs, sys.inputs)
            yield inputs, state

    # Technical debt where sys_actions and env_actions
    # are two different lists.
    demos = [list(zip(*to_demo(etrc))) for etrc in encoded]

    tree = prefix_tree(sys, demos)
    tree.write_dot('foo.dot')

    cspec = concretize(spec, sys, 3)
    ctrl = fit(cspec, 0.7, bv=True)
    lprob = tree.log_likelihood(ctrl, actions_only=True)
    assert lprob < 0

    assert tree.psat(cspec) == 1/2

    lprob2 = tree.log_likelihood(ctrl, actions_only=False)
    assert lprob2 < lprob
