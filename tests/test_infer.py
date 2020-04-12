import funcy as fn

from mce import infer
from mce.test_scenarios import sys1, sys3, SPEC1, SPEC2, SPEC3


def create_demos(n):
    actions1 = states1 = 3*[{'a': (True, )}]
    actions2 = states2 = 3*[{'a': (False, )}]
    actions3 = states3 = [
        {'a': (True, )},
        {'a': (False, )},
        {'a': (False, )},
    ]

    return [
        (actions2, states2),
        (actions3, states3),
    ] + n*[(actions1, states1)]


SPECS = [
    SPEC1,   # Historically a
    ~SPEC1,  # Once ~a
    SPEC2,   # True
    ~SPEC2   # False
]


def test_infer_log_likelihood1():
    mdp = sys1()
    demos = create_demos(10)

    spec, spec2score = infer.spec_mle(mdp, demos, SPECS)
    assert max(spec2score.values()) == spec2score[spec]
    assert max(spec2score.values()) == spec2score[SPEC1]


def create_demos2(n):
    def create_states(actions, coins):
        coins_next = [{'c_next': v['c']} for v in coins]
        coins = [{'c': (True,)}] + coins
        return [
            fn.merge(x, y, z) for x, y, z in zip(actions, coins, coins_next)
        ]

    actions1 = [{'a': (True, )}, {'a': (False, )}, {'a': (True, )}]
    states1 = [
        {'a': (True,), 'c': (True,), 'c_next': (False,)},
        {'a': (False,), 'c': (False,), 'c_next': (True,)},
        {'a': (True,), 'c': (True,), 'c_next': (True,)},
    ]

    actions2 = 3*[{'a': (True, )}]
    coins2 = [{'c': (True, )}, {'c': (True, )}, {'c': (False, )}]
    states2 = create_states(actions2, coins2)

    actions3 = 3*[{'a': (False, )}]
    coins3 = [{'c': (True, )}, {'c': (True, )}, {'c': (True, )}]
    states3 = create_states(actions3, coins3)

    return [
        (actions2, states2),
        (actions3, states3),
    ] + n*[(actions1, states1)]


SPECS2 = [
    SPEC3,   # Historically a = c
    ~SPEC3,  # Once a != c
    SPEC2,   # True
    ~SPEC2   # False
]


def test_infer_log_likelihood2():
    mdp = sys3()
    demos = create_demos2(10)

    spec, spec2score = infer.spec_mle(mdp, demos, SPECS2)
    assert max(spec2score.values()) == spec2score[spec]
    assert max(spec2score.values()) == spec2score[SPEC3]
