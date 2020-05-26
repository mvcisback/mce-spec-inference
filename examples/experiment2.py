#!/usr/bin/env python

import aiger as A
import aiger_bv as BV
import aiger_coins as C
import aiger_gridworld as GW
import aiger_ptltl as LTL
import funcy as fn
import termplotlib as tpl
from bidict import bidict
from blessings import Terminal


from mce.infer import spec_mle


# ================= World ========================

TERM = Terminal()


X = BV.atom(8, 'x', signed=False)
Y = BV.atom(8, 'y', signed=False)


def mask_test(xmask, ymask):
    return ((X & xmask) !=0) & ((Y & ymask) != 0)


APS = {
    'yellow': mask_test(0b1100_0000, 0b0000_0001),
    'red': mask_test(0b0011_1111, 0b1111_1111)
}


def create_sensor(aps):
    sensor = BV.aig2aigbv(A.empty())
    for name, ap in APS.items():
        sensor |= ap.with_output(name).aigbv
    return sensor


SENSOR = create_sensor(APS)
DYN = GW.gridworld(8, start=(8, 8), compressed_inputs=True)

SLIP = BV.atom(1, 'c', signed=False).repeat(2) & BV.atom(2, 'a', signed=False)
SLIP = SLIP.with_output('a').aigbv
DYN2 = C.coin((31, 32), 'c') >> C.circ2mdp(DYN << SLIP)


def encode_state(x, y):
    x, y = [BV.encode_int(8, 1 << (v - 1), signed=False) for v in (x, y)]
    return {'x': tuple(x), 'y': tuple(y)}


def ap_at_state(x, y, in_ascii=False):
    state = encode_state(x, y)
    obs = SENSOR(state)[0]

    if not in_ascii:
        return obs

    for k, code in {'yellow': 3, 'red': 1}.items():
        if obs[k][0]:
            return TERM.on_color(code)(' ')
    return TERM.on_color(7)(' ')


def print_map():
    order = range(1, 9)
    for y in order:
        chars = (ap_at_state(x, y, in_ascii=True) for x in order)
        print(''.join(chars))


# ==============  TRACES ========================

ACTION2ARROW = bidict({
    GW.NORTH_C: '↑',
    GW.SOUTH_C: '↓',
    GW.WEST_C: '←',
    GW.EAST_C: '→',
})


def print_trc(trc):
    actions, states = trc
    obs = (ap_at_state(*pos, in_ascii=True) for pos in states)
    print(''.join(''.join(x) for x in zip(actions, obs)))


def str2actions(vals):
    return [ACTION2ARROW.inv[c] for c in vals]


ACTIONS0 = "↑↑→↑↑↑→←←←←←←←←"
STATES0 = (
    (8, 7), (7, 7), (8, 7), (8, 6),
    (8, 5), (7, 5), (6, 5), 
    (5, 5), (4, 5),(3, 5), (2, 5),
    (1, 5), (1, 5),(1, 5), (1, 5),
)
TRC0 = (ACTIONS0, STATES0)

print(len(ACTIONS0))
print(len(STATES0))

TRACES = [TRC0]


def encode_trace(trc):
    actions, states = trc
    actions = str2actions(actions)
    actions = [{'a': a} for a in actions]
    states = [encode_state(*s) for s in states]
    return actions, states


def decode_states(observations):
    observations = [
        {k: v.index(True) + 1 for k, v in obs[0].items()} for obs in observations
    ]
    return [(obs['x'], obs['y']) for obs in observations]


def validate_trace(trc):
    actions, states = encode_trace(trc)
    states2 = DYN.simulate(actions)
    states2 = [s for s, l in states2]
    #states2 = decode_states(states2)

    print(f"  consistent states: {states == states2}")


# ============== Specifications ====================


LAVA, RECHARGE = map(LTL.atom, ['red', 'yellow'])

EVENTUALLY_RECHARGE = RECHARGE.once()
AVOID_LAVA = (~LAVA).historically()

CONST_TRUE = LTL.atom(True)


SPECS = [
    CONST_TRUE,
    AVOID_LAVA, 
    EVENTUALLY_RECHARGE,
    AVOID_LAVA & EVENTUALLY_RECHARGE,
]


SPEC_NAMES = [
    "CONST_TRUE", 
    "AVOID_LAVA",
    "EVENTUALLY_RECHARGE",
    "AVOID_LAVA & EVENTUALLY_RECHARGE",
]


def spec2monitor(spec):
    monitor = spec.aig | A.sink(['red', 'yellow'])
    monitor = monitor['o', {spec.output: 'sat'}]
    monitor = BV.aig2aigbv(monitor)
    return SENSOR >> monitor
    

SPEC2MONITORS = {
    spec: spec2monitor(spec) for spec in SPECS
}


def eval_monitors(trc):
    actions, _ = encode_trace(trc)
    out = {}
    for i, monitor in enumerate(SPEC2MONITORS.values()):
        out[i] = (DYN >> monitor).simulate(actions)[-1][0]['sat'][0]
    print(out)


# ================= Infererence ==============

def infer():
    trcs = [encode_trace(trc) for trc in TRACES]
    mdp = DYN2
    best, spec2score = spec_mle(
        mdp, trcs, SPEC2MONITORS.values(), parallel=False, psat=0.9
    )

    breakpoint()
    def normalize(score):
        return int(round(score - spec2score[SPEC2MONITORS[CONST_TRUE]]))

    best_score = normalize(spec2score[best])

    fig = tpl.figure()
    fig.barh(
        fn.lmap(normalize, spec2score.values()),
        labels=SPEC_NAMES,
        force_ascii=False,
        show_vals=True,
    )


    print('\n' + "="*80)
    print('        (log likelihood(spec) - log_likelihood(True))'.rjust(40) + '\n')
    print('(higher is better)'.rjust(41))
    print("="*80)
    fig.show()
    print(f"\n\nbest score: {abs(best_score)}")
    
    return best


# ================== Main ====================


def main():
    print_map()

    for i, trc in enumerate(set(TRACES)):
        print()
        print(f'trace {i}')
        print_trc(trc)

    infer()


if __name__ == '__main__':
    main()
