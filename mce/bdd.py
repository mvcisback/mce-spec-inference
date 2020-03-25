import re

import attr
import aiger_bv
import aiger_bdd
import funcy as fn

from mce.order import BitOrder


TIMED_INPUT_MATCHER = re.compile(r'(.*)##time_(\d+)\[(\d+)\]')


def to_bdd(mdp, horizon, output=None, manager=None):
    circ = mdp.aigbv

    if '##valid' in circ.outputs:  # TODO: handle ##valid.
        circ >>= aiger_bv.sink(1, ['##valid'])

    unrolled = circ.unroll(horizon, only_last_outputs=True)  # HACK

    if output is not None:
        bdl = unrolled.omap[f"{output}##time_{horizon}"]
        assert bdl.size == 1
        output = bdl[0]

    inputs, env_inputs = mdp.inputs, circ.inputs - mdp.inputs
    imap = circ.imap

    def causal_order():

        def flattened(t):
            # TODO: clean up based on bundles
            def fmt(k):
                idxs = range(imap[k].size)
                return [f"{k}##time_{t}[{i}]" for i in idxs]

            actions = fn.lmapcat(fmt, inputs)
            coin_flips = fn.lmapcat(fmt, env_inputs)
            return actions + coin_flips

        unrolled_inputs = fn.lmapcat(flattened, range(horizon))
        return {k: i for i, k in enumerate(unrolled_inputs)}

    bdd, manager, input2var = aiger_bdd.to_bdd(
        unrolled, output=output, manager=manager,
        renamer=lambda _, x: x, levels=causal_order(),
    )

    def count_bits(inputs):
        return sum(imap[i].size for i in inputs)

    order = BitOrder(count_bits(inputs), count_bits(env_inputs), horizon)
    return bdd, manager, input2var, order
