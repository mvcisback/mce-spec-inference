import re

import aiger_bv
import aiger_bdd
import funcy as fn

from mce.order import BitOrder


TIMED_INPUT_MATCHER = re.compile(r'(.*)\[(\d+)\]##time_(\d+)')


def to_bdd(mdp, horizon):
    circ = mdp.aigbv 
    circ >>= aiger_bv.sink(1, ['##valid'])  # TODO: handle ##valid.
    unrolled = circ.aig.unroll(horizon, only_last_outputs=True)  # HACK

    bdd, manager, input2var = aiger_bdd.to_bdd(unrolled)

    # Force order to be causal.
    inputs = mdp.inputs
    env_inputs = circ.inputs - mdp.inputs

    imap = circ.imap
    def relabeled(t):
        def fmt(k):
            idxs = range(imap[k].size)
            return [
                input2var[f"{k}[{i}]##time_{t}"] for i in idxs
            ]
        
        actions = fn.lmapcat(fmt, inputs)
        coin_flips = fn.lmapcat(fmt, env_inputs)
        return actions + coin_flips

    unrolled_inputs = fn.lmapcat(relabeled, range(horizon))
    levels = {k: i for i, k in enumerate(unrolled_inputs)}
    manager.reorder(levels)
    manager.configure(reordering=False)

    dbits = sum(imap[i].size for i in inputs)
    cbits = sum(imap[i].size for i in env_inputs)
    order = BitOrder(dbits, cbits, horizon)
    return bdd, manager, input2var, order
