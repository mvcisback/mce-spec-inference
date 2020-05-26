__all__ = ['to_bdd2']

import re

import aiger_bv as BV
import aiger_bdd
import funcy as fn

from dd.autoref import BDD

from mce.order import BitOrder
from mce.utils import cone, Atom, Literal


def to_bdd2(mdp, horizon, output=None, manager=None):
    if output is None:
        assert len(mdp.outputs) == 1
        output = fn.first(mdp.outputs)

    circ = cone(mdp.aigbv, output)
    inputs, env_inputs = mdp.inputs, circ.inputs - mdp.inputs
    imap = circ.imap

    def flattened(t):
        def fmt(k):
            idxs = range(imap[k].size)            
            return [f"{k}##time_{t}[{i}]" for i in idxs]

        actions = fn.lmapcat(fmt, inputs)
        coin_flips = fn.lmapcat(fmt, env_inputs)
        return actions + coin_flips

    unrolled_inputs = fn.lmapcat(flattened, range(horizon))
    levels = {k: i for i, k in enumerate(unrolled_inputs)}

    circ2 = BV.AIGBV(circ.aig.lazy_aig).unroll(horizon, only_last_outputs=True)
    bexpr, *_ = aiger_bdd.to_bdd(circ2, levels=levels, renamer=lambda _, x: x)

    def count_bits(inputs):
        return sum(imap[i].size for i in inputs)

    order = BitOrder(count_bits(inputs), count_bits(env_inputs), horizon)

    return bexpr, bexpr.bdd, order



