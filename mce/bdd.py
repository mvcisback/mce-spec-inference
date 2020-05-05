__all__ = ['to_bdd2']

import re

import aiger_bdd
import funcy as fn

from dd.autoref import BDD

from mce.order import BitOrder
from mce.utils import cone, interpret_unrolled, Atom, Literal


def ordered_bdd_atom_fact(mdp, output, horizon, manager=None) -> Atom:
    """
    Factory that creates a function that creates BDD atoms whose
    variables are causally ordered versions of circ.
    """
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

    manager = BDD() if manager is None else manager
    manager.declare(*levels.keys())
    manager.reorder(levels)
    manager.configure(reordering=False)

    def atom(x: Literal):
        if isinstance(x, str):
            return manager.var(x)
        return manager.true if x else manager.false

    def count_bits(inputs):
        return sum(imap[i].size for i in inputs)

    order = BitOrder(count_bits(inputs), count_bits(env_inputs), horizon)
    return atom, order


def to_bdd2(mdp, horizon, output=None, manager=None):
    if output is None:
        assert len(mdp.outputs) == 1
        output = fn.first(mdp.outputs)

    # 3. Create BDD atom
    atom, order = ordered_bdd_atom_fact(mdp, output, horizon, manager)
    bexpr = interpret_unrolled(mdp, horizon, atom, output=output)
    return bexpr, bexpr.bdd, order



