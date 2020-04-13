__all__ = ['TIMED_INPUT_MATCHER', 'to_bdd2']

import re

import attr
import aiger
import aiger_bv as BV
import aiger_bdd
import funcy as fn
from aiger import common as cmn

try:
    from dd.cudd import BDD
except ImportError:
    from dd.autoref import BDD

from mce.order import BitOrder


TIMED_INPUT_MATCHER = re.compile(r'(.*)##time_(\d+)\[(\d+)\]')
INPUT_MATCHER = re.compile(r'(.*)\[(\d+)\]')


def cone(circ: BV.AIGBV, output: str) -> BV.AIGBV:
    """Return cone of influence aigbv output."""
    for out in circ.outputs - {output}:
        size = circ.omap[out].size
        circ >>= BV.sink(size, [out])
    assert len(circ.outputs) == 1
    assert fn.first(circ.outputs) == output
    return circ


def to_bdd2(mdp, horizon, output=None, manager=None):
    if output is None:
        assert len(mdp.outputs) == 1
        output = fn.first(mdp.outputs)

    # 1. Convert MDP to AIGBV circuit.
    circ = cone(mdp.aigbv, output)

    # 2. Define Causal Order
    inputs, env_inputs = mdp.inputs, circ.inputs - mdp.inputs
    imap = circ.imap

    def count_bits(inputs):
        return sum(imap[i].size for i in inputs)

    order = BitOrder(count_bits(inputs), count_bits(env_inputs), horizon)

    # 4. Create topological ordering on variables.
    step, old2new_lmap = circ.cutlatches()
    init = dict(old2new_lmap.values())
    init = step.imap.blast(init)
    states = set(init.keys())

    def flattened(t):
        def fmt(k):
            idxs = range(imap[k].size)            
            return [f"{k}##time_{t}[{i}]" for i in idxs]

        actions = fn.lmapcat(fmt, inputs)
        coin_flips = fn.lmapcat(fmt, env_inputs)
        return actions + coin_flips

    unrolled_inputs = fn.lmapcat(flattened, range(horizon))

    levels = {k: i for i, k in enumerate(unrolled_inputs)}

    # 3. Create BDD

    manager = BDD() if manager is None else manager
    manager.declare(*levels.keys())
    manager.reorder(levels)
    manager.configure(reordering=False)

    def bdd_bool(x):
        return manager.true if x else manager.false

    gate_nodes = {aiger.aig.Input(k): bdd_bool(v) for k, v in init.items()}
    state_inputs = [aiger.aig.Input(k) for k in init.keys()]

    for time in range(horizon):
        # Only remember states.
        gate_nodes = fn.project(gate_nodes, state_inputs)

        for gate in cmn.eval_order(step.aig):
            if isinstance(gate, aiger.aig.ConstFalse):
                gate_nodes[gate] = manager.false
            elif isinstance(gate, aiger.aig.Inverter):
                gate_nodes[gate] = ~gate_nodes[gate.input]
            elif isinstance(gate, aiger.aig.Input):
                if gate.name in states:
                    continue

                name, idx = INPUT_MATCHER.match(gate.name).groups()
                name = f'{name}##time_{time}[{idx}]'
                gate_nodes[gate] = manager.var(name)

            elif isinstance(gate, aiger.aig.AndGate):
                gate_nodes[gate] = gate_nodes[gate.left] & gate_nodes[gate.right]

        for si in state_inputs:
            gate_nodes[si] = gate_nodes[step.aig.node_map[si.name]]

    assert step.omap[output].size == 1
    output = step.omap[output][0]

    return gate_nodes[step.aig.node_map[output]], manager, order
