__all__ = ['interpret_unrolled', 'cone', 'Literal', 'Atom']

import re
from typing import Any, Union, Callable

import aiger
import aiger_bv as BV
import funcy as fn
from aiger import common as cmn


TIMED_INPUT_MATCHER = re.compile(r'(.*)##time_(\d+)\[(\d+)\]')
INPUT_MATCHER = re.compile(r'(.*)\[(\d+)\]')


Literal = Union[bool, str]
Atom = Callable[[Literal], Any]


def cone(circ: BV.AIGBV, output: str) -> BV.AIGBV:
    """Return cone of influence aigbv output."""
    for out in circ.outputs - {output}:
        size = circ.omap[out].size
        circ >>= BV.sink(size, [out])
    assert len(circ.outputs) == 1
    assert fn.first(circ.outputs) == output
    return circ


def interpret_unrolled(mdp, horizon: int, atom: Atom, output=None):
    """
    Reinterpets the horizon-unrolled mdp circuit using atom's boolean
    algebra.
    """
    if output is None:
        assert len(mdp.outputs) == 1
        output = fn.first(mdp.outputs)

    circ = cone(mdp.aigbv, output)
    inputs, env_inputs = mdp.inputs, circ.inputs - mdp.inputs
    imap = circ.imap

    step, old2new_lmap = circ.cutlatches()
    init = dict(old2new_lmap.values())
    init = step.imap.blast(init)
    states = set(init.keys())

    gate_nodes = {aiger.aig.Input(k): atom(v) for k, v in init.items()}
    state_inputs = [aiger.aig.Input(k) for k in init.keys()]

    for time in range(horizon):
        # Only remember states.
        gate_nodes = fn.project(gate_nodes, state_inputs)

        for gate in cmn.eval_order(step.aig):
            if isinstance(gate, aiger.aig.ConstFalse):
                gate_nodes[gate] = atom(False)
            elif isinstance(gate, aiger.aig.Inverter):
                gate_nodes[gate] = ~gate_nodes[gate.input]
            elif isinstance(gate, aiger.aig.Input):
                if gate.name in states:
                    continue

                name, idx = INPUT_MATCHER.match(gate.name).groups()
                name = f'{name}##time_{time}[{idx}]'
                gate_nodes[gate] = atom(name)

            elif isinstance(gate, aiger.aig.AndGate):
                gate_nodes[gate] = gate_nodes[gate.left] & gate_nodes[gate.right]

        for si in state_inputs:
            gate_nodes[si] = gate_nodes[step.aig.node_map[si.name]]

    assert step.omap[output].size == 1
    output = step.omap[output][0]

    return gate_nodes[step.aig.node_map[output]]
