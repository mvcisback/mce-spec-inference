"""
Code for checking if two trajectories index the same time unrolled state.
"""
from typing import Iterator

import attr
import aiger
import aiger_sat
import aiger_bv as BV


def bmc_equiv(circ1, circ2, horizon, assume=None) -> Iterator[bool]:
    """
    Perform bounded model checking up to horizon to see if circ1 and
    circ2 are equivilent.
    """
    # Create distinguishing predicate.
    expr = BV.uatom(1, val=0)
    for o1 in circ1.outputs:
        o2 = f'{o1}##copy'
        size = circ1.omap[o1].size
        expr |= BV.uatom(size, o1) != BV.uatom(size, o2)
    expr.with_output('distinguished')
    
    monitor = ((circ1 | circ2) >> expr.aigbv).aig
    assert len(monitor.outputs) == 1

    # Make underlying aig lazy.
    monitor = monitor.lazy_aig

    # BMC loop
    for t in range(horizon):
        delta = horizon - t
        unrolled = monitor.unroll(delta, only_last_outputs=True)
        assert len(unrolled.outputs) == 1
        unrolled = aiger.BoolExpr(unrolled)

        if assume is not None:
            unrolled |= assume

        yield aiger_sat.is_sat(unrolled)


def equiv_states(mdp, horizon, time, state1, state2):
    """
    Return whether state1 and state2 correspond to the same state at
    t=`time`.
    """
    if state1 == state2:
        return True

    circ1 = mdp.aigbv

    # Reinitialize circuits to start at respective states.
    circ2 = circ1.reinit(state2)
    circ1 = circ1.reinit(state1)

    # Give circ2 different symbol names from circ1.
    circ2 = circ2['o', {o: f'{o}##copy' for o in circ1.outputs}]
    circ2 = circ2['l', {l: f'{l}##copy' for l in circ1.latches}]

    return not any(bmc_equiv(circ1, circ2, horizon - time))


__all__ = ['equiv_states']
