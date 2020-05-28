__all__ = ['cone']

import re

import aiger_bv as BV
import funcy as fn


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
