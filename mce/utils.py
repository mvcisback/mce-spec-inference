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
