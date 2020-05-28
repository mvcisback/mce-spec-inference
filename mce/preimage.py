__all__ = ["preimage"]

import operator as op
from functools import reduce
from typing import Mapping, Tuple, List, Iterable

import attr
import aiger_bv as BV


CircVal = Mapping[str, Tuple[bool]]
AtomicPred = CircVal
AtomicPreds = List[AtomicPred]
BVExpr = BV.UnsignedBVExpr


def gen_equiv_checks(aps: AtomicPreds) -> Iterable[BVExpr]:
    for t, ap in enumerate(aps):
        for name, val in ap.items():
            size = len(val)
            timed_name = f"{name}##time_{t + 1}"
            sym = BV.atom(size, timed_name, signed=False)
            sym_val = BV.atom(size, val, signed=False)

            yield sym == sym_val


def preimage(aps: AtomicPreds, mdp: BV.AIGBV) -> BVExpr:
    """
    Returns a circuit which checks if an action sequence results in a
    given sequence of observations/atomic predicates (aps).
    """
    assert len(aps) > 1
    mdp = attr.evolve(mdp, aig=mdp.aig.lazy_aig)  # Make AIGBV lazy.

    unrolled = mdp.unroll(len(aps))

    check_val = reduce(op.and_, gen_equiv_checks(aps)).aigbv

    test = unrolled >> check_val
    assert test.inputs == unrolled.inputs
    assert len(test.outputs) == 1

    return BVExpr(test)
