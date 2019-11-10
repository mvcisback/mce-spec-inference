# TODO:
# 1. Support invalid inputs (decision and chance).
#    - Maintain validity BDD (pg 104)
# 2. Implement the dynamic programming on the CompressedDTree.
#    - Perhaps implement more generally first.
# 3. Force variable ordering to be correct
# https://github.com/tulip-control/dd/blob/master/examples/cudd_configure_reordering.py

from math import log

import aiger_bv
import aiger_coins
import aiger_bdd
import attr
import funcy as fn
import theano
import theano.tensor as T
from fold_bdd import fold_path, post_order


@attr.s(frozen=True, auto_attribs=True)
class BitOrder:
    decision_bits: int
    chance_bits: int
    horizon: int

    @property
    def total_bits(self) -> int:
        return self.decision_bits + self.chance_bits

    def round_index(self, lvl) -> int:
        return lvl % self.total_bits

    def is_decision(self, lvl) -> bool:
        return self.round_index(level) >= self.decision_bits

    def skipped_decisions(self, lvl1, lvl2) -> int:
        ri1, ri2 = self.round_index(lvl1), self.round_index(lvl2)
        tb = self.total_bits

        skipped_rounds = ((lvl2 - ri2) - (lvl1 - (tb - ri1)))
        assert skipped_rounds % tb == 0
        skipped_rounds //= tb

        start = self.is_decision(lvl1) * (self.decision_bits - ri1)
        middle = self.decision_bits * skipped_rounds
        end = self.is_decision(lvl2) * ri2

        total = start + middle + end
        assert total < (lvl2 - lvl1)
        return total

    def delta(self, lvl1, lvl2, val):
        return val + self.skipped_decisions*log(2)

    def itvl(self, lvl):
        # TODO
        raise NotImplementedError


def to_bdd(mdp, horizon):
    circ = mdp.aigbv 
    circ >>= aiger_bv.sink(1, ['##valid'])  # TODO: handle ##valid.
    circ = circ.unroll(horizon, only_last_outputs=True)

    bdd, manager, input2var = aiger_bdd.to_bdd(circ)

    inputs = mdp.inputs
    env_inputs = circ.inputs - mdp.inputs
    # Force order to be causal.
    def relabeled(t):
        def fmt(i):
            return f"{input2var[i]}##time_{t}"
        
        actions = fn.lmap(fmt, inputs)
        coin_flips = fn.lmap(fmt, env_inputs)
        return actions + coin_flips

    unrolled_inputs = fn.mapcat(relabeled, range(horizon))
    manager.reorder({k: i for i, k in enumerate(unrolled_inputs)})
    manager.configure(reordering=False)

    order = BitOrder(len(inputs), len(env_inputs), horizon)
    return bdd, manager, input2var, order
 

def policy(mdp, spec, horizon, coeff):
    mdp >>= spec
    bdd, manager, input2var, order = to_bdd(mdp, horizon)

    
# TODO:
# 1. propogate expectation computation.
# 2. return dictionary of outputs including Q and V values.
# 3. index Q and V values by node.
def to_func(root, ctree):
    rationality = theano.dscalar('rationality')
    discount = theano.dscalar('discount')
    max_lvl = len(ctree.manager.vars)

    node2expr = {}
    def child_val(node, child):
        # true and false are at MAXINT level.
        child_level = min(child.level, max_lvl + 1)
        return ctree.delta(node.level, child_level, _to_func(child))

    def reward(node):
        r = 2*(node == ctree.manager.true) - 1

        return r

    def _to_func(node):
        if node in (ctree.manager.true, ctree.manager.false):
            expr = reward(node)
        else:
            left, right = [child_val(node, c) for c in (node.low, now.high)]

            if ctree.is_decision(node):
                expr = T.log(T.exp(left) + T.exp(right))
            else:
                expr = (left + right) / 2
        node2expr[node] = -rationality*x - k*discount*T.log(2)
        return expr

    _to_func(root)

    return node2expr

