# TODO: switch to python3.8 for cached properties
from typing import TypeVar, Tuple

import attr
import funcy as fn
import numpy as np
from fold_bdd import post_order
from fold_bdd.folds import Context
from pyrsistent import pmap
from pyrsistent.typing import PMap

from mce.order import BitOrder


def softmax(x, y):
    m = max(x, y)
    x2, y2 = x - m, y - m
    return np.log(np.exp(x2) + np.exp(y2)) + m


BExpr = TypeVar('BExpr')


@attr.s(frozen=True, auto_attribs=True)
class PolicyTable:
    coeff: float
    order: BitOrder
    bdd: BExpr
    tbl: PMap[Tuple[BExpr, bool], float] = None

    @property
    def horizon(self):
        return self.order.horizon

    def keys(self):
        return self.tbl.keys()

    def __getitem__(self, key):
        if isinstance(key, Context):
            key = key.node, key.path_negated

        return self.tbl[key]

    @property
    def lsat(self):
        order = self.order

        def merge(ctx, low, high):
            q = np.log(self[ctx])
            if ctx.is_leaf:
                l = 0 if ctx.node_val ^ ctx.path_negated else -float('inf')
            else:
                l = softmax(low, high)
                if not order.is_decision(ctx):
                    l -= np.log(2)
                elif order.on_boundary(ctx):
                    l -= q

            first_decision = order.first_real_decision(ctx)
            prev_was_decision = order.prev_was_decision(ctx)
                
            if not first_decision and prev_was_decision:
                l += order.decision_entropy(ctx) + q

            return l
        
        return post_order(self.bdd, merge)

    @property
    def psat(self):
        return np.exp(self.lsat)

    def log_likelihood_ratio(self, trc):
        """
        Compute the ratio of the likelihood of the abstract trace
        using this policy over the uniformly random policy.
        """

        def delta(ctx):
            return ctx.is_leaf - self.order.first_real_decision(ctx)

        def log_prob(ctx, val, acc):
            return acc \
                + delta(ctx) * self[ctx] \
                - self.order.decision_entropy(ctx)
        
        return fold_path(merge=log_prob, bexpr=self.bdd, vals=trc, initial=0)


def policy_tbl(bdd, order, coeff):
    def merge(ctx, low, high):
        negated = ctx.path_negated
        if ctx.is_leaf:
            val = coeff if ctx.node_val ^ negated else -coeff
            val = np.exp(val)
            tbl = {(ctx.node, negated): val}
        else:
            (tbl_l, val_l), (tbl_r, val_r) = low, high
            tbl = fn.merge(tbl_l, tbl_r)
            
            decision = order.is_decision(ctx)
            val = (val_l + val_r) if decision else np.sqrt(val_l*val_r)
            tbl[ctx.node, negated] = val

        val *= 2**order.decisions_on_edge(ctx)
        return tbl, val

    tbl = post_order(bdd, merge)[0]
    return PolicyTable(coeff=coeff, order=order, bdd=bdd, tbl=tbl)
