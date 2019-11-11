# TODO:
# 1. Support invalid inputs (decision and chance).
#    - Maintain validity BDD (pg 104)
# 2. Implement the dynamic programming on the CompressedDTree.
#    - Perhaps implement more generally first.
# 3. Force variable ordering to be correct
# https://github.com/tulip-control/dd/blob/master/examples/cudd_configure_reordering.py
# rationality = theano.dscalar('rationality')
# TODO: switch to theano
import attr
import funcy as fn
import theano
import theano.tensor as T
import numpy as np
from fold_bdd import fold_path, post_order
from scipy.optimize import brentq

from mce.order import BitOrder
from mce.bdd import to_bdd
 

def softmax(x, y):
    return T.log(T.exp(x) + T.exp(y))


def avg(x, y):
    return (x + y) / 2


def policy(mdp, spec, horizon, coeff="coeff"):
    mdp >>= spec
    bdd, _, relabels, order = to_bdd(mdp, horizon)
    
    coeff = T.dscalar(coeff)
    def merge(ctx, low, high):
        if ctx.is_leaf:
            val = coeff*(2*ctx.node_val - 1)
            tbl = {ctx.node: val}
        else:
            (tbl_l, val_l), (tbl_r, val_r) = low, high
            tbl = fn.merge(tbl_l, tbl_r)

            op = softmax if order.is_decision(ctx.curr_lvl) else avg
            val = op(val_l, val_r)
            tbl[ctx.node] = val

        val = ctx.delta(ctx.curr_lvl, ctx.prev_lvl, val)
        return tbl, val

    tbl = post_order(bdd, merge)[0]
    return Policy(coeff, tbl, order, bdd, relabels)


def _post_process(p, q, citvl, pitvl, ctx, order):
    if citvl == pitvl:  # Not at decision boundary.
        return p, q
    elif not order.is_decision(ctx.prev_lvl):  # Equiv decision bump.
        return p, ctx.delta(ctx.curr_lvl, ctx.prev_lvl, q)

    # Decision boundary. Weight by (prob of action) * (num actions).
    p *= T.exp(q)*(ctx.prev_lvl - pitvl[0] - 1)
    return p, None


@attr.s
class Policy:
    coeff = attr.ib()
    tbl = attr.ib()
    order = attr.ib()
    bdd = attr.ib()
    relabels = attr.ib()
    _fitted = attr.ib(default=False)

    def psat(self):
        def merge(ctx, low, high):
            citvl = self.order.interval(ctx.curr_lvl)
            pitvl = self.order.interval(ctx.prev_lvl)
            decision = self.order.is_decision(ctx.curr_lvl)

            if ctx.is_leaf:
                p = q = int(ctx.node_val)
                return _post_process(p, q, citvl, pitvl, ctx, self.order)

            (pl, ql), (pr, qr) = low, high
            if not decision:
                p = avg(pl, ph)
                q = self.tbl[ctx.node]
                return post_process(p, q, citvl, pitvl, ctx, self.order)

            p = pl + ph
            if citvl == pitvl:
                p *= ctx.skipped_decisions
            else:
                p *= (citvl[1] - ctx. curr_lvl - 1)
                p /= T.exp(tbl[ctx.node])  # Normalize.

            return post_process(p, q, citvl, pitvl, ctx, self.order)

        return post_order(self.bdd, merge)[0]

    def fit(self, sat_prob, top=100, fudge=1e-3):
        # TODO: binary search or root find.
        sat_prob = min(sat_prob, 1 - fudge)
        f = tensor.function([self.coeff], self.psat() - sat_prob)
        coeff = brentq(f, 0, top)
        self._fitted = True
        # TODO: transform tbl to use correct coeff.
        raise NotImplementedError()

    def likelihood(self, trcs):
        return np.product(map(self._likelihood, trcs))

    def _likelihood(self, trc):
        def prob(ctx, val, acc):
            pval = 1  # TODO: compute pval given ctx and policy.
            return acc * pval

        return fold_path(merge=prob, bexpr=self.bdd, vals=trc, initial=1)
