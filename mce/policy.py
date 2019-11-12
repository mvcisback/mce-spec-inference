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
from mce.bdd import to_bdd, TIMED_INPUT_MATCHER
 

def softmax(x, y):
    return T.log(T.exp(x) + T.exp(y))


def avg(x, y):
    return (x + y) / 2


def policy(mdp, spec, horizon, coeff="coeff"):
    orig_mdp = mdp
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

        prev_lvl = -1 if ctx.prev_lvl is None else ctx.prev_lvl
        val = order.delta(prev_lvl, ctx.curr_lvl, val)
        return tbl, val

    tbl = post_order(bdd, merge)[0]
    return Policy(coeff, tbl, order, bdd, orig_mdp, spec, relabels)


@attr.s
class Policy:
    coeff = attr.ib()
    tbl = attr.ib()
    order = attr.ib()
    bdd = attr.ib()
    mdp = attr.ib()
    spec = attr.ib()
    relabels = attr.ib()
    _fitted = attr.ib(default=False)

    def psat(self):
        exp = np.exp if self._fitted else T.exp
        def merge(ctx, low, high):
            prev_lvl = -1 if ctx.prev_lvl is None else ctx.prev_lvl
            q = self.tbl[ctx.node]
            if ctx.is_leaf:
                p = int(ctx.node_val)
            elif self.order.is_decision(ctx.curr_lvl):
                p = low + high

                on_boundary = self.order.interval(ctx.curr_lvl) != \
                    self.order.interval(prev_lvl)

                if on_boundary:
                    p /= exp(q)
            else:
                # TODO: Replace with avg after testing.
                p = (low + high) / 2 
                
            if prev_lvl == -1 or self.order.is_decision(prev_lvl):
                skipped = self.order.skipped_decisions(prev_lvl, ctx.curr_lvl)
                p *= 2**skipped

            if prev_lvl != -1 and self.order.is_decision(prev_lvl):
                p *= exp(q)                

            return p
        
        return post_order(self.bdd, merge)

    def empirical_sat_prob(self, trcs):
        circ = self.mdp.aigbv
        name = fn.first(self.mdp.outputs)
        trcs = [self.mdp.encode_trc(*v) for v in trcs]
        n_sat = sum(circ.simulate(trc)[-1][0][name][0] for trc in trcs)
        return n_sat / len(trcs)

    def fit(self, sat_prob_or_trcs, top=100, fudge=1e-3):
        if not isinstance(sat_prob_or_trcs, float):
            sat_prob = empirical_sat_prob(sat_prob_or_trcs)
        else:
            sat_prob = sat_prob_or_trcs

        assert not self._fitted
        # TODO: binary search or root find.
        sat_prob = min(sat_prob, 1 - fudge)
        f = theano.function([self.coeff], self.psat() - sat_prob)
        coeff = brentq(f, 0, top)
        self._fitted = True
        self.fix_coeff(coeff)

    def fix_coeff(self, coeff):
        for k, val in self.tbl.items():
            self.tbl[k] = theano.function([self.coeff], val)(coeff)

        self.coeff = coeff
        self._fitted = True

    def _encode_trc(self, trc):
        for lvl in range(self.order.horizon*self.order.total_bits):
            var = self.bdd.bdd.var_at_level(lvl)
            var = self.relabels.inv[var]
            t1 = self.order.time_step(lvl)

            name, idx, t2 = TIMED_INPUT_MATCHER.match(var).groups()
            assert t1 == int(t2)
            yield trc[t1][name][int(idx)]

    def encode_trc(self, sys_actions, states):
        trc = self.mdp.encode_trc(sys_actions, states)
        return list(self._encode_trc(trc))

    def log_likelihood(self, trcs):
        trcs = [self.encode_trc(*v) for v in trcs]
        return sum(fn.lmap(self._log_likelihood, trcs))

    def _log_likelihood(self, trc):
        assert self._fitted
        def prob(ctx, val, acc):
            q = self.tbl[ctx.node]
            if ctx.is_leaf:
                return acc * np.exp(q)
            elif not self.order.is_decision(ctx.curr_lvl):
                return acc / 2  # Flip fair coin
            
            prev_lvl = -1 if ctx.prev_lvl is None else ctx.prev_lvl

            on_boundary = self.order.interval(ctx.curr_lvl) != \
                self.order.interval(prev_lvl)

            if on_boundary:
                acc /= np.exp(q)
                
            if prev_lvl == -1 or self.order.is_decision(prev_lvl):
                skipped = self.order.skipped_decisions(prev_lvl, ctx.curr_lvl)
                acc /= 2**skipped

            if prev_lvl != -1 and self.order.is_decision(prev_lvl):
                acc *= np.exp(q)                

            return acc
        
        _prob = fold_path(merge=prob, bexpr=self.bdd, vals=trc, initial=1)
        return np.log(_prob)
