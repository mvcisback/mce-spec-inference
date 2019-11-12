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

            op = softmax if order.is_decision(ctx) else avg
            val = op(val_l, val_r)
            tbl[ctx.node] = val

        equiv_decision_bump = order.decisions_on_edge(ctx)*T.log(2)
        return tbl, val + equiv_decision_bump

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
        order = self.order
        def merge(ctx, low, high):
            q = self.tbl[ctx.node]
            if ctx.is_leaf:
                p = int(ctx.node_val)
            elif not order.is_decision(ctx):
                p = avg(low, high)
            else:
                p = (low + high) / (int(order.on_boundary(ctx))*exp(q))

            first_decision = order.first_real_decision(ctx)
            prev_was_decision = order.prev_was_decision(ctx)
                
            if first_decision or prev_was_decision:
                skipped = order.decisions_on_edge(ctx)
                p *= 2**skipped

            if not first_decision and prev_was_decision:
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
        order = self.order

        def prob(ctx, val, acc):
            q = self.tbl[ctx.node]
            if ctx.is_leaf:
                return acc + q
            elif not order.is_decision(ctx):
                return acc - np.log(2)  # Flip fair coin

            if order.on_boundary(ctx):
                acc -= q
            
            first_decision = order.first_real_decision(ctx)
            prev_was_decision = order.prev_was_decision(ctx)

            if first_decision or prev_was_decision:
                acc -= order.decisions_on_edge(ctx)*np.log(2)

            if (not first_decision) and prev_was_decision:
                acc += q

            return acc
        
        return fold_path(merge=prob, bexpr=self.bdd, vals=trc, initial=0)
