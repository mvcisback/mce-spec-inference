# TODO:
# 1. Support invalid inputs (decision and chance).
#    - Maintain validity BDD (pg 104)
# 2. Implement the dynamic programming on the CompressedDTree.
#    - Perhaps implement more generally first.
# 3. Force variable ordering to be correct
# https://github.com/tulip-control/dd/blob/master/examples/cudd_configure_reordering.py
# rationality = theano.dscalar('rationality')
# TODO: switch to theano
import aiger_bv as BV
import aiger_coins as C
import attr
import funcy as fn
import theano
import theano.tensor as T
import numpy as np
from fold_bdd import fold_path, post_order
from scipy.optimize import brentq

from mce.order import BitOrder
from mce.bdd import to_bdd, TIMED_INPUT_MATCHER
from mce.utils import empirical_sat_prob
 

def softmax(x, y):
    m = T.max([x, y])
    x2, y2 = x - m, y - m
    return T.log(T.exp(x2) + T.exp(y2)) + m


def avg(x, y):
    return (x + y) / 2


def function(*args, **kwargs):
    kwargs['on_unused_input'] = 'ignore'
    kwargs['mode'] = theano.Mode(optimizer="stabilize")
    return theano.function(*args, **kwargs)


def policy(mdp, spec, horizon, coeff="coeff"):
    orig_mdp = mdp
    spec_circ = BV.aig2aigbv(spec.aig)
    mdp >>= C.circ2mdp(spec_circ)
    # HACK. TODO fix
    for out, size in orig_mdp._aigbv.omap.items():
        if out not in mdp.outputs:
            continue
        mdp >>= C.circ2mdp(BV.sink(size, [out]))
    assert len(mdp.outputs) == 1

    output = spec_circ.omap[spec.output][0]

    bdd, _, relabels, order = to_bdd(mdp, horizon, output=output)
    
    coeff = T.dscalar(coeff)
    def merge(ctx, low, high):
        negated = ctx.path_negated
        if ctx.is_leaf:
            val = coeff*(2*ctx.node_val - 1)
            if negated:
                val *= -1
            tbl = {(ctx.node, negated): val}
        else:
            (tbl_l, val_l), (tbl_r, val_r) = low, high
            tbl = fn.merge(tbl_l, tbl_r)

            op = softmax if order.is_decision(ctx) else avg
            tbl[ctx.node, negated] = val = op(val_l, val_r)

        equiv_decision_bump = order.decisions_on_edge(ctx)*T.log(2)
        return tbl, val + equiv_decision_bump

    tbl = post_order(bdd, merge)[0]
    return Policy(coeff, tbl, order, bdd, orig_mdp, spec, spec_circ, relabels)


@attr.s
class Policy:
    coeff = attr.ib()
    tbl = attr.ib()
    order = attr.ib()
    bdd = attr.ib()
    mdp = attr.ib()
    spec = attr.ib()
    monitor = attr.ib()
    relabels = attr.ib()
    _fitted = attr.ib(default=False)

    def value(self, ctx):
        return self.tbl[ctx.node, ctx.path_negated]

    def psat(self):
        exp = np.exp if self._fitted else T.exp
        const = (lambda x: x) if self._fitted else T.constant
        order = self.order

        def merge(ctx, low, high):
            q = self.value(ctx)
            if ctx.is_leaf:
                p = const(int(ctx.node_val))
                if ctx.path_negated:
                    p = 1 - p
            elif not order.is_decision(ctx):
                p = avg(low, high)
            else:
                p = (low + high) / (int(order.on_boundary(ctx))*exp(q))

            first_decision = order.first_real_decision(ctx)
            prev_was_decision = order.prev_was_decision(ctx)
                
            if not first_decision and prev_was_decision:
                skipped = order.decisions_on_edge(ctx)
                p *= 2**skipped

            if not first_decision and prev_was_decision:
                p *= exp(q)                

            return p
        
        return post_order(self.bdd, merge)

    def empirical_sat_prob(self, trcs):
        return empirical_sat_prob(self.monitor, trcs)

    def fit(self, sat_prob_or_trcs, top=100, fudge=1e-3, encoded_trcs=None):
        if not isinstance(sat_prob_or_trcs, float):
            sat_prob = self.empirical_sat_prob(sat_prob_or_trcs)
        else:
            sat_prob = sat_prob_or_trcs

        assert not self._fitted
        f = function([self.coeff], self.psat() - sat_prob)
        if f(-top) > 0:
            coeff = 0
        elif f(top) < 0 :
            coeff = top
        else:
            coeff = brentq(f, -top, top)
        if coeff < 0:
            coeff = 0

        self.fix_coeff(coeff)
        
        if not isinstance(sat_prob_or_trcs, float):
            if encoded_trcs is not None:
                return self.log_likelihood(encoded_trcs, encoded=True)
            
            return self.log_likelihood(sat_prob_or_trcs)

    def fix_coeff(self, coeff):
        for k, val in self.tbl.items():
            self.tbl[k] = function([self.coeff], val)(coeff)

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

    def encode_trcs(self, trcs):
        return [self.encode_trc(*v) for v in trcs]

    def encode_trc(self, sys_actions, states):
        trc = self.mdp.encode_trc(sys_actions, states)
        return list(self._encode_trc(trc))

    def log_likelihood(self, trcs, encoded=False):
        if not encoded:
            trcs = self.encode_trcs(trcs)
        lls = fn.lmap(self._log_likelihood, trcs)
        return sum(lls)

    def _log_likelihood(self, trc):
        assert self._fitted
        order = self.order

        def prob(ctx, val, acc):
            if not order.is_decision(ctx):
                # TODO: Document this!
                return acc  # Only Want Decision Likelihoods            

            q = self.value(ctx)
            first_decision = order.first_real_decision(ctx)
            prev_was_decision = order.prev_was_decision(ctx)

            if ctx.is_leaf:
                acc += q
                if not first_decision:
                    return acc

            if order.on_boundary(ctx):
                acc -= q
            
            if first_decision or prev_was_decision:
                acc -= order.decisions_on_edge(ctx)*np.log(2)

            if (not first_decision) and prev_was_decision:
                acc += q
            return acc
        
        return fold_path(merge=prob, bexpr=self.bdd, vals=trc, initial=0)
