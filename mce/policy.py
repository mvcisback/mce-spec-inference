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
 

def avg(x, y):
    return (x + y) / 2


def function(*args, **kwargs):
    kwargs['on_unused_input'] = 'ignore'
    kwargs['mode'] = theano.Mode(optimizer="stabilize")
    return theano.function(*args, **kwargs)


def make_tbl(bdd, order, coeff):
    base1 = T.exp(coeff)
    base2 = T.exp(-coeff)

    def merge(ctx, low, high):
        negated = ctx.path_negated
        if ctx.is_leaf:
            val = base1 if ctx.node_val ^ negated else base2
            tbl2 = {(ctx.node, negated): val}
        else:
            (tbl2_l, val_l), (tbl2_r, val_r) = low, high
            tbl2 = fn.merge(tbl2_l, tbl2_r)
            
            decision = order.is_decision(ctx)
            val = (val_l + val_r) if decision else T.sqrt(val_l*val_r)
            tbl2[ctx.node, negated] = val

        val *= 2**order.decisions_on_edge(ctx)
        return tbl2, val

    tbl2, _ = post_order(bdd, merge)
    return tbl2


def policy(mdp, spec, horizon, coeff="coeff"):
    orig_mdp = mdp
    if not isinstance(spec, BV.AIGBV):
        spec_circ = BV.aig2aigbv(spec.aig)
    else:
        spec_circ = spec

    assert len(spec_circ.outputs)

    mdp >>= C.circ2mdp(spec_circ)
    # HACK. TODO fix
    for out, size in orig_mdp._aigbv.omap.items():
        if out not in mdp.outputs:
            continue
        mdp >>= C.circ2mdp(BV.sink(size, [out]))
    assert len(mdp.outputs) == 1

    output = spec_circ.omap[fn.first(spec_circ.outputs)][0]

    bdd, _, relabels, order = to_bdd(mdp, horizon, output=output)

    coeff = T.dscalar(coeff)
    tbl2 = make_tbl(bdd, order, coeff)

    return Policy(coeff, tbl2, order, bdd, orig_mdp, spec, spec_circ, relabels)


@attr.s
class Policy:
    coeff = attr.ib()
    tbl2 = attr.ib()
    order = attr.ib()
    bdd = attr.ib()
    mdp = attr.ib()
    spec = attr.ib()
    monitor = attr.ib()
    relabels = attr.ib()
    _fitted = attr.ib(default=False)

    def value(self, ctx):
        op = np.log if self._fitted else T.log
        return op(self.value2(ctx))

    def value2(self, ctx):
        return self.tbl2[ctx.node, ctx.path_negated]


    def psat(self, return_log=False):
        log = np.log if self._fitted else T.log
        exp = np.exp if self._fitted else T.exp
        const = (lambda x: x) if self._fitted else T.constant
        order = self.order
        max = np.max if self._fitted else T.max

        def softmax(x, y):
            m = max([x, y])
            x2, y2 = x - m, y - m
            return log(exp(x2) + exp(y2)) + m

        def merge(ctx, low, high):
            q = self.value2(ctx)
            if ctx.is_leaf:
                l = 0 if ctx.node_val ^ ctx.path_negated else -float('inf')
                l = const(l)
            else:
                l = softmax(low, high)
                if not order.is_decision(ctx):
                    l -= log(2)
                elif order.on_boundary(ctx):
                    l -= log(q)

            first_decision = order.first_real_decision(ctx)
            prev_was_decision = order.prev_was_decision(ctx)
                
            if not first_decision and prev_was_decision:
                skipped = order.decisions_on_edge(ctx)
                l += skipped*log(2)

            if not first_decision and prev_was_decision:
                l += log(q)

            return l
        
        val = post_order(self.bdd, merge)
        return val if return_log else exp(val)

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
        for k, val in self.tbl2.items():
            self.tbl2[k] = function([self.coeff], val)(coeff)

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

        return sum(fn.map(self._log_likelihood, trcs))

    def _log_likelihood(self, trc):
        assert self._fitted

        def delta(ctx):
            return ctx.is_leaf - self.order.first_real_decision(ctx)

        def decision_entropy(ctx):
            return np.log(2)*self.order.decisions_on_edge(ctx)

        def log_prob(ctx, val, acc):
            return acc + delta(ctx) * self.value(ctx) - decision_entropy(ctx)
        
        return fold_path(merge=log_prob, bexpr=self.bdd, vals=trc, initial=0)
