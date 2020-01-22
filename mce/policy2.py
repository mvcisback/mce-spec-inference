# TODO: switch to python3.8 for cached properties
from typing import TypeVar, Tuple, Dict

import aiger_bv as BV
import aiger_coins as C
import attr
import funcy as fn
import numpy as np
from fold_bdd import fold_path, post_order
from fold_bdd.folds import Context
from scipy.optimize import brentq

from mce.bdd import to_bdd, TIMED_INPUT_MATCHER
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
    tbl: Dict[Tuple[BExpr, bool], float] = None

    @property
    def horizon(self):
        return self.order.horizon

    def keys(self):
        return self.tbl.keys()

    def items(self):
        return self.tbl.items()

    def __getitem__(self, key):
        if isinstance(key, Context):
            key = key.node, key.path_negated

        return self.tbl[key]

    @property
    def lsat(self):
        # TODO: replace with ctx.first_lvl
        first_lvl = self.bdd.level

        def bump(ctx, q, acc):
            first_decision = ctx.curr_lvl == first_lvl
            prev_was_decision = self.order.prev_was_decision(ctx)

            if not first_decision and prev_was_decision:
                acc += self.order.decision_entropy(ctx)
                acc += self.order.on_boundary(ctx)*q
            return acc


        def merge(ctx, low, high):
            q = np.log(self[ctx])

            if ctx.is_leaf:
                acc = 0 if ctx.node_val ^ ctx.path_negated else -float('inf')
            else:
                low, l_ctx, l_q = low
                high, h_ctx, h_q = high
                low = bump(l_ctx, l_q, low)
                high = bump(h_ctx, h_q, high)

                acc = softmax(low, high)
                if not self.order.is_decision(ctx):
                    acc -= np.log(2)
                elif self.order.on_boundary(ctx):
                    acc -= q

            #acc = bump(ctx, q, acc)
            return acc, ctx, q

        return post_order(self.bdd, merge)[0]

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
            acc += delta(ctx) * np.log(self[ctx]) \
                - self.order.decision_entropy(ctx)
            return acc

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


def policy(mdp, spec, horizon, coeff=None, manager=None, psat=None):
    assert (coeff is None) ^ (psat is None)
    if coeff is None:
        coeff = 0

    monitor = spec if isinstance(spec, BV.AIGBV) else BV.aig2aigbv(spec.aig)
    output = monitor.omap[fn.first(monitor.outputs)][0]
    composed = mdp >> C.circ2mdp(monitor)

    # HACK. TODO fix. Remove extra outputs.
    for out, size in mdp._aigbv.omap.items():
        if out not in composed.outputs:
            continue
        composed >>= C.circ2mdp(BV.sink(size, [out]))

    assert len(composed.outputs) == 1

    bdd, (*__), order = to_bdd(
        composed, horizon, output=output, manager=manager
    )
    ctrl = Policy(mdp, spec, horizon, policy_tbl(bdd, order, coeff))
    if psat is not None:
        ctrl.fit(psat)
    return ctrl


@attr.s
class Policy:
    mdp = attr.ib()
    spec = attr.ib()
    horizon = attr.ib()
    tbl = attr.ib()

    @property
    def coeff(self):
        return self.tbl.coeff

    @coeff.setter
    def coeff(self, val):
        self.tbl = policy_tbl(self.tbl.bdd, self.tbl.order, val)

    @property
    def psat(self):
        return self.tbl.psat

    @property
    def lsat(self):
        return self.tbl.lsat

    @property
    def order(self):
        return self.tbl.order

    def fit(self, sat_prob, top=100):

        def f(coeff):
            self.coeff = coeff
            return self.psat - sat_prob

        if f(-top) > 0:
            coeff = 0
        elif f(top) < 0:
            coeff = top
        else:
            coeff = brentq(f, -top, top)

        self.coeff = coeff

        if coeff < 0:
            self.coeff = 0

    def log_likelihood_ratio(self, trc):
        """Log Likelihood policy  - log likelihood dynamics"""
        return self.tbl.log_likelihood_ratio(trc)

    def _encode_trc(self, trc):
        for lvl in range(self.order.horizon*self.order.total_bits):
            var = self.tbl.bdd.bdd.var_at_level(lvl)
            t1 = self.order.time_step(lvl)

            name, idx, t2 = TIMED_INPUT_MATCHER.match(var).groups()
            assert t1 == int(t2)
            yield trc[t1][name][int(idx)]

    def encode_trcs(self, trcs):
        return [self.encode_trc(*v) for v in trcs]

    def encode_trc(self, sys_actions, states):
        trc = self.mdp.encode_trc(sys_actions, states)
        return list(self._encode_trc(trc))
