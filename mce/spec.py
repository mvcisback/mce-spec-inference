__all__ = ["ConcreteSpec", "concretize"]

from typing import FrozenSet, Sequence

import attr
import aiger_bdd
import aiger_bv as BV
import aiger_coins as C
import funcy as fn
from aiger_bv.bundle import BundleMap
from bdd2dfa.b2d import to_dfa, BNode
from bidict import bidict
from dfa import DFA

from mce.preimage import preimage
from mce.order import BitOrder
from mce.bdd import to_bdd


def xor(x, y):
    return (x | y) & ~(x & y)


@attr.s(frozen=True, auto_attribs=True)
class ConcreteSpec:
    """
    Models an concrete specification over sequences of
    system/environment action pairs encoded as bitvectors.
    """

    bexpr: "BDD"
    order: BitOrder  # TODO: Make this a derived quantity.
    dyn: BV.AIGBV
    sys_inputs: FrozenSet[str] = attr.ib(converter=frozenset)

    @property
    def imap(self) -> BundleMap:
        """System input map."""
        return self.dyn.imap.project(self.sys_inputs)

    @property
    def emap(self) -> BundleMap:
        """Environment input map."""
        return self.dyn.imap.omit(self.sys_inputs)

    @property
    def horizon(self) -> int:
        return self.order.horizon

    @property
    def manager(self):
        return self.bexpr.bdd

    def flatten(self, actions) -> Sequence[bool]:
        """
        Converts structured sequence of (sys, env) actions to a
        sequence of bits that this concrete specification recognizes.
        """
        manager = self.manager

        timed_actions = {}
        bmap = self.imap + self.emap
        for t, action in enumerate(actions):
            old2new = {k: f'{k}##time_{t}' for k in bmap.keys()}
            bmap_t = bmap.relabel(old2new)
            action_t = fn.walk_keys(old2new.get, action)
            timed_actions.update(bmap_t.blast(action_t))

        idx2key = bidict(self.bexpr.bdd.vars).inv
        return [timed_actions[idx2key[i]] for i in range(len(idx2key))]
        
    def accepts(self, actions) -> bool:
        """Does this spec accept the given sequence of (sys, env) actions."""
        flattened = self.flatten(actions)
        assert len(flattened) == self.order.horizon * self.order.total_bits
        return self._as_dfa(qdd=True).label(flattened)

    def toggle(self, actions):
        """Toggles a sequence of (sys, env) actions."""
        assert len(actions) == self.horizon
        aps = fn.lpluck(0, self.dyn.simulate(actions))
        expr = preimage(aps=aps, mdp=self._unrolled(), is_unrolled=True)
        bexpr, *_ = aiger_bdd.to_bdd(
            expr, manager=self.manager, renamer=lambda _, x: x
        )

        return attr.evolve(self, bexpr=xor(self.bexpr, bexpr))

    @fn.cache(60 * 5)  # Evict after 5 minutes.
    def _unrolled(self) -> BV.AIGBV:
        return self.dyn.unroll(self.horizon)

    @fn.cache(60 * 1)  # Evict after 1 minute.
    def _as_dfa(self, qdd=False) -> DFA:
        """
        Returns a dfa with binary alphabet which models the
        ConcreteSpecification with the order given by self.bexpr.
        """
        return to_dfa(self.bexpr, qdd=qdd)

    def abstract_trace(self, actions) -> Sequence[BNode]:
        """Path a set of (sys, env) actions takes through BDD."""
        return self._as_dfa().trace(self.flatten(actions))


def concretize(
        monitor: C.MDP, sys: C.MDP, horizon: int, manager=None
) -> ConcreteSpec:
    """
    Convert an abstract specification monitor and a i/o transition
    system into a concrete specification over the horizion.
    """

    bexpr, manager, _, order = to_bdd(sys >> monitor, horizon)
    return ConcreteSpec(bexpr, order, sys_inputs=sys.inputs, dyn=sys.aigbv)
