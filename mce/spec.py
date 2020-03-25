from typing import FrozenSet

import attr
import aiger_bdd
import aiger_bv as BV
import aiger_coins as C
import funcy as fn
from aiger_bv.bundle import BundleMap

from mce.preimage import preimage
from mce.order import BitOrder
from mce.bdd import to_bdd


def xor(x, y):
    return (x | y) & ~(x & y)


@attr.s(frozen=True, auto_attribs=True)
class ConcreteSpec:
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

    def accepts(self, actions) -> bool:
        """Does this spec accept the given sequence of (sys, env) actions."""
        manager = self.manager

        timed_actions = {}
        for t, action in enumerate(actions):
            c, a = action['c'], action['a']
            timed_actions.update(
                {f'c##time_{t}[0]': c[0], f'a##time_{t}[0]': a[0]}
            )

        assert timed_actions.keys() == manager.vars.keys()
        tmp = manager.let(timed_actions, self.bexpr)
        assert tmp in (manager.true, manager.false)
        return tmp == manager.true

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


def concretize(
        monitor: C.MDP, sys: C.MDP, horizon: int, manager=None
) -> ConcreteSpec:
    """
    Convert an abstract specification monitor and a i/o transition
    system into a concrete specification over the horizion.
    """

    bexpr, manager, _, order = to_bdd(sys >> monitor, horizon)
    return ConcreteSpec(bexpr, order, sys_inputs=sys.inputs, dyn=sys.aigbv)
