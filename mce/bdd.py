import aiger_bv
import aiger_bdd
import funcy as fn


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
