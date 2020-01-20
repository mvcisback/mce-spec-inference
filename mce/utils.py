import aiger_bv as BV
import funcy as fn


def empirical_sat_prob(monitor, trcs):
    if not isinstance(monitor, BV.AIGBV):
        if hasattr(monitor, "aigbv"):
            circ = monitor.aigbv
        else:
            circ = BV.aig2aigbv(monitor.aig)
    else:
        circ = monitor
    trcs2 = fn.pluck(1, trcs)  # Only care about state observations.
    # Only care about subset of inputs.
    trcs2 = [
        fn.lmap(lambda x: fn.project(x, circ.inputs), trc) for trc in trcs2
    ]

    assert len(circ.outputs) == 1

    name = fn.first(circ.outputs)
    n_sat = sum(circ.simulate(trc)[-1][0][name][0] for trc in trcs2)
    return n_sat / len(trcs)
