import funcy as fn

from mce.policy import policy


def spec_mle(mdp, demos, specs, top=100):
    horizon, encode_trcs = len(demos[0][0]), ctrl.encode_trcs(demos)

    @fn.memoize
    def score(spec):
        nonlocal spec2score
        ctrl = policy(mdp, spec, horizon=horizon)
        spec2score[spec] = ctrl.fit(demos, top=top, encoded_trcs=encoded_demos)
        return spec2score

    return max(specs, key=score), score.memory
