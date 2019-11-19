import funcy as fn

from mce.policy import policy


def spec_mle(mdp, demos, specs, top=100):
    
    horizon, encoded_trcs = len(demos[0][0]), None

    spec2score = {}
    @fn.memoize
    def score(spec):
        nonlocal spec2score
        nonlocal encoded_trcs

        ctrl = policy(mdp, spec, horizon=horizon)
        if encoded_trcs is None:
            encoded_trcs = ctrl.encode_trcs(demos)

        spec2score[spec] = ctrl.fit(demos, top=top, encoded_trcs=encoded_trcs)
        return spec2score

    return max(specs, key=score), score.memory
