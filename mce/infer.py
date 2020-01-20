from multiprocess import Pool

import funcy as fn

from mce.policy2 import policy
from mce.utils import empirical_sat_prob


def spec_mle(mdp, demos, specs, top=100, parallel=False, psat=None):
    horizon = len(demos[0][0])
    specs = list(specs)

    @fn.memoize
    def score(spec):
        ctrl = policy(mdp, spec, horizon=horizon, coeff=0)
        encoded_trcs = ctrl.encode_trcs(demos)
        if psat is None:
            sat_prob = empirical_sat_prob(spec, demos)
        else:
            sat_prob = psat
        
        ctrl.fit(sat_prob)
        return sum(map(ctrl.log_likelihood_ratio, encoded_trcs))

    if parallel:
        _specs = list(enumerate(specs))

        def score2(spec):
            i, spec = spec
            return i, score(spec)

        spec2score = dict(Pool().map(score2, _specs))
        spec2score = fn.walk_keys(lambda idx: specs[idx], spec2score)
        best_spec = max(specs, key=spec2score.get)
        
    else:
        best_spec = max(specs, key=score)
        spec2score = fn.walk_keys(lambda x: x[0], score.memory)
    return best_spec, spec2score
