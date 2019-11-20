from multiprocessing import Pool

import funcy as fn

from mce.policy2 import policy
from mce.utils import empirical_sat_prob, ltl2monitor


def spec_mle(mdp, demos, specs, top=100, parallel=False, psat=None):
    horizon = len(demos[0][0])

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
        def score2(spec):
            return score(spec), spec

        _, best_spec = max(Pool().map(score2, specs), key=lambda x: x[0])
    else:
        best_spec = max(specs, key=score)
        spec2score = fn.walk_keys(lambda x: x[0], score.memory)
    return best_spec, spec2score
