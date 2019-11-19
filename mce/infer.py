import funcy as fn

from mce.policy2 import policy
from mce.utils import empirical_sat_prob, ltl2monitor


def spec_mle(mdp, demos, specs, top=100):    
    horizon = len(demos[0][0])

    @fn.memoize
    def score(spec):
        ctrl = policy(mdp, spec, horizon=horizon, coeff=0)
        encoded_trcs = ctrl.encode_trcs(demos)
        ctrl.fit(empirical_sat_prob(ltl2monitor(spec), demos))
        return sum(map(ctrl.log_likelihood_ratio, encoded_trcs))

    best_spec = max(specs, key=score)
    return best_spec, fn.walk_keys(lambda x: x[0], score.memory)
