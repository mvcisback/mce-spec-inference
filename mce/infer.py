import time
from multiprocess import Pool

import funcy as fn

from mce.policy2 import policy
from mce.utils import empirical_sat_prob


def spec_mle(mdp, demos, specs, top=100, parallel=False, psat=None):
    horizon = len(demos[0][0])
    specs = list(specs)

    @fn.memoize
    def score(spec):
        start_time = time.time()
        times = {}
        print("building policy")
        ctrl = policy(mdp, spec, horizon=horizon, coeff=0)        
        times["build"] = time.time() - start_time
        print(f"done building policy")

        start_time = time.time()
        print("encoding traces")
        encoded_trcs = ctrl.encode_trcs(demos)
        print(f"done encoding traces")
        times["encode"] = time.time() - start_time

        if psat is None:
            sat_prob = empirical_sat_prob(spec, demos)
        else:
            sat_prob = psat

        start_time = time.time()
        print("fitting")
        ctrl.fit(sat_prob)
        print(f"done fitting")
        times["fit"] = time.time() - start_time

        start_time = time.time()
        print(f"compute likelihoods")
        logl = sum(map(ctrl.log_likelihood_ratio, encoded_trcs))
        times["trc_likelihood"] = time.time() - start_time

        print("\n----------------------------\n")
        print(f"ROBDD size: {ctrl.tbl.bdd.dag_size}")
        print(f"TBL size: {len(ctrl.tbl.tbl)}")
        print("\n".join(f"{key}: {val:.2}s" for key, val in times.items()))
        print("\n----------------------------\n")

        return logl

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
