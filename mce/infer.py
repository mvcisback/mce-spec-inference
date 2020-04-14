import time
from multiprocess import Pool

import funcy as fn

from mce.policy3 import fit
from mce.spec import concretize
from mce.demos import encode_trcs, log_likelihoods


def spec_mle(mdp, demos, specs, top=100, parallel=False, psat=None):
    """
    Searches for the most likely specification in specs given
    demonstations, demos, from an agent operating in mdp.
    """
    horizon = len(demos[0][0])
    specs = list(specs)

    start_time = time.time()
    print("encoding traces")
    encoded_trcs = encode_trcs(mdp, demos)
    print(f"done encoding traces")

    @fn.memoize
    def score(spec):
        start_time = time.time()
        times = {}

        print("concretizing spec")
        cspec = concretize(spec, mdp, horizon)
        print("done spec")
        times["build spec"] = time.time() - start_time

        if psat is None:
            sat_prob = sum(cspec.accepts(trc) for trc in encoded_trcs)
            sat_prob /= len(encoded_trcs)
        else:
            sat_prob = psat

        start_time = time.time()
        print("fitting policy")
        ctrl = fit(cspec, sat_prob)
        print(f"done fitting")
        times["fit"] = time.time() - start_time

        start_time = time.time()
        print(f"compute log likelihood of demos")
        lprob = log_likelihoods(ctrl, encoded_trcs)

        times["surprise"] = time.time() - start_time

        print("\n----------------------------\n")
        print(f"BDD size: {cspec.bexpr.dag_size}")
        print(f"Controller Size: {ctrl.size}")
        print(f"log_prob: {lprob}")
        print("\n".join(f"{key}: {val:.2}s" for key, val in times.items()))
        print("\n----------------------------\n")

        print(times)

        return lprob

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
