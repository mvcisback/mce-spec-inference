from mce.policy import policy


def spec_mle(mdp, demos, specs, top=100):
    horizon = len(demos[0][0])
    assert all(len(demo[0]) == horizon for demo in demos)

    spec2score = {}
    best_spec = specs[0]
    best_score = -float('inf')
    encoded_demos = None  # HACK
    for spec in specs:
        ctrl = policy(mdp, spec, horizon=horizon)
        if encoded_demos is None:
            encoded_demos = ctrl.encode_trcs(demos)

        spec2score[spec] = ctrl.fit(demos, top=top, encoded_trcs=encoded_demos)
        if spec2score[spec] > best_score:
            best_score = spec2score[spec]
            best_spec = spec

    return best_spec, spec2score
