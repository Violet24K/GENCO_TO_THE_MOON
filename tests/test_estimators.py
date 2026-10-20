import numpy as np

from genco_market.estimators.multifidelity import (
    multifidelity,
    optimal_ratio,
    single_fidelity,
    split_budget,
    variance_ratio,
)


def _synthetic(N, K=5, rho=0.95, bias=3.0, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.linspace(0, 2, K)
    common = rng.normal(size=(N, 1)) * 5  # scenario effect shared by candidates
    noise = rng.normal(size=(N, K))
    hi = mu + common + noise
    lo = rho * hi + np.sqrt(1 - rho**2) * rng.normal(size=(N, K)) * hi.std() + bias
    return mu, hi, lo


def test_mf_unbiased_despite_lo_bias():
    errs = []
    for s in range(300):
        mu, hi, lo = _synthetic(400, seed=s)
        est = multifidelity(hi[:20], lo, alpha=1.0)
        errs.append(est.mean - (mu + 0))
    # scenario effect has mean 0, so the MF mean must center on mu
    assert np.all(np.abs(np.mean(errs, axis=0)) < 0.5)


def test_mf_beats_hi_only_at_same_n_hi():
    v_mf, v_hi = [], []
    for s in range(300):
        mu, hi, lo = _synthetic(2000, seed=s)
        v_mf.append(multifidelity(hi[:20], lo).mean)
        v_hi.append(single_fidelity(hi[:20]).mean)
    assert np.var(v_mf, axis=0).mean() < 0.5 * np.var(v_hi, axis=0).mean()


def test_se_matches_empirical_spread():
    means, ses = [], []
    for s in range(400):
        _, hi, lo = _synthetic(500, seed=s)
        e = multifidelity(hi[:30], lo)
        means.append(e.mean)
        ses.append(e.se)
    ratio = np.std(means, axis=0) / np.mean(ses, axis=0)
    assert np.all((ratio > 0.8) & (ratio < 1.25))


def test_ratio_formulas():
    assert optimal_ratio(0.0, 50) == 1.0
    assert abs(variance_ratio(0.9, 50) - 0.317) < 0.01
    assert abs(1 / variance_ratio(0.99, 50) - 12.6) < 0.2
    n, N = split_budget(5.0, cost_hi=0.02, cost_lo=0.0005, rho=0.9, K=6)
    assert N > n >= 2


def test_multi_cv_beats_single_when_errors_differ():
    from genco_market.estimators.multifidelity import multifidelity_multi

    v1, v2 = [], []
    for s in range(300):
        rng = np.random.default_rng(s)
        N, K = 1000, 4
        hi = rng.normal(size=(N, 1)) * 3 + rng.normal(size=(N, K)) + np.arange(K)
        e1, e2 = rng.normal(size=(N, K)), rng.normal(size=(N, K))
        lo1, lo2 = hi + e1 + 2.0, hi + e2 - 1.0  # independent errors, different biases
        v1.append(multifidelity(hi[:20], lo1).mean)
        v2.append(multifidelity_multi(hi[:20], [lo1, lo2]).mean)
    assert np.var(v2, axis=0).mean() < 0.8 * np.var(v1, axis=0).mean()
    assert np.all(np.abs(np.mean(v2, axis=0) - np.arange(4)) < 0.2)  # unbiased
