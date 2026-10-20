"""Profit estimators over K candidate bids with common random numbers.

Inputs are profit matrices of shape (n_scenarios, K): row i = scenario i
(shared by all candidates), column k = candidate bid k.

- ``single_fidelity``: plain Monte Carlo mean (IPOPT-only, GENCO-only, DC-only).
- ``multifidelity``: control-variate estimator (README result 3). The
  high-fidelity model is evaluated on the first n scenarios, the low-fidelity
  one on all N >= n. Unbiased for the high-fidelity mean for any alpha.
- ``optimal_ratio``: N/n that minimizes variance for a fixed cost budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Estimate:
    mean: np.ndarray  # (K,)
    se: np.ndarray  # (K,) standard error
    n_hi: int
    n_lo: int
    alpha: np.ndarray | None = None
    rho: np.ndarray | None = None

    def best(self) -> int:
        return int(np.nanargmax(self.mean))

    def ci(self, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
        return self.mean - z * self.se, self.mean + z * self.se


def single_fidelity(profits: np.ndarray) -> Estimate:
    p = np.asarray(profits, dtype=float)
    n = p.shape[0]
    se = p.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(p.shape[1], np.inf)
    return Estimate(mean=p.mean(axis=0), se=se, n_hi=n, n_lo=0)


ALPHA_CLIP = (0.0, 1.0)  # estimated alpha is clipped: with few paired samples and heavy-tailed
# surrogate errors (e.g. scarcity prices) the raw OLS alpha can explode and wreck the estimate.


def multifidelity(hi: np.ndarray, lo: np.ndarray, alpha: float | np.ndarray | None = None,
                  alpha_clip: tuple[float, float] | None = ALPHA_CLIP) -> Estimate:
    """hi: (n, K) high-fidelity profits on scenarios 0..n-1.
    lo: (N, K) low-fidelity profits on scenarios 0..N-1 (same first n scenarios).
    alpha=None uses the per-candidate optimal alpha = cov(hi, lo) / var(lo)
    estimated on the paired samples (a small, standard O(1/n) bias), clipped
    to ``alpha_clip``. Any fixed alpha keeps the estimator exactly unbiased."""
    hi = np.asarray(hi, dtype=float)
    lo = np.asarray(lo, dtype=float)
    n, N = hi.shape[0], lo.shape[0]
    assert N >= n >= 2 and hi.shape[1] == lo.shape[1]
    lo_n = lo[:n]
    var_hi = hi.var(axis=0, ddof=1)
    var_lo = lo_n.var(axis=0, ddof=1)
    cov = ((hi - hi.mean(0)) * (lo_n - lo_n.mean(0))).sum(0) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(var_hi * var_lo > 0, cov / np.sqrt(var_hi * var_lo), 0.0)
        if alpha is None:
            a = np.where(var_lo > 0, cov / var_lo, 0.0)
            if alpha_clip is not None:
                a = np.clip(a, *alpha_clip)
        else:
            a = np.broadcast_to(alpha, hi.shape[1]).astype(float)
    mean = hi.mean(0) + a * (lo.mean(0) - lo_n.mean(0))
    # Var = var_hi/n + a^2 var_lo (1/n - 1/N) - 2 a cov (1/n - 1/N)
    d = 1.0 / n - 1.0 / N
    var = var_hi / n + (a**2 * var_lo - 2 * a * cov) * d
    return Estimate(mean=mean, se=np.sqrt(np.maximum(var, 0.0)), n_hi=n, n_lo=N, alpha=a, rho=rho)


def multifidelity_multi(hi: np.ndarray, los: list[np.ndarray]) -> Estimate:
    """Several low-fidelity models as joint control variates (all evaluated on
    the same N >= n scenarios). Per candidate, alpha = OLS of hi on the lo's
    over the n paired rows; unbiased for fixed alpha, small O(1/n) bias when
    estimated. Useful when the lo models err in different conditions."""
    hi = np.asarray(hi, dtype=float)
    L = np.stack([np.asarray(x, dtype=float) for x in los], axis=-1)  # (N, K, J)
    n, N, K, J = hi.shape[0], L.shape[0], hi.shape[1], L.shape[2]
    assert N >= n > J + 1
    mean, var, rho = np.empty(K), np.empty(K), np.empty(K)
    a_all = np.empty((K, J))
    for k in range(K):
        X = L[:n, k, :] - L[:n, k, :].mean(0)
        y = hi[:, k] - hi[:, k].mean()
        a, *_ = np.linalg.lstsq(X, y, rcond=None)
        a = np.clip(a, *ALPHA_CLIP)
        if a.sum() > ALPHA_CLIP[1]:
            a = a * ALPHA_CLIP[1] / a.sum()
        a_all[k] = a
        delta = L[:, k, :].mean(0) - L[:n, k, :].mean(0)
        mean[k] = hi[:, k].mean() + a @ delta
        resid = y - X @ a
        r2 = 1 - resid.var() / y.var() if y.var() > 0 else 0.0
        rho[k] = np.sqrt(max(r2, 0.0))
        # Var = var_hi/n * (1 - R^2 (1 - n/N))
        var[k] = y.var(ddof=1) / n * (1 - max(r2, 0.0) * (1 - n / N))
    return Estimate(mean=mean, se=np.sqrt(np.maximum(var, 0)), n_hi=n, n_lo=N, alpha=a_all, rho=rho)


def optimal_ratio(rho: float, r: float) -> float:
    """N/n minimizing variance at fixed budget (Peherstorfer et al. 2016),
    r = cost_hi / cost_lo. Returns 1.0 when the low fidelity does not help."""
    rho2 = min(float(rho) ** 2, 1 - 1e-9)
    return max(1.0, float(np.sqrt(r * rho2 / (1 - rho2))))


def variance_ratio(rho: float, r: float) -> float:
    """Var(MF) / Var(hi-only) at equal budget, with the optimal ratio."""
    return (np.sqrt(1 - rho**2) + np.sqrt(rho**2 / r)) ** 2


def split_budget(budget_s: float, cost_hi: float, cost_lo: float, rho: float, K: int) -> tuple[int, int]:
    """Number of scenarios (n_hi, N_lo) per candidate that fit in budget_s,
    assuming each scenario costs K evaluations per fidelity."""
    ratio = optimal_ratio(rho, cost_hi / cost_lo)
    per_n = K * (cost_hi + ratio * cost_lo)
    n = max(2, int(budget_s / per_n))
    return n, max(n, int(n * ratio))
