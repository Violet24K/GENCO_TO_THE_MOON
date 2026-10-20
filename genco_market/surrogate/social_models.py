"""Social-cost models for Layer 4 (outage coordination), at any fidelity.

    model.evaluate(scenarios, timeout_s=None) -> dict(cost [n], shed [n], ok [n])
    model.cost_per_eval_s()

``cost`` = true generation cost + VOLL * shed ($/h, env/welfare.py), with
everyone bidding cost (markup 1). Failed / unfinished entries are NaN.

- ``PoolSocial("ac")``: AC-OPF market clearing (truth; incl. its DC fallback).
- ``PoolSocial("dc_highs")``: DC-OPF LP with load shedding at VOLL.
- ``GencoSocial``: GENCO dispatch. GENCO was trained on feasible OPF solutions
  only and has no notion of load shedding, so its social cost is
  sum(c1 * pg) + VOLL * max(0, sum(pd) + loss_ref - sum(pg)), i.e. any power
  GENCO "forgets" to produce beyond normal losses is charged as shed. The raw
  pieces are kept in ``last`` so other estimates can be tried offline
  (experiments/l4_offline.py reports how each correlates with AC).
"""

from __future__ import annotations

import time

import numpy as np

from genco_market.core.schemas import MarketRules, Scenario
from genco_market.env.grid import GridInfo, scenario_to_network

VOLL = 2000.0
LOSS_FRAC_REF = 0.0  # GENCO shortfall reference: sum(pg) >= sum(pd) * (1 + LOSS_FRAC_REF); set from data


class SocialModel:
    name = "base"
    prior_s = 1e-2

    def __init__(self, info: GridInfo, voll: float = VOLL):
        self.info, self.voll = info, voll
        self._wall, self._n = 0.0, 0
        self.last: dict = {}

    def cost_per_eval_s(self) -> float:
        return self._wall / self._n if self._n else self.prior_s

    def _account(self, t0: float, n: int) -> None:
        self._wall += time.perf_counter() - t0
        self._n += n


class PoolSocial(SocialModel):
    def __init__(self, pool, info: GridInfo, task: str = "ac", voll: float = VOLL):
        super().__init__(info, voll)
        self.pool, self.task = pool, task
        self.name = {"ac": "AC", "dc_highs": "DC"}.get(task, task)
        self.prior_s = 1 / 50 if task == "ac" else 1 / 2000

    def submit(self, scenarios: list[Scenario]) -> list[int]:
        return [self.pool.submit(self.task, s, 1.0, MarketRules(), voll=self.voll) for s in scenarios]

    def gather(self, scenarios, jobs, timeout_s=None, t0=None) -> dict:
        t0 = time.perf_counter() if t0 is None else t0
        got = self.pool.collect(jobs, timeout_s)
        if timeout_s is not None and len(got) < len(jobs):
            self.pool.cancel([j for j in jobs if j not in got])
        n = len(scenarios)
        cost, shed = np.full(n, np.nan), np.full(n, np.nan)
        status = np.full(n, "NOT_DONE", dtype=object)
        for i, (s, j) in enumerate(zip(scenarios, jobs)):
            r = got.get(j)
            if r is None:
                continue
            if not r:
                status[i] = "TASK_ERROR"
                continue
            status[i] = r.status
            if r.status != "FAILED" and np.all(np.isfinite(r.pg)):
                shed[i] = r.load_shed_mw
                cost[i] = float(np.dot(np.asarray(s.gen_c1), np.asarray(r.pg))) + self.voll * r.load_shed_mw
        self._account(t0, int((status != "NOT_DONE").sum()))
        self.last = dict(cost=cost, shed=shed, status=status)
        return dict(cost=cost, shed=shed, ok=np.isfinite(cost))

    def evaluate(self, scenarios: list[Scenario], timeout_s: float | None = None) -> dict:
        t0 = time.perf_counter()
        return self.gather(scenarios, self.submit(scenarios), timeout_s, t0)


class GencoSocial(SocialModel):
    name = "GENCO"
    prior_s = 1e-3

    def __init__(self, service, info: GridInfo, voll: float = VOLL, loss_frac_ref: float = LOSS_FRAC_REF,
                 batch_size: int = 1024):
        super().__init__(info, voll)
        self.svc, self.loss_frac_ref, self.batch_size = service, loss_frac_ref, batch_size

    def evaluate(self, scenarios: list[Scenario], timeout_s: float | None = None) -> dict:
        t0 = time.perf_counter()
        nets = [scenario_to_network(s, None, self.info.cfg.grid) for s in scenarios]
        out = self.svc.predict_arrays(nets, self.batch_size)
        pg = out["pg"].astype(float)
        c1 = np.array([s.gen_c1 for s in scenarios], float)
        pd = np.array([sum(s.bus_pd) for s in scenarios], float)
        gen_cost = (c1 * pg).sum(1)
        sum_pg = pg.sum(1)
        short = np.maximum(0.0, pd * (1 + self.loss_frac_ref) - sum_pg)
        cost = gen_cost + self.voll * short
        self._account(t0, len(scenarios))
        self.last = dict(gen_cost=gen_cost, sum_pg=sum_pg, sum_pd=pd, res_p=out["res_p_mw"].astype(float),
                         pbe_max=out["pbe_max"].astype(float), short=short)
        ok = np.isfinite(cost)
        return dict(cost=np.where(ok, cost, np.nan), shed=short, ok=ok)


class GencoOpSocial(GencoSocial):
    """Operator GENCO: GENCO's generation cost plus a learned social-cost residual (load shedding and
    generation-cost bias) from a SocialHead on GENCO's bus embeddings (surrogate/social_head.py)."""

    name = "GENCO-op"

    def __init__(self, service, info: GridInfo, head, voll: float = VOLL, batch_size: int = 1024):
        super().__init__(service, info, voll, batch_size=batch_size)
        self.head = head

    def evaluate(self, scenarios: list[Scenario], timeout_s: float | None = None) -> dict:
        from genco_market.surrogate.social_head import head_features

        t0 = time.perf_counter()
        nets = [scenario_to_network(s, None, self.info.cfg.grid) for s in scenarios]
        out = self.svc.predict_arrays(nets, self.batch_size, embeddings=True)
        pg = out["pg"].astype(float)
        c1 = np.array([s.gen_c1 for s in scenarios], float)
        pd = np.array([s.bus_pd for s in scenarios], float)
        off = np.array([[x == 0 for x in s.branch_status] for s in scenarios])
        gen_cost = (c1 * pg).sum(1)
        cost = gen_cost + self.head.predict_residual(out["emb_bus"], head_features(pd, off, self.info))
        self._account(t0, len(scenarios))
        self.last = dict(gen_cost=gen_cost, sum_pg=pg.sum(1), sum_pd=pd.sum(1), res_p=out["res_p_mw"].astype(float),
                         pbe_max=out["pbe_max"].astype(float), sc=cost)
        ok = np.isfinite(cost)
        return dict(cost=np.where(ok, cost, np.nan), shed=np.zeros(len(scenarios)), ok=ok)
