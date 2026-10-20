"""Profit models: the strategic firm's profit for (scenario, markup) pairs at any fidelity.

    model.evaluate(scenarios, markups, timeout_s=None) -> (n, K) $/h
    model.cost_per_eval_s()  -> measured wall s per (scenario, markup)

Profit is always settled at TRUE cost with the model's own LMP and dispatch
(``profit_from`` = env/market.firm_profits for the strategic firm), after the
market's mitigation rule and price cap (``rules``).

Failures: a scenario row is all-NaN when any of its K evaluations failed
(solver FAILED, worker error, non-finite LMP/dispatch) or did not finish
before ``timeout_s``; a partial row would break common random numbers.
``model.last`` keeps per-entry diagnostics of the latest call: ``lmp``
[n, K, n_bus], ``pg`` [n, K, n_gen], ``ok`` [n, K] and (pool models)
``status``.

Cost accounting: ``cost_per_eval_s`` = total wall time of all evaluate calls
/ evaluations done, for the hardware the model was built on (pool size for
AC/DC, one GPU for GENCO); before the first call it returns ``prior_s``.
"""

from __future__ import annotations

import time

import numpy as np

from genco_market.core.config import Config
from genco_market.core.schemas import MarketRules, Scenario
from genco_market.env.grid import GridInfo, scenario_to_network
from genco_market.env.market import DEFAULT_VOLL, effective_markup, firm_profits


def profit_from(lmp, pg, scn: Scenario, info: GridInfo, rules: MarketRules | None = None) -> float:
    """Strategic firm's profit ($/h, true cost) from a full LMP vector and dispatch."""
    return firm_profits(np.asarray(lmp), np.asarray(pg), scn, info, rules or MarketRules())[info.strategic]


def profit_matrix(lmp: np.ndarray, pg: np.ndarray, scenarios: list[Scenario], info: GridInfo,
                  rules: MarketRules | None = None) -> np.ndarray:
    """Vectorized profit_from over lmp [n, K, n_bus], pg [n, K, n_gen] -> [n, K]
    (strategic firm's gens only; identical to firm_profits)."""
    rules = rules or MarketRules()
    gens = info.firm_gens[info.strategic]
    bus = info.net.gens[gens, 0].astype(int)
    price = lmp[..., bus]
    if rules.price_cap is not None:
        price = np.minimum(price, rules.price_cap)
    c1 = np.array([np.asarray(s.gen_c1)[gens] for s in scenarios])[:, None, :]
    return ((price - c1) * pg[..., gens]).sum(-1)


def firm_profit_tensor(lmp: np.ndarray, pg: np.ndarray, scenarios: list[Scenario], info: GridInfo, firms: list[str],
                       rules: MarketRules | None = None) -> np.ndarray:
    """Profit of each firm in ``firms`` from lmp [n, P, n_bus], pg [n, P, n_gen] -> [n, P, F]."""
    rules = rules or MarketRules()
    price_all = np.minimum(lmp, rules.price_cap) if rules.price_cap is not None else lmp
    out = []
    for f in firms:
        gens = info.firm_gens[f]
        bus = info.net.gens[gens, 0].astype(int)
        c1 = np.array([np.asarray(s.gen_c1)[gens] for s in scenarios])[:, None, :]
        out.append(((price_all[..., bus] - c1) * pg[..., gens]).sum(-1))
    return np.stack(out, -1)


def _rowwise_nan(profit: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = profit.copy()
    bad = ~ok.all(1) | ~np.isfinite(profit).all(1)
    out[bad] = np.nan
    return out


class ProfitModel:
    name: str = "base"
    prior_s: float = 1.0

    def __init__(self, info: GridInfo | None = None, rules: MarketRules | None = None):
        self.info = info or GridInfo(Config.load())
        self.rules = rules or MarketRules()
        self._wall, self._n = 0.0, 0
        self.last: dict = {}

    def evaluate(self, scenarios: list[Scenario], markups: list[float], timeout_s: float | None = None) -> np.ndarray:
        raise NotImplementedError

    def cost_per_eval_s(self) -> float:
        return self._wall / self._n if self._n else self.prior_s

    def evaluate_profiles(self, scenarios: list[Scenario], profiles: list[dict[str, float]], firms: list[str],
                          timeout_s: float | None = None) -> np.ndarray:
        """Multi-firm bids: profits [n, P, F] of ``firms`` for each bid profile
        {firm: markup}. Rows with any failed profile are NaN (common random numbers)."""
        self.evaluate(scenarios, profiles, timeout_s)
        lmp, pg, ok = self.last["lmp"], self.last["pg"], self.last["ok"]
        out = firm_profit_tensor(np.nan_to_num(lmp), np.nan_to_num(pg), scenarios, self.info, firms, self.rules)
        out[~ok.all(1)] = np.nan
        return out

    def _account(self, t0: float, n_evals: int) -> None:
        self._wall += time.perf_counter() - t0
        self._n += n_evals

    def _finish(self, scenarios, lmp, pg, ok, status=None) -> np.ndarray:
        self.last = dict(lmp=lmp, pg=pg, ok=ok)
        if status is not None:
            self.last["status"] = status
        raw = profit_matrix(np.nan_to_num(lmp), np.nan_to_num(pg), scenarios, self.info, self.rules)
        raw[~ok] = np.nan
        self.last["profit_raw"] = raw
        return _rowwise_nan(raw, ok)


# --------------------------------------------------------------------------
# solver-pool models (AC truth, DC)
# --------------------------------------------------------------------------
class _PoolModel(ProfitModel):
    task = ""

    def __init__(self, pool, info: GridInfo | None = None, rules: MarketRules | None = None,
                 voll: float | None = DEFAULT_VOLL):
        if info is None and pool is not None:
            info = GridInfo(Config.model_validate(pool._cfg))
        super().__init__(info, rules)
        self.pool, self.voll = pool, voll

    def _results(self, scenarios, markups, timeout_s):
        """ClearingResult (or None) per (i, k); scenario-major submission so
        the earliest scenarios complete first under a timeout."""
        jobs = [self.pool.submit(self.task, s, m, self.rules, voll=self.voll) for s in scenarios for m in markups]
        got = self.pool.collect(jobs, timeout_s)
        if timeout_s is not None and len(got) < len(jobs):
            self.pool.cancel([j for j in jobs if j not in got])  # only this call's jobs
        return [got.get(j) for j in jobs]

    def evaluate(self, scenarios, markups, timeout_s=None):
        t0 = time.perf_counter()
        n, K, info = len(scenarios), len(markups), self.info
        res = self._results(scenarios, markups, timeout_s)
        lmp = np.full((n * K, info.n_bus), np.nan)
        pg = np.full((n * K, info.n_gen), np.nan)
        ok = np.zeros(n * K, dtype=bool)
        status = np.full(n * K, "NOT_DONE", dtype=object)
        for j, r in enumerate(res):
            if r is None:
                continue
            if not r:  # TaskError
                status[j] = "TASK_ERROR"
                continue
            status[j] = r.status
            if r.status != "FAILED" and np.all(np.isfinite(r.lmp)) and np.all(np.isfinite(r.pg)):
                lmp[j], pg[j], ok[j] = r.lmp, r.pg, True
        self._account(t0, int((status != "NOT_DONE").sum()))
        return self._finish(scenarios, lmp.reshape(n, K, -1), pg.reshape(n, K, -1), ok.reshape(n, K),
                            status.reshape(n, K))


class ACModel(_PoolModel):
    """Truth: AC-OPF clearing exactly as the market (env/market.clear, incl.
    its retry and DC fallback; status "DC_FALLBACK" entries are kept since the
    market settles on them)."""

    name = "AC"
    task = "ac"
    prior_s = 1 / 50.0  # 16 workers


class DCModel(_PoolModel):
    """DC-OPF clearing, LMP from LP duals. backend "highs" (scipy LP, runs
    in-process when ``pool`` is None) or "ipopt" (PowerModels DCP, pool only)."""

    prior_s = 1 / 2200.0

    def __init__(self, pool=None, backend: str = "highs", info=None, rules=None, voll=DEFAULT_VOLL):
        assert backend in ("highs", "ipopt")
        assert pool is not None or backend == "highs", "DC-Ipopt needs a SolverPool"
        super().__init__(pool, info, rules, voll)
        self.backend = backend
        self.task = "dc_highs" if backend == "highs" else "dc"
        self.name = f"DC-{'HiGHS' if backend == 'highs' else 'IPOPT'}" + ("" if pool is not None else "-1core")

    def _results(self, scenarios, markups, timeout_s):
        if self.pool is not None:
            return super()._results(scenarios, markups, timeout_s)
        from genco_market.env.market import clear_dc

        return [clear_dc(s, m, self.info, self.rules, backend="highs", voll=self.voll) for s in scenarios for m in markups]


# --------------------------------------------------------------------------
# GENCO
# --------------------------------------------------------------------------
LMP_METHODS = ("dc_hybrid", "autodiff", "head")


class GencoModel(ProfitModel):
    """GENCO dispatch + an LMP method (surrogate/lmp.py):

    - "dc_hybrid": DC-OPF (HiGHS) prices of the same scenario and bid; ``dc``
      is the DCModel used for them (default: in-process HiGHS). Its cost is
      included in cost_per_eval_s.
    - "autodiff": d(bid cost)/dPd through GENCO.
    - "head": LMPHead on frozen bus embeddings; ``head`` is an LMPHead or a
      path (default: models/lmp_head_{size}.pt under the scratch dir).

    All K markups of all scenarios go through GENCO in one batched call.
    ``timeout_s`` is ignored (a call takes well under a second for
    hundreds of scenarios).
    """

    prior_s = 1e-3

    def __init__(self, service, lmp_method: str = "head", info=None, rules=None, head=None, dc: DCModel | None = None,
                 batch_size: int = 1024):
        super().__init__(info, rules)
        assert lmp_method in LMP_METHODS, lmp_method
        self.svc, self.lmp_method, self.batch_size = service, lmp_method, batch_size
        size = service.ckpt_dir.parent.name
        self.name = f"GENCO-{size}+{lmp_method}"
        self.dc = None
        self.head = None
        if lmp_method == "dc_hybrid":
            self.dc = dc or DCModel(None, "highs", info=self.info, rules=self.rules)
        if lmp_method == "head":
            from genco_market.surrogate.lmp import LMPHead, head_dir

            if head is None or not hasattr(head, "predict_lmp"):
                head = LMPHead.load(head or head_dir() / f"lmp_head_{size}.pt", service.device)
            trained_for = getattr(head, "meta", {}).get("strategic")
            if trained_for is not None and trained_for != self.info.strategic and not head.meta.get("firms"):
                raise ValueError(f"LMP head was trained for firm {trained_for}, not {self.info.strategic}")
            self.head = head

    def _nets(self, scenarios, markups):
        ms = [effective_markup(m, self.rules) for m in markups]
        return [scenario_to_network(s, self.info.bid_c1(s, m), self.info.cfg.grid) for s in scenarios for m in ms]

    def evaluate(self, scenarios, markups, timeout_s=None):
        from genco_market.surrogate.lmp import autodiff_lmp, head_features, head_features_profile

        t0 = time.perf_counter()
        n, K = len(scenarios), len(markups)
        nets = self._nets(scenarios, markups)
        if self.lmp_method == "autodiff":
            out = autodiff_lmp(self.svc, nets, batch_size=min(self.batch_size, 256))
            lmp = out["lmp"]
        else:
            out = self.svc.predict_arrays(nets, self.batch_size, embeddings=self.lmp_method == "head")
            if self.lmp_method == "head":
                pd = np.array([s.bus_pd for s in scenarios for _ in markups])
                eff = [effective_markup(m, self.rules) for _ in scenarios for m in markups]
                hf = getattr(self.head, "meta", {}).get("firms")
                if hf:  # multi-firm head: per-firm markup features
                    prof = np.array([[(e.get(f, 1.0) if isinstance(e, dict) else (e if f == self.info.strategic else 1.0))
                                      for f in hf] for e in eff])
                    feat = head_features_profile(pd, prof, hf, self.info)
                else:
                    assert not isinstance(eff[0], dict), "single-firm LMP head cannot price bid profiles"
                    feat = head_features(pd, np.array(eff), self.info)
                lmp = self.head.predict_lmp(out["emb_bus"], feat)
            else:
                lmp = None
        pg = out["pg"].astype(float).reshape(n, K, -1)
        self._account(t0, n * K)
        if self.lmp_method == "dc_hybrid":
            self.dc.evaluate(scenarios, markups)  # accounted in self.dc
            lmp3, ok = self.dc.last["lmp"], self.dc.last["ok"].copy()
        else:
            lmp3 = lmp.reshape(n, K, -1)
            ok = np.ones((n, K), dtype=bool)
        ok &= np.isfinite(lmp3).all(-1) & np.isfinite(pg).all(-1)
        res = self._finish(scenarios, lmp3, pg, ok)
        if "res_p_mw" in out:  # label-free power-balance residual (not in the autodiff path)
            self.last["res_p_mw"] = out["res_p_mw"].reshape(n, K)
        return res

    def cost_per_eval_s(self) -> float:
        own = super().cost_per_eval_s()
        return own + (self.dc.cost_per_eval_s() if self.dc is not None else 0.0)
