"""Fixed-flow bidding agent.

perceive (LLM / prior / L0 / oracle belief) -> sample scenarios from the belief
-> anytime profit estimation under a compute budget -> argmax -> bid.
The bid value always comes from the estimator, never from the LLM.
"""

from __future__ import annotations

import json
import time

import numpy as np

from genco_market.agent.perception import perceive
from genco_market.core.schemas import BidAction, Observation, PhysicalState, ScenarioDistribution
from genco_market.core.seeding import rng
from genco_market.core.timing import Deadline
from genco_market.env.grid import GridInfo
from genco_market.env.physical import ood_flags, sample_scenario
from genco_market.estimators.multifidelity import multifidelity, multifidelity_multi, single_fidelity, split_budget

MARGIN_S = 0.5  # reserved for argmax + submit
ROUTE_FILE = "routing_table.json"  # configs/: belief signature -> low-fidelity model (from Exp 2)


def belief_signature(b: PhysicalState) -> str:
    """Coarse situation label the agent can infer from its own belief:
    outage > heat > storm > gas > none (first match wins)."""
    if any(p >= 0.9 for p in b.branch_outage_prob.values()):
        return "outage"
    if (b.system_load_std > 0 or b.gen_outage_prob or any(m < 1 for m in b.branch_rate_mult.values())
            or any(v >= 1.03 for v in b.zone_load_mean.values())):
        return "heat"
    if b.branch_outage_prob:
        return "storm"
    if any(v > 1 for v in b.rival_cost_mult.values()):
        return "gas"
    return "none"


def load_routing_table() -> dict:
    from genco_market.core.config import CONFIGS

    p = CONFIGS / ROUTE_FILE
    return json.loads(p.read_text()) if p.exists() else {}
MAX_SCENARIOS = 4000
LO_CHUNK = 256
PRIOR_LOAD_STD = 0.06


def prior_belief(info: GridInfo, t: int) -> ScenarioDistribution:
    """Generic prior: normal conditions with wider load uncertainty, no events."""
    z = list(info.zones)
    return ScenarioDistribution(t=t, zone_load_mean={k: 1.0 for k in z}, zone_load_std={k: PRIOR_LOAD_STD for k in z},
                                notes="prior")


def l0_belief(obs: Observation) -> ScenarioDistribution:
    items = [it for it in obs.bulletin.items if obs.t in it.refers_to_t]
    return ScenarioDistribution.model_validate(json.loads(items[-1].text))


class _LazyScenarios:
    """List-like scenario stream: draws are made only when a slice is first
    accessed (sampling costs ~0.2 ms each, so drawing MAX_SCENARIOS upfront
    would eat ~1 s of the compute budget). Draw order is fixed by the RNG."""

    def __init__(self, draw, r, n: int):
        self._draw, self._r, self._n, self._buf = draw, r, n, []

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, sl: slice):
        stop = min(self._n, sl.stop if sl.stop is not None else self._n)
        while len(self._buf) < stop:
            self._buf.append(self._draw(self._r))
        return self._buf[sl]


class PipelineAgent:
    """belief: "llm" | "prior" | "l0" | "oracle";  method: "cost" | "fixed:<m>" | "ac" | "genco" | "dc" |
    "mf_genco" | "mf_dc" | "mf_both" (both surrogates as joint control variates) | "mf_route" (picks the
    low-fidelity method from configs/routing_table.json by belief_signature; falls back to DC when the belief
    is outside GENCO's training distribution, else GENCO).  ``models`` maps "ac"/"genco"/"dc" to
    ProfitModel objects."""

    def __init__(self, info: GridInfo, markups: list[float], belief: str, method: str, models: dict | None = None,
                 llm=None, seed: int = 0, assumed_rho: float = 0.7, oracle_fn=None, llm_timeout_s: float = 8.0,
                 shared_beliefs: dict | None = None):
        self.info, self.markups, self.belief, self.method = info, list(markups), belief, method
        self.models, self.llm, self.seed = models or {}, llm, seed
        self.assumed_rho, self.oracle_fn, self.llm_timeout_s = assumed_rho, oracle_fn, llm_timeout_s
        self.episode = 0
        self.last_bid: float | None = None
        # (episode, t) -> (belief, extraction, latency): agents reading the same bulletin
        # share one LLM call (fair comparison, lower API cost); latency is still charged
        self.shared_beliefs = shared_beliefs

    @property
    def name(self) -> str:
        return f"{self.belief}/{self.method}"

    def reset(self, episode: int) -> None:
        self.episode, self.last_bid = episode, None

    # --------------------------------------------------------------- belief
    def _belief(self, obs: Observation, dl: Deadline, trace: dict) -> ScenarioDistribution:
        t0 = time.perf_counter()
        try:
            if self.belief == "prior":
                return prior_belief(self.info, obs.t)
            if self.belief == "l0":
                return l0_belief(obs)
            if self.belief == "oracle":
                return ScenarioDistribution.model_validate(self.oracle_fn(obs.t).model_dump())
            if self.belief == "llm":
                key = (self.episode, obs.t)
                if self.shared_beliefs is not None and key in self.shared_beliefs:
                    d, ext, lat = self.shared_beliefs[key]
                    trace["extraction"] = ext.model_dump()
                    trace["belief_shared"], trace["belief_charge_s"] = True, lat
                    return d.model_copy(deep=True)
                budget = dl.remaining() - obs.compute_budget_s - MARGIN_S
                d, ext = perceive(self.llm, obs.bulletin, self.info, obs.t,
                                  timeout_s=max(1.0, min(self.llm_timeout_s, budget)), cache_mode="off")
                trace["extraction"] = ext.model_dump()
                if self.shared_beliefs is not None:
                    self.shared_beliefs[key] = (d.model_copy(deep=True), ext, time.perf_counter() - t0)
                return d
            raise ValueError(self.belief)
        except Exception as e:
            trace["belief_fallback"] = f"{type(e).__name__}: {e}"[:200]
            return prior_belief(self.info, obs.t)
        finally:
            trace["t_belief_s"] = time.perf_counter() - t0

    # ------------------------------------------------------------ estimation
    def _scenarios(self, belief: PhysicalState, obs: Observation, n: int):
        return _LazyScenarios(lambda r: sample_scenario(belief, self.info, obs.baseline_zone_load_mw, r),
                              rng(self.seed, "agent", self.episode, obs.t), n)

    def _lo_anytime(self, model, scns, budget_s: float) -> np.ndarray:
        """Evaluate chunks until the next one would not finish inside the budget (the first always runs)."""
        t0, rows, chunk_s = time.perf_counter(), [], 0.0
        c = LO_CHUNK if budget_s >= 2 else 32  # a 256-scenario chunk alone takes ~0.7 s with GENCO
        for i in range(0, len(scns), c):
            if rows and time.perf_counter() - t0 + chunk_s > budget_s:
                break
            t1 = time.perf_counter()
            rows.append(model.evaluate(scns[i:i + c], self.markups))
            chunk_s = max(chunk_s, time.perf_counter() - t1)
        return np.vstack(rows)

    def _estimate(self, belief, obs: Observation, budget_s: float, trace: dict):
        m = self.method
        for mdl in self.models.values():  # price each offer under the market rules the agent believes apply
            if hasattr(mdl, "rules"):
                mdl.rules = belief.market_rules
        scns = self._scenarios(belief, obs, MAX_SCENARIOS)
        if m in ("genco", "dc"):
            P = self._lo_anytime(self.models[m], scns, budget_s)
            P = P[~np.isnan(P).any(1)]
            return single_fidelity(P)
        if m == "ac":
            ac = self.models["ac"]
            cap = int(2 * budget_s / (len(self.markups) * ac.cost_per_eval_s())) + 8  # ~2x what fits
            P = ac.evaluate(scns[:min(cap, len(scns))], self.markups, timeout_s=budget_s)
            P = P[~np.isnan(P).any(1)]
            return single_fidelity(P) if len(P) >= 2 else None
        if m == "mf_route":
            sig = belief_signature(belief)
            table = load_routing_table()
            m = table.get(sig) or ("mf_dc" if trace.get("belief_ood") else "mf_genco")
            trace.update(routed_to=m, signature=sig)
        if m == "mf_both":
            dc, gc, hi = self.models["dc"], self.models["genco"], self.models["ac"]
            n, N = split_budget(budget_s, hi.cost_per_eval_s(), dc.cost_per_eval_s() + gc.cost_per_eval_s(),
                                self.assumed_rho, len(self.markups))
            t0 = time.perf_counter()
            Ld = self._lo_anytime(dc, scns[:N], budget_s / 2)
            Lg = gc.evaluate(scns[:len(Ld)], self.markups)
            H = hi.evaluate(scns[:len(Ld)], self.markups, timeout_s=max(0.1, budget_s - (time.perf_counter() - t0)))
            done = ~np.isnan(H).any(1)
            n_hi = len(done) if done.all() else int(np.argmin(done))
            ok = ~(np.isnan(Ld).any(1) | np.isnan(Lg).any(1))
            hi_rows = [i for i in range(n_hi) if ok[i]]
            lo_rows = hi_rows + [i for i in range(n_hi, len(Ld)) if ok[i]]
            trace.update(planned_n=n, planned_N=N)
            if len(hi_rows) < 4:
                return single_fidelity(Ld[ok])
            return multifidelity_multi(H[hi_rows], [Ld[lo_rows], Lg[lo_rows]])
        if m.startswith("mf_"):
            lo, hi = self.models[m[3:]], self.models["ac"]
            n, N = split_budget(budget_s, hi.cost_per_eval_s(), lo.cost_per_eval_s(), self.assumed_rho, len(self.markups))
            t0 = time.perf_counter()
            L = self._lo_anytime(lo, scns[:N], budget_s)
            H = hi.evaluate(scns[:len(L)], self.markups, timeout_s=max(0.1, budget_s - (time.perf_counter() - t0)))
            # use the leading block of finished AC rows (finish order may correlate with solve
            # difficulty, so arbitrary finished rows could bias the estimate)
            done = ~np.isnan(H).any(1)
            n_hi = len(done) if done.all() else int(np.argmin(done))
            ok_lo = ~np.isnan(L).any(1)
            trace.update(planned_n=n, planned_N=N)
            hi_rows = [i for i in range(n_hi) if ok_lo[i]]
            if len(hi_rows) < 2:
                return single_fidelity(L[ok_lo])
            lo_rest = [i for i in range(n_hi, len(L)) if ok_lo[i]]
            return multifidelity(H[hi_rows], L[hi_rows + lo_rest])
        raise ValueError(m)

    # ------------------------------------------------------------------ act
    def act(self, obs: Observation) -> tuple[BidAction, dict]:
        dl = Deadline(obs.deadline_s)
        trace: dict = {"agent": self.name, "t": obs.t}
        if self.method == "cost" or self.method.startswith("fixed:"):
            m = 1.0 if self.method == "cost" else float(self.method.split(":")[1])
            return BidAction(markup=m, method=self.method), trace
        belief = self._belief(obs, dl, trace)
        for g, c in obs.own_marginal_cost.items():  # the firm knows its own costs
            belief.gen_cost_mult[g] = c / float(self.info.base_c1[g])
        trace["belief_ood"] = ood_flags(belief, self.info, obs.baseline_zone_load_mw)
        budget = max(0.0, min(obs.compute_budget_s, dl.remaining() - MARGIN_S))
        t0 = time.perf_counter()
        est = None
        try:
            est = self._estimate(belief, obs, budget, trace)
        except Exception as e:
            trace["estimate_error"] = f"{type(e).__name__}: {e}"[:200]
        trace["t_compute_s"] = time.perf_counter() - t0
        if est is None or not np.isfinite(est.mean).any():
            m = self.last_bid if self.last_bid is not None else 1.0
            trace["t_total_s"] = dl.elapsed()
            return BidAction(markup=m, method=self.method, fallback_used=True), trace
        k = est.best()
        trace.update(est_mean=est.mean.tolist(), est_se=est.se.tolist(), n_hi=est.n_hi, n_lo=est.n_lo,
                     rho=None if est.rho is None else est.rho.tolist(), t_total_s=dl.elapsed(),
                     deadline_missed=dl.expired())
        self.last_bid = self.markups[k]
        return BidAction(markup=self.markups[k], method=self.method), trace
