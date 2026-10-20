"""Market clearing: AC-OPF (Ipopt via PowerModels) with LMPs from KCL duals.

Sign/units (verified by finite differences, tests/test_env_market.py):
PowerModels reports ``lam_kcl_r`` in $/h per p.u. of injection, and a load
increase raises the objective, so LMP [$/MWh] = -lam_kcl_r / baseMVA.

Every public function returns a ClearingResult and never raises; the status
tells what happened: Ipopt's status on success, "DC_FALLBACK" when the AC
solve failed twice and the DC-OPF cleared the market, "FAILED" otherwise.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from gridfm_datakit.utils.idx_brch import RATE_A
from gridfm_datakit.utils.idx_gen import GEN_BUS, PMAX

from genco_market.core.schemas import ClearingResult, MarketRules, Scenario
from genco_market.env.grid import GridInfo, scenario_to_network

AC_MAX_ITER = 300  # retry uses 2x; converged case118 solves need ~20-40
MAX_WALL_S = 8.0  # per Ipopt attempt; infeasible cases otherwise crawl for 30 s+
OK_STATUS = ("LOCALLY_SOLVED", "ALMOST_LOCALLY_SOLVED", "OPTIMAL")
BINDING_FRAC = 0.99
DEFAULT_VOLL: float | None = 2000.0  # $/MWh load shedding (scarcity pricing); None = hard OPF
_JULIA_FILE = Path(__file__).parent / "julia" / "market.jl"
_JL = None


def get_jl():
    """Julia session for this process (datakit init + market.jl), ~30 s once."""
    global _JL
    if _JL is None:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        from gridfm_datakit.process.process_network import init_julia

        jl = init_julia(max_iter=AC_MAX_ITER)
        jl.seval("using LinearAlgebra; BLAS.set_num_threads(1)")
        jl.include(str(_JULIA_FILE))
        _JL = jl
    return _JL


def effective_markup(markup: float | dict[str, float], rules: MarketRules) -> float | dict[str, float]:
    """Markup after the simple cap rule (per firm for a bid profile)."""
    cap = rules.mitigation_markup_cap
    if isinstance(markup, dict):
        return {f: (1.0 if cap is not None and m > cap else m) for f, m in markup.items()}
    return 1.0 if cap is not None and markup > cap else markup


def _solve(scn: Scenario, bid_c1: np.ndarray, info: GridInfo, kind: str, attempt: int, voll: float | None = None) -> dict:
    """One raw solve; returns arrays in MW and $/MWh (shed: MW per bus)."""
    from gridfm_datakit.process.solvers import _julia_pm_data

    jl = get_jl()
    net = scenario_to_network(scn, bid_c1, info.cfg.grid)
    rate = net.branches[:, RATE_A].copy()
    # keep base ratings in the cached parse (no reparse); push the real ones below
    net.branches[:, RATE_A] = info.base_rate_a
    data = _julia_pm_data(net, jl)
    mva = float(net.baseMVA)
    jl.gc_set_limits_b(data, rate / mva, net.gens[:, PMAX] / mva)
    bus_ids = net._solver_cache["bus_ids"]
    out = jl.gc_clear(data, kind, attempt, AC_MAX_ITER, MAX_WALL_S, float(voll or 0.0), bus_ids, info.n_gen, info.n_branch)
    status, obj = str(out[0]), float(out[1])
    lam, pg, pf, qf, pt, qt, shed = (np.asarray(x, dtype=float) for x in out[2:])
    if kind == "ac":
        flow = np.maximum(np.hypot(pf, qf), np.hypot(pt, qt)) * mva
    else:
        flow = np.abs(pf) * mva
    on = np.asarray(scn.branch_status) == 1
    binding = np.where(on & (rate > 0) & (np.nan_to_num(flow) >= BINDING_FRAC * rate))[0]
    return dict(
        status=status,
        objective=obj,
        lmp=-lam / mva,
        pg=np.nan_to_num(pg * mva, nan=0.0),
        flow=flow,
        binding=binding.tolist(),
        shed=shed * mva,
    )


def firm_profits(lmp: np.ndarray, pg: np.ndarray, scn: Scenario, info: GridInfo, rules: MarketRules) -> dict[str, float]:
    """Profit per firm at TRUE cost, LMP clipped at the price cap."""
    price = np.asarray(lmp, dtype=float)
    if rules.price_cap is not None:
        price = np.minimum(price, rules.price_cap)
    gen_bus = info.net.gens[:, GEN_BUS].astype(int)
    margin = (price[gen_bus] - np.asarray(scn.gen_c1)) * np.asarray(pg)
    return {f: float(margin[g].sum()) for f, g in info.firm_gens.items()}


def _result(raw: dict, status: str, scn, info, rules, t0, dc: bool, log: dict | None) -> ClearingResult:
    shed = np.asarray(raw.get("shed", np.zeros(info.n_bus)))
    if log is not None:
        log.update(load_shed_mw=float(shed.sum()), shed_buses=np.nonzero(shed > 1e-3)[0].tolist())
    return ClearingResult(
        status=status,
        lmp=raw["lmp"].tolist(),
        pg=raw["pg"].tolist(),
        objective=raw["objective"],
        solve_s=time.perf_counter() - t0,
        binding_branches=raw["binding"],
        used_dc_fallback=dc,
        firm_profit=firm_profits(raw["lmp"], raw["pg"], scn, info, rules),
        load_shed_mw=float(shed.sum()),
    )


def _failed(info: GridInfo, t0: float, status: str = "FAILED") -> ClearingResult:
    return ClearingResult(
        status=status,
        lmp=[float("nan")] * info.n_bus,
        pg=[float("nan")] * info.n_gen,
        objective=float("nan"),
        solve_s=time.perf_counter() - t0,
        firm_profit={f: float("nan") for f in info.firm_gens},
    )


def _ok(raw: dict) -> bool:
    return raw["status"] in OK_STATUS and np.all(np.isfinite(raw["lmp"]))


def clear(
    scn: Scenario,
    markup: float,
    info: GridInfo,
    rules: MarketRules | None = None,
    voll: float | None = DEFAULT_VOLL,
    log: dict | None = None,
) -> ClearingResult:
    """AC-OPF market clearing with the strategic firm bidding c1 * markup.
    Failure chain: retry (warm start, other Ipopt options) -> DC-OPF (HiGHS LP)
    -> FAILED. ``voll`` ($/MWh) allows load shedding at that price, so LMPs
    spike to at most VOLL instead of the OPF going infeasible. ``log`` (optional)
    receives load_shed_mw and shed_buses."""
    rules = rules or MarketRules()
    t0 = time.perf_counter()
    bid = info.bid_c1(scn, effective_markup(markup, rules))
    for attempt in (1, 2):
        try:
            raw = _solve(scn, bid, info, "ac", attempt, voll)
            if _ok(raw):
                res = _result(raw, raw["status"], scn, info, rules, t0, False, log)
                res.effective_markup = effective_markup(markup, rules)
                return res
        except Exception:
            pass
    try:
        from genco_market.env.dc_lp import solve_dc_lp

        raw = solve_dc_lp(scn, bid, info, voll)  # same model as Ipopt DCP, exact and fast
        if _ok(raw):
            res = _result(raw, "DC_FALLBACK", scn, info, rules, t0, True, log)
            res.effective_markup = effective_markup(markup, rules)
            return res
    except Exception:
        pass
    return _failed(info, t0)


def clear_dc(
    scn: Scenario,
    markup: float,
    info: GridInfo,
    rules: MarketRules | None = None,
    backend: str = "ipopt",
    voll: float | None = DEFAULT_VOLL,
    log: dict | None = None,
) -> ClearingResult:
    """DC-OPF clearing (lossless, LMP from duals). backend: "ipopt"
    (PowerModels DCP, as in the GENCO paper) or "highs" (scipy LP, no Julia)."""
    rules = rules or MarketRules()
    t0 = time.perf_counter()
    bid = info.bid_c1(scn, effective_markup(markup, rules))
    try:
        if backend == "highs":
            from genco_market.env.dc_lp import solve_dc_lp

            raw = solve_dc_lp(scn, bid, info, voll)
        else:
            raw = _solve(scn, bid, info, "dc", 1, voll)
        if _ok(raw):
            return _result(raw, raw["status"], scn, info, rules, t0, False, log)
    except Exception:
        pass
    return _failed(info, t0)
