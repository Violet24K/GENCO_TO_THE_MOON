"""Pool task for the three-firm payoff tensors (experiments/l3_abd.py): AC clearing like
env.market.clear, but keeping branch flows (MVA) and shed MW. Falls back to the DC LP like the
market does."""

from __future__ import annotations

import numpy as np

from genco_market.core.schemas import MarketRules, Scenario
from genco_market.env.grid import GridInfo


def ac_detail(scn: Scenario, markup, *, info: GridInfo, voll: float | None = 2000.0) -> dict:
    from genco_market.env.dc_lp import solve_dc_lp
    from genco_market.env.market import OK_STATUS, _solve, firm_profits

    bid = info.bid_c1(scn, markup)
    raw = None
    for attempt in (1, 2):
        try:
            r = _solve(scn, bid, info, "ac", attempt, voll)
            if r["status"] in OK_STATUS and np.all(np.isfinite(r["lmp"])):
                raw = r
                break
        except Exception:
            pass
    if raw is None:
        raw = solve_dc_lp(scn, bid, info, voll)
        raw["status"] = "DC_FALLBACK"
    raw["profit"] = firm_profits(raw["lmp"], raw["pg"], scn, info, MarketRules())
    return raw
