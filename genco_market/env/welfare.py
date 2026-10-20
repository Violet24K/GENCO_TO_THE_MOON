"""Welfare accounting for one cleared market interval.

Demand is inelastic, so consumers' gross value is a constant and
    social cost   = true generation cost + VOLL * shed
    social welfare = -social cost (up to that constant).
A markup moves money from consumers to producers (a transfer) and creates a
deadweight loss only if it changes the dispatch (costlier units run, or load
is shed).
"""

from __future__ import annotations

import numpy as np

from genco_market.core.schemas import Scenario
from genco_market.env.grid import GridInfo
from genco_market.env.market import DEFAULT_VOLL

GEN_BUS = 0


def welfare_terms(lmp, pg, scn: Scenario, info: GridInfo, shed_mw=None, voll: float = DEFAULT_VOLL) -> dict:
    """lmp [n_bus] $/MWh, pg [n_gen] MW, shed_mw [n_bus] MW (or a total, or None = 0).
    Returns $/h: true_cost, shed_mw, shed_cost, social_cost, consumer_payment,
    producer_revenue, producer_profit (all firms), congestion_rent, and
    profit per firm."""
    lmp, pg = np.asarray(lmp, float), np.asarray(pg, float)
    c1 = np.asarray(scn.gen_c1, float)
    pd = np.asarray(scn.bus_pd, float)
    bus = info.net.gens[:, GEN_BUS].astype(int)
    if shed_mw is None:
        shed = np.zeros_like(pd)
    elif np.ndim(shed_mw) == 0:  # total only: attribute pro rata to load
        shed = pd / max(pd.sum(), 1e-9) * float(shed_mw)
    else:
        shed = np.asarray(shed_mw, float)
    served = pd - shed
    true_cost = float((c1 * pg).sum())
    revenue = float((lmp[bus] * pg).sum())
    payment = float((lmp * served).sum())
    shed_tot = float(shed.sum())
    out = dict(true_cost=true_cost, shed_mw=shed_tot, shed_cost=voll * shed_tot,
               social_cost=true_cost + voll * shed_tot, consumer_payment=payment,
               producer_revenue=revenue, producer_profit=revenue - true_cost, congestion_rent=payment - revenue)
    margin = (lmp[bus] - c1) * pg
    out["firm_profit"] = {f: float(margin[g].sum()) for f, g in info.firm_gens.items()}
    return out
