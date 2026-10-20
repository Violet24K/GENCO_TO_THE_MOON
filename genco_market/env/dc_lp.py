"""DC-OPF as an LP solved by HiGHS (scipy.optimize.linprog), no Julia.

Same model as PowerModels' DCPPowerModel OPF: lossless B-theta flows
p_ij = -b_ij (va_i - va_j) with b = imag(1/(r + jx)) (taps and shifts ignored),
|p_ij| <= rate_a, angle-difference limits, gs shunts at vm = 1, reference
angle fixed at 0. LMP = -(dual of the bus balance row), in $/MWh.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from gridfm_datakit.utils.idx_brch import ANGMAX, ANGMIN, BR_R, BR_X, F_BUS, T_BUS
from gridfm_datakit.utils.idx_bus import BUS_TYPE, GS
from gridfm_datakit.utils.idx_gen import GEN_BUS, PMIN

from genco_market.core.schemas import Scenario
from genco_market.env.grid import GridInfo

REF = 3


def solve_dc_lp(scn: Scenario, bid_c1: np.ndarray, info: GridInfo, voll: float | None = None) -> dict:
    """voll > 0 adds load shedding s_b in [0, Pd_b] at that price ($/MWh)."""
    net = info.net
    mva = float(net.baseMVA)
    nb = info.n_bus
    br = np.where(np.asarray(scn.branch_status) == 1)[0]
    gens = np.where((np.asarray(scn.gen_status) == 1))[0]
    ng, nl = len(gens), len(br)

    f = net.branches[br, F_BUS].astype(int)
    t = net.branches[br, T_BUS].astype(int)
    r, x = net.branches[br, BR_R], net.branches[br, BR_X]
    bsus = x / (r**2 + x**2)  # = -imag(1/(r+jx)); flow_pu = bsus * (va_f - va_t)
    A = sp.csr_matrix((np.r_[np.ones(nl), -np.ones(nl)], (np.r_[np.arange(nl), np.arange(nl)], np.r_[f, t])), shape=(nl, nb))
    Bf = sp.diags(bsus * mva) @ A  # MW per rad
    Bbus = A.T @ Bf
    Cg = sp.csr_matrix((np.ones(ng), (net.gens[gens, GEN_BUS].astype(int), np.arange(ng))), shape=(nb, ng))

    pd = np.asarray(scn.bus_pd) + net.buses[:, GS]
    loads = np.nonzero(np.asarray(scn.bus_pd) > 0)[0] if voll else np.zeros(0, dtype=int)
    ns = len(loads)
    Cs = sp.csr_matrix((np.ones(ns), (loads, np.arange(ns))), shape=(nb, ns))
    # x = [pg (MW), va (rad), shed (MW)]
    c = np.r_[np.asarray(bid_c1)[gens], np.zeros(nb), np.full(ns, voll or 0.0)]
    A_eq = sp.hstack([-Cg, Bbus, -Cs]).tocsr()
    b_eq = -pd
    rate = np.asarray(scn.branch_rate_a)[br]
    lim = rate > 0
    angmin, angmax = np.deg2rad(net.branches[br, ANGMIN]), np.deg2rad(net.branches[br, ANGMAX])
    Zl, Zs = sp.csr_matrix((nl, ng)), sp.csr_matrix((nl, ns))
    A_ub = sp.vstack(
        [sp.hstack([Zl, Bf, Zs])[lim], sp.hstack([Zl, -Bf, Zs])[lim], sp.hstack([Zl, A, Zs]), sp.hstack([Zl, -A, Zs])]
    ).tocsr()
    b_ub = np.r_[rate[lim], rate[lim], angmax, -angmin]
    ref = int(np.where(net.buses[:, BUS_TYPE] == REF)[0][0])
    bounds = [(net.gens[g, PMIN], scn.gen_pmax[g]) for g in gens] + [(None, None)] * nb
    bounds += [(0.0, scn.bus_pd[b]) for b in loads]
    bounds[ng + ref] = (0.0, 0.0)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    status = "OPTIMAL" if res.status == 0 else f"HIGHS_{res.status}"
    pg = np.zeros(info.n_gen)
    flow = np.full(info.n_branch, np.nan)
    lmp = np.full(nb, np.nan)
    shed = np.zeros(nb)
    if res.status == 0:
        pg[gens] = res.x[:ng]
        flow[br] = np.abs(Bf @ res.x[ng : ng + nb])
        shed[loads] = res.x[ng + nb :]
        lmp = -res.eqlin.marginals
    binding = [int(b) for b in br[lim & (np.nan_to_num(flow[br]) >= 0.99 * rate)]] if res.status == 0 else []
    return dict(
        status=status,
        objective=float(res.fun) if res.status == 0 else float("nan"),
        lmp=lmp,
        pg=pg,
        flow=flow,
        binding=binding,
        shed=shed,
    )
