"""PhysicalState -> Scenario sampling, ERCOT baseline, OOD flags.

GENCO training envelope (official dataset gridfm/opf_small_case118_ieee,
args.log + scenarios_agg_load_profile.log): datakit ``agg_load_profile`` with
the "default" profile, global_range 0.4, sigma 0.2, change_reactive_power.
``find_largest_scaling_factor`` stopped at u = 1.30 (1.40 was infeasible), so
l = 0.6 u = 0.78. Every bus in training is ``p_i * ref_k * U(0.8, 1.2)`` with
the common factor ref_k in [0.78, 1.30] (5/50/95% = 0.83/0.94/1.12), i.e. a
system load of 3,309-5,515 MW around the 4,242 MW base, and per-bus ratios in
[0.624, 1.56]. Topology: up to k = 10 random branch/gen outages. Generator
costs were *permuted* (cost_permutation), never scaled; ratings and pmax
were never changed.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from gridfm_datakit.utils.idx_brch import F_BUS, T_BUS
from gridfm_datakit.utils.idx_gen import GEN_STATUS

from genco_market.core.schemas import PhysicalState, Scenario
from genco_market.env.grid import GridInfo

TRAIN_SCALE = (0.78, 1.30)  # global load factor range seen in training
# "Normal" hours are mapped into a narrower band so that weather events
# (up to about +18% load) stay inside TRAIN_SCALE.
BASELINE_SCALE = (0.78, 1.10)
TRAIN_BUS_NOISE = 0.2  # uniform per-bus noise half-width in training
BUS_RATIO_RANGE = (TRAIN_SCALE[0] * (1 - TRAIN_BUS_NOISE), TRAIN_SCALE[1] * (1 + TRAIN_BUS_NOISE))
MAX_TRAIN_OUTAGES = 10
MAX_OUTAGE_REDRAWS = 100


ERCOT_HOUR_OFFSET = 11  # row i of the datakit ERCOT files is local hour i - 11 (see _profile)


def _profile(name: str) -> np.ndarray:
    """ERCOT zone load by local hour of 2024 (index 0 = Jan 1 00:00-01:00).

    The datakit files are offset: their daily minimum sits at row-hour ~15 and
    the maximum at ~3-4 all summer, and the winter morning ramp at ~17. Rolling
    by ERCOT_HOUR_OFFSET puts the summer peak at 17:00 and the trough at 04:00,
    as in ERCOT. One 24-row block (around the Nov 3 DST change) holds doubled
    values; it is halved."""
    path = resources.files("gridfm_datakit.load_profiles").joinpath(f"{name}.csv")
    p = pd.read_csv(path, header=None).to_numpy(dtype=float).ravel().copy()
    prev = np.roll(p, 24)
    bad = (p > 1.6 * prev) & (np.arange(len(p)) >= 24)
    p[bad] /= 2.0
    return np.roll(p, -ERCOT_HOUR_OFFSET)


@lru_cache(maxsize=1)
def _train_curve() -> np.ndarray:
    agg = pd.read_csv(resources.files("gridfm_datakit.load_profiles").joinpath("default.csv")).to_numpy().ravel()
    lo, hi = BASELINE_SCALE
    return lo + (agg - agg.min()) * (hi - lo) / (agg.max() - agg.min())


@lru_cache(maxsize=16)
def _zone_factor_curve(profile: str) -> np.ndarray:
    p = _profile(f"ercot_load_act_hr_2024_{profile}")
    q = (np.argsort(np.argsort(p, kind="stable"), kind="stable") + 0.5) / len(p)
    return np.quantile(_train_curve(), q)


def baseline_zone_load(info: GridInfo, hour_of_year: int) -> dict[str, float]:
    """Public "normal" load per zone (MW) for an hour of 2024.

    Mapping: each zone follows its own ERCOT 2024 zone profile (``ercot_profile``
    in the firms yaml), quantile-mapped onto the distribution of the training
    load factor: factor_z(h) = Q_train(F_z(P_z(h))), where F_z is the empirical
    CDF of the zone's ERCOT load over the year and Q_train the quantile function
    of the reference curve (datakit default profile min-max scaled to
    BASELINE_SCALE). The ERCOT timing (daily/seasonal shape, cross-zone
    correlation) is kept exactly; the marginal distribution of each zone factor
    follows the training curve rescaled to BASELINE_SCALE (peak hour 1.10), so a "normal" hour
    is in-distribution and only events push zones out. MW = factor * zone base
    load (sum of PGLib Pd in the zone). The ERCOT files have 8,268 hours; the
    hour index wraps around.
    """
    out = {}
    for z, v in info.zones.items():
        curve = _zone_factor_curve(v["ercot_profile"])
        out[z] = float(curve[hour_of_year % len(curve)] * info.base_pd[v["buses"]].sum())
    return out


def _zone_base(info: GridInfo) -> dict[str, float]:
    return {z: float(info.base_pd[v["buses"]].sum()) for z, v in info.zones.items()}


def _lognormal(rng: np.random.Generator, mean: float, std: float) -> float:
    if std <= 0:
        return mean
    s2 = np.log1p((std / mean) ** 2)
    return float(rng.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2)))


def is_connected(info: GridInfo, branch_status) -> bool:
    on = np.asarray(branch_status) == 1
    f = info.net.branches[on, F_BUS].astype(int)
    t = info.net.branches[on, T_BUS].astype(int)
    adj = sp.csr_matrix((np.ones(len(f)), (f, t)), shape=(info.n_bus, info.n_bus))
    return connected_components(adj, directed=False, return_labels=False) == 1


def sample_scenario(
    state: PhysicalState,
    info: GridInfo,
    baseline: dict[str, float],
    rng: np.random.Generator,
    log: dict | None = None,
) -> Scenario:
    """One realization of P_t. ``baseline`` = zone MW from baseline_zone_load.

    Load: bus Pd = base Pd * (baseline_z / base_z) * C * M_z * (1 + eps_b) with
    C ~ lognormal(1, system_load_std) shared by all zones, M_z ~ lognormal(mean, std) and eps_b ~ N(0, bus_load_noise_std) clipped
    at +-3 std; Qd scales with the same per-bus ratio (constant power factor), times
    zone_qd_mult[z] when given.
    Costs: c1 = base c1 * rival_cost_mult[firm] (never the strategic firm)
    * gen_cost_mult[gen]. Renewables are subtracted from Pd and clipped at 0. Outages are
    independent Bernoulli draws; draws that island the grid are redrawn.
    ``log`` (optional) receives diagnostics: zone multipliers, clipped
    renewable MW, number of redraws.
    """
    zbase = _zone_base(info)
    ratio = np.ones(info.n_bus)
    zone_mult = {}
    # common factor first, only when present (keeps RNG streams of older states unchanged)
    common = _lognormal(rng, 1.0, state.system_load_std) if state.system_load_std > 0 else 1.0
    for z, v in info.zones.items():
        m = common * _lognormal(rng, state.zone_load_mean.get(z, 1.0), state.zone_load_std.get(z, 0.0))
        zone_mult[z] = m
        ratio[v["buses"]] = baseline.get(z, zbase[z]) / zbase[z] * m
    sd = state.bus_load_noise_std
    if sd > 0:
        ratio *= 1.0 + np.clip(rng.normal(0.0, sd, info.n_bus), -3 * sd, 3 * sd)
    pd = info.base_pd * ratio
    qd = info.base_qd * ratio
    for z, m in state.zone_qd_mult.items():  # e.g. heatwave cooling load at a lower power factor
        qd[info.zones[z]["buses"]] *= m
    clipped = 0.0
    ren = dict(state.renewable_mw)
    if state.renewable_std_mw:  # drawn only when present (keeps RNG streams of older states unchanged)
        for b, sd in state.renewable_std_mw.items():
            mw = ren.get(b, 0.0)
            # generation (mw > 0) stays >= 0; extra load (mw < 0) stays <= 0
            ren[b] = mw + sd * float(rng.standard_normal())
            ren[b] = max(ren[b], 0.0) if mw >= 0 else min(ren[b], 0.0)
    for b, mw in ren.items():
        net_load = pd[b] - mw
        if net_load < 0:
            clipped += -net_load
        pd[b] = max(net_load, 0.0)

    base_on = info.net.gens[:, GEN_STATUS].astype(int)
    br_ids = np.array(list(state.branch_outage_prob.keys()), dtype=int)
    br_p = np.array(list(state.branch_outage_prob.values()), dtype=float)
    shocks = state.common_shocks
    shock_hit = [False] * len(shocks)
    redraws = 0
    while True:
        branch_status = np.ones(info.n_branch, dtype=int)
        if len(br_ids):
            branch_status[br_ids[rng.random(len(br_ids)) < br_p]] = 0
        for j, sh in enumerate(shocks):  # correlated outages (only drawn when present)
            shock_hit[j] = bool(rng.random() < sh.prob)
            if shock_hit[j] and sh.branch_prob:
                ids = np.array(list(sh.branch_prob), dtype=int)
                ps = np.array(list(sh.branch_prob.values()), dtype=float)
                branch_status[ids[rng.random(len(ids)) < ps]] = 0
        if is_connected(info, branch_status):
            break
        redraws += 1
        if redraws >= MAX_OUTAGE_REDRAWS:  # only certain (p=1) outages that keep the grid connected
            branch_status = np.ones(info.n_branch, dtype=int)
            for b in br_ids[br_p >= 1.0]:
                branch_status[b] = 0
                if not is_connected(info, branch_status):
                    branch_status[b] = 1
            break
    gen_status = base_on.copy()
    for g, p in state.gen_outage_prob.items():
        if rng.random() < p:
            gen_status[g] = 0
    for j, sh in enumerate(shocks):
        if shock_hit[j]:
            for g, p in sh.gen_prob.items():
                if rng.random() < p:
                    gen_status[g] = 0

    rate = info.base_rate_a.copy()
    for b, m in state.branch_rate_mult.items():
        rate[b] *= m
    pmax = info.base_pmax.copy()
    for g, m in state.gen_pmax_mult.items():
        pmax[g] *= m
    c1 = info.base_c1.copy()
    for f, m in state.rival_cost_mult.items():
        if f != info.strategic:
            c1[info.firm_gens[f]] *= m
    for g, m in state.gen_cost_mult.items():
        c1[g] *= m

    if log is not None:
        log.update(zone_mult=zone_mult, common_mult=common, renewable_clipped_mw=float(clipped), outage_redraws=redraws,
                   shocks_hit=[sh.label for sh, h in zip(shocks, shock_hit) if h])
    return Scenario(
        t=state.t,
        seed=int(rng.integers(2**31)),
        bus_pd=pd.tolist(),
        bus_qd=qd.tolist(),
        branch_status=branch_status.tolist(),
        branch_rate_a=rate.tolist(),
        gen_status=gen_status.tolist(),
        gen_pmax=pmax.tolist(),
        gen_c1=c1.tolist(),
    )


def ood_flags(x: PhysicalState | Scenario, info: GridInfo, baseline: dict[str, float] | None = None) -> list[str]:
    """Which parts of a state/scenario fall outside GENCO's training data. The bid markup itself is the caller's to flag.

    For a PhysicalState the zone factor is (baseline_z / base_z) * mean_z
    (baseline defaults to the PGLib base) and is compared with [0.78, 1.30];
    for a Scenario the system and every zone factor are compared with
    [0.78, 1.30] and each bus ratio with [0.624, 1.56]. (In training a zone's
    factor stays within ~0.03 of the common factor, since per-bus noise
    averages out, so zone factors outside the range never occurred.)
    """
    lo, hi = TRAIN_SCALE
    flags: list[str] = []
    if isinstance(x, Scenario):
        pd = np.asarray(x.bus_pd)
        sys_f = pd.sum() / info.base_pd.sum()
        loaded = info.base_pd > 0
        r = pd[loaded] / info.base_pd[loaded]
        zone_f = [pd[v["buses"]].sum() / info.base_pd[v["buses"]].sum() for v in info.zones.values()]
        if (
            not lo <= sys_f <= hi
            or not lo <= min(zone_f) <= max(zone_f) <= hi
            or r.min() < BUS_RATIO_RANGE[0]
            or r.max() > BUS_RATIO_RANGE[1]
        ):
            flags.append("load_outside_envelope")
        if not np.allclose(x.branch_rate_a, info.base_rate_a):
            flags.append("rate_a_changed")
        on = np.asarray(x.gen_status) == 1
        if not np.allclose(np.sort(np.asarray(x.gen_c1)), np.sort(info.base_c1)):
            flags.append("rival_cost_scaled")
        if not np.allclose(x.gen_pmax, info.base_pmax):
            flags.append("pmax_changed")
        base_on = info.net.gens[:, GEN_STATUS] == 1
        n_out = int((np.asarray(x.branch_status) == 0).sum() + (base_on & ~on).sum())
        if n_out > MAX_TRAIN_OUTAGES:
            flags.append("outages_gt_10")
        return flags

    zbase = _zone_base(info)
    baseline = baseline or zbase
    factor = {z: baseline.get(z, zbase[z]) / zbase[z] * x.zone_load_mean.get(z, 1.0) for z in info.zones}
    for z, f in factor.items():
        if not lo <= f <= hi:
            flags.append(f"load_outside_envelope:{z}")
    if any(m != 1.0 for m in x.branch_rate_mult.values()):
        flags.append("rate_a_changed")
    if any(m != 1.0 for f, m in x.rival_cost_mult.items() if f != info.strategic) or any(
        m != 1.0 for m in x.gen_cost_mult.values()
    ):
        flags.append("rival_cost_scaled")
    if any(m != 1.0 for m in x.gen_pmax_mult.values()):
        flags.append("pmax_changed")
    expected_out = sum(x.branch_outage_prob.values()) + sum(x.gen_outage_prob.values())
    if expected_out > MAX_TRAIN_OUTAGES:
        flags.append("outages_gt_10")
    if any(mw > info.base_pd[b] * factor[info.bus_zone[b]] for b, mw in x.renewable_mw.items()):
        flags.append("negative_net_load")
    return flags
