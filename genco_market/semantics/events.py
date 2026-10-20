"""Event taxonomy and the effects map that turns events into
PhysicalState channel deltas.

Only MVP types are implemented; the rest are declared so the taxonomy is
explicit and can be filled in later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import networkx as nx
import yaml
from gridfm_datakit.utils.idx_brch import F_BUS, T_BUS

from genco_market.core.config import CONFIGS
from genco_market.core.schemas import CommonShock, EventRecord, PhysicalState
from genco_market.env.grid import GridInfo

CHANNELS = ("load", "load_std", "branch_outage", "branch_rate", "gen_outage", "gen_pmax", "renewable", "rival_cost", "market_rules",
            "common_shock")


@dataclass(frozen=True)
class EventType:
    name: str
    category: str
    channels: tuple[str, ...]
    description: str
    implemented: bool = False
    primary_source: str = "news"


_TYPES = [
    # weather
    EventType("heatwave", "weather", ("load", "load_std", "system_load_std", "branch_rate", "branch_outage", "gen_pmax", "gen_outage"),
              "Heat raises cooling load, lowers line/transformer ratings, derates and trips thermal units, "
              "trips heavily loaded lines, and makes the system-wide load forecast less certain.", True, "weather"),
    EventType("cold_snap", "weather", ("load", "load_std", "system_load_std", "gen_outage", "rival_cost", "common_shock"),
              "Heating load up; units trip on freezing; a gas-supply curtailment can trip many gas units at once; "
              "gas prices spike (cf. Uri 2021).", True, "weather"),
    EventType("thunderstorm", "weather", ("branch_outage", "load", "load_std"),
              "Lightning and gusts raise outage probability of overhead lines in a zone.", True, "weather"),
    EventType("high_wind", "weather", ("branch_outage", "renewable"),
              "Wind output above forecast in a wind zone (less net load, uncertain); strong gusts threaten lines.", True, "weather"),
    EventType("wildfire_psps", "weather", ("branch_outage",),
              "Wildfire risk triggers deterministic preventive de-energization of named lines.", True, "iso"),
    EventType("cool_windy", "weather", ("branch_rate",),
              "Cool, windy conditions raise dynamic line ratings (good news)."),
    EventType("solar_drop", "weather", ("renewable",),
              "Dust, a cloud deck or an eclipse cuts solar output below forecast, raising net load (uncertain).", True, "weather"),
    EventType("drought", "weather", ("gen_pmax",), "Slow-moving derate of thermal and hydro units."),
    EventType("flood", "weather", ("branch_outage",), "Substation flooding: several correlated branch outages (N-k)."),
    EventType("geomagnetic_storm", "weather", ("common_shock",),
              "Space weather: correlated transformer (and generator step-up) trips; low probability, high impact.", True, "weather"),
    # load and social
    EventType("holiday", "load", ("load",), "Holiday, school closure or weekend: commercial load down.", True, "news"),
    EventType("major_event", "load", ("load", "load_std"), "Sports final or concert: short local load spike.", True, "news"),
    EventType("industrial_shutdown", "load", ("load",), "Plant maintenance or strike: local load drop."),
    EventType("data_center", "load", ("renewable",),
              "Data center energized: steady extra MW at one bus (negative renewable_mw).", True, "iso"),
    EventType("demand_response", "load", ("load", "load_std"),
              "Conservation appeal: issued because demand is near peak; the appeal trims load by an uncertain amount, "
              "but net load is still above normal.", True, "iso"),
    EventType("ev_charging", "load", ("load",), "Clustered EV charging raises evening load."),
    # grid and operations
    EventType("planned_outage", "grid", ("branch_outage",),
              "Scheduled branch/transformer maintenance with a time window; may be postponed or cancelled (L3).", True, "iso"),
    EventType("forced_outage", "grid", ("branch_outage", "gen_outage"), "Forced outage report, known only after the fact."),
    EventType("new_equipment", "grid", ("branch_outage", "gen_pmax"), "New line or unit enters service."),
    EventType("emergency_alert", "grid", ("load_std", "market_rules"), "Operating reserve shortage / emergency: price upside risk."),
    # fuel and economics
    EventType("gas_price", "fuel", ("rival_cost",),
              "Gas hub price spike or pipeline problem raises rival gas units' marginal cost.", True, "news"),
    EventType("carbon_price", "fuel", ("rival_cost",), "Carbon price change shifts rival costs by unit type."),
    # competition and regulation
    EventType("rival_maintenance", "competition", ("gen_outage", "gen_pmax"), "Rival unit maintenance lowers available rival capacity."),
    EventType("rival_bidding", "competition", (), "Rumors about rival bidding strategy; usually noise."),
    EventType("market_power_review", "competition", ("market_rules",),
              "Market monitor tightens mitigation: offers above a markup cap are reset to cost.", True, "iso"),
    # security
    EventType("cyber_physical", "security", ("common_shock",),
              "Credible threat against a substation: if it happens, all lines at one bus trip together.", True, "iso"),
]
EVENT_TYPES: dict[str, EventType] = {e.name: e for e in _TYPES}
IMPLEMENTED = [e.name for e in _TYPES if e.implemented]


def load_effects(name: str = "effects.yaml") -> dict:
    return _load_effects(str(CONFIGS / name))


@lru_cache(maxsize=4)
def _load_effects(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# static grid facts derived for the semantics layer
# --------------------------------------------------------------------------
def fuel_of_units(info: GridInfo, eff: dict) -> dict[int, str]:
    """Firm unit -> "coal" | "gas" | "oil" by the c1 rule in effects.yaml."""
    r = eff["fuel_rule"]
    out = {}
    for g in sorted(info.gen_firm):
        c1 = float(info.base_c1[g])
        out[g] = "oil" if c1 >= r["oil_c1"] else "gas" if c1 >= r["gas_c1"] else "coal"
    return out


def bridge_branches(info: GridInfo) -> set[int]:
    """Branches whose loss islands part of the grid (never outaged by events)."""
    G = nx.MultiGraph()
    G.add_nodes_from(range(info.n_bus))
    br = info.net.branches
    for i in range(info.n_branch):
        G.add_edge(int(br[i, F_BUS]), int(br[i, T_BUS]), key=i)
    out = set()
    for i in range(info.n_branch):
        f, t = int(br[i, F_BUS]), int(br[i, T_BUS])
        if G.number_of_edges(f, t) == 1:
            G.remove_edge(f, t, key=i)
            if not nx.has_path(G, f, t):
                out.add(i)
            G.add_edge(f, t, key=i)
    return out


def keeps_connected(info: GridInfo, removed: set[int]) -> bool:
    br = info.net.branches
    G = nx.MultiGraph()
    G.add_nodes_from(range(info.n_bus))
    for i in range(info.n_branch):
        if i not in removed:
            G.add_edge(int(br[i, F_BUS]), int(br[i, T_BUS]))
    return nx.is_connected(G)


def heat_shape(hour_of_day: int) -> float:
    return 0.6 + 0.4 * math.cos(2 * math.pi * (hour_of_day - 16) / 24)


# --------------------------------------------------------------------------
# effects: events active at t -> PhysicalState
# --------------------------------------------------------------------------
def neutral_state(info: GridInfo, eff: dict, t: int) -> PhysicalState:
    std = eff["neutral"]["load_std"]
    return PhysicalState(
        t=t,
        zone_load_mean={z: 1.0 for z in info.zones},
        zone_load_std={z: std for z in info.zones},
        rival_cost_mult={z: 1.0 for z in info.zones if z != info.strategic},
    )


def is_active(ev: EventRecord, t: int) -> bool:
    return ev.occurs and ev.start_t <= t <= ev.end_t


def apply_events(info: GridInfo, eff: dict, events: list[EventRecord], t: int) -> PhysicalState:
    """Combine all events active at t. Load deltas add; rate/pmax/cost
    multipliers multiply; outage probabilities combine as independent."""
    b = eff["bounds"]
    s = neutral_state(info, eff, t)
    load = dict(s.zone_load_mean)
    std = dict(s.zone_load_std)
    rate: dict[int, float] = {}
    pmax: dict[int, float] = {}
    pout: dict[int, float] = {}
    cost = dict(s.rival_cost_mult)
    gcost: dict[int, float] = {}
    ren: dict[int, float] = {}
    ren_sd: dict[int, float] = {}
    gout: dict[int, float] = {}
    shocks: list[CommonShock] = []
    caps: list[float] = []
    heat_extra: dict[str, float] = {}  # zone -> cooling load added by heatwaves, fraction of normal
    sys_std = 0.0

    for ev in events:
        if not is_active(ev, t):
            continue
        p = ev.params
        if ev.event_type == "heatwave":
            c = eff["heatwave"]
            for z in p["zones"]:
                a = p["anomaly_c"][z][t]
                load[z] += c["load_mult_per_c"] * a
                std[z] += c["load_std_per_c"] * a
                heat_extra[z] = heat_extra.get(z, 0.0) + c["load_mult_per_c"] * a
            for br, a in _per_item_anomaly(p, "branches", t):
                rate[br] = rate.get(br, 1.0) * (1 + c["rate_mult_per_c"] * a)
            for g, a in _per_item_anomaly(p, "gens", t):
                pmax[g] = pmax.get(g, 1.0) * (1 + c["pmax_mult_per_c"] * a)
                q = p["unit_trip_per_c"][str(g)] * a
                gout[g] = 1 - (1 - gout.get(g, 0.0)) * (1 - q)
            a_max = max(p["anomaly_c"][z][t] for z in p["zones"])
            for br in p.get("heat_lines", []):
                pout[br] = 1 - (1 - pout.get(br, 0.0)) * (1 - c["line_trip"]["per_c"] * a_max)
            sys_std = math.hypot(sys_std, heat_system_std(c, a_max, p["lead"], p["load_share"]))
        elif ev.event_type == "planned_outage":
            pout[p["branch"]] = 1.0
        elif ev.event_type == "thunderstorm":
            c = eff["thunderstorm"]["severity"][int(ev.severity)]
            for br in p["lines"]:
                pout[br] = 1 - (1 - pout.get(br, 0.0)) * (1 - c["outage_prob"])
            load[ev.zone] += c["load_delta"]
            std[ev.zone] += eff["thunderstorm"]["load_std_add"] * ev.severity
        elif ev.event_type == "gas_price":
            # per-unit truth (gas units only); firm-level averages stay in params for reference
            for g, m in p["unit_cost_mult"].items():
                gcost[int(g)] = gcost.get(int(g), 1.0) * m
        elif ev.event_type == "major_event":
            load[ev.zone] += p["load_delta"]
            std[ev.zone] += eff["major_event"]["load_std_add"]
        elif ev.event_type == "holiday":
            for z in p["zones"]:
                load[z] += p["load_delta"]
        elif ev.event_type == "data_center":
            ren[p["bus"]] = ren.get(p["bus"], 0.0) - p["mw"]
        elif ev.event_type == "cold_snap":
            c = eff["cold_snap"]
            sv = c["severity"][int(ev.severity)]
            a = p["anomaly_c"]
            for z in p["zones"]:
                load[z] += c["load_mult_per_c"] * a
                std[z] += c["load_std_per_c"] * a
                for g in info.firm_gens[z]:
                    if info.base_pmax[g] > 0:
                        q = c["unit_trip_per_c"] * a
                        gout[g] = 1 - (1 - gout.get(g, 0.0)) * (1 - q)
            sys_std = math.hypot(sys_std, c["system_load_std_per_c"] * a * p["load_share"])
            for g in p["gas_units"]:
                gcost[g] = gcost.get(g, 1.0) * (1 + c["pass_through"] * (sv["gas_mult"] - 1))
            shocks.append(CommonShock(label=f"{ev.event_id}:gas_curtailment", prob=sv["gas_shock_prob"],
                                      gen_prob={g: sv["gas_trip_given_shock"] for g in p["gas_units"]}))
        elif ev.event_type == "wildfire_psps":
            for br in p["lines"]:
                pout[br] = 1.0
        elif ev.event_type in ("high_wind", "solar_drop"):
            sign = 1.0 if ev.event_type == "high_wind" else -1.0  # wind: more generation; solar loss: more net load
            for bus, share in p["bus_share"].items():
                ren[int(bus)] = ren.get(int(bus), 0.0) + sign * p["mw"] * share
                ren_sd[int(bus)] = math.hypot(ren_sd.get(int(bus), 0.0), p["mw_std"] * share)
            for br in p.get("gust_lines", []):
                pout[br] = 1 - (1 - pout.get(br, 0.0)) * (1 - p["line_trip_prob"])
        elif ev.event_type == "demand_response":
            for z in load:
                load[z] += p["tight_delta"] + p["response_delta"]
                std[z] += eff["demand_response"]["load_std_add"]
        elif ev.event_type == "market_power_review":
            caps.append(p["cap"])
        elif ev.event_type == "geomagnetic_storm":
            sv = eff["geomagnetic_storm"]["severity"][int(ev.severity)]
            shocks.append(CommonShock(label=f"{ev.event_id}:gic", prob=sv["shock_prob"],
                                      branch_prob={b: sv["tx_trip_given_shock"] for b in p["transformers"]},
                                      gen_prob={g: sv["gen_trip_given_shock"] for g in p["units"]}))
        elif ev.event_type == "cyber_physical":
            shocks.append(CommonShock(label=f"{ev.event_id}:attack", prob=p["attack_prob"],
                                      branch_prob={b: eff["cyber_physical"]["line_trip_given_attack"] for b in p["lines"]}))
        else:
            raise NotImplementedError(f"effects for event type {ev.event_type!r}")

    clip = lambda x, lohi: round(min(max(x, lohi[0]), lohi[1]), 4)  # noqa: E731
    s.zone_load_mean = {z: clip(v, b["load_mult"]) for z, v in load.items()}
    s.zone_load_std = {z: clip(v, b["load_std"]) for z, v in std.items()}
    s.branch_rate_mult = {k: clip(v, b["rate_mult"]) for k, v in sorted(rate.items()) if round(v, 4) != 1.0}
    s.gen_pmax_mult = {k: clip(v, b["pmax_mult"]) for k, v in sorted(pmax.items()) if round(v, 4) != 1.0}
    s.branch_outage_prob = {k: round(v, 4) for k, v in sorted(pout.items()) if v > 0}
    s.rival_cost_mult = {z: clip(v, b["cost_mult"]) for z, v in cost.items()}
    s.gen_outage_prob = {k: clip(v, b["gen_outage_prob"]) for k, v in sorted(gout.items()) if v > 0}
    s.system_load_std = clip(sys_std, b["system_load_std"])
    s.renewable_mw = {k: round(v, 3) for k, v in sorted(ren.items())}
    s.gen_cost_mult = {k: clip(v, b["cost_mult"]) for k, v in sorted(gcost.items()) if round(v, 4) != 1.0}
    s.renewable_std_mw = {k: round(v, 3) for k, v in sorted(ren_sd.items()) if v > 0}
    s.common_shocks = shocks
    pf = eff["heatwave"].get("cooling_pf")
    if pf and heat_extra:  # extra cooling load draws reactive power at power factor pf
        s.zone_qd_mult = {z: float(round(_qd_mult(info, z, x, pf), 4)) for z, x in sorted(heat_extra.items())}
    if caps:
        s.market_rules = s.market_rules.model_copy(update={"mitigation_markup_cap": min(caps)})
    assert len(s.branch_outage_prob) <= b["max_outage_branches"], s.branch_outage_prob
    return s


def heat_system_std(c: dict, anomaly: float, lead: int, share: float) -> float:
    lf = c["system_load_lead_factor"]
    return c["system_load_std_per_c"] * anomaly * lf[min(lead, len(lf) - 1)] * share


def _per_item_anomaly(p: dict, key: str, t: int):
    """Heatwave params map each zone to its branches/gens; a branch on the
    boundary of two hot zones takes the larger anomaly."""
    best: dict[int, float] = {}
    for z in p["zones"]:
        a = p["anomaly_c"][z][t]
        for i in p[key][z]:
            best[i] = max(best.get(i, 0.0), a)
    return sorted(best.items())


def _qd_mult(info: GridInfo, z: str, extra: float, pf: float) -> float:
    """Reactive-load multiplier of zone z when a fraction ``extra`` of its
    normal real load is added at power factor ``pf`` (the rest keeps the
    zone's base power factor): (1 + extra * tan_pf / tan_base) / (1 + extra)."""
    b = info.zones[z]["buses"]
    tan_base = info.base_qd[b].sum() / info.base_pd[b].sum()
    tan_pf = math.tan(math.acos(pf))
    return (1 + extra * tan_pf / tan_base) / (1 + extra)
