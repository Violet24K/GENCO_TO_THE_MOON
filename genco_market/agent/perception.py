"""Step 1 of a round: bulletin text -> ScenarioDistribution.

The LLM fills ``BeliefExtraction`` (list-shaped, since structured outputs do
not allow free-form dict keys); ``to_distribution`` maps it onto 0-based grid
indices. On timeout/failure the caller falls back to the prior.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from genco_market.core.schemas import Bulletin, CommonShock, ScenarioDistribution
from genco_market.env.grid import GridInfo

ZoneId = Literal["A", "B", "C", "D", "E", "ALL"]


class ZoneLoad(BaseModel):
    zone: ZoneId
    mult_low: float = Field(description="low end of load multiplier vs normal, e.g. 1.15 for +15%")
    mult_high: float
    confidence: float = Field(description="0..1, how reliable the sources are")


class BranchOutage(BaseModel):
    branch: str = Field(description="PGLib line name 'from-to' exactly as written, including a '#k' circuit "
                                    "suffix when present, e.g. '38-37' or '49-54#2'")
    prob: float = Field(description="probability the line is out during the interval, 1.0 if planned")
    confidence: float


class ZoneLineRisk(BaseModel):
    zone: ZoneId
    event_probability: float = Field(description="chance the storm/hazard actually hits the zone in the target interval")
    severity: Literal["low", "moderate", "high"]
    confidence: float


MAX_BUS_DELTA_MW = 300.0  # sanity bound on a single bus load change (largest case118 bus load is ~280 MW)

# Agent's prior on hazard impact: expected number of line trips in the zone if
# the hazard hits (spread uniformly over lines touching the zone).
EXPECTED_TRIPS_IF_HIT = {"low": 0.2, "moderate": 0.4, "high": 0.7}


class LineDerate(BaseModel):
    zone: ZoneId
    rate_mult: float = Field(description="thermal rating multiplier, <1 when hot, >1 when cool and windy")


class RivalCost(BaseModel):
    firm: ZoneId = Field(description="rival firm/zone, or ALL for all rivals")
    cost_mult: float = Field(description="multiplier on rivals' marginal cost, e.g. 1.3 for gas price +30%")
    confidence: float


class BusLoad(BaseModel):
    bus: int = Field(description="PGLib bus number")
    extra_mw: float = Field(description="additional steady load at that bus in MW (e.g. a new data center); negative "
                                        "for less load. Convert household equivalents to MW: an average home draws "
                                        "roughly 1-1.5 kW, so 10,000 homes is about 10-15 MW")


class UnitOutageRisk(BaseModel):
    zone: ZoneId
    prob_per_unit: float = Field(description="probability that each generating unit in the zone trips during the interval")
    confidence: float


class GasSupplyRisk(BaseModel):
    probability: float = Field(description="chance that gas deliveries to power plants are curtailed in the interval")
    share_of_gas_units_tripping: float = Field(description="if curtailment happens, the share (0..1) of gas-fired "
                                                           "units that go offline together")
    confidence: float


class TransformerRisk(BaseModel):
    probability: float = Field(description="chance that a geomagnetic storm (or similar hazard) trips large transformers")
    share_of_transformers_tripping: float = Field(description="if it happens, the share (0..1) of large transformers "
                                                              "that trip together")
    confidence: float


class SubstationRisk(BaseModel):
    bus: int = Field(description="PGLib bus number of the threatened substation")
    probability: float = Field(description="chance the substation is knocked out (all its lines trip together)")
    confidence: float


class RenewableChange(BaseModel):
    zone: ZoneId
    mw_low: float = Field(description="low end of the change in renewable output vs forecast, MW; positive = more "
                                      "wind/solar than forecast (less net load), negative = less (more net load)")
    mw_high: float
    confidence: float


class BeliefExtraction(BaseModel):
    zone_load: list[ZoneLoad]
    system_load_uncertainty: float = Field(
        description="1-sigma relative error of the system-wide load forecast shared by all zones, e.g. 0.03 for "
                    "+-3%; 0 if nothing suggests unusual forecast uncertainty")
    bus_loads: list[BusLoad]
    unit_outage_risk: list[UnitOutageRisk]
    branch_outages: list[BranchOutage]
    zone_line_risk: list[ZoneLineRisk]
    line_derates: list[LineDerate]
    rival_costs: list[RivalCost]
    gas_supply_risk: list[GasSupplyRisk]
    transformer_risk: list[TransformerRisk]
    substation_risk: list[SubstationRisk]
    renewable_changes: list[RenewableChange]
    markup_cap: float | None = Field(description="if a market rule resets offers above some multiple of cost to cost, "
                                                 "that multiple (e.g. 2.0); null if no such rule is announced")
    ignored_item_ids: list[str] = Field(description="items judged irrelevant or unreliable")
    notes: str = Field(description="one or two sentences on the key drivers")


SYSTEM = """You are the market analyst of power producer {firm} (zone {firm}) in a nodal electricity market.
Read the bulletin items and estimate conditions for the target interval ONLY, relative to a normal day.
Rules:
- Use only information that applies to the target interval (check each item's validity window).
- Later items that supersede earlier ones override them.
- Weigh sources: ISO notices are reliable; weather forecasts are uncertain; social media and rumors are often false.
- Omit anything not mentioned (empty lists); never invent events. Zones are A, B, C, D, E.
- Translate indirect information into load effects when it is clearly implied (e.g. big events, holidays,
  new large consumers); new large consumers at a specific bus go into bus_loads.
- Load multipliers: 1.0 = normal. Planned outages (and announced shutoffs of named lines) have prob 1.0.
- Correlated hazards go into their own fields: gas supply curtailment (gas_supply_risk), geomagnetic storms
  (transformer_risk), threats to a specific substation (substation_risk). Wind/solar output different from
  forecast goes into renewable_changes, not zone_load."""


def render_bulletin(bulletin: Bulletin, target_t: int) -> str:
    lines = [f"Target interval: t={target_t}", "Bulletin items:"]
    for it in bulletin.items:
        sup = f" (supersedes {it.supersedes})" if it.supersedes else ""
        lines.append(
            f"- [{it.item_id}] source={it.source} published t={it.published_t} "
            f"applies to t={it.refers_to_t}{sup}: {it.text}"
        )
    return "\n".join(lines)


def to_distribution(ext: BeliefExtraction, info: GridInfo, t: int, prior: ScenarioDistribution | None = None) -> ScenarioDistribution:
    d = (prior.model_copy(deep=True) if prior else ScenarioDistribution(t=t))
    d.t = t
    zones = list(info.zones)

    def expand(z: str) -> list[str]:
        return zones if z == "ALL" else [z]

    for zl in ext.zone_load:
        lo, hi = sorted((zl.mult_low, zl.mult_high))
        for z in expand(zl.zone):
            d.zone_load_mean[z] = (lo + hi) / 2
            d.zone_load_std[z] = max(0.02, (hi - lo) / 4)
            d.confidence[f"load:{z}"] = zl.confidence
    d.system_load_std = max(0.0, min(0.15, ext.system_load_uncertainty))
    for ur in ext.unit_outage_risk:
        for z in expand(ur.zone):
            for g in info.firm_gens[z]:
                d.gen_outage_prob[g] = min(1.0, max(d.gen_outage_prob.get(g, 0.0), ur.prob_per_unit))
            d.confidence[f"unit_risk:{z}"] = ur.confidence
    bus_of = {int(b): i for i, b in enumerate(info.net.original_bus_indices)}
    for bl in ext.bus_loads:
        if bl.bus in bus_of:  # extra load = negative renewable injection
            i = bus_of[bl.bus]
            mw = max(-MAX_BUS_DELTA_MW, min(MAX_BUS_DELTA_MW, bl.extra_mw))
            if mw != bl.extra_mw:
                d.notes += f" [clipped bus {bl.bus} extra load {bl.extra_mw:g} MW to {mw:g}]"
            d.renewable_mw[i] = d.renewable_mw.get(i, 0.0) - mw
    for zr in ext.zone_line_risk:
        for z in expand(zr.zone):
            lines = info.zone_branches[z]
            p = min(1.0, max(0.0, zr.event_probability)) * EXPECTED_TRIPS_IF_HIT[zr.severity] / len(lines)
            for b in lines:
                d.branch_outage_prob[b] = 1 - (1 - d.branch_outage_prob.get(b, 0.0)) * (1 - p)
            d.confidence[f"line_risk:{z}"] = zr.confidence
    for bo in ext.branch_outages:
        for b in info.branch_by_name.get(bo.branch.replace(" ", ""), []):
            d.branch_outage_prob[b] = min(1.0, max(0.0, bo.prob))
            d.confidence[f"outage:{bo.branch}"] = bo.confidence
    for ld in ext.line_derates:
        for z in expand(ld.zone):
            for b in info.zone_branches[z]:
                d.branch_rate_mult[b] = ld.rate_mult
    for rc in ext.rival_costs:
        for z in expand(rc.firm):
            if z != info.strategic:
                d.rival_cost_mult[z] = rc.cost_mult
                d.confidence[f"rival:{z}"] = rc.confidence
    if ext.gas_supply_risk:
        from genco_market.semantics.events import fuel_of_units, load_effects

        gas = [g for g, f in fuel_of_units(info, load_effects()).items() if f == "gas"]
        for k, gr in enumerate(ext.gas_supply_risk):
            q = min(1.0, max(0.0, gr.share_of_gas_units_tripping))
            d.common_shocks.append(CommonShock(label=f"belief:gas{k}", prob=min(1.0, max(0.0, gr.probability)),
                                               gen_prob={g: q for g in gas}))
            d.confidence[f"gas_supply:{k}"] = gr.confidence
    if ext.transformer_risk:
        from gridfm_datakit.utils.idx_brch import TAP

        tx = [b for b in range(info.n_branch) if info.net.branches[b, TAP] != 0]
        for k, tr in enumerate(ext.transformer_risk):
            q = min(1.0, max(0.0, tr.share_of_transformers_tripping))
            d.common_shocks.append(CommonShock(label=f"belief:gic{k}", prob=min(1.0, max(0.0, tr.probability)),
                                               branch_prob={b: q for b in tx}))
            d.confidence[f"transformer:{k}"] = tr.confidence
    br = info.net.branches
    for k, sr in enumerate(ext.substation_risk):
        if sr.bus in bus_of:
            i = bus_of[sr.bus]
            lines = [b for b in range(info.n_branch) if int(br[b, 0]) == i or int(br[b, 1]) == i]
            d.common_shocks.append(CommonShock(label=f"belief:sub{sr.bus}", prob=min(1.0, max(0.0, sr.probability)),
                                               branch_prob={b: 0.9 for b in lines[:-1]}))  # the bus keeps one line
            d.confidence[f"substation:{sr.bus}"] = sr.confidence
    for rc in ext.renewable_changes:
        lo, hi = sorted((rc.mw_low, rc.mw_high))
        for z in expand(rc.zone):
            bs = [b for b in info.zones[z]["buses"] if info.base_pd[b] > 0]
            tot = float(sum(info.base_pd[b] for b in bs))
            mid, sd = (lo + hi) / 2 / (len(zones) if rc.zone == "ALL" else 1), (hi - lo) / 4
            for b in bs:
                w = float(info.base_pd[b]) / tot
                d.renewable_mw[b] = d.renewable_mw.get(b, 0.0) + mid * w
                d.renewable_std_mw[b] = float(np.hypot(d.renewable_std_mw.get(b, 0.0), sd * w))
            d.confidence[f"renewable:{z}"] = rc.confidence
    if ext.markup_cap is not None and ext.markup_cap >= 1.0:
        d.market_rules = d.market_rules.model_copy(update={"mitigation_markup_cap": float(ext.markup_cap)})
    d.notes = ext.notes + d.notes
    return d


def perceive(llm, bulletin: Bulletin, info: GridInfo, t: int, timeout_s: float | None = None,
             cache_mode: str | None = None) -> tuple[ScenarioDistribution, BeliefExtraction]:
    ext = llm.parse(SYSTEM.format(firm=info.strategic), render_bulletin(bulletin, t), BeliefExtraction, purpose="perceive",
                    timeout_s=timeout_s, cache_mode=cache_mode)
    return to_distribution(ext, info, t), ext
