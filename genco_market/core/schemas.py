"""Data contracts shared by all modules.

Index convention: buses, gens and branches are 0-based row indices of the
datakit ``Network`` arrays (``net.buses``, ``net.gens``, ``net.branches``).
PGLib bus numbers appear only in configs (``*_orig``) and in rendered text.
Power is in MW, prices in $/MWh, costs in $/MWh (linear term c1).
Zones are firm names "A".."E" (configs/firms_case118.yaml).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# A <- B: the hidden truth for one market interval
# --------------------------------------------------------------------------
class MarketRules(BaseModel):
    price_cap: float | None = None  # $/MWh; LMPs are clipped at settlement
    mitigation_markup_cap: float | None = None  # bids above this markup are reset to cost


class CommonShock(BaseModel):
    """A correlated outage hazard (gas curtailment, geomagnetic storm, attack):
    first draw whether the shock happens (prob), then, only if it does, each
    listed device fails independently with its conditional probability."""

    label: str = ""
    prob: float  # chance the shock happens in the interval
    branch_prob: dict[int, float] = Field(default_factory=dict)  # branch -> P(out | shock)
    gen_prob: dict[int, float] = Field(default_factory=dict)  # gen -> P(trip | shock)


class PhysicalState(BaseModel):
    """True scenario distribution P_t. Multipliers are relative to the
    "normal" conditions of interval t (env supplies the normal baseline)."""

    t: int
    zone_load_mean: dict[str, float] = Field(default_factory=dict)  # zone -> load multiplier
    zone_load_std: dict[str, float] = Field(default_factory=dict)  # zone -> std of multiplier
    system_load_std: float = 0.0  # std of a multiplier shared by all zones (e.g. temperature forecast error)
    bus_load_noise_std: float = 0.03  # independent per-bus multiplicative noise
    renewable_mw: dict[int, float] = Field(default_factory=dict)  # bus -> MW, applied as negative load
    renewable_std_mw: dict[int, float] = Field(default_factory=dict)  # bus -> std of the renewable MW (forecast error)
    branch_outage_prob: dict[int, float] = Field(default_factory=dict)  # branch -> p (1.0 = planned outage)
    branch_rate_mult: dict[int, float] = Field(default_factory=dict)  # branch -> rate_a multiplier
    gen_outage_prob: dict[int, float] = Field(default_factory=dict)  # gen -> p
    gen_pmax_mult: dict[int, float] = Field(default_factory=dict)  # gen -> pmax multiplier
    rival_cost_mult: dict[str, float] = Field(default_factory=dict)  # zone/firm -> c1 multiplier
    gen_cost_mult: dict[int, float] = Field(default_factory=dict)  # gen -> c1 multiplier (applied on top of rival_cost_mult)
    zone_qd_mult: dict[str, float] = Field(default_factory=dict)  # zone -> reactive-load multiplier on top of the real-load ratio (lower power factor)
    market_rules: MarketRules = Field(default_factory=MarketRules)
    common_shocks: list[CommonShock] = Field(default_factory=list)  # correlated outages, on top of the independent ones


class ScenarioDistribution(PhysicalState):
    """The agent's belief about P_t. Same fields as PhysicalState (missing
    entries mean "use the prior"), plus per-field confidence in [0, 1]."""

    confidence: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class Scenario(BaseModel):
    """One realization drawn from a PhysicalState (full arrays, before bids)."""

    t: int
    seed: int
    bus_pd: list[float]
    bus_qd: list[float]
    branch_status: list[int]
    branch_rate_a: list[float]  # MVA
    gen_status: list[int]
    gen_pmax: list[float]  # MW
    gen_c1: list[float]  # true marginal cost (rival multipliers applied), $/MWh


# --------------------------------------------------------------------------
# B -> C: public text
# --------------------------------------------------------------------------
Source = Literal["iso", "weather", "news", "social", "rival"]


class BulletinItem(BaseModel):
    item_id: str
    source: Source
    published_t: int
    refers_to_t: list[int]
    text: str
    supersedes: str | None = None


class Bulletin(BaseModel):
    t: int
    items: list[BulletinItem]


# B -> D: ground truth for each bulletin item
class EventRecord(BaseModel):
    event_id: str
    event_type: str
    zone: str | None = None
    start_t: int
    end_t: int  # inclusive
    severity: float = 1.0
    occurs: bool = True  # False for rumors / cancelled events
    params: dict = Field(default_factory=dict)


class ItemLabel(BaseModel):
    item_id: str
    event_id: str | None  # None for pure distractors
    is_true: bool
    is_distractor: bool = False
    superseded_by: str | None = None


class BulletinLabels(BaseModel):
    t: int
    items: list[ItemLabel]
    active_events: list[EventRecord]
    current_item_ids: list[str] = Field(default_factory=list)  # true, non-superseded items published by t


# --------------------------------------------------------------------------
# C -> A: the bid; A -> C/D: the market outcome
# --------------------------------------------------------------------------
class BidAction(BaseModel):
    markup: float  # multiplies c1 of every unit of the strategic firm
    method: str = ""  # estimator that produced it
    risk: str = "neutral"
    fallback_used: bool = False
    rationale: str = ""


class ClearingResult(BaseModel):
    status: str  # e.g. "LOCALLY_SOLVED", "DC_FALLBACK", "FAILED"
    lmp: list[float]  # per bus, $/MWh (before price cap)
    pg: list[float]  # per gen, MW
    objective: float  # $/h, using bid costs
    solve_s: float
    binding_branches: list[int] = Field(default_factory=list)
    used_dc_fallback: bool = False
    firm_profit: dict[str, float] = Field(default_factory=dict)  # firm -> $/h, at true cost
    load_shed_mw: float = 0.0  # MW shed at VOLL (scarcity)
    effective_markup: float | dict[str, float] | None = None  # markup (or bid profile) after mitigation


class PublicRoundResult(BaseModel):
    """What the strategic firm learns after interval t settles."""

    t: int
    markup: float
    own_dispatch_mw: dict[int, float]  # gen -> MW
    own_lmp: dict[int, float]  # gen -> $/MWh at its bus
    profit: float
    system_load_mw: float


class Observation(BaseModel):
    t: int
    bulletin: Bulletin
    baseline_zone_load_mw: dict[str, float]  # public "normal" forecast for interval t
    own_marginal_cost: dict[int, float] = Field(default_factory=dict)  # gen -> c1 of the firm's own units (known privately)
    history: list[PublicRoundResult] = Field(default_factory=list)
    deadline_s: float
    compute_budget_s: float
