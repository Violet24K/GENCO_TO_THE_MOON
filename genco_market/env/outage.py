"""Layer 4: ISO outage coordination.

One episode = one hidden world over H = 8 daytime hours + one transmission
owner's request to take a line out for d in {1, 2} consecutive hours. At t=0
the ISO picks a start s in {0..H-d} or defers the work past the episode.

Everything here is deterministic from (seed, episode):
- ``l4_start_hour``: a day (biased to summer, where heat waves can happen) and
  a start hour in [08:00, 10:00], so the 8 hours fit in 08:00-18:00.
- ``l4_event_types``: the events of the episode, biased toward heat waves and
  thunderstorms (they create outage risk). No planned_outage events: the
  outage is now the ISO's decision.
- ``L4WorldGenerator``: the normal WorldGenerator, except every event is
  announced at t=0 (its ``lead`` is raised to at least its start), so the
  t=0 bulletin carries the day-ahead forecast of every event. Physics and
  text use the longer lead consistently (larger forecast errors, larger
  shared load-forecast error for heat waves).
- ``sample_request``: the outage request and its ISO bulletin item.

Option cost (social cost, $): only the hours with the line out differ, so
    cost(s) = sum_{t=s}^{s+d-1} Delta_t,   Delta_t = E[SC(t, out) - SC(t, in)],
    cost(defer) = deferral_cost,
where SC = true generation cost + VOLL * shed (env/welfare.py).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import yaml

from genco_market.core.config import CONFIGS, Config
from genco_market.core.schemas import BulletinItem, Scenario
from genco_market.core.seeding import rng
from genco_market.env.grid import GridInfo
from genco_market.semantics import render as R
from genco_market.semantics.world import WorldGenerator

DEFER = -1  # option id of "defer past the episode"
HAZARDS = ("heatwave", "thunderstorm")


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass
class L4Cfg:
    horizon: int = 8
    start_hours: tuple[int, int] = (8, 10)  # inclusive range of the clock hour at t=0
    summer_prob: float = 0.7  # chance the day is drawn from the summer window
    summer_doy: tuple[int, int] = (150, 250)  # day-of-year range (heat waves need normal high >= 30 C)
    hazard_prob: float = 0.75  # chance one hazard (heatwave / thunderstorm) is forced into the episode
    heat_share: float = 0.6  # of forced hazards, heat wave when the season allows it
    extra_events: tuple[int, int] = (0, 2)  # number of other events added
    event_pool: list[str] = field(default_factory=lambda: ["heatwave", "thunderstorm", "gas_price", "major_event",
                                                           "holiday", "data_center"])
    candidates: dict[str, float] = field(default_factory=dict)  # line -> weight (default: effects.yaml planned_outage)
    durations: list[int] = field(default_factory=lambda: [1, 2])
    deferral_cost: float = 4000.0  # $; cost of postponing the work past the episode
    voll: float = 2000.0
    oracle_scenarios: int = 300
    oracle_hold_out: bool = True  # E4.2: second half of the cached scenarios is the oracle


def load_l4(cfg_name: str = "l4.yaml") -> tuple[Config, L4Cfg]:
    p = CONFIGS / cfg_name
    raw = yaml.safe_load(p.read_text())
    cfg = Config.model_validate(raw)
    l4 = L4Cfg(**{k: (tuple(v) if isinstance(getattr(L4Cfg, k, None), tuple) else v)
                  for k, v in (raw.get("l4") or {}).items()})
    l4.horizon = cfg.episode.horizon
    if not l4.candidates:
        from genco_market.semantics.events import load_effects

        l4.candidates = dict(load_effects()["planned_outage"]["candidates"])
    return cfg, l4


# --------------------------------------------------------------------------
# episode sampling
# --------------------------------------------------------------------------
def l4_start_hour(seed: int, ep: int, l4: L4Cfg) -> int:
    """Hour of year 2024 of t=0. ERCOT files cover 8,268 h = 344 days."""
    r = rng(seed, "l4_start", ep)
    if r.random() < l4.summer_prob:
        day = int(r.integers(l4.summer_doy[0], l4.summer_doy[1] + 1)) - 1
    else:
        day = int(r.integers(0, 8268 // 24 - 1))
    return day * 24 + int(r.integers(l4.start_hours[0], l4.start_hours[1] + 1))


def heat_eligible(gen: WorldGenerator, start_hour: int) -> bool:
    return gen._normal_high(start_hour) >= gen.eff["heatwave"]["season_min_normal_high_c"]


def l4_event_types(seed: int, ep: int, l4: L4Cfg, gen: WorldGenerator, start_hour: int) -> list[str]:
    """Forced event list for the episode (WorldGenerator.sample_world(event_types=...))."""
    r = rng(seed, "l4_events", ep)
    pool = [e for e in l4.event_pool if e != "heatwave" or heat_eligible(gen, start_hour)]
    types: list[str] = []
    if r.random() < l4.hazard_prob:
        heat_ok = "heatwave" in pool
        types.append("heatwave" if heat_ok and r.random() < l4.heat_share else "thunderstorm")
    n_extra = int(r.integers(l4.extra_events[0], l4.extra_events[1] + 1))
    if not types:
        n_extra = max(1, n_extra)
    rest = [e for e in pool if e not in types]
    if n_extra:
        types += [str(x) for x in r.choice(rest, size=min(n_extra, len(rest)), replace=False)]
    return types


class L4WorldGenerator(WorldGenerator):
    """Every event is announced at t=0 (day-ahead forecast): lead >= start_t.
    Later items (forecast revisions, echoes, rumors) still arrive later, but
    the ISO decides at t=0."""

    def _announce_at_0(self, ev):
        ev.params["lead"] = max(int(ev.params.get("lead", 0)), int(ev.start_t))
        return ev

    def _ev_heatwave(self, *a):
        return self._announce_at_0(super()._ev_heatwave(*a))

    def _ev_thunderstorm(self, *a):
        return self._announce_at_0(super()._ev_thunderstorm(*a))

    def _ev_gas_price(self, *a):
        return self._announce_at_0(super()._ev_gas_price(*a))

    def _ev_major_event(self, *a):
        return self._announce_at_0(super()._ev_major_event(*a))

    def _ev_data_center(self, *a):
        return self._announce_at_0(super()._ev_data_center(*a))

    def _ev_holiday(self, *a):
        return self._announce_at_0(super()._ev_holiday(*a))


# --------------------------------------------------------------------------
# the outage request
# --------------------------------------------------------------------------
REQUEST_TEMPLATES = [
    "ISO outage coordination: the transmission owner requests a planned outage of {kind} {branch} ({zone_short}) "
    "for {d} h for {purpose}, requested window {window}. The ISO may approve, reschedule within today's "
    "08:00-18:00 study period, or defer the work.",
    "Outage request received: {kind} {branch} in {zone_short}, {d}-hour outage for {purpose}; requested start "
    "{start} ({window}). Pending ISO approval.",
]


@dataclass
class OutageRequest:
    episode: int
    line: str  # candidate name, e.g. "49-54"
    branch: int  # 0-based branch row (one circuit)
    branch_name: str  # unique name, e.g. "49-54#2"
    duration: int
    s_req: int
    purpose: str
    text: str = ""

    def options(self, horizon: int) -> list[int]:
        """Start hours 0..H-d, then DEFER."""
        return list(range(horizon - self.duration + 1)) + [DEFER]

    def window(self, s: int) -> list[int]:
        return [] if s == DEFER else list(range(s, s + self.duration))

    def to_dict(self) -> dict:
        return asdict(self)


def sample_request(seed: int, ep: int, l4: L4Cfg, info: GridInfo, start_hour: int) -> OutageRequest:
    r = rng(seed, "l4_request", ep)
    names = sorted(l4.candidates)
    p = np.array([l4.candidates[n] for n in names], float)
    line = str(r.choice(names, p=p / p.sum()))
    rows = info.branch_by_name[line]
    br = int(rows[int(r.integers(len(rows)))])
    d = int(r.choice(l4.durations))
    s_req = int(r.integers(0, l4.horizon - d + 1))
    tpl = R.load_templates()
    purpose = str(r.choice(tpl["planned_outage"]["purposes"]))
    zone = str(info.bus_zone[int(info.net.branches[br, 1])])
    from gridfm_datakit.utils.idx_brch import TAP

    slots = {**R.zone_slots(tpl["common"], zone), "kind": "transformer" if info.net.branches[br, TAP] != 0 else "line",
             "branch": info.branch_name[br], "d": d, "purpose": purpose,
             "window": R.fmt_window(tpl["common"], start_hour, s_req, s_req + d - 1),
             "start": R.clock(start_hour, s_req).strftime("%H:%M")}
    text = REQUEST_TEMPLATES[int(r.integers(len(REQUEST_TEMPLATES)))].format_map(slots)
    return OutageRequest(ep, line, br, info.branch_name[br], d, s_req, purpose, text)


def request_item(req: OutageRequest, horizon: int) -> BulletinItem:
    """The request as an ISO bulletin item (for display / replay). It is the
    ISO's own decision input, so it is NOT fed to perception (the belief is
    about conditions with the line in service)."""
    return BulletinItem(item_id=f"ep{req.episode}-request", source="iso", published_t=0,
                        refers_to_t=list(range(horizon)), text=req.text)


def with_line_out(scn: Scenario, branch: int) -> Scenario:
    st = list(scn.branch_status)
    st[branch] = 0
    return scn.model_copy(update={"branch_status": st})


def option_costs(delta: np.ndarray, req: OutageRequest, deferral_cost: float) -> np.ndarray:
    """delta [H] $ per hour -> cost of each option in req.options(H) order."""
    H = len(delta)
    return np.array([deferral_cost if s == DEFER else float(np.sum(delta[s:s + req.duration]))
                     for s in req.options(H)])


def option_label(s: int, start_hour: int) -> str:
    return "defer" if s == DEFER else R.clock(start_hour, s).strftime("%H:%M")


def social_cost(pg, scn: Scenario, shed_mw: float, voll: float = 2000.0) -> float:
    """True generation cost + VOLL * shed, $/h (same as env/welfare.welfare_terms)."""
    return float(np.dot(np.asarray(scn.gen_c1, float), np.nan_to_num(np.asarray(pg, float)))) + voll * float(shed_mw)


def nan_safe_mean(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else math.nan


def reset_episode(env, l4: L4Cfg, ep: int) -> OutageRequest:
    """Reset a MarketEnv (built with an L4WorldGenerator) to L4 episode ``ep``
    and return its outage request."""
    seed = env.cfg.seed
    sh = l4_start_hour(seed, ep, l4)
    env.reset(ep, sh, event_types=l4_event_types(seed, ep, l4, env.gen, sh))
    return sample_request(seed, ep, l4, env.info, sh)
