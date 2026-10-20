"""WorldGenerator: hidden world (weather + event schedule) -> PhysicalState,
bulletin text and labels per interval.

A World is fully sampled up front (events, weather, every rendered item), so
it can be frozen to JSON and replayed; physical_state/bulletin/labels are pure
functions of the World. Event sampling (incl. schedule revisions) uses streams
independent of the text level, so L0-L3 of the same seed share the same
events and physics; levels differ only in what the text reveals and when.

L3 adds: the original planned-outage notice plus a later update
(postponed / extended / cancelled, via ``supersedes``); storm forecasts revised
as lead time shrinks; rumors later denied by an official source; false
contradicting items from unreliable sources; indirect cause-only wording for
major_event / holiday / data_center.

Item truth semantics: ``is_true`` = the item was accurate when published
(False for rumors and contradictions); ``superseded_by`` = the later item that
replaces it. An item is current truth at t if is_true and not superseded by an
item published at or before t (``current_item_ids``).
"""

from __future__ import annotations

import math

import numpy as np
from gridfm_datakit.utils.idx_brch import T_BUS, TAP
from pydantic import BaseModel, Field

from genco_market.core.config import CONFIGS, Config
from genco_market.core.schemas import (
    Bulletin,
    BulletinItem,
    BulletinLabels,
    EventRecord,
    ItemLabel,
    PhysicalState,
    Source,
)
from genco_market.core.seeding import rng
from genco_market.env.grid import GridInfo
from genco_market.semantics import render as R
from genco_market.semantics.events import (
    EVENT_TYPES,
    IMPLEMENTED,
    apply_events,
    bridge_branches,
    fuel_of_units,
    heat_shape,
    heat_system_std,
    is_active,
    keeps_connected,
    load_effects,
)
from genco_market.semantics.paraphrase import IdentityParaphraser, Paraphraser

WORLD_VERSION = 2
LEVELS = ("L0", "L1", "L2", "L3")
INDIRECT = ("major_event", "holiday", "data_center", "demand_response", "market_power_review")


class ItemRecord(BaseModel):
    """Full record of one rendered item (public text + hidden metadata)."""

    item_id: str
    source: Source
    published_t: int
    refers_to_t: list[int]
    text: str
    template_text: str  # before paraphrase
    template_key: str
    kind: str  # l0 | primary | advisory | update | echo | contradiction | distractor | rumor | denial
    event_id: str | None = None
    is_true: bool = True
    is_distractor: bool = False
    supersedes: str | None = None
    superseded_by: str | None = None
    claims: dict = Field(default_factory=dict)  # numbers/hedges stated in the text and their hidden ranges


class WeatherTrack(BaseModel):
    temp_normal_c: dict[str, list[float]]
    temp_c: dict[str, list[float]]
    storm_prob: dict[str, list[float]]  # forecast thunderstorm probability per zone and interval


class World(BaseModel):
    version: int = WORLD_VERSION
    effects_version: int
    templates_version: int
    seed: int
    episode_id: int
    horizon: int
    start_hour: int  # hour of year 2024 of interval t=0
    level: str
    weather: WeatherTrack
    events: list[EventRecord]  # incl. non-occurring forecasts and rumors (occurs=False)
    items: list[ItemRecord]


class WorldGenerator:
    def __init__(self, cfg: Config, info: GridInfo, paraphraser: Paraphraser | None = None):
        self.cfg, self.info = cfg, info
        self.eff = load_effects()
        self.tpl = R.load_templates()
        self.common = self.tpl["common"]
        self.paraphraser = paraphraser or IdentityParaphraser()
        self.fuel = fuel_of_units(info, self.eff)
        self.bridges = bridge_branches(info)
        self.is_tx = {i: bool(info.net.branches[i, TAP] != 0) for i in range(info.n_branch)}
        self.loading = {int(k): float(v) for k, v in R.load_yaml(CONFIGS / "line_loading_case118.yaml")["loading"].items()}

    # ------------------------------------------------------------------ API
    def sample_world(self, episode_id: int, horizon: int, start_hour: int, *, event_types: list[str] | None = None) -> World:
        """``event_types`` forces exactly these events (tests, ablations);
        otherwise 1..max_events types are drawn from cfg.semantics.event_types."""
        level = self.cfg.semantics.level
        if level not in LEVELS:
            raise NotImplementedError(f"text level {level}")
        key = (self.cfg.seed, "semantics", episode_id)
        normal = self._normal_temps(horizon, start_hour)
        events = self._sample_events(key, horizon, start_hour, normal, event_types)
        weather = self._weather(key, horizon, events, normal)
        world = World(
            effects_version=self.eff["version"],
            templates_version=self.common["version"],
            seed=self.cfg.seed,
            episode_id=episode_id,
            horizon=horizon,
            start_hour=start_hour,
            level=level,
            weather=weather,
            events=events,
            items=[],
        )
        world.items = self._render_items(key, world)
        return world

    def physical_state(self, world: World, t: int) -> PhysicalState:
        assert 0 <= t < world.horizon, t
        return apply_events(self.info, self.eff, world.events, t)

    def bulletin(self, world: World, t: int) -> Bulletin:
        items = [
            BulletinItem(item_id=i.item_id, source=i.source, published_t=i.published_t,
                         refers_to_t=i.refers_to_t, text=i.text, supersedes=i.supersedes)
            for i in world.items if i.published_t == t
        ]
        return Bulletin(t=t, items=items)

    def labels(self, world: World, t: int) -> BulletinLabels:
        extra = "superseded_by" in ItemLabel.model_fields  # filled once the schema has the field
        items = [
            ItemLabel(item_id=i.item_id, event_id=i.event_id, is_true=i.is_true, is_distractor=i.is_distractor,
                      **({"superseded_by": i.superseded_by} if extra else {}))
            for i in world.items if i.published_t == t
        ]
        lab = BulletinLabels(t=t, items=items, active_events=[e for e in world.events if is_active(e, t)])
        if "current_item_ids" in BulletinLabels.model_fields:
            lab.current_item_ids = self.current_item_ids(world, t)
        return lab

    def current_item_ids(self, world: World, t: int) -> list[str]:
        """Items published by t that are true and not yet superseded at t."""
        pub = {i.item_id: i.published_t for i in world.items}
        return [i.item_id for i in world.items if i.published_t <= t and i.is_true and not i.is_distractor
                and (i.superseded_by is None or pub[i.superseded_by] > t)]

    # ------------------------------------------------------------ weather
    def _normal_temps(self, horizon: int, start_hour: int) -> dict[str, list[float]]:
        w = self.eff["weather"]
        out = {}
        for z in self.info.zones:
            row = []
            for t in range(horizon):
                c = R.clock(start_hour, t)
                doy = c.timetuple().tm_yday
                v = (w["annual_mean_c"] + w["seasonal_amp_c"] * math.sin(2 * math.pi * (doy - 110) / 365)
                     + w["diurnal_amp_c"] * math.cos(2 * math.pi * (c.hour - 16) / 24) + w["zone_offset_c"][z])
                row.append(round(v, 2))
            out[z] = row
        return out

    def _normal_high(self, start_hour: int) -> float:
        w = self.eff["weather"]
        doy = R.clock(start_hour, 0).timetuple().tm_yday
        return w["annual_mean_c"] + w["seasonal_amp_c"] * math.sin(2 * math.pi * (doy - 110) / 365) + w["diurnal_amp_c"]

    def _weather(self, key, horizon: int, events: list[EventRecord], normal: dict) -> WeatherTrack:
        w = self.eff["weather"]
        temp, storm = {}, {}
        for z in self.info.zones:
            r = rng(*key, "weather", z)
            e, row = 0.0, []
            for t in range(horizon):
                e = w["noise_ar"] * e + w["noise_sd_c"] * math.sqrt(1 - w["noise_ar"] ** 2) * r.standard_normal()
                row.append(normal[z][t] + e)
            temp[z], storm[z] = row, [0.0] * horizon
        for ev in events:
            if ev.event_type == "heatwave" and ev.occurs:
                for z in ev.params["zones"]:
                    for t in range(horizon):
                        temp[z][t] += ev.params["anomaly_c"][z][t]
            elif ev.event_type == "cold_snap" and ev.occurs:
                for z in ev.params["zones"]:
                    for t in range(ev.start_t, ev.end_t + 1):
                        temp[z][t] -= ev.params["anomaly_c"]
            elif ev.event_type == "thunderstorm":
                for t in range(ev.start_t, ev.end_t + 1):
                    storm[ev.zone][t] = ev.params["forecast_prob"]
                    if ev.occurs:
                        temp[ev.zone][t] -= self.eff["thunderstorm"]["temp_drop_c"]
        temp = {z: [round(v, 2) for v in row] for z, row in temp.items()}
        return WeatherTrack(temp_normal_c=normal, temp_c=temp, storm_prob=storm)

    # ------------------------------------------------------------- events
    def _sample_events(self, key, horizon, start_hour, normal, forced) -> list[EventRecord]:
        if forced is not None:
            types = list(forced)
        else:
            sc = self.cfg.semantics
            elig = [e for e in sc.event_types if e in IMPLEMENTED]
            if self._normal_high(start_hour) < self.eff["heatwave"]["season_min_normal_high_c"]:
                elig = [e for e in elig if e != "heatwave"]
            if "cold_snap" in self.eff and self._normal_high(start_hour) > self.eff["cold_snap"]["season_max_normal_high_c"]:
                elig = [e for e in elig if e != "cold_snap"]
            r = rng(*key, "event_count")
            n = min(int(r.integers(1, sc.max_events + 1)), len(elig))
            types = [str(x) for x in r.choice(elig, size=n, replace=False)] if n else []
        for e in types:
            if e not in EVENT_TYPES:
                raise ValueError(f"unknown event type {e!r}")
            if e not in IMPLEMENTED:
                raise NotImplementedError(f"event type {e!r} is declared but not implemented")
        # outage budget priority: planned > storm > heat lines
        prio = {"planned_outage": 0, "wildfire_psps": 0, "thunderstorm": 1, "high_wind": 1, "heatwave": 2}
        order = sorted(range(len(types)), key=lambda i: prio.get(types[i], 3))
        events: list[EventRecord | None] = [None] * len(types)
        taken: set[int] = set()
        for i in order:
            r = rng(*key, "event", i, types[i])
            ev = getattr(self, f"_ev_{types[i]}")(r, f"ev{i}-{types[i]}", horizon, start_hour, normal, taken)
            self._revise(rng(*key, "revision", i, types[i]), ev, horizon)
            if ev.event_type == "planned_outage":
                taken.add(ev.params["branch"])
            elif ev.event_type == "thunderstorm":
                taken.update(ev.params["lines"])
            elif ev.event_type == "heatwave":
                taken.update(ev.params["heat_lines"])
            elif ev.event_type == "wildfire_psps":
                taken.update(ev.params["lines"])
            elif ev.event_type == "high_wind":
                taken.update(ev.params["gust_lines"])
            events[i] = ev
        return events  # type: ignore[return-value]

    def _window(self, r, horizon: int, dur_range) -> tuple[int, int]:
        s = int(r.integers(0, horizon))
        d = int(r.integers(dur_range[0], dur_range[1] + 1))
        return s, min(s + d - 1, horizon - 1)

    def _zone(self, r) -> str:
        zs = list(self.eff["zone_weights"])
        p = np.array([self.eff["zone_weights"][z] for z in zs], dtype=float)
        return str(r.choice(zs, p=p / p.sum()))

    def _ev_heatwave(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["heatwave"]
        sev = int(r.integers(1, 4))
        system = bool(r.random() < c["system_prob"])
        zone = None if system else self._zone(r)
        zones = sorted(self.info.zones) if system else [zone]
        s, e = self._window(r, horizon, c["duration_h"])
        peak = c["severity"][sev]["peak_anomaly_c"]
        anomaly = {}
        for z in zones:
            pz = peak * (float(r.uniform(0.85, 1.1)) if system else 1.0)
            anomaly[z] = [round(pz * heat_shape(R.clock(start_hour, t).hour), 2) if s <= t <= e else 0.0 for t in range(horizon)]
        gens = {z: sorted(self.info.firm_gens[z]) for z in zones}
        # heat line trips: most loaded eligible lines, within the remaining outage budget
        lt = c["line_trip"]
        cand = {b for z in zones for b in self.info.zone_branches[z]
                if not self.is_tx[b] and b not in self.bridges and b not in taken
                and self.loading.get(b, 0.0) >= lt["min_loading"]}
        n = int(r.integers(lt["n_lines"][0], lt["n_lines"][1] + 1))
        n = max(0, min(n, self.eff["bounds"]["max_outage_branches"] - len(taken)))
        heat_lines: list[int] = []
        for b in sorted(cand, key=lambda b: (-self.loading[b], b)):
            if len(heat_lines) == n:
                break
            if keeps_connected(self.info, taken | set(heat_lines) | {b}):
                heat_lines.append(b)
        zload = {z: float(self.info.zones[z]["base_load_mw"]) for z in self.info.zones}
        params = {
            "zones": zones,
            "peak_anomaly_c": peak,
            "anomaly_c": anomaly,
            "branches": {z: sorted(self.info.zone_branches[z]) for z in zones},
            "gens": gens,
            "lead": int(r.integers(0, 3)),
            "unit_trip_per_c": {str(g): c["unit_trip_per_c"][self.fuel[g]] for z in zones for g in gens[z]},
            "heat_lines": sorted(heat_lines),
            "heat_line_names": [self.info.branch_name[b] for b in sorted(heat_lines)],
            "load_share": round(sum(zload[z] for z in zones) / sum(zload.values()), 4),
        }
        return EventRecord(event_id=eid, event_type="heatwave", zone=zone, start_t=s, end_t=e, severity=float(sev), params=params)

    def _ev_planned_outage(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["planned_outage"]
        names = list(c["candidates"])
        p = np.array([c["candidates"][n] for n in names], dtype=float)
        name = str(r.choice(names, p=p / p.sum()))
        rows = [b for b in self.info.branch_by_name[name] if b not in self.bridges and b not in taken]
        assert rows, f"planned outage candidate {name} is a bridge or already out"
        br = rows[int(r.integers(len(rows)))]
        assert keeps_connected(self.info, taken | {br})
        s, e = self._window(r, horizon, c["duration_h"])
        zone = str(self.info.bus_zone[int(self.info.net.branches[br, T_BUS])])  # zone of the to-bus
        params = {"branch": br, "branch_name": self.info.branch_name[br], "is_transformer": self.is_tx[br],
                  "lead": int(r.integers(c["lead_h"][0], c["lead_h"][1] + 1)),
                  "purpose": str(r.choice(self.tpl["planned_outage"]["purposes"]))}
        return EventRecord(event_id=eid, event_type="planned_outage", zone=zone, start_t=s, end_t=e, severity=1.0, params=params)

    def _ev_thunderstorm(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["thunderstorm"]
        sev = int(r.integers(1, 4))
        zone = self._zone(r)
        s, e = self._window(r, horizon, c["duration_h"])
        q = float(r.choice(c["forecast_prob"]))
        occurs = bool(r.random() < q)
        # overhead lines only (no transformers), never bridges or branches already scheduled out
        cand = [b for b in self.info.zone_branches[zone] if not self.is_tx[b] and b not in self.bridges and b not in taken]
        n = int(r.integers(c["n_lines"][0], c["n_lines"][1] + 1))
        n = max(0, min(n, self.eff["bounds"]["max_outage_branches"] - len(taken), len(cand)))
        lines: list[int] = []
        for b in (int(x) for x in r.permutation(cand)):
            if len(lines) == n:
                break
            if keeps_connected(self.info, taken | set(lines) | {b}):
                lines.append(b)
        params = {"lines": sorted(lines), "line_names": [self.info.branch_name[b] for b in sorted(lines)],
                  "forecast_prob": q, "gust_kmh": 60 + 20 * sev, "lead": int(r.integers(0, 2))}
        return EventRecord(event_id=eid, event_type="thunderstorm", zone=zone, start_t=s, end_t=e,
                           severity=float(sev), occurs=occurs, params=params)

    def _ev_gas_price(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["gas_price"]
        sev = int(r.integers(1, 4))
        gm = c["severity"][sev]["gas_mult"]
        s = int(r.integers(0, horizon))
        unit = {g: (1 + c["pass_through"] * (gm - 1) if f == "gas" else 1.0) for g, f in self.fuel.items()}
        firm = {}
        for z, gs in self.info.firm_gens.items():
            if z == self.info.strategic:
                continue
            cap = np.array([self.info.base_pmax[g] for g in gs])
            firm[z] = round(float(np.dot(cap, [unit[g] for g in gs]) / cap.sum()), 4)
        params = {"gas_mult": gm, "unit_cost_mult": {str(g): round(m, 4) for g, m in unit.items()},
                  "firm_cost_mult": firm, "lead": 0, "hub": str(r.choice(self.tpl["gas_price"]["hubs"]))}
        return EventRecord(event_id=eid, event_type="gas_price", zone=None, start_t=s, end_t=horizon - 1,
                           severity=float(sev), params=params)

    def _ev_major_event(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c, tp = self.eff["major_event"], self.tpl["major_event"]
        zone = self._zone(r)
        s, e = self._window(r, horizon, c["duration_h"])
        params = {"load_delta": round(float(r.uniform(*c["load_delta"])), 3),
                  "lead": int(r.integers(c["lead_h"][0], c["lead_h"][1] + 1)),
                  "what": str(r.choice(tp["whats"])), "venue": str(r.choice(tp["venues"])),
                  "crowd": int(r.integers(20, 81)) * 1000}
        return EventRecord(event_id=eid, event_type="major_event", zone=zone, start_t=s, end_t=e, params=params)

    def _ev_holiday(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["holiday"]
        system = bool(r.random() < c["system_prob"])
        zone = None if system else self._zone(r)
        params = {"zones": sorted(self.info.zones) if system else [zone],
                  "load_delta": round(float(r.uniform(*c["load_delta"])), 3), "lead": 0,
                  "why": str(r.choice(self.tpl["holiday"]["whys"]))}
        return EventRecord(event_id=eid, event_type="holiday", zone=zone, start_t=0, end_t=horizon - 1, params=params)

    def _ev_data_center(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["data_center"]
        zone = self._zone(r)
        zb = self.info.zones[zone]
        cand = [(b, o) for b, o in zip(zb["buses"], zb["buses_orig"]) if self.info.base_pd[b] > 0]
        bus, bus_orig = cand[int(r.integers(len(cand)))]
        mw = int(r.integers(c["mw"][0], c["mw"][1] + 1))
        params = {"bus": int(bus), "bus_orig": int(bus_orig), "mw": mw, "homes": mw * c["homes_per_mw"],
                  "lead": int(r.integers(0, 2)), "company": str(r.choice(self.tpl["data_center"]["companies"]))}
        return EventRecord(event_id=eid, event_type="data_center", zone=zone, start_t=int(r.integers(0, horizon)),
                           end_t=horizon - 1, params=params)

    # ------------------------------------------------- extension II events
    def _lead(self, r, c) -> int:
        return int(r.integers(c["lead_h"][0], c["lead_h"][1] + 1))

    def _pick_lines(self, cand, n, taken) -> list[int]:
        """Up to n candidate branches (in the given order) that keep the grid connected, within the outage budget."""
        n = max(0, min(n, self.eff["bounds"]["max_outage_branches"] - len(taken), len(cand)))
        out: list[int] = []
        for b in cand:
            if len(out) == n:
                break
            if keeps_connected(self.info, set(taken) | set(out) | {b}):
                out.append(b)
        return sorted(out)

    def _overhead(self, zone, taken) -> list[int]:
        return [b for b in self.info.zone_branches[zone] if not self.is_tx[b] and b not in self.bridges and b not in taken]

    def _bus_share(self, zone) -> dict[str, float]:
        bs = [b for b in self.info.zones[zone]["buses"] if self.info.base_pd[b] > 0]
        tot = float(sum(self.info.base_pd[b] for b in bs))
        return {str(b): round(float(self.info.base_pd[b]) / tot, 5) for b in bs}

    def _ev_cold_snap(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["cold_snap"]
        sev = int(r.integers(1, 4))
        system = bool(r.random() < c["system_prob"])
        zone = None if system else self._zone(r)
        zones = sorted(self.info.zones) if system else [zone]
        s, e = self._window(r, horizon, c["duration_h"])
        zload = {z: float(self.info.zones[z]["base_load_mw"]) for z in self.info.zones}
        params = {"zones": zones, "anomaly_c": c["severity"][sev]["anomaly_c"], "lead": self._lead(r, c),
                  "gas_units": sorted(g for g, f in self.fuel.items() if f == "gas"),
                  "load_share": round(sum(zload[z] for z in zones) / sum(zload.values()), 4)}
        return EventRecord(event_id=eid, event_type="cold_snap", zone=zone, start_t=s, end_t=e, severity=float(sev), params=params)

    def _ev_wildfire_psps(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["wildfire_psps"]
        zone = self._zone(r)
        cand = [int(b) for b in r.permutation(self._overhead(zone, taken))]
        n = int(r.integers(c["n_lines"][0], c["n_lines"][1] + 1))
        lines = self._pick_lines(cand, n + 1, taken)  # one spare line for an "expanded" revision
        main, spare = lines[:n], lines[n:]
        s, e = self._window(r, horizon, c["duration_h"])
        params = {"lines": main, "line_names": [self.info.branch_name[b] for b in main], "spare": spare,
                  "lead": self._lead(r, c)}
        return EventRecord(event_id=eid, event_type="wildfire_psps", zone=zone, start_t=s, end_t=e, params=params)

    def _ev_high_wind(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["high_wind"]
        sev = int(r.integers(1, 4))
        zs = list(self.eff["wind_zone_weights"])
        pz = np.array([self.eff["wind_zone_weights"][z] for z in zs], float)
        zone = str(r.choice(zs, p=pz / pz.sum()))
        sv = c["severity"][sev]
        mw = round(float(r.uniform(*sv["extra_mw"])), 1)
        gust: list[int] = []
        if sv["line_trip_prob"] > 0:
            n = int(r.integers(c["n_lines"][0], c["n_lines"][1] + 1))
            gust = self._pick_lines([int(b) for b in r.permutation(self._overhead(zone, taken))], n, taken)
        s, e = self._window(r, horizon, c["duration_h"])
        params = {"mw": mw, "mw_std": round(c["std_frac"] * mw, 1), "bus_share": self._bus_share(zone),
                  "gust_lines": gust, "gust_line_names": [self.info.branch_name[b] for b in gust],
                  "line_trip_prob": sv["line_trip_prob"], "gust_kmh": 50 + 15 * sev, "lead": self._lead(r, c)}
        return EventRecord(event_id=eid, event_type="high_wind", zone=zone, start_t=s, end_t=e, severity=float(sev), params=params)

    def _ev_solar_drop(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["solar_drop"]
        sev = int(r.integers(1, 4))
        zone = self._zone(r)
        mw = round(float(r.uniform(*c["severity"][sev]["lost_mw"])), 1)
        s, e = self._window(r, horizon, c["duration_h"])
        params = {"mw": mw, "mw_std": round(c["std_frac"] * mw, 1), "bus_share": self._bus_share(zone),
                  "cause": str(r.choice(c["causes"])), "lead": self._lead(r, c)}
        return EventRecord(event_id=eid, event_type="solar_drop", zone=zone, start_t=s, end_t=e, severity=float(sev), params=params)

    def _ev_demand_response(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["demand_response"]
        s, e = self._window(r, horizon, c["duration_h"])
        params = {"tight_delta": round(float(r.uniform(*c["tight_load_delta"])), 3),
                  "response_delta": round(float(r.uniform(*c["response_delta"])), 3), "lead": self._lead(r, c)}
        return EventRecord(event_id=eid, event_type="demand_response", zone=None, start_t=s, end_t=e, params=params)

    def _ev_market_power_review(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["market_power_review"]
        sev = int(r.integers(1, 4))
        s = int(r.integers(0, horizon))
        params = {"cap": c["severity"][sev]["cap"], "lead": self._lead(r, c)}
        return EventRecord(event_id=eid, event_type="market_power_review", zone=None, start_t=s, end_t=horizon - 1,
                           severity=float(sev), params=params)

    def _ev_geomagnetic_storm(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["geomagnetic_storm"]
        sev = int(r.integers(1, 4))
        s, e = self._window(r, horizon, c["duration_h"])
        tx = sorted(b for b in range(self.info.n_branch) if self.is_tx[b] and b not in self.bridges)
        units = sorted(g for g in self.info.gen_firm if self.info.base_pmax[g] > 0)
        params = {"scale": c["severity"][sev]["scale"], "transformers": tx, "units": units, "lead": self._lead(r, c),
                  "shock_prob": c["severity"][sev]["shock_prob"]}
        return EventRecord(event_id=eid, event_type="geomagnetic_storm", zone=None, start_t=s, end_t=e,
                           severity=float(sev), params=params)

    def _ev_cyber_physical(self, r, eid, horizon, start_hour, normal, taken) -> EventRecord:
        c = self.eff["cyber_physical"]
        sev = int(r.integers(1, 4))
        zone = self._zone(r)
        br = self.info.net.branches
        best = None
        # an attacked substation trips all but one of its lines (the bus stays connected, its neighbours lose those ties)
        for b in [int(x) for x in r.permutation(self.info.zones[zone]["buses"])] + list(self.info.zones[zone]["buses"]):
            lines = sorted(i for i in range(self.info.n_branch) if (int(br[i, 0]) == b or int(br[i, 1]) == b)
                           and i not in self.bridges)
            for drop in range(len(lines)):
                keep = lines[:drop] + lines[drop + 1:]
                if len(keep) >= 2 and keeps_connected(self.info, set(keep)):
                    best = (b, keep)
                    break
            if best:
                break
        assert best is not None, f"no attackable substation in zone {zone}"
        bus, lines = best
        s, e = self._window(r, horizon, c["duration_h"])
        params = {"bus": bus, "bus_orig": int(self.info.net.original_bus_indices[bus]), "lines": lines,
                  "line_names": [self.info.branch_name[i] for i in lines],
                  "attack_prob": c["severity"][sev]["attack_prob"], "lead": self._lead(r, c)}
        return EventRecord(event_id=eid, event_type="cyber_physical", zone=zone, start_t=s, end_t=e,
                           severity=float(sev), params=params)

    # ---------------------------------------------------------- revisions
    def _revise(self, r, ev: EventRecord, H: int) -> None:
        """Level-independent schedule/forecast evolution. Planned-outage
        revisions change the truth; storm forecast paths are text only."""
        if ev.event_type == "planned_outage":
            c = self.eff["planned_outage"]
            kinds = list(c["revision"])
            kind = str(r.choice(kinds, p=np.array([c["revision"][k] for k in kinds]) / sum(c["revision"].values())))
            s0, e0 = ev.start_t, ev.end_t
            pub = max(0, s0 - ev.params["lead"])
            d = int(r.integers(c["revision_shift_h"][0], c["revision_shift_h"][1] + 1))
            rev = None
            if kind in ("postponed", "cancelled") and pub < s0:
                rev = {"kind": kind, "t": int(r.integers(pub + 1, s0 + 1)), "shift": d}
            elif kind == "extended" and e0 < H - 1 and max(pub + 1, s0) <= e0:
                rev = {"kind": kind, "t": int(r.integers(max(pub + 1, s0), e0 + 1)), "shift": d}
            ev.params["announced"] = [s0, e0]
            ev.params["revision"] = rev
            if rev and kind == "postponed":
                ev.start_t, ev.end_t = s0 + d, (min(e0 + d, H - 1) if s0 + d <= H - 1 else e0 + d)
            elif rev and kind == "extended":
                ev.end_t = min(e0 + d, H - 1)
            elif rev and kind == "cancelled":
                ev.occurs = False
        elif ev.event_type == "wildfire_psps":
            c = self.eff["wildfire_psps"]
            kinds = list(c["revision"])
            kind = str(r.choice(kinds, p=np.array([c["revision"][k] for k in kinds]) / sum(c["revision"].values())))
            s0, e0 = ev.start_t, ev.end_t
            pub = max(0, s0 - ev.params["lead"])
            d = int(r.integers(c["revision_shift_h"][0], c["revision_shift_h"][1] + 1))
            rev = None
            if kind == "cancelled" and pub < s0:
                rev = {"kind": kind, "t": int(r.integers(pub + 1, s0 + 1))}
            elif kind == "expanded" and ev.params["spare"] and pub < s0:
                rev = {"kind": kind, "t": int(r.integers(pub + 1, s0 + 1))}
            elif kind == "extended" and e0 < H - 1 and max(pub + 1, s0) <= e0:
                rev = {"kind": kind, "t": int(r.integers(max(pub + 1, s0), e0 + 1)), "shift": d}
            ev.params["announced"] = [s0, e0]
            ev.params["announced_lines"] = list(ev.params["line_names"])
            ev.params["revision"] = rev
            if rev and kind == "cancelled":
                ev.occurs = False
            elif rev and kind == "expanded":
                ev.params["lines"] = sorted(ev.params["lines"] + ev.params["spare"][:1])
                ev.params["line_names"] = [self.info.branch_name[b] for b in ev.params["lines"]]
            elif rev and kind == "extended":
                ev.end_t = min(e0 + d, H - 1)
        elif ev.event_type == "thunderstorm":
            c = self.eff["thunderstorm"]
            q, s = ev.params["forecast_prob"], ev.start_t
            p0 = max(0, s - ev.params["lead"] - c["forecast_path_lead"])
            path = []
            for t in range(p0, s + 1):
                qt = q if t == s else float(np.clip(round(q + c["forecast_path_sd"] * (s - t) * r.standard_normal(), 1), 0.1, 0.9))
                path.append([t, round(qt, 2)])
            ev.params["forecast_path"] = path

    # -------------------------------------------------------------- items
    def _render_items(self, key, world: World) -> list[ItemRecord]:
        H, lvl = world.horizon, world.level
        drafts: list[dict] = []
        ADVISORY = {"heatwave": self._advisory, "cold_snap": self._gas_notice}
        if lvl == "L0":
            for t in range(H):
                txt = R.render_l0(self.physical_state(world, t))
                drafts.append(dict(source="iso", published_t=t, refers_to_t=[t], text=txt, template_key="l0", kind="l0"))
        elif lvl == "L1":
            for ev in world.events:
                if ev.occurs or ev.event_type == "thunderstorm":  # a missed storm was still forecast
                    drafts.append(self._primary(world, ev, "clean"))
                if ev.event_type in ADVISORY:
                    drafts.append(ADVISORY[ev.event_type](world, ev, "clean"))
        else:
            for ev in list(world.events):
                if lvl == "L3":
                    drafts += self._l3_event_items(world, ev)
                elif ev.occurs or ev.event_type == "thunderstorm":
                    drafts.append(self._primary(world, ev, "hedged"))
                if ev.event_type in ADVISORY:
                    drafts.append(ADVISORY[ev.event_type](world, ev, "indirect" if lvl == "L3" else "hedged"))
            for ev in world.events:
                r = rng(*key, "echo", ev.event_id)
                if r.random() < self.common["echo_prob"] and (ev.occurs or ev.event_type == "thunderstorm"):
                    drafts.append(self._echo(world, ev, r))
            n_d = self.cfg.semantics.n_distractors
            n_d = self.common["n_distractors_default"] if n_d is None else n_d
            used: set[str] = set()
            for k in range(n_d):
                r = rng(*key, "distractor", k)
                for _ in range(10):  # avoid repeating the same distractor template
                    d = self._distractor(world, r)
                    if d["template_key"] not in used:
                        break
                used.add(d["template_key"])
                drafts.append(d)
            r = rng(*key, "rumor")
            if r.random() < self.common["rumor_prob_l3" if lvl == "L3" else "rumor_prob"]:
                d, ev = self._rumor(world, r)
                world.events.append(ev)
                drafts.append(d)
                if lvl == "L3" and d["published_t"] < H - 1 and r.random() < self.common["denial_prob"]:
                    drafts.append(self._denial(world, ev, d, r))
        # deterministic ids: by publish time, then (L2/L3) shuffled within an interval
        out: list[ItemRecord] = []
        ids: dict[str, str] = {}
        for t in range(H):
            group = [d for d in drafts if d["published_t"] == t]
            if lvl in ("L2", "L3"):
                group = [group[i] for i in rng(*key, "order", t).permutation(len(group))]
            for k, d in enumerate(group):
                iid = f"ep{world.episode_id}-t{t}-{k:02d}"
                d = dict(d)
                ids[d.pop("dkey", iid)] = iid
                text = d["text"]
                if lvl in ("L2", "L3"):
                    text = self.paraphraser(text, item_id=iid, seed=world.seed, source=d["source"])
                out.append(ItemRecord(item_id=iid, template_text=d["text"], **{**d, "text": text}))
        by_id = {i.item_id: i for i in out}
        for it in out:
            if it.supersedes is not None:
                it.supersedes = ids[it.supersedes]
                assert by_id[it.supersedes].published_t < it.published_t
                by_id[it.supersedes].superseded_by = it.item_id
        return out

    def _refers(self, world: World, s: int, e: int) -> list[int]:
        return [t for t in range(s, e + 1) if 0 <= t < world.horizon]

    def _conf(self, lead: int) -> float:
        tab = self.common["conf_by_lead"]
        return tab[min(lead, len(tab) - 1)]

    def _slots(self, world: World, ev: EventRecord, r, lead: int, claims: dict, *, window: tuple[int, int] | None = None,
               q: float | None = None, distort: float = 0.0) -> dict:
        """All slots for any template of this event; numbers recorded in claims."""
        s, e = window or (ev.start_t, ev.end_t)
        c0 = R.clock(world.start_hour, s)
        sl = {**R.zone_slots(self.common, ev.zone), "window": R.fmt_window(self.common, world.start_hour, s, e),
              "since": R.fmt_since(self.common, world.start_hour, s), "start": c0.strftime("%H:%M"),
              "day": c0.strftime("%a %b %d")}
        p, et = ev.params, ev.event_type
        if et == "heatwave":
            sl.update(self._heat_numbers(world, ev, lead, r, claims, distort=distort))
            sl["conf"], claims["conf_range"] = R.pick_phrase(self.common["conf_hedges"], self._conf(lead), r)
        elif et == "planned_outage":
            sl.update(self._outage_slots(ev))
            claims.update(branch=p["branch_name"])
        elif et == "thunderstorm":
            q = p["forecast_prob"] if q is None else q
            sl.update(pct=int(round(100 * q)), gust=p["gust_kmh"])
            sl["hedge"], claims["prob_range"] = R.pick_phrase(self.common["prob_hedges"], q, r)
            sl["hedge_cap"] = sl["hedge"][0].upper() + sl["hedge"][1:]
            claims.update(prob=q, hedge=sl["hedge"])
        elif et == "gas_price":
            true = 100 * (p["gas_mult"] - 1)
            pct = true * (1 + distort * abs(float(r.standard_normal()))) if distort else true
            sl.update(pct=int(round(pct)), hub=p["hub"])
            claims["gas_price_pct"] = sl["pct"]
        elif et == "major_event":
            sl.update(pct=int(round(100 * p["load_delta"])), what=p["what"], venue=p["venue"], crowd=f"{p['crowd']:,}")
            claims["load_pct"] = round(100 * p["load_delta"], 1)
        elif et == "holiday":
            sl.update(pct=int(round(-100 * p["load_delta"])), why=p["why"])
            claims["load_pct"] = round(100 * p["load_delta"], 1)
        elif et == "data_center":
            sl.update(bus=p["bus_orig"], mw=p["mw"], homes=f"{p['homes']:,}", company=p["company"])
            claims["mw"] = p["mw"]
        elif et == "cold_snap":
            c = self.eff["cold_snap"]
            sv = c["severity"][int(ev.severity)]
            a = p["anomaly_c"] - (p.get("_understate", 0.0))
            a_said = max(2.0, a * (1 + distort * abs(float(r.standard_normal())))) if distort else a
            z0 = p["zones"][0]
            lo_t = min(world.weather.temp_normal_c[z0][t] for t in range(s, e + 1)) - a_said
            mid = 100 * c["load_mult_per_c"] * a_said
            hw = self.common["forecast_load_halfwidth_pct"]["base"] + self.common["forecast_load_halfwidth_pct"]["per_lead"] * lead
            sl.update(temp_lo=int(round(lo_t)), anom=int(round(a_said)), load_lo=max(0, int(round(mid - hw))),
                      load_hi=int(round(mid + hw)), load_mid=int(round(mid)),
                      gas_pct=int(round(100 * sv["gas_shock_prob"])), gas_trip_pct=int(round(100 * sv["gas_trip_given_shock"])),
                      gas_price_pct=int(round(100 * (sv["gas_mult"] - 1))),
                      unit_pct=f"{100 * c['unit_trip_per_c'] * p['anomaly_c']:.1f}")
            sl["conf"], claims["conf_range"] = R.pick_phrase(self.common["conf_hedges"], self._conf(lead), r)
            claims.update(anom_c=a_said, true_anom_c=p["anomaly_c"], gas_shock_prob=sv["gas_shock_prob"])
        elif et == "wildfire_psps":
            names = p["line_names"]
            sl.update(lines=" and ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + " and " + names[-1],
                      n=len(names))
            claims["lines"] = list(names)
        elif et in ("high_wind", "solar_drop"):
            mw_said = p["mw"] * (1 + distort * float(r.standard_normal())) if distort else p["mw"]
            sd = p["mw_std"]
            sl.update(mw=int(round(mw_said)), mw_lo=max(0, int(round(mw_said - 2 * sd))), mw_hi=int(round(mw_said + 2 * sd)))
            if et == "high_wind":
                gl = p["gust_line_names"]
                sl.update(gust=p["gust_kmh"], line_pct=int(round(100 * p["line_trip_prob"])),
                          gust_lines=" and ".join(gl))
                sl["line_clause"] = self.tpl["high_wind"]["line_clause"]["hedged"].format_map(sl) if gl else ""
            else:
                sl.update(cause=p["cause"])
            claims.update(mw=round(mw_said, 1), mw_std=sd)
        elif et == "demand_response":
            sl.update(tight_pct=int(round(100 * p["tight_delta"])), resp_pct=int(round(-100 * p["response_delta"])),
                      net_pct=int(round(100 * (p["tight_delta"] + p["response_delta"]))))
            claims.update(tight=p["tight_delta"], response=p["response_delta"])
        elif et == "market_power_review":
            sl.update(cap=f"{p['cap']:g}", cap_pct=int(round(100 * (p["cap"] - 1))))
            claims["cap"] = p["cap"]
        elif et == "geomagnetic_storm":
            sv = self.eff["geomagnetic_storm"]["severity"][int(ev.severity)]
            sl.update(scale=p["scale"], pct=int(round(100 * sv["shock_prob"])), tx_pct=int(round(100 * sv["tx_trip_given_shock"])),
                      n_tx=len(p["transformers"]))
            sl["hedge"], claims["prob_range"] = R.pick_phrase(self.common["prob_hedges"], sv["shock_prob"], r)
            claims.update(shock_prob=sv["shock_prob"])
        elif et == "cyber_physical":
            sl.update(bus=p["bus_orig"], n=len(p["lines"]), pct=int(round(100 * p["attack_prob"])),
                      lines=", ".join(p["line_names"]))
            sl["hedge"], claims["prob_range"] = R.pick_phrase(self.common["prob_hedges"], p["attack_prob"], r)
            claims.update(attack_prob=p["attack_prob"], lines=list(p["line_names"]))
        return sl

    def _item(self, group: dict, src: str | None, r, tkey: str, sl: dict, **fields) -> dict:
        src = src if src in group else sorted(group)[0]
        tmpl, key = R.choose_template(group, src, r, tkey)
        return dict(source=src, text=R.fill(tmpl, sl), template_key=key, **fields)

    def _primary(self, world: World, ev: EventRecord, group: str, *, window=None, pub=None, dkey=None, q=None) -> dict:
        r = rng(world.seed, "semantics", world.episode_id, "primary", ev.event_id)
        s, e = window or (ev.start_t, ev.end_t)
        pub = max(0, s - ev.params.get("lead", 0)) if pub is None else pub
        claims: dict = {"lead": s - pub}
        sl = self._slots(world, ev, r, s - pub, claims, window=(s, e), q=q)
        tpl = self.tpl[ev.event_type]
        group = group if group in tpl else "hedged"
        if ev.event_type == "high_wind" and group == "clean" and ev.params["gust_lines"]:
            sl["line_clause"] = tpl["line_clause"]["clean"].format_map(sl)
        d = self._item(tpl[group], EVENT_TYPES[ev.event_type].primary_source, r, f"{ev.event_type}/{group}", sl,
                       published_t=pub, refers_to_t=self._refers(world, s, e), kind="primary",
                       event_id=ev.event_id, is_true=True, claims=claims)
        if dkey:
            d["dkey"] = dkey
        return d

    def _advisory(self, world: World, ev: EventRecord, group: str) -> dict:
        """ISO notice on heat risks: unit trips, line sag trips, shared load-forecast error (at the window peak)."""
        r = rng(world.seed, "semantics", world.episode_id, "advisory", ev.event_id)
        c, p = self.eff["heatwave"], ev.params
        pub = max(0, ev.start_t - p["lead"])
        a_pk = max(p["anomaly_c"][z][t] for z in p["zones"] for t in range(ev.start_t, ev.end_t + 1))
        rates = list(p["unit_trip_per_c"].values())
        unit = 100 * a_pk * sum(rates) / len(rates)
        sigma = 100 * heat_system_std(c, a_pk, p["lead"], p["load_share"])
        fc = self.common["forecast_temp_sd_c"]
        sl = {**R.zone_slots(self.common, ev.zone), "window": R.fmt_window(self.common, world.start_hour, ev.start_t, ev.end_t),
              "unit_pct": f"{unit:.1f}", "unit_pct_r": max(1, int(round(unit))), "sigma_pct": f"{sigma:.1f}",
              "sigma_r": max(1, int(round(sigma))), "temp_sd": f"{fc['base'] + fc['per_lead'] * p['lead']:.1f}",
              "lines": " and ".join(p["heat_line_names"]), "line_pct": f"{100 * c['line_trip']['per_c'] * a_pk:.1f}"}
        sl["line_clause"] = self.tpl["heatwave"]["line_clause"][group].format_map(sl) if p["heat_lines"] else ""
        claims = {"unit_trip_pct_peak": float(sl["unit_pct"]), "system_load_std_pct_peak": float(sl["sigma_pct"]),
                  "heat_lines": p["heat_line_names"], "line_trip_pct_peak": float(sl["line_pct"])}
        return self._item(self.tpl["heatwave"]["advisory"][group], "iso", r, f"heatwave/advisory/{group}", sl,
                          published_t=pub, refers_to_t=self._refers(world, ev.start_t, ev.end_t), kind="advisory",
                          event_id=ev.event_id, is_true=True, claims=claims)

    def _gas_notice(self, world: World, ev: EventRecord, group: str) -> dict:
        """ISO / pipeline notice on gas-supply curtailment risk during a cold snap (correlated gas-unit trips)."""
        r = rng(world.seed, "semantics", world.episode_id, "gasnotice", ev.event_id)
        pub = max(0, ev.start_t - ev.params["lead"])
        claims: dict = {"lead": ev.start_t - pub}
        sl = self._slots(world, ev, r, ev.start_t - pub, claims)
        return self._item(self.tpl["cold_snap"]["gas_notice"][group], "iso", r, f"cold_snap/gas_notice/{group}", sl,
                          published_t=pub, refers_to_t=self._refers(world, ev.start_t, ev.end_t), kind="advisory",
                          event_id=ev.event_id, is_true=True, claims=claims)

    def _l3_event_items(self, world: World, ev: EventRecord) -> list[dict]:
        """Original notice (+ updates superseding it) + optional contradiction."""
        r = rng(world.seed, "semantics", world.episode_id, "l3", ev.event_id)
        p, et, H = ev.params, ev.event_type, world.horizon
        group = "indirect" if et in INDIRECT else "hedged"
        out: list[dict] = []
        if et == "planned_outage":
            s0, e0 = p["announced"]
            first = self._primary(world, ev, group, window=(s0, e0), dkey=f"{ev.event_id}/0")
            out.append(first)
            rev = p["revision"]
            if rev:
                claims = {"revision": rev["kind"], "announced": [s0, e0]}
                sl = self._slots(world, ev, r, 0, claims)
                sl["old_window"] = R.fmt_window(self.common, world.start_hour, s0, e0)
                refers = sorted(set(self._refers(world, s0, e0)) | (set() if rev["kind"] == "cancelled"
                                                                    else set(self._refers(world, ev.start_t, ev.end_t))))
                out.append(self._item(self.tpl[et]["update"][rev["kind"]], "iso", r, f"{et}/update/{rev['kind']}", sl,
                                      published_t=rev["t"], refers_to_t=refers, kind="update", event_id=ev.event_id,
                                      is_true=True, supersedes=f"{ev.event_id}/0", dkey=f"{ev.event_id}/1", claims=claims))
        elif et == "thunderstorm":
            path = p["forecast_path"]
            t0, q0 = path[0]
            out.append(self._primary(world, ev, group, pub=t0, dkey=f"{ev.event_id}/0", q=q0))
            last, k = q0, 0
            for t, q in path[1:]:
                if abs(q - last) < 0.05:
                    continue
                claims = {"prob": q, "lead": ev.start_t - t}
                sl = self._slots(world, ev, r, ev.start_t - t, claims, q=q)
                sl["direction"] = "upgraded" if q > last else "downgraded"
                out.append(self._item(self.tpl[et]["update"]["revised"], "weather", r, f"{et}/update/revised", sl,
                                      published_t=t, refers_to_t=self._refers(world, ev.start_t, ev.end_t), kind="update",
                                      event_id=ev.event_id, is_true=True, supersedes=f"{ev.event_id}/{k}",
                                      dkey=f"{ev.event_id}/{k + 1}", claims=claims))
                last, k = q, k + 1
        elif et == "heatwave":
            first = self._primary(world, ev, group, dkey=f"{ev.event_id}/0")
            out.append(first)
            if first["published_t"] < ev.start_t:
                claims = {"lead": 0}
                sl = self._slots(world, ev, r, 0, claims)
                out.append(self._item(self.tpl[et]["update"]["revised"], "weather", r, f"{et}/update/revised", sl,
                                      published_t=ev.start_t, refers_to_t=self._refers(world, ev.start_t, ev.end_t),
                                      kind="update", event_id=ev.event_id, is_true=True,
                                      supersedes=f"{ev.event_id}/0", dkey=f"{ev.event_id}/1", claims=claims))
        elif et == "cold_snap":
            # first forecast understates the cold, then an upgrade at the start supersedes it
            p["_understate"] = self.eff["cold_snap"]["l3_initial_anomaly_c"]
            first = self._primary(world, ev, group, dkey=f"{ev.event_id}/0")
            p.pop("_understate")
            out.append(first)
            if first["published_t"] < ev.start_t:
                claims = {"lead": 0}
                sl = self._slots(world, ev, r, 0, claims)
                out.append(self._item(self.tpl[et]["update"]["revised"], "weather", r, f"{et}/update/revised", sl,
                                      published_t=ev.start_t, refers_to_t=self._refers(world, ev.start_t, ev.end_t),
                                      kind="update", event_id=ev.event_id, is_true=True,
                                      supersedes=f"{ev.event_id}/0", dkey=f"{ev.event_id}/1", claims=claims))
        elif et == "wildfire_psps":
            s0, e0 = p["announced"]
            final_lines = p["line_names"]
            p["line_names"] = p["announced_lines"]
            out.append(self._primary(world, ev, group, window=(s0, e0), dkey=f"{ev.event_id}/0"))
            p["line_names"] = final_lines
            rev = p["revision"]
            if rev:
                claims = {"revision": rev["kind"], "announced": [s0, e0], "lines": list(final_lines)}
                sl = self._slots(world, ev, r, 0, claims)
                sl["old_window"] = R.fmt_window(self.common, world.start_hour, s0, e0)
                if rev["kind"] == "expanded":
                    added = sorted(set(final_lines) - set(p["announced_lines"]))
                    sl["added"] = " and ".join(added)
                refers = sorted(set(self._refers(world, s0, e0)) | (set() if rev["kind"] == "cancelled"
                                                                    else set(self._refers(world, ev.start_t, ev.end_t))))
                out.append(self._item(self.tpl[et]["update"][rev["kind"]], "iso", r, f"{et}/update/{rev['kind']}", sl,
                                      published_t=rev["t"], refers_to_t=refers, kind="update", event_id=ev.event_id,
                                      is_true=True, supersedes=f"{ev.event_id}/0", dkey=f"{ev.event_id}/1", claims=claims))
        elif ev.occurs:
            out.append(self._primary(world, ev, group))
        # false contradiction from an unreliable source, same interval as the first notice
        cgroup = "contradiction_miss" if et == "thunderstorm" and not ev.occurs else "contradiction"
        cancelled = et in ("planned_outage", "wildfire_psps") and not ev.occurs
        if out and cgroup in self.tpl[et] and not cancelled and r.random() < self.common["contradiction_prob"]:
            sl = self._slots(world, ev, r, 0, {})
            out.append(self._item(self.tpl[et][cgroup], str(r.choice(sorted(self.tpl[et][cgroup]))), r,
                                  f"{et}/{cgroup}", sl, published_t=out[0]["published_t"],
                                  refers_to_t=out[0]["refers_to_t"], kind="contradiction", event_id=ev.event_id,
                                  is_true=False, claims={"contradicts": out[0].get("dkey") or "primary"}))
        return out

    def _heat_numbers(self, world: World, ev: EventRecord, lead: int, r, claims: dict, distort: float = 0.0) -> dict:
        """Forecast of the window peak: error grows with lead time (and source distortion)."""
        c, fc = self.eff["heatwave"], self.common
        z, t_pk = max(((z, t) for z in ev.params["zones"] for t in range(ev.start_t, ev.end_t + 1)),
                      key=lambda zt: ev.params["anomaly_c"][zt[0]][zt[1]])
        true_anom = ev.params["anomaly_c"][z][t_pk]
        sd = fc["forecast_temp_sd_c"]["base"] + fc["forecast_temp_sd_c"]["per_lead"] * lead + 3.0 * distort
        f_anom = max(0.5, true_anom + sd * float(r.standard_normal()))
        mid = 100 * c["load_mult_per_c"] * f_anom
        hw = fc["forecast_load_halfwidth_pct"]["base"] + fc["forecast_load_halfwidth_pct"]["per_lead"] * lead
        out = {"temp_hi": int(round(world.weather.temp_normal_c[z][t_pk] + f_anom)), "anom": int(round(f_anom)),
               "load_lo": max(0, int(round(mid - hw))), "load_hi": int(round(mid + hw)), "load_mid": int(round(mid))}
        claims.update(true_peak_anom_c=true_anom, true_peak_load_pct=round(100 * c["load_mult_per_c"] * true_anom, 1),
                      stated=dict(out))
        return out

    def _outage_slots(self, ev: EventRecord) -> dict:
        return {"branch": ev.params["branch_name"], "kind": "transformer" if ev.params["is_transformer"] else "line",
                "purpose": ev.params["purpose"]}

    def _echo(self, world: World, ev: EventRecord, r) -> dict:
        src = str(r.choice(["news", "social"]))
        echoes = self.tpl[ev.event_type]["echo"]
        src = src if src in echoes else sorted(echoes)[0]
        rel = self.common["sources"][src]["reliability"]
        pub0 = max(0, ev.start_t - ev.params.get("lead", 0))
        pub = int(r.integers(pub0, max(pub0, min(ev.start_t, world.horizon - 1)) + 1))  # before or at the start
        claims: dict = {"echo_of": ev.event_type, "reliability": rel}
        sl = self._slots(world, ev, r, max(0, ev.start_t - pub), claims,
                         distort=(1 - rel) * self.common["distortion_scale"])
        return self._item(echoes, src, r, f"{ev.event_type}/echo", sl, published_t=pub,
                          refers_to_t=self._refers(world, ev.start_t, ev.end_t), kind="echo", event_id=ev.event_id,
                          is_true=True, claims=claims)

    def _distractor(self, world: World, r) -> dict:
        tpl = self.tpl["distractors"]
        kind = str(r.choice(sorted(tpl)))
        src = str(r.choice(sorted(tpl[kind])))
        z = str(r.choice(sorted(self.info.zones)))
        br = int(r.choice(self.info.zone_branches[z]))
        sl = {**R.zone_slots(self.common, z), "bus": int(r.choice(self.info.zones[z]["buses_orig"])),
              "branch": self.info.branch_name[br], "kind": "transformer" if self.is_tx[br] else "line"}
        return self._item(tpl[kind], src, r, f"distractors/{kind}", sl, published_t=int(r.integers(0, world.horizon)),
                          refers_to_t=[], kind="distractor", event_id=None, is_true=True, is_distractor=True,
                          claims={"distractor_kind": kind})

    def _rumor(self, world: World, r) -> tuple[dict, EventRecord]:
        tpl = self.tpl["rumors"]
        hot = {z for e in world.events if e.event_type == "heatwave" for z in e.params["zones"]}
        out_br = {b for t in range(world.horizon) for b in self.physical_state(world, t).branch_outage_prob}
        kinds = ["false_outage", "rival_bidding"] + (["false_heat"] if len(hot) < len(self.info.zones) else [])
        kind = str(r.choice(kinds))
        t = int(r.integers(0, world.horizon))
        sl: dict = {}
        if kind == "false_outage":
            z = self.info.strategic
            cand = [b for b in self.info.zone_branches[z] if b not in out_br and b not in self.bridges]
            br = int(r.choice(cand))
            sl.update(branch=self.info.branch_name[br], kind="transformer" if self.is_tx[br] else "line",
                      **R.zone_slots(self.common, z))
            ev = EventRecord(event_id="rumor-false_outage", event_type="forced_outage", zone=z, start_t=t,
                             end_t=world.horizon - 1, occurs=False, params={"rumor": True, "branch": br})
        elif kind == "false_heat":
            z = str(r.choice(sorted(set(self.info.zones) - hot)))
            sl.update(temp_hi=int(round(max(world.weather.temp_normal_c[z]) + 8)), **R.zone_slots(self.common, z))
            ev = EventRecord(event_id="rumor-false_heat", event_type="heatwave", zone=z, start_t=t,
                             end_t=world.horizon - 1, occurs=False, params={"rumor": True})
        else:
            z = str(r.choice(sorted(set(self.info.zones) - {self.info.strategic})))
            sl.update(firm=self.common["firm_names"][z])
            ev = EventRecord(event_id="rumor-rival_bidding", event_type="rival_bidding", zone=z, start_t=t,
                             end_t=world.horizon - 1, occurs=False, params={"rumor": True})
        ev.params["slots"] = {k: v for k, v in sl.items()}
        ev.params["kind"] = kind
        d = self._item(tpl[kind], str(r.choice(sorted(tpl[kind]))), r, f"rumors/{kind}", sl, published_t=t,
                       refers_to_t=list(range(t, world.horizon)), kind="rumor", event_id=ev.event_id, is_true=False,
                       claims={"rumor_kind": kind}, dkey=ev.event_id)
        return d, ev

    def _denial(self, world: World, ev: EventRecord, rumor: dict, r) -> dict:
        kind = ev.params["kind"]
        grp = self.tpl["rumors"]["denial"][kind]
        t = int(r.integers(rumor["published_t"] + 1, world.horizon))
        return self._item(grp, sorted(grp)[0], r, f"rumors/denial/{kind}", ev.params["slots"], published_t=t,
                          refers_to_t=rumor["refers_to_t"], kind="denial", event_id=ev.event_id, is_true=True,
                          supersedes=rumor["dkey"], claims={"denies": kind})
