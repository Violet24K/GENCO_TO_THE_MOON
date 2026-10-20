import json
import re

import pytest

from genco_market.core.config import Config
from genco_market.core.schemas import PhysicalState
from genco_market.env.grid import GridInfo
from genco_market.semantics.events import EVENT_TYPES, IMPLEMENTED, keeps_connected, load_effects
from genco_market.semantics.world import INDIRECT, LEVELS, World, WorldGenerator

SUMMER = 5390  # Mon Aug 12, 14:00
WINTER = 300  # Jan 13, 12:00
H = 4


@pytest.fixture(scope="module")
def info():
    return GridInfo(Config.load())


def make(info, level="L2", seed=0, **sem):
    cfg = Config.load()
    cfg.seed = seed
    cfg.semantics.level = level
    for k, v in sem.items():
        setattr(cfg.semantics, k, v)
    return WorldGenerator(cfg, info)


def dump_episode(gen, w):
    return json.dumps([
        [gen.bulletin(w, t).model_dump(), gen.labels(w, t).model_dump(), gen.physical_state(w, t).model_dump()]
        for t in range(w.horizon)
    ], sort_keys=True)


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3"])
def test_same_seed_byte_identical(info, level):
    w1 = make(info, level, seed=7).sample_world(3, H, SUMMER)
    g2 = make(info, level, seed=7)
    w2 = g2.sample_world(3, H, SUMMER)
    assert w1.model_dump_json() == w2.model_dump_json()
    assert dump_episode(g2, w1) == dump_episode(g2, w2)
    w3 = make(info, level, seed=8).sample_world(3, H, SUMMER)
    assert w3.model_dump_json() != w1.model_dump_json()


def test_freeze_and_replay(info, tmp_path):
    gen = make(info, "L2", seed=1)
    w = gen.sample_world(0, H, SUMMER)
    p = tmp_path / "world.json"
    p.write_text(w.model_dump_json())
    w2 = World.model_validate_json(p.read_text())
    assert w2.model_dump_json() == w.model_dump_json()
    assert dump_episode(gen, w2) == dump_episode(gen, w)


def test_physics_independent_of_level(info):
    states = {}
    for ep in range(10):
        for level in LEVELS:
            gen = make(info, level, seed=4, event_types=IMPLEMENTED, max_events=3)
            w = gen.sample_world(ep, H, SUMMER)
            states[level] = [gen.physical_state(w, t).model_dump_json() for t in range(H)]
        assert states["L0"] == states["L1"] == states["L2"] == states["L3"]


@pytest.mark.parametrize("etype", IMPLEMENTED)
@pytest.mark.parametrize("level", ["L1", "L2", "L3"])
def test_each_type_alone(info, etype, level):
    gen = make(info, level)
    for seed_ep in range(8):
        w = gen.sample_world(seed_ep, H, SUMMER, event_types=[etype])
        real = [e for e in w.events if not e.event_id.startswith("rumor")]
        assert [e.event_type for e in real] == [etype]
        ev = real[0]
        prim = [i for i in w.items if i.kind == "primary"]
        cancelled = etype in ("planned_outage", "wildfire_psps") and not ev.occurs
        if cancelled and level != "L3":
            assert prim == []  # L1/L2 only show the final schedule
        else:
            assert len(prim) == 1 and prim[0].event_id == ev.event_id
            if not (level == "L3" and etype in INDIRECT):
                assert prim[0].source == EVENT_TYPES[etype].primary_source
        if ev.occurs:
            s = gen.physical_state(w, ev.start_t)
            changed = {
                "heatwave": lambda: any(v > 1 for v in s.zone_load_mean.values()) and s.branch_rate_mult and s.gen_pmax_mult,
                "planned_outage": lambda: s.branch_outage_prob.get(ev.params["branch"]) == 1.0,
                "thunderstorm": lambda: len(s.branch_outage_prob) >= 1,
                "gas_price": lambda: any(v > 1 for v in s.gen_cost_mult.values()),
                "major_event": lambda: s.zone_load_mean[ev.zone] > 1,
                "holiday": lambda: all(s.zone_load_mean[z] < 1 for z in ev.params["zones"]),
                "data_center": lambda: s.renewable_mw == {ev.params["bus"]: -ev.params["mw"]},
                "cold_snap": lambda: all(s.zone_load_mean[z] > 1 for z in ev.params["zones"]) and s.common_shocks
                and any(v > 1 for v in s.gen_cost_mult.values()),
                "wildfire_psps": lambda: all(s.branch_outage_prob.get(b) == 1.0 for b in ev.params["lines"]),
                "high_wind": lambda: sum(s.renewable_mw.values()) > 0 and s.renewable_std_mw,
                "solar_drop": lambda: sum(s.renewable_mw.values()) < 0 and s.renewable_std_mw,
                "demand_response": lambda: all(v > 1 for v in s.zone_load_mean.values()),
                "market_power_review": lambda: s.market_rules.mitigation_markup_cap == ev.params["cap"],
                "geomagnetic_storm": lambda: len(s.common_shocks) == 1 and s.common_shocks[0].branch_prob,
                "cyber_physical": lambda: len(s.common_shocks) == 1 and len(s.common_shocks[0].branch_prob) >= 2,
            }[etype]()
            assert changed, (etype, s)


def test_unimplemented_raise(info):
    gen = make(info)
    with pytest.raises(NotImplementedError):
        gen.sample_world(0, H, SUMMER, event_types=["flood"])
    with pytest.raises(NotImplementedError):
        make(info, "L4").sample_world(0, H, SUMMER)
    assert len(EVENT_TYPES) > 20 and set(IMPLEMENTED) == {
        "heatwave", "planned_outage", "thunderstorm", "gas_price", "major_event", "holiday", "data_center",
        "cold_snap", "wildfire_psps", "high_wind", "solar_drop", "demand_response", "market_power_review",
        "geomagnetic_storm", "cyber_physical"}


def test_no_events_is_neutral(info):
    gen = make(info, "L1")
    w = gen.sample_world(0, H, SUMMER, event_types=[])
    for t in range(H):
        s = gen.physical_state(w, t)
        assert set(s.zone_load_mean.values()) == {1.0} and set(s.zone_load_std.values()) == {0.03}
        assert not s.branch_outage_prob and not s.branch_rate_mult and not s.gen_pmax_mult
        assert set(s.rival_cost_mult.values()) == {1.0} and info.strategic not in s.rival_cost_mult
    assert gen.bulletin(w, 0).items == []


def test_bounds_and_connectivity(info):
    b = load_effects()["bounds"]
    for level in ["L1", "L2", "L3"]:
        gen = make(info, level, max_events=3, event_types=IMPLEMENTED)
        for ep in range(40):
            for start in (SUMMER, WINTER, 4000):
                w = gen.sample_world(ep, 6, start)
                for t in range(w.horizon):
                    s = gen.physical_state(w, t)
                    assert all(b["load_mult"][0] <= v <= b["load_mult"][1] for v in s.zone_load_mean.values())
                    assert all(b["rate_mult"][0] <= v <= b["rate_mult"][1] for v in s.branch_rate_mult.values())
                    assert all(b["pmax_mult"][0] <= v <= 1.0 for v in s.gen_pmax_mult.values())
                    assert all(0 < v <= 1 for v in s.branch_outage_prob.values())
                    assert len(s.branch_outage_prob) <= b["max_outage_branches"]
                    assert keeps_connected(info, set(s.branch_outage_prob))
                    PhysicalState.model_validate(s.model_dump())


def test_heatwave_not_sampled_in_winter(info):
    gen = make(info, "L1", max_events=4)
    for ep in range(30):
        w = gen.sample_world(ep, H, WINTER)
        assert "heatwave" not in [e.event_type for e in w.events]


def test_labels_traceable(info):
    for level in LEVELS:
        gen = make(info, level, event_types=IMPLEMENTED, max_events=3)
        for ep in range(10):
            w = gen.sample_world(ep, H, SUMMER)
            ev_ids = {e.event_id for e in w.events}
            seen = set()
            for t in range(H):
                bul, lab = gen.bulletin(w, t), gen.labels(w, t)
                assert [i.item_id for i in bul.items] == [l.item_id for l in lab.items]
                for it, l in zip(bul.items, lab.items):
                    assert it.published_t == t and it.item_id not in seen
                    seen.add(it.item_id)
                    if l.is_distractor:
                        assert l.event_id is None and level in ("L2", "L3")
                    elif level != "L0":
                        assert l.event_id in ev_ids
                for e in lab.active_events:
                    assert e.occurs and e.start_t <= t <= e.end_t
            assert len(seen) == len(w.items)


def test_level_contents(info):
    # L0: JSON of the true state
    gen = make(info, "L0")
    w = gen.sample_world(0, H, SUMMER)
    for t in range(H):
        (it,) = gen.bulletin(w, t).items
        assert PhysicalState.model_validate_json(it.text) == gen.physical_state(w, t)
    # L1: one clean item per event, no distractors/rumors
    gen = make(info, "L1")
    for ep in range(20):
        w = gen.sample_world(ep, H, SUMMER)
        assert len(w.items) == sum(e.occurs or e.event_type == "thunderstorm" for e in w.events) + sum(
            e.event_type == "heatwave" for e in w.events)  # + ISO heat advisory
        assert all(i.kind in ("primary", "advisory") and i.is_true and i.source in ("iso", "weather", "news") for i in w.items)
    # L2: distractors, unreliable sources and false rumors appear across episodes
    gen = make(info, "L2")
    items = [i for ep in range(20) for i in gen.sample_world(ep, H, SUMMER).items]
    assert any(i.is_distractor for i in items)
    assert any(not i.is_true and i.source in ("social", "rival") for i in items)
    assert any(i.kind == "echo" for i in items)


def test_text_identifiers(info):
    gen = make(info, "L1")
    for ep in range(20):
        w = gen.sample_world(ep, H, SUMMER, event_types=["planned_outage", "heatwave"])
        po = next(e for e in w.events if e.event_type == "planned_outage")
        hw = next(e for e in w.events if e.event_type == "heatwave")
        txt = {i.event_id: i.text for i in w.items}
        if not po.occurs:
            continue
        assert po.params["branch_name"] in txt[po.event_id]
        assert info.branch_by_name[po.params["branch_name"]]
        if hw.zone:
            assert f"Zone {hw.zone}" in txt[hw.event_id]
        assert re.search(r"t=\d", txt[po.event_id])


def test_hedge_matches_hidden_prob(info):
    gen = make(info, "L2")
    n = 0
    for ep in range(30):
        w = gen.sample_world(ep, H, SUMMER, event_types=["thunderstorm"])
        for it in w.items:
            if it.kind == "primary":
                lo, hi = it.claims["prob_range"]
                assert lo <= it.claims["prob"] < hi
                n += 1
    assert n == 30


def test_planned_outage_key_case(info):
    gen = make(info, "L1")
    branches = [gen.sample_world(ep, H, SUMMER, event_types=["planned_outage"]).events[0].params["branch"] for ep in range(40)]
    assert branches.count(info.named_branches["tx_38_37"]) >= 3
    assert sum(b in info.branch_by_name["65-68"] for b in branches) >= 3
