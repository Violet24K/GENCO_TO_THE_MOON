"""Layer 4 (outage coordination): config, episode sampling, request, options."""

import numpy as np
import pytest

from genco_market.core.config import Config
from genco_market.core.schemas import Bulletin
from genco_market.env.grid import GridInfo, nominal_scenario
from genco_market.env.outage import (DEFER, L4WorldGenerator, l4_event_types, l4_start_hour, load_l4, option_costs,
                                        option_label, request_item, sample_request, social_cost, with_line_out)
from genco_market.semantics.render import clock
from genco_market.semantics.world import WorldGenerator


@pytest.fixture(scope="module")
def setup():
    cfg, l4 = load_l4("l4.yaml")
    cfg.semantics.paraphrase = False
    info = GridInfo(cfg)
    return cfg, l4, info, L4WorldGenerator(cfg, info)


def test_config(setup):
    cfg, l4, _, _ = setup
    assert cfg.episode.horizon == l4.horizon == 8
    assert "planned_outage" not in cfg.semantics.event_types
    assert set(l4.candidates) == {"65-68", "38-37", "49-54", "49-66", "56-57", "15-33"}
    assert l4.durations == [1, 2] and l4.deferral_cost > 0


def test_start_hours_daytime_and_deterministic(setup):
    cfg, l4, _, _ = setup
    hs = [l4_start_hour(cfg.seed, ep, l4) for ep in range(60)]
    assert hs == [l4_start_hour(cfg.seed, ep, l4) for ep in range(60)]
    for h in hs:
        c0, c_end = clock(h, 0), clock(h, l4.horizon)
        assert 8 <= c0.hour <= 10 and c_end.hour <= 18 and c_end.date() == c0.date()
        assert h < 8268


def test_events_biased_no_planned_outage(setup):
    cfg, l4, _, gen = setup
    lists = [l4_event_types(cfg.seed, ep, l4, gen, l4_start_hour(cfg.seed, ep, l4)) for ep in range(80)]
    assert all(lst for lst in lists)
    assert not any("planned_outage" in lst for lst in lists)
    hazard = np.mean([bool({"heatwave", "thunderstorm"} & set(lst)) for lst in lists])
    assert hazard > 0.6
    for ep, lst in enumerate(lists):  # heat waves only in heat season
        if "heatwave" in lst:
            from genco_market.env.outage import heat_eligible
            assert heat_eligible(gen, l4_start_hour(cfg.seed, ep, l4))


def test_all_events_announced_at_t0(setup):
    cfg, l4, info, gen = setup
    c = cfg.model_copy(deep=True)
    c.semantics.level = "L1"
    g1 = L4WorldGenerator(c, info)
    for ep in range(10):
        sh = l4_start_hour(cfg.seed, ep, l4)
        w = g1.sample_world(ep, 8, sh, event_types=l4_event_types(cfg.seed, ep, l4, g1, sh))
        for it in w.items:
            if it.kind in ("primary", "advisory"):
                assert it.published_t == 0, (ep, it.template_key)
    # the base generator would publish late events later
    base = WorldGenerator(c, info)
    late = 0
    for ep in range(10):
        sh = l4_start_hour(cfg.seed, ep, l4)
        w = base.sample_world(ep, 8, sh, event_types=l4_event_types(cfg.seed, ep, l4, base, sh))
        late += sum(it.published_t > 0 for it in w.items if it.kind == "primary")
    assert late > 0


def test_request(setup):
    cfg, l4, info, _ = setup
    reqs = [sample_request(cfg.seed, ep, l4, info, l4_start_hour(cfg.seed, ep, l4)) for ep in range(40)]
    assert reqs[3].to_dict() == sample_request(cfg.seed, 3, l4, info, l4_start_hour(cfg.seed, 3, l4)).to_dict()
    for r in reqs:
        assert r.line in l4.candidates and r.branch in info.branch_by_name[r.line]
        assert r.duration in (1, 2) and 0 <= r.s_req <= l4.horizon - r.duration
        assert r.branch_name in r.text
        opts = r.options(l4.horizon)
        assert opts[-1] == DEFER and len(opts) == l4.horizon - r.duration + 2
        it = request_item(r, l4.horizon)
        assert it.published_t == 0 and it.source == "iso"
        Bulletin(t=0, items=[it])
    assert len({r.line for r in reqs}) >= 4


def test_option_costs_and_line_out(setup):
    cfg, l4, info, _ = setup
    r = sample_request(cfg.seed, 0, l4, info, l4_start_hour(cfg.seed, 0, l4))
    r.duration = 2
    delta = np.arange(8, dtype=float) * 10
    oc = option_costs(delta, r, 55.0)
    assert oc[:7].tolist() == [10, 30, 50, 70, 90, 110, 130] and oc[-1] == 55.0
    assert option_label(DEFER, 0) == "defer"
    scn = nominal_scenario(info)
    out = with_line_out(scn, r.branch)
    assert out.branch_status[r.branch] == 0 and scn.branch_status[r.branch] == 1
    assert sum(out.branch_status) == sum(scn.branch_status) - 1
    pg = np.zeros(info.n_gen)
    pg[0] = 100
    assert social_cost(pg, scn, 2.0) == pytest.approx(100 * scn.gen_c1[0] + 4000)


def test_l4_config_is_a_valid_config():
    # the l4: block must not break the generic loader
    cfg = Config.load("l4.yaml")
    assert cfg.episode.horizon == 8
