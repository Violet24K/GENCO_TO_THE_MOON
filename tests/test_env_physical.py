"""PhysicalState -> Scenario sampling, ERCOT baseline, OOD flags (no Julia)."""

import numpy as np
import pytest

from genco_market.core.config import Config
from genco_market.core.schemas import PhysicalState
from genco_market.core.seeding import rng
from genco_market.env.grid import GridInfo, nominal_scenario
from genco_market.env.physical import (
    TRAIN_SCALE,
    baseline_zone_load,
    is_connected,
    ood_flags,
    sample_scenario,
)


@pytest.fixture(scope="module")
def info():
    return GridInfo(Config.load())


def _zone_base(info):
    return {z: info.base_pd[v["buses"]].sum() for z, v in info.zones.items()}


def test_baseline_in_envelope_and_follows_ercot(info):
    zb = _zone_base(info)
    f = np.array([[baseline_zone_load(info, h)[z] / zb[z] for z in info.zones] for h in range(0, 8268, 7)])
    assert f.min() >= TRAIN_SCALE[0] - 1e-9 and f.max() <= TRAIN_SCALE[1] + 1e-9
    assert 0.85 < np.median(f) < 1.0  # training median is 0.94
    day = [baseline_zone_load(info, 24 * 200 + h)["C"] for h in range(24)]
    assert max(day) / min(day) > 1.1  # a summer day has a visible daily shape


def test_sampling_is_deterministic(info):
    st = PhysicalState(t=3, zone_load_mean={"C": 1.1}, zone_load_std={"C": 0.05}, branch_outage_prob={43: 0.5})
    base = baseline_zone_load(info, 5000)
    a = sample_scenario(st, info, base, rng(0, "scn", 3))
    b = sample_scenario(st, info, base, rng(0, "scn", 3))
    c = sample_scenario(st, info, base, rng(1, "scn", 3))
    assert a == b and a != c


def test_load_mean_and_qd(info):
    st = PhysicalState(t=0, zone_load_mean={"C": 1.2}, bus_load_noise_std=0.0)
    s = sample_scenario(st, info, _zone_base(info), rng(0))
    C = info.zones["C"]["buses"]
    A = info.zones["A"]["buses"]
    assert np.sum(np.asarray(s.bus_pd)[C]) == pytest.approx(1.2 * info.base_pd[C].sum())
    assert np.sum(np.asarray(s.bus_pd)[A]) == pytest.approx(info.base_pd[A].sum())
    np.testing.assert_allclose(np.asarray(s.bus_qd)[C], 1.2 * info.base_qd[C])
    # lognormal mean matches the requested mean
    st = PhysicalState(t=0, zone_load_mean={"C": 1.1}, zone_load_std={"C": 0.1}, bus_load_noise_std=0.0)
    m = [np.sum(np.asarray(sample_scenario(st, info, _zone_base(info), rng(i)).bus_pd)[C]) for i in range(400)]
    assert np.mean(m) / info.base_pd[C].sum() == pytest.approx(1.1, abs=0.015)


def test_renewables_clip_and_log(info):
    b = 44
    st = PhysicalState(t=0, renewable_mw={b: info.base_pd[b] + 50.0}, bus_load_noise_std=0.0)
    log = {}
    s = sample_scenario(st, info, _zone_base(info), rng(0), log)
    assert s.bus_pd[b] == 0.0 and log["renewable_clipped_mw"] == pytest.approx(50.0)
    assert "negative_net_load" in ood_flags(st, info)


def test_outages_never_island(info):
    radial = [b for b in range(info.n_branch) if not is_connected(info, [int(i != b) for i in range(info.n_branch)])]
    assert radial, "case118 has radial branches"
    st = PhysicalState(t=0, branch_outage_prob={radial[0]: 1.0, 50: 1.0, 43: 0.5})
    for i in range(20):
        s = sample_scenario(st, info, _zone_base(info), rng(i))
        assert s.branch_status[radial[0]] == 1 and s.branch_status[50] == 0
        assert is_connected(info, s.branch_status)


def test_costs_ratings_pmax(info):
    st = PhysicalState(
        t=0,
        rival_cost_mult={"A": 1.3, info.strategic: 2.0},
        gen_cost_mult={29: 1.5},
        branch_rate_mult={50: 0.5},
        gen_pmax_mult={20: 0.5},
        gen_outage_prob={4: 1.0},
    )
    s = sample_scenario(st, info, _zone_base(info), rng(0))
    c1 = np.asarray(s.gen_c1)
    np.testing.assert_allclose(c1[info.firm_gens["A"]], 1.3 * info.base_c1[info.firm_gens["A"]])
    np.testing.assert_allclose(c1[info.firm_gens[info.strategic]], info.base_c1[info.firm_gens[info.strategic]])  # strategic untouched
    assert c1[29] == pytest.approx(1.5 * info.base_c1[29])
    assert s.branch_rate_a[50] == pytest.approx(0.5 * info.base_rate_a[50])
    assert s.gen_pmax[20] == pytest.approx(0.5 * info.base_pmax[20])
    assert s.gen_status[4] == 0
    flags = ood_flags(s, info)
    assert {"rate_a_changed", "rival_cost_scaled", "pmax_changed"} <= set(flags)
    assert {"rate_a_changed", "rival_cost_scaled", "pmax_changed"} <= set(ood_flags(st, info))


def test_ood_flags_load_and_outages(info):
    assert ood_flags(nominal_scenario(info), info) == []
    assert ood_flags(PhysicalState(t=0), info) == []
    assert ood_flags(PhysicalState(t=0, gen_cost_mult={29: 1.2}), info) == ["rival_cost_scaled"]
    hot = PhysicalState(t=0, zone_load_mean={"C": 1.4})
    assert "load_outside_envelope:C" in ood_flags(hot, info)
    s = sample_scenario(hot, info, _zone_base(info), rng(0))
    assert "load_outside_envelope" in ood_flags(s, info)
    many = PhysicalState(t=0, branch_outage_prob={b: 1.0 for b in range(10, 60, 4)})
    assert "outages_gt_10" in ood_flags(many, info)


def test_system_load_factor_is_common(info):
    st = PhysicalState(t=0, system_load_std=0.05, bus_load_noise_std=0.0)
    r = rng(3)
    totals, zone_a = [], []
    for _ in range(400):
        s = sample_scenario(st, info, _zone_base(info), r)
        pd_ = np.asarray(s.bus_pd)
        totals.append(pd_.sum())
        zone_a.append(pd_[info.zones["A"]["buses"]].sum())
    # all zones move together: total load spread ~ 5%, zone A perfectly correlated with total
    assert 0.035 < np.std(totals) / np.mean(totals) < 0.065
    assert np.corrcoef(totals, zone_a)[0, 1] > 0.99
