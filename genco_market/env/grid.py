"""Base grid, zones, and Scenario -> datakit Network conversion.

This is the single place that knows how a Scenario (and a bid) maps onto the
datakit ``Network`` arrays. Both the market (env/market.py) and the GENCO
adapter (surrogate/) must build networks through ``scenario_to_network``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from gridfm_datakit.network import Network, load_net_from_pglib
from gridfm_datakit.utils.idx_brch import BR_STATUS, RATE_A
from gridfm_datakit.utils.idx_bus import PD, QD
from gridfm_datakit.utils.idx_cost import COST, NCOST
from gridfm_datakit.utils.idx_gen import GEN_STATUS, PMAX

from genco_market.core.config import Config
from genco_market.core.schemas import Scenario

C1 = COST  # all case118 gencosts are linear (NCOST=2): columns [c1, c0]


@lru_cache(maxsize=4)
def base_network(grid: str = "case118_ieee") -> Network:
    net = load_net_from_pglib(grid)
    assert np.all(net.gencosts[:, NCOST] == 2), "expected linear costs"
    return net


class GridInfo:
    """Static facts about the grid + firm partition, all 0-based indices."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.net = base_network(cfg.grid)
        f = cfg.firms()
        self.strategic = f["strategic_firm"]
        self.zones: dict[str, dict] = f["zones"]
        self.named_branches: dict[str, int] = f.get("named_branches", {})
        n_bus = self.net.buses.shape[0]
        self.bus_zone = np.empty(n_bus, dtype=object)
        for z, v in self.zones.items():
            self.bus_zone[v["buses"]] = z
        self.firm_gens: dict[str, list[int]] = {z: v["gens"] for z, v in self.zones.items()}
        self.gen_firm: dict[int, str] = {g: z for z, gs in self.firm_gens.items() for g in gs}
        self.base_pd = self.net.buses[:, PD].copy()
        self.base_qd = self.net.buses[:, QD].copy()
        self.base_c1 = self.net.gencosts[:, C1].copy()
        self.base_pmax = self.net.gens[:, PMAX].copy()
        self.base_rate_a = self.net.branches[:, RATE_A].copy()
        orig = self.net.original_bus_indices
        from gridfm_datakit.utils.idx_brch import F_BUS, T_BUS

        # Unique names: "38-37"; parallel circuits get "#k" ("49-54#1", "49-54#2").
        # branch_by_name maps both directions, and a bare "49-54" maps to all its circuits.
        self.branch_name: list[str] = []
        self.branch_by_name: dict[str, list[int]] = {}
        pairs = [(int(orig[int(br[F_BUS])]), int(orig[int(br[T_BUS])])) for br in self.net.branches]
        count: dict[frozenset, int] = {}
        for f, t in pairs:
            count[frozenset((f, t))] = count.get(frozenset((f, t)), 0) + 1
        seen: dict[frozenset, int] = {}
        for i, (f, t) in enumerate(pairs):
            key = frozenset((f, t))
            seen[key] = seen.get(key, 0) + 1
            suffix = f"#{seen[key]}" if count[key] > 1 else ""
            self.branch_name.append(f"{f}-{t}{suffix}")
            for a, b in ((f, t), (t, f)):
                self.branch_by_name.setdefault(f"{a}-{b}", []).append(i)
                if suffix:
                    self.branch_by_name.setdefault(f"{a}-{b}{suffix}", []).append(i)
        fb, tb = self.net.branches[:, F_BUS].astype(int), self.net.branches[:, T_BUS].astype(int)
        # branches touching a zone (internal or boundary)
        self.zone_branches: dict[str, list[int]] = {
            z: [i for i in range(len(fb)) if self.bus_zone[fb[i]] == z or self.bus_zone[tb[i]] == z]
            for z in self.zones
        }

    @property
    def n_bus(self) -> int:
        return self.net.buses.shape[0]

    @property
    def n_gen(self) -> int:
        return self.net.gens.shape[0]

    @property
    def n_branch(self) -> int:
        return self.net.branches.shape[0]

    def bid_c1(self, scn: Scenario, markup: float | dict[str, float]) -> np.ndarray:
        """Bid cost vector. ``markup`` is the strategic firm's markup (float) or a
        bid profile {firm: markup} (firms not listed bid cost)."""
        c1 = np.asarray(scn.gen_c1, dtype=float).copy()
        prof = markup if isinstance(markup, dict) else {self.strategic: markup}
        for f, m in prof.items():
            c1[self.firm_gens[f]] *= m
        return c1


def scenario_to_network(scn: Scenario, bid_c1: np.ndarray | None = None, grid: str = "case118_ieee") -> Network:
    """Fresh Network for one realization. ``bid_c1`` overrides scn.gen_c1
    (the costs the ISO sees); defaults to true costs."""
    net = base_network(grid).copy_for_perturbation()
    net.buses[:, PD] = scn.bus_pd
    net.buses[:, QD] = scn.bus_qd
    net.branches[:, RATE_A] = scn.branch_rate_a
    net.gens[:, PMAX] = scn.gen_pmax
    net.gencosts[:, C1] = scn.gen_c1 if bid_c1 is None else bid_c1
    off_br = np.where(np.asarray(scn.branch_status) == 0)[0]
    if len(off_br):
        net.branches[off_br, BR_STATUS] = 0
    off_gen = np.where(np.asarray(scn.gen_status) == 0)[0]
    if len(off_gen):
        net.deactivate_gens(off_gen)  # also turns PV buses without gens into PQ
    return net


def nominal_scenario(info: GridInfo, t: int = 0) -> Scenario:
    """The unperturbed PGLib case as a Scenario (handy for tests)."""
    return Scenario(
        t=t,
        seed=0,
        bus_pd=info.base_pd.tolist(),
        bus_qd=info.base_qd.tolist(),
        branch_status=[1] * info.n_branch,
        branch_rate_a=info.base_rate_a.tolist(),
        gen_status=info.net.gens[:, GEN_STATUS].astype(int).tolist(),
        gen_pmax=info.base_pmax.tolist(),
        gen_c1=info.base_c1.tolist(),
    )
