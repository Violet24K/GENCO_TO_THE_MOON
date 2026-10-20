"""Gym-like market environment.

One episode = one hidden World (semantics) over ``horizon`` intervals. Each
step: realize the true scenario of interval t (independent of the bid, so
every agent faces the same realizations), clear the market with the bid via
AC-OPF, settle, and publish the next bulletin.
"""

from __future__ import annotations

from genco_market.core.config import Config
from genco_market.core.schemas import (
    BidAction,
    Bulletin,
    ClearingResult,
    MarketRules,
    Observation,
    PhysicalState,
    PublicRoundResult,
    Scenario,
)
from genco_market.core.seeding import rng
from genco_market.env.grid import GridInfo
from genco_market.env.physical import baseline_zone_load, sample_scenario
from genco_market.env.pool import SolverPool, TaskError


class MarketEnv:
    def __init__(self, cfg: Config, info: GridInfo, world_gen, pool: SolverPool):
        self.cfg, self.info, self.gen, self.pool = cfg, info, world_gen, pool
        self.world = None

    # ------------------------------------------------------------- truth
    def baseline(self, t: int) -> dict[str, float]:
        return baseline_zone_load(self.info, self.world.start_hour + t)

    def true_state(self, t: int) -> PhysicalState:
        return self.gen.physical_state(self.world, t)

    def rules(self, t: int) -> MarketRules:
        return self.true_state(t).market_rules

    def realized(self, t: int) -> Scenario:
        scn = sample_scenario(self.true_state(t), self.info, self.baseline(t),
                              rng(self.cfg.seed, "realize", self.world.episode_id, t))
        scn.t = t
        return scn

    # ------------------------------------------------------------- loop
    def reset(self, episode_id: int, start_hour: int, event_types: list[str] | None = None, world=None) -> Observation:
        self.world = world or self.gen.sample_world(episode_id, self.cfg.episode.horizon, start_hour,
                                                    event_types=event_types)
        self.t = 0
        self.history: list[PublicRoundResult] = []
        self.seen_items = []
        return self._obs()

    def _obs(self) -> Observation:
        self.seen_items = self.seen_items + self.gen.bulletin(self.world, self.t).items
        a = self.cfg.agent
        st = self.true_state(self.t)
        own = {g: float(self.info.base_c1[g] * st.gen_cost_mult.get(g, 1.0)) for g in self.info.firm_gens[self.info.strategic]}
        return Observation(t=self.t, bulletin=Bulletin(t=self.t, items=list(self.seen_items)),
                           baseline_zone_load_mw=self.baseline(self.t), own_marginal_cost=own, history=list(self.history),
                           deadline_s=a.deadline_s, compute_budget_s=a.compute_budget_s)

    def clear(self, t: int, markup: float) -> ClearingResult:
        out = self.pool.map("ac", [(self.realized(t), markup, self.rules(t))])[0]
        if isinstance(out, TaskError) or out is None:
            raise RuntimeError(f"clearing task failed: {out}")
        return out

    def step(self, action: BidAction):
        t = self.t
        scn = self.realized(t)
        res = self.clear(t, action.markup)
        firm = self.info.strategic
        gens = self.info.firm_gens[firm]
        bus = self.info.net.gens[:, 0].astype(int)  # GEN_BUS column
        self.history.append(PublicRoundResult(
            t=t, markup=action.markup,
            own_dispatch_mw={g: res.pg[g] for g in gens},
            own_lmp={g: res.lmp[bus[g]] for g in gens},
            profit=res.firm_profit.get(firm, float("nan")),
            system_load_mw=float(sum(scn.bus_pd)),
        ))
        reward = res.firm_profit.get(firm, float("nan"))
        self.t += 1
        done = self.t >= self.world.horizon
        info = {"scenario": scn, "clearing": res, "state": self.true_state(t)}
        return (None if done else self._obs()), reward, done, info
