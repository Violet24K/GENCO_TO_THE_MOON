"""Run agents through episodes and log one row per (agent, episode, interval).

Every agent faces the same worlds and the same realized scenarios. Expected
regret is scored against the oracle table (true P_t, M AC-OPF scenarios).

    python experiments/run_episode.py --config mvp.yaml --agents cost fixed:1.5 l0/ac l0/mf_genco llm/mf_genco
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from genco_market.core.config import REPO, Config, scratch_dir
from genco_market.core.seeding import sample_start_hour
from genco_market.env.env import MarketEnv
from genco_market.env.grid import GridInfo
from genco_market.env.pool import SolverPool
from genco_market.eval.oracle import oracle_table
from genco_market.semantics.world import WorldGenerator

SUMMER_PEAK = 5390  # Mon Aug 12 2024, 14:00



def replayed_beliefs(info, path):
    """(episode, t) -> (belief, extraction, latency) from saved LLM judgments (data/llm_judgments_L3.jsonl: one
    line per decision with the LLM's structured extraction and its latency), so llm/* agents reuse exactly the
    LLM judgments of our runs (no API calls). Intervals whose LLM call failed are absent and fall back to the
    prior, as they did originally."""
    from genco_market.agent.perception import BeliefExtraction, to_distribution

    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        # judgments saved before the extension-II events lack their fields; those episodes have none of them
        old = {"gas_supply_risk": [], "transformer_risk": [], "substation_risk": [], "renewable_changes": [],
               "markup_cap": None}
        ext = BeliefExtraction.model_validate(old | r["extraction"])
        out[(r["episode"], r["t"])] = (to_distribution(ext, info, r["t"]), ext, r["t_belief_s"])
    return out


def build_agents(specs, info, cfg, pool, env, need_models, shared_beliefs=False, genco_size="small", replay=None):
    from genco_market.agent.pipeline_agent import PipelineAgent

    models, llm = {}, None
    if need_models & {"ac"}:
        from genco_market.surrogate.profit_models import ACModel
        models["ac"] = ACModel(pool, info)
    if need_models & {"dc"}:
        from genco_market.surrogate.profit_models import DCModel
        models["dc"] = DCModel(pool, "highs", info=info)
    if need_models & {"genco"}:
        from genco_market.surrogate.genco_service import GencoService
        from genco_market.surrogate.profit_models import GencoModel
        models["genco"] = GencoModel(GencoService(size=genco_size), info=info)
    if any(s.startswith("llm/") for s in specs) and replay is None:
        from genco_market.agent.llm import LLM
        llm = LLM()
    agents, shared = [], (replay if replay is not None else ({} if shared_beliefs else None))
    for s in specs:
        belief, method = s.split("/") if "/" in s else ("prior", s)
        agents.append(PipelineAgent(info, cfg.agent.markups, belief, method, models=models, llm=llm,
                                    seed=cfg.seed, oracle_fn=env.true_state, shared_beliefs=shared))
    return agents, llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mvp.yaml")
    ap.add_argument("--agents", nargs="+", default=["cost", "fixed:1.5"])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--level", default=None)
    ap.add_argument("--events", nargs="*", default=None)
    ap.add_argument("--start-hours", nargs="*", type=int, default=None)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--budget", type=float, default=None, help="override agent.compute_budget_s")
    ap.add_argument("--oracle-scenarios", type=int, default=None)
    ap.add_argument("--episode-offset", type=int, default=0, help="run episodes offset..offset+episodes-1")
    ap.add_argument("--shared-belief", action="store_true", help="llm/* agents share one LLM call per interval")
    ap.add_argument("--genco-size", default="small", help="GENCO checkpoint, e.g. small_ft2 (fine-tuned)")
    ap.add_argument("--replay-beliefs", default=None,
                    help="saved LLM judgments that llm/* agents reuse, e.g. data/llm_judgments_L3.jsonl (no API calls)")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    if a.level:
        cfg.semantics.level = a.level
    if a.budget:
        cfg.agent.compute_budget_s = a.budget
    if a.oracle_scenarios:
        cfg.oracle_scenarios = a.oracle_scenarios
    n_ep = a.episodes or cfg.episode.n_episodes
    if any(s.startswith("l0/") for s in a.agents) and cfg.semantics.level != "L0":
        raise SystemExit("l0/* agents read the L0 JSON bulletin; run with --level L0 (or use oracle/*)")
    info = GridInfo(cfg)
    para = None
    if cfg.semantics.paraphrase and cfg.semantics.level in ("L2", "L3"):
        from genco_market.semantics.paraphrase import CachedLLMParaphraser
        para = CachedLLMParaphraser(offline=True)
    gen = WorldGenerator(cfg, info, paraphraser=para)
    methods = {s.split("/")[-1] for s in a.agents}
    need = {m for m in ("ac", "genco", "dc") if methods & {m, f"mf_{m}"}}
    if methods & {"mf_both", "mf_route"}:
        need |= {"genco", "dc"}
    if any(m.startswith("mf_") for m in methods):
        need |= {"ac"}
    rows = []
    with SolverPool(cfg.solver_workers, cfg) as pool:
        env = MarketEnv(cfg, info, gen, pool)
        replay = replayed_beliefs(info, a.replay_beliefs) if a.replay_beliefs else None
        agents, llm = build_agents(a.agents, info, cfg, pool, env, need, shared_beliefs=a.shared_belief,
                                   genco_size=a.genco_size, replay=replay)
        for ep in range(a.episode_offset, a.episode_offset + n_ep):
            start = a.start_hours[ep % len(a.start_hours)] if a.start_hours else sample_start_hour(cfg.seed, ep)
            env.reset(ep, start, event_types=a.events)
            world = env.world
            events = [(e.event_type, e.zone, e.start_t, e.end_t, e.occurs) for e in world.events]
            tables = {t: oracle_table(env, t, cfg.agent.markups, cfg.oracle_scenarios) for t in range(world.horizon)}
            for agent in agents:
                agent.reset(ep)
                obs = env.reset(ep, start, world=world)
                done = False
                while not done:
                    t = obs.t
                    t0 = time.perf_counter()
                    action, trace = agent.act(obs)
                    wall = time.perf_counter() - t0 + float(trace.get("belief_charge_s", 0.0))
                    obs, reward, done, info_ = env.step(action)
                    tab = tables[t]
                    k = tab["markups"].index(action.markup) if action.markup in tab["markups"] else None
                    best = float(np.max(tab["mean"]))
                    exp_profit = tab["mean"][k] if k is not None else float("nan")
                    res = info_["clearing"]
                    rows.append(dict(
                        tag=a.tag, level=cfg.semantics.level, budget_s=cfg.agent.compute_budget_s, episode=ep, t=t, agent=agent.name, markup=action.markup,
                        fallback=action.fallback_used, wall_s=wall, deadline_s=cfg.agent.deadline_s,
                        deadline_missed=wall > cfg.agent.deadline_s, realized_profit=reward,
                        expected_profit=exp_profit, oracle_profit=best,
                        oracle_markup=tab["markups"][int(np.argmax(tab["mean"]))],
                        regret=best - exp_profit, oracle_n=tab["n_ok"], clearing_status=res.status,
                        load_shed_mw=res.load_shed_mw, events=json.dumps(events),
                        trace=json.dumps(trace, default=str),
                    ))
                    print(f"ep{ep} t{t} {agent.name:18s} m={action.markup:.1f} wall={wall:5.2f}s "
                          f"profit={reward:9.0f} E={exp_profit:9.0f} oracle={best:9.0f} (m*={rows[-1]['oracle_markup']})",
                          flush=True)
    df = pd.DataFrame(rows)
    out_dir = scratch_dir("results")
    path = out_dir / f"episodes_{a.tag}.parquet"
    df.to_parquet(path)
    summ = df.groupby("agent").agg(regret=("regret", "mean"), exp_profit=("expected_profit", "mean"),
                                   realized=("realized_profit", "mean"), wall=("wall_s", "mean"),
                                   missed=("deadline_missed", "mean"), fallback=("fallback", "mean"))
    print(summ.round(2))
    summ.round(3).to_csv(REPO / "results" / f"episodes_{a.tag}_summary.csv")
    if llm is not None:
        print(f"LLM tokens {llm.stats.tokens}, calls {len(llm.stats.calls)}")
    print("->", path)


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    main()
