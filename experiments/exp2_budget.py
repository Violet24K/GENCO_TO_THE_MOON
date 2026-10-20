"""Exp 2 (README 5.2, main figure): expected regret vs compute budget.

Phase 1 (expensive, cached): for S "world states" (true P_t of random worlds
and intervals), draw N scenarios and compute firm profit matrices (N, K) with
AC (truth), GENCO and DC on the same scenarios. Rows [0, N/2) form the pool
estimators draw from; rows [N/2, N) are held out to define the oracle.

Phase 2 (cheap, offline): for each budget T and method, the number of rows a
method can afford is T / (K * cost_per_eval) with costs measured on this
machine; estimate, pick argmax, score regret against the held-out oracle.
Repeated R times with random row subsets.

    python experiments/exp2_budget.py --states 30 --n 400            # phase 1 + 2
    python experiments/exp2_budget.py --offline                      # phase 2 only
"""

from __future__ import annotations

import argparse
import json
import zlib

import numpy as np
import pandas as pd

from genco_market.core.config import REPO, Config, scratch_dir
from genco_market.core.seeding import rng, sample_start_hour
from genco_market.estimators.multifidelity import multifidelity, multifidelity_multi, single_fidelity, split_budget

BUDGETS = [0.25, 0.5, 1, 2, 5, 10, 20, 50]


def state_scenarios(cfg: Config, info, env, s: int, n: int, event_types: list[str] | None = None):
    """(t, true state, baseline, n scenarios) of world state s; ``env`` is reset to its world."""
    from genco_market.env.physical import sample_scenario

    if event_types:  # forced events: one type per state (cycled), t inside its window
        et = event_types[s % len(event_types)]
        start = sample_start_hour(cfg.seed + 11, s)
        if et == "cold_snap":  # winter morning/evening hours
            rr = rng(cfg.seed, "exp2_winter", s)
            start = int(rr.integers(0, 59)) * 24 + int(rr.choice([6, 7, 8, 17, 18, 19]))
        env.reset(5000 + s, start, event_types=[et])
        win = [t for e in env.world.events if e.occurs for t in range(e.start_t, e.end_t + 1)
               if t < env.world.horizon]
        t = int(rng(cfg.seed, "exp2_t", s).choice(win)) if win else 0
    else:
        env.reset(1000 + s, sample_start_hour(cfg.seed + 7, s))
        t = int(rng(cfg.seed, "exp2_t", s).integers(env.world.horizon))
    state, base = env.true_state(t), env.baseline(t)
    return t, state, base, [sample_scenario(state, info, base, rng(cfg.seed, "exp2", s, i)) for i in range(n)]


def phase1(cfg: Config, n_states: int, n: int, out_dir, event_types: list[str] | None = None,
           cooling_pf: float | None = None) -> None:
    from genco_market.env.env import MarketEnv
    from genco_market.env.grid import GridInfo
    from genco_market.env.physical import ood_flags
    from genco_market.env.pool import SolverPool
    from genco_market.semantics.world import WorldGenerator
    from genco_market.surrogate.genco_service import GencoService
    from genco_market.surrogate.profit_models import ACModel, DCModel, GencoModel

    info = GridInfo(cfg)
    gen = WorldGenerator(cfg, info)
    if cooling_pf:  # heatwave cooling load at this power factor (events.apply_events)
        gen.eff["heatwave"]["cooling_pf"] = cooling_pf
    K = cfg.agent.markups
    with SolverPool(cfg.solver_workers, cfg) as pool:
        env = MarketEnv(cfg, info, gen, pool)
        models = {"ac": ACModel(pool, info), "dc": DCModel(pool, "highs", info=info),
                  "genco": GencoModel(GencoService(size="small"), info=info)}
        for s in range(n_states):
            path = out_dir / f"state{s:03d}.npz"
            if path.exists():
                continue
            t, state, base, scns = state_scenarios(cfg, info, env, s, n, event_types)
            for m in models.values():  # e.g. a mitigation rule announced by a market_power_review event
                m.rules = state.market_rules
            mats = {k: m.evaluate(scns, K) for k, m in models.items()}
            ev = sorted({e.event_type for e in env.world.events if e.occurs and e.start_t <= t <= e.end_t})
            meta = dict(events=ev, ood=ood_flags(state, info, base), load=float(sum(base.values())),
                        cost={k: m.cost_per_eval_s() for k, m in models.items()})
            np.savez(path, **mats, meta=json.dumps(meta))
            print(f"state {s}: events={ev} AC nan rows={np.isnan(mats['ac']).any(1).sum()}", flush=True)


def phase2(out_dir, reps: int, assumed_rho: float, ac_cost_mult: float = 1.0) -> pd.DataFrame:
    rows = []
    files = sorted(out_dir.glob("state*.npz"))
    # per-eval costs: median over states (robust to load spikes on the shared machine)
    costs = [json.loads(str(np.load(f)["meta"]))["cost"] for f in files]
    c = {m: float(np.median([x[m] for x in costs])) for m in costs[0]}
    c["ac"] *= ac_cost_mult  # what-if: a larger grid where exact AC-OPF is this much slower
    budgets = [b * ac_cost_mult for b in BUDGETS]
    print("median cost per eval (ms):", {m: round(1e3 * v, 3) for m, v in c.items()})
    for f in files:
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        ac, gc, dc = z["ac"], z["genco"], z["dc"]
        ok = ~(np.isnan(ac).any(1) | np.isnan(gc).any(1) | np.isnan(dc).any(1))
        ac, gc, dc = ac[ok], gc[ok], dc[ok]
        half = len(ac) // 2
        oracle = ac[half:].mean(0)
        best = oracle.max()
        pool = {"ac": ac[:half], "genco": gc[:half], "dc": dc[:half]}
        K = ac.shape[1]
        r = np.random.default_rng(zlib.crc32(f.name.encode()))
        for T in budgets:
            for rep in range(reps):
                perm = r.permutation(half)
                res = {}
                for m in ("ac", "genco", "dc"):
                    k = max(2, min(half, int(T / (K * c[m]))))
                    res[m] = single_fidelity(pool[m][perm[:k]])
                for lo in ("genco", "dc"):
                    n_hi, n_lo = split_budget(T, c["ac"], c[lo], assumed_rho, K)
                    n_hi, n_lo = min(n_hi, half), min(n_lo, half)
                    if n_hi >= 2:
                        res[f"mf_{lo}"] = multifidelity(pool["ac"][perm[:n_hi]], pool[lo][perm[:n_lo]])
                # both surrogates as joint control variates (each scenario costs c_dc + c_genco)
                n_hi, n_lo = split_budget(T, c["ac"], c["dc"] + c["genco"], assumed_rho, K)
                n_hi, n_lo = min(n_hi, half), min(n_lo, half)
                if n_hi >= 4:
                    res["mf_both"] = multifidelity_multi(pool["ac"][perm[:n_hi]],
                                                         [pool["dc"][perm[:n_lo]], pool["genco"][perm[:n_lo]]])
                for m, e in res.items():
                    k_hat = int(np.argmax(e.mean))
                    rows.append(dict(state=f.stem, budget=T, ac_cost_mult=ac_cost_mult, method=m, rep=rep, regret=best - oracle[k_hat],
                                     regret_pct=100 * (best - oracle[k_hat]) / max(abs(best), 1.0),
                                     events="+".join(meta["events"]) or "none", ood=bool(meta["ood"])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mvp.yaml")
    ap.add_argument("--states", type=int, default=30)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--assumed-rho", type=float, default=0.7)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--ac-cost-mult", type=float, default=1.0,
                    help="what-if: scale AC cost (and budgets) by this factor; output tag gets _acx<mult>")
    ap.add_argument("--tag", default="", help="separate output dir/files, e.g. 'heatv3'")
    ap.add_argument("--events", default="", help="comma list: force one of these event types per state")
    ap.add_argument("--cooling-pf", type=float, default=None, help="power factor of heatwave cooling load (default: none)")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    out_dir = scratch_dir("exp2" + (f"_{a.tag}" if a.tag else ""))
    if not a.offline:
        phase1(cfg, a.states, a.n, out_dir, [e for e in a.events.split(',') if e] or None, a.cooling_pf)
    df = phase2(out_dir, a.reps, a.assumed_rho, a.ac_cost_mult)
    sfx = (f"_{a.tag}" if a.tag else "") + (f"_acx{a.ac_cost_mult:g}" if a.ac_cost_mult != 1 else "")
    df.to_parquet(scratch_dir("results") / f"exp2_regret{sfx}.parquet")
    summ = df.groupby(["method", "budget"]).regret.agg(["mean", "sem"]).unstack("budget").round(1)
    print(summ)
    summ.to_csv(REPO / "results" / f"exp2_regret_summary{sfx}.csv")


if __name__ == "__main__":
    main()
