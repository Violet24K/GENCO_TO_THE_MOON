"""Budget vs. regret for outage
coordination, offline, in the style of experiments/exp2_budget.py.

Phase 1 (expensive, cached by genco_market.eval.outage_oracle): for every
episode and hour, N scenarios from the true P_t, social cost with the line in
and out from AC, DC and GENCO on the same scenarios.
Phase 2 (cheap): rows [0, N/2) are the pool estimators draw from, rows
[N/2, N) define the oracle. A budget T is split equally over the H hours; per
hour a method affords T / (H * 2 * cost_per_eval) scenario pairs (median
measured costs). Estimate Delta_t per hour, pick the cheapest option (start
hour or defer), score regret against the hold-out oracle. R repetitions.

    CUDA_VISIBLE_DEVICES=7 python experiments/l4_offline.py --episodes 0 20      # phase 1 (+2)
    python experiments/l4_offline.py --offline                                   # phase 2 only
"""

from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np
import pandas as pd

from genco_market.core.config import REPO, scratch_dir
from genco_market.env.outage import DEFER, OutageRequest, load_l4, option_costs
from genco_market.estimators.multifidelity import multifidelity, multifidelity_multi, single_fidelity, split_budget

BUDGETS = [0.5, 1, 2, 5, 10, 20, 50]
METHODS = ["ac", "dc", "genco", "mf_dc", "mf_genco", "mf_both"]
DEFERRAL_SENS = [2000, 4000, 10000, 20000]


def phase1(cfg_name: str, eps: range, n: int, workers: int, genco_size: str = "small", sub: str = "l4/oracle",
           op: bool = False) -> None:
    from genco_market.env.env import MarketEnv
    from genco_market.env.grid import GridInfo
    from genco_market.env.outage import L4WorldGenerator, reset_episode
    from genco_market.env.pool import SolverPool
    from genco_market.eval.outage_oracle import compute_episode
    from genco_market.surrogate.genco_service import GencoService
    from genco_market.surrogate.social_models import GencoSocial, PoolSocial

    cfg, l4 = load_l4(cfg_name)
    cfg.semantics.paraphrase = False  # physics only; text does not matter here
    info = GridInfo(cfg)
    gen = L4WorldGenerator(cfg, info)
    with SolverPool(workers, cfg) as pool:
        env = MarketEnv(cfg, info, gen, pool)
        models = {"ac": PoolSocial(pool, info, "ac", l4.voll), "dc": PoolSocial(pool, info, "dc_highs", l4.voll),
                  "genco": GencoSocial(GencoService(size=genco_size), info, l4.voll)}
        if op:
            from genco_market.surrogate.social_head import SocialHead, head_path
            from genco_market.surrogate.social_models import GencoOpSocial

            svc = models["genco"].svc
            models["genco"] = GencoOpSocial(svc, info, SocialHead.load(head_path(genco_size), svc.device), l4.voll)
        for ep in eps:
            req = reset_episode(env, l4, ep)
            if sub != "l4/oracle":
                regenco(env, req, n, models["genco"], sub)
                continue
            compute_episode(env, req, n, models, log=lambda s: print(s, flush=True))


def regenco(env, req, n: int, model, sub: str) -> None:
    """Copy the cached AC/DC rows of one episode into `sub` and redo only the GENCO fields with `model`
    (same scenarios: same evaluation RNG stream as outage_oracle.compute_episode)."""
    from genco_market.core.seeding import rng
    from genco_market.env.outage import with_line_out
    from genco_market.env.physical import sample_scenario
    from genco_market.eval.outage_oracle import FIELDS_GENCO, load_cache

    # match the source cache by episode id (the key hashes the state schema, which has gained fields since);
    # the sum_pd check below confirms the scenarios are the same
    ep = env.world.episode_id
    src = sorted(scratch_dir("l4", "oracle").glob(f"ep{ep:04d}_*.npz"))
    assert len(src) == 1, src
    dst = scratch_dir(*sub.split("/")) / src[0].name
    if dst.exists():
        return
    z = load_cache(src[0])
    H, info = env.world.horizon, env.info
    flat = []
    for t in range(H):
        st, base = env.true_state(t), env.baseline(t)
        for i in range(n):
            s = sample_scenario(st, info, base, rng(env.cfg.seed, "l4_oracle", ep, t, i))
            flat += [s, with_line_out(s, req.branch)]
    model.evaluate(flat)
    for f in FIELDS_GENCO + (("sc",) if "sc" in model.last else ()):
        v = np.asarray(model.last[f]).reshape(H, n, 2)
        z[f"genco_{f}_in"], z[f"genco_{f}_out"] = v[..., 0], v[..., 1]
    assert np.allclose(z["sum_pd"], np.reshape(model.last["sum_pd"], (H, n, 2))[..., 0])
    meta = z.pop("meta")
    meta["cost_per_eval_s"]["genco"] = model.cost_per_eval_s()
    np.savez(dst, **z, meta=json.dumps(meta))
    print(f"[regenco] ep{ep} -> {dst.name}", flush=True)


# ---------------------------------------------------------------- phase 2
def genco_social(z: dict, side: str, variant: str, voll: float = 2000.0, loss_ref: float = 0.0) -> np.ndarray:
    """GENCO social-cost estimates (GENCO has no load shedding):
    'cost'  = sum c1*pg;
    'short' = cost + VOLL * max(0, sum_pd*(1+loss_ref) - sum_pg);
    'op'    = the operator GENCO's social cost (cache made with --op)."""
    c = z[f"genco_gen_cost_{side}"]
    if variant == "op":  # operator GENCO: learned social cost (surrogate/social_head.py)
        return z[f"genco_sc_{side}"]
    if variant == "cost":
        return c
    if variant == "short":
        return c + voll * np.maximum(0.0, z["sum_pd"] * (1 + loss_ref) - z[f"genco_sum_pg_{side}"])
    raise ValueError(variant)


def deltas(z: dict, genco_variant: str = "cost", loss_ref: float = 0.0) -> dict[str, np.ndarray]:
    """[H, N] per-scenario Delta = SC(out) - SC(in) for each fidelity."""
    return {"ac": z["ac_cost_out"] - z["ac_cost_in"], "dc": z["dc_cost_out"] - z["dc_cost_in"],
            "genco": genco_social(z, "out", genco_variant, loss_ref=loss_ref) - genco_social(z, "in", genco_variant, loss_ref=loss_ref)}


def estimate_hour(method: str, D: dict, perm: np.ndarray, T_h: float, c: dict, rho: float) -> float:
    """Estimate one hour's Delta under per-hour budget T_h; D[m] = clean pool rows [n_pool]."""
    half = len(perm)
    if method in ("ac", "dc", "genco"):
        k = max(2, min(half, int(T_h / (2 * c[method]))))
        return float(single_fidelity(D[method][perm[:k], None]).mean[0])
    if method in ("mf_dc", "mf_genco"):
        lo = method[3:]
        n_hi, n_lo = split_budget(T_h, c["ac"], c[lo], rho, 2)
        n_hi, n_lo = min(n_hi, half), min(n_lo, half)
        return float(multifidelity(D["ac"][perm[:n_hi], None], D[lo][perm[:n_lo], None]).mean[0])
    if method == "mf_both":
        n_hi, n_lo = split_budget(T_h, c["ac"], c["dc"] + c["genco"], rho, 2)
        n_hi, n_lo = min(max(n_hi, 4), half), min(max(n_lo, 4), half)
        return float(multifidelity_multi(D["ac"][perm[:n_hi], None],
                                         [D["dc"][perm[:n_lo], None], D["genco"][perm[:n_lo], None]]).mean[0])
    raise ValueError(method)


def load_all(sub: str = "l4/oracle") -> list[tuple[str, dict]]:
    from genco_market.eval.outage_oracle import load_cache

    files = sorted(scratch_dir(*sub.split("/")).glob("ep*.npz"))
    return [(f.stem, load_cache(f)) for f in files]


def clean_rows(z: dict, genco_variant="cost", loss_ref=0.0) -> list[dict[str, np.ndarray]]:
    """Per hour: rows where every fidelity finished (pool and oracle halves split later)."""
    D = deltas(z, genco_variant, loss_ref)
    out = []
    for t in range(D["ac"].shape[0]):
        ok = np.isfinite(D["ac"][t]) & np.isfinite(D["dc"][t]) & np.isfinite(D["genco"][t])
        out.append({m: D[m][t][ok] for m in D})
    return out


def phase2(data, reps: int, rho: float, deferral_costs: list[float], genco_variant="cost", loss_ref=0.0,
           ac_cost_mult: float = 1.0, budgets=BUDGETS) -> pd.DataFrame:
    costs = [d["meta"]["cost_per_eval_s"] for _, d in data]
    c = {m: float(np.median([x[m] for x in costs])) for m in costs[0]}
    c["ac"] *= ac_cost_mult
    print("median cost per eval (ms):", {m: round(1e3 * v, 3) for m, v in c.items()})
    rows = []
    for stem, z in data:
        meta = z["meta"]
        req = OutageRequest(**meta["request"])
        hours = clean_rows(z, genco_variant, loss_ref)
        H = len(hours)
        half = [len(h["ac"]) // 2 for h in hours]
        pools = [{m: h[m][:k] for m in h} for h, k in zip(hours, half)]
        oracle = np.array([h["ac"][k:].mean() for h, k in zip(hours, half)])
        r = np.random.default_rng(zlib.crc32(stem.encode()))
        ev = "+".join(sorted({e["type"] for e in meta["events"] if e["occurs"]})) or "none"
        for T in budgets:
            T_h = T * ac_cost_mult / H
            for rep in range(reps):
                perms = [r.permutation(k) for k in half]
                est = {m: np.array([estimate_hour(m, pools[t], perms[t], T_h, c, rho) for t in range(H)]) for m in METHODS}
                for dc_ in deferral_costs:
                    oc = option_costs(oracle, req, dc_)
                    best = oc.min()
                    opts = req.options(H)
                    for m, e in est.items():
                        k = int(np.argmin(option_costs(e, req, dc_)))
                        rows.append(dict(episode=stem, budget=T, method=m, rep=rep, deferral_cost=dc_,
                                         regret=oc[k] - best, best_cost=best, chose_best=bool(oc[k] - best <= 1e-9),
                                         deferred=opts[k] == DEFER, oracle_defer=opts[int(np.argmin(oc))] == DEFER,
                                         line=req.line, duration=req.duration, events=ev, ac_cost_mult=ac_cost_mult))
    return pd.DataFrame(rows)


def correlations(data, genco_variant="cost", loss_ref=0.0) -> pd.DataFrame:
    """Within-hour correlation of per-scenario Delta (DC/GENCO vs AC) and bias of the hour means."""
    rows = []
    for stem, z in data:
        meta = z["meta"]
        for t, h in enumerate(clean_rows(z, genco_variant, loss_ref)):
            a = h["ac"]
            row = dict(episode=stem, t=t, line=meta["request"]["line"], ac_mean=a.mean(), ac_std=a.std(),
                       n=len(a), shed_out=float(np.nanmean(z["ac_shed_out"][t])),
                       events="+".join(sorted({e["type"] for e in meta["events"] if e["occurs"]
                                               and e["start"] <= t <= e["end"]})) or "none")
            for m in ("dc", "genco"):
                x = h[m]
                row[f"{m}_mean"] = x.mean()
                row[f"{m}_rho"] = np.corrcoef(a, x)[0, 1] if a.std() > 0 and x.std() > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="l4.yaml")
    ap.add_argument("--episodes", nargs=2, type=int, default=[0, 40], help="range [a, b)")
    ap.add_argument("--n", type=int, default=None, help="scenarios per hour (default: l4.oracle_scenarios)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-phase2", action="store_true")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--assumed-rho", type=float, default=0.7)
    ap.add_argument("--genco-variant", default="cost")
    ap.add_argument("--loss-ref", type=float, default=0.0)
    ap.add_argument("--ac-cost-mult", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--genco-size", default="small", help="GENCO checkpoint; not 'small' -> GENCO-only redo in l4/oracle_<size>")
    ap.add_argument("--op", action="store_true", help="operator GENCO (social-cost head on --genco-size)")
    a = ap.parse_args()
    sub = "l4/oracle" if a.genco_size == "small" and not a.op else f"l4/oracle_{a.genco_size}" + ("_op" if a.op else "")
    if a.op:
        a.genco_variant = "op"
    cfg, l4 = load_l4(a.config)
    n = a.n or l4.oracle_scenarios
    if not a.offline:
        phase1(a.config, range(*a.episodes), n, a.workers, a.genco_size, sub, a.op)
    if a.no_phase2:
        return
    data = load_all(sub)
    print(f"{len(data)} episodes cached")
    sfx = (f"_{a.tag}" if a.tag else "") + (f"_acx{a.ac_cost_mult:g}" if a.ac_cost_mult != 1 else "")
    cor = correlations(data, a.genco_variant, a.loss_ref)
    cor.to_csv(REPO / "results" / f"l4_e42_correlation{sfx}.csv", index=False)
    df = phase2(data, a.reps, a.assumed_rho, sorted(set(DEFERRAL_SENS) | {l4.deferral_cost}), a.genco_variant,
                a.loss_ref, a.ac_cost_mult)
    df.to_parquet(scratch_dir("results") / f"l4_e42{sfx}.parquet")
    main_ = df[df.deferral_cost == l4.deferral_cost]
    summ = main_.groupby(["method", "budget"]).regret.mean().unstack("budget").round(1)
    print(summ)
    summ.to_csv(REPO / "results" / f"l4_e42_regret{sfx}.csv")


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    main()
