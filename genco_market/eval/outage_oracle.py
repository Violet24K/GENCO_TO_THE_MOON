"""Layer 4 oracle + offline cache: per (episode, hour) social costs with the
requested line in / out on the same scenarios (common random numbers), from
AC (truth), DC and GENCO, drawn from the TRUE P_t with an evaluation-only RNG
stream (never the agent's). Cached on disk (npz), keyed by the truth itself.

The same cache serves
- E4.1: oracle Delta_t = mean over all rows of AC(out) - AC(in);
- E4.2: rows [0, N/2) are the pool estimators draw from, [N/2, N) the oracle.
It also holds the AC clearing of the realized scenario of every hour (line in
and out) for the replay page.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from genco_market.core.config import scratch_dir
from genco_market.core.seeding import rng
from genco_market.env.outage import OutageRequest, with_line_out
from genco_market.env.physical import is_connected, sample_scenario

FIELDS_POOL = ("cost", "shed")
FIELDS_GENCO = ("gen_cost", "sum_pg", "res_p", "pbe_max")


def cache_key(env, req: OutageRequest, n: int) -> str:
    H = env.world.horizon
    blob = json.dumps([[env.true_state(t).model_dump(mode="json") for t in range(H)],
                       [env.baseline(t) for t in range(H)], req.branch, n, env.cfg.seed, env.world.episode_id],
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def cache_path(env, req: OutageRequest, n: int, sub: str = "l4/oracle"):
    return scratch_dir(*sub.split("/")) / f"ep{env.world.episode_id:04d}_{cache_key(env, req, n)}.npz"


def load_cache(path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(str(z["meta"]))
    return out


def compute_episode(env, req: OutageRequest, n: int, models: dict, sub: str = "l4/oracle", log=print) -> dict:
    """models: {"ac": PoolSocial, "dc": PoolSocial, "genco": GencoSocial} (dc/genco optional).
    Returns (and caches) arrays [H, n] per model and field, with 'in'/'out' suffixes."""
    path = cache_path(env, req, n, sub)
    if path.exists():
        return load_cache(path)
    info, H, ep = env.info, env.world.horizon, env.world.episode_id
    scns, outs = [], []
    for t in range(H):
        st, base = env.true_state(t), env.baseline(t)
        for i in range(n):
            s = sample_scenario(st, info, base, rng(env.cfg.seed, "l4_oracle", ep, t, i))
            scns.append(s)
            outs.append(with_line_out(s, req.branch))
    flat = [x for pair in zip(scns, outs) for x in pair]  # in, out, in, out, ...
    res: dict[str, np.ndarray] = {}
    meta: dict = {"episode": ep, "n": n, "branch": req.branch, "cost_per_eval_s": {}}
    # realized scenario per hour (what the market actually sees), line in and out
    real = [env.realized(t) for t in range(H)]
    real_flat = [x for s in real for x in (s, with_line_out(s, req.branch))]
    for name, m in models.items():
        if name in ("ac", "dc"):
            r = m.evaluate(flat + (real_flat if name == "ac" else []))
            st = m.last["status"]
            k = len(flat)
            for f in FIELDS_POOL:
                v = r[f][:k].reshape(H, n, 2)
                res[f"{name}_{f}_in"], res[f"{name}_{f}_out"] = v[..., 0], v[..., 1]
            if name == "ac":
                for f in FIELDS_POOL:
                    v = r[f][k:].reshape(H, 2)
                    res[f"real_{f}_in"], res[f"real_{f}_out"] = v[:, 0], v[:, 1]
            meta["cost_per_eval_s"][name] = m.cost_per_eval_s()
            u, c = np.unique(st.astype(str), return_counts=True)
            meta[f"{name}_status"] = dict(zip(u.tolist(), c.tolist()))
        else:
            m.evaluate(flat)
            for f in FIELDS_GENCO:
                v = np.asarray(m.last[f]).reshape(H, n, 2)
                res[f"genco_{f}_in"], res[f"genco_{f}_out"] = v[..., 0], v[..., 1]
            meta["cost_per_eval_s"][name] = m.cost_per_eval_s()
    res["sum_pd"] = np.array([sum(s.bus_pd) for s in scns]).reshape(H, n)
    res["real_load"] = np.array([sum(s.bus_pd) for s in real])
    res["disconnected_out"] = np.array([not is_connected(info, s.branch_status) for s in outs]).reshape(H, n)
    meta["events"] = [dict(type=e.event_type, zone=e.zone, start=e.start_t, end=e.end_t, occurs=e.occurs,
                           severity=e.severity) for e in env.world.events]
    meta["request"] = req.to_dict()
    np.savez(path, **res, meta=json.dumps(meta))
    a = res["ac_cost_out"] - res["ac_cost_in"]
    log(f"[oracle] ep{ep} {req.branch_name} d={req.duration}: Delta_t = "
        f"{np.round(np.nanmean(a, 1)).astype(int).tolist()} nan={int(np.isnan(a).sum())} "
        f"shed_out={np.round(np.nanmean(res['ac_shed_out'], 1), 1).tolist()}")
    return load_cache(path)


def oracle_delta(cache: dict, rows: slice = slice(None)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Delta_t mean [H], se [H], shed_delta MW [H]) from AC rows."""
    d = (cache["ac_cost_out"] - cache["ac_cost_in"])[:, rows]
    s = (cache["ac_shed_out"] - cache["ac_shed_in"])[:, rows]
    mean = np.array([np.nanmean(x) for x in d])
    se = np.array([np.nanstd(x, ddof=1) / np.sqrt(np.isfinite(x).sum()) for x in d])
    shed = np.array([np.nanmean(x) for x in s])
    return mean, se, shed
