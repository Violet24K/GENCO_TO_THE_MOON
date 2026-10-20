"""Oracle bid for an interval: true P_t known, unlimited AC-OPF budget.

Expected profit of every candidate is estimated on M scenarios drawn from the
true distribution with an evaluation-only RNG stream (never the agent's), and
cached on disk. Used to score agents (expected regret), never by agents.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from genco_market.core.config import scratch_dir
from genco_market.core.seeding import rng
from genco_market.env.physical import sample_scenario
from genco_market.env.pool import TaskError


def oracle_table(env, t: int, markups: list[float], M: int) -> dict:
    """{'markups', 'mean', 'se', 'n_ok'} of firm-C profit under the true P_t."""
    w = env.world
    state, base = env.true_state(t), env.baseline(t)
    # key on the truth itself, so worlds with different events never share an entry
    blob = json.dumps([state.model_dump(mode="json"), base, markups, M, env.cfg.seed, w.episode_id, t], sort_keys=True)
    path = scratch_dir("oracle_cache") / f"{hashlib.sha256(blob.encode()).hexdigest()[:24]}.json"
    if path.exists():
        return json.loads(path.read_text())
    scns = [sample_scenario(state, env.info, base, rng(env.cfg.seed, "oracle", w.episode_id, t, i)) for i in range(M)]
    args = [(s, m, state.market_rules) for s in scns for m in markups]
    outs = env.pool.map("ac", args)
    firm = env.info.strategic
    P = np.array([np.nan if (o is None or isinstance(o, TaskError)) else o.firm_profit.get(firm, np.nan) for o in outs])
    P = P.reshape(M, len(markups))
    P = P[~np.isnan(P).any(axis=1)]
    out = {"markups": list(markups), "mean": P.mean(0).tolist(),
           "se": (P.std(0, ddof=1) / np.sqrt(len(P))).tolist(), "n_ok": int(len(P))}
    path.write_text(json.dumps(out))
    return out
