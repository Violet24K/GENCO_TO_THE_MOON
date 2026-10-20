"""Training data for the operator GENCO (a social-cost head on frozen GENCO bus
embeddings, genco_market/surrogate/social_head.py).

Layer 4 training episodes are disjoint from the test episodes (test: 0-199;
here 1000+). Per episode and hour, n scenarios from the TRUE P_t on a separate
RNG stream ("l4_train"), each cleared three times: line in service, the
episode's requested line out, and one other candidate line out (for diversity).
Labels: AC social cost (true generation cost + VOLL * shed) and shed MW.
Features: GENCO bus embeddings (float16), GENCO generation cost, bus loads.

    CUDA_VISIBLE_DEVICES=6 python experiments/l4_op_data.py --episodes 1000 1600 --n 4 --workers 32
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from genco_market.core.config import scratch_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="l4.yaml")
    ap.add_argument("--episodes", nargs=2, type=int, default=[1000, 1600])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--genco-size", default="small_ft2")
    ap.add_argument("--chunk", type=int, default=50, help="episodes per output file")
    a = ap.parse_args()

    from genco_market.core.seeding import rng
    from genco_market.env.env import MarketEnv
    from genco_market.env.grid import GridInfo, scenario_to_network
    from genco_market.env.outage import L4WorldGenerator, load_l4, reset_episode, with_line_out
    from genco_market.env.physical import sample_scenario
    from genco_market.env.pool import SolverPool
    from genco_market.surrogate.genco_service import GencoService
    from genco_market.surrogate.social_models import PoolSocial

    cfg, l4 = load_l4(a.config)
    cfg.semantics.paraphrase = False
    info = GridInfo(cfg)
    gen = L4WorldGenerator(cfg, info)
    svc = GencoService(size=a.genco_size)
    names = sorted(l4.candidates)
    out_dir = scratch_dir("l4", f"op_train_{a.genco_size}")
    with SolverPool(a.workers, cfg) as pool:
        env = MarketEnv(cfg, info, gen, pool)
        ac = PoolSocial(pool, info, "ac", l4.voll)
        for c0 in range(a.episodes[0], a.episodes[1], a.chunk):
            path = out_dir / f"ep{c0:04d}.npz"
            if path.exists():
                continue
            scns, meta = [], []
            for ep in range(c0, min(c0 + a.chunk, a.episodes[1])):
                req = reset_episode(env, l4, ep)
                r = rng(cfg.seed, "l4_train_line", ep)
                other = [n for n in names if n != req.line]
                ob = info.branch_by_name[str(r.choice(other))]
                br2 = int(ob[int(r.integers(len(ob)))])
                for t in range(env.world.horizon):
                    st, base = env.true_state(t), env.baseline(t)
                    for i in range(a.n):
                        s = sample_scenario(st, info, base, rng(cfg.seed, "l4_train", ep, t, i))
                        scns += [s, with_line_out(s, req.branch), with_line_out(s, br2)]
                        meta.append((ep, t, i, req.branch, br2))
            res = ac.evaluate(scns)
            g = svc.predict_arrays([scenario_to_network(s, None, cfg.grid) for s in scns], 512, embeddings=True)
            c1 = np.array([s.gen_c1 for s in scns], float)
            np.savez(path, meta=np.array(meta), ac_cost=res["cost"], ac_shed=res["shed"],
                     emb=g["emb_bus"].astype(np.float16), genco_cost=(c1 * g["pg"]).sum(1),
                     bus_pd=np.array([s.bus_pd for s in scns], np.float32),
                     off=np.array([[int(x == 0) for x in s.branch_status] for s in scns], np.int8))
            print(f"{path.name}: {len(scns)} nets, AC ok {np.isfinite(res['cost']).mean():.3f}, "
                  f"shed>1MW {np.nanmean(res['shed'] > 1):.3f}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    main()
