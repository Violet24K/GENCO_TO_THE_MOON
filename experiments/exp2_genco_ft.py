"""Exp 2 with the fine-tuned GENCO (experiments/genco_finetune.py): same world states, same
scenarios, same AC and DC profit matrices; only the GENCO column is recomputed.

Scenarios are regenerated with exp2_budget.state_scenarios and checked against the cached
matrices by re-solving DC-HiGHS on the first scenarios (must match the stored DC profits).
Output: <scratch>/exp2_<src>_ft/stateNNN.npz (_ft2 for --size small_ft2) (ac, dc, genco = fine-tuned, genco_frozen), then
phase 2 as in exp2_budget, and tables

    results/exp2_regret_pct_ft_<name>.csv   regret % by method x budget
    results/exp2_by_event_ft_<name>.csv     regret % by event x budget x method

    CUDA_VISIBLE_DEVICES=4 python experiments/exp2_genco_ft.py --src heatv4,heatv4_hs --name all
    CUDA_VISIBLE_DEVICES=4 python experiments/exp2_genco_ft.py --src ext2 --name ext2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp2_budget import phase2, state_scenarios  # noqa: E402

from genco_market.core.config import REPO, Config, scratch_dir  # noqa: E402

EVENTS = {"heatv4": None, "heatv4_hs": ["heatwave", "thunderstorm"],
          "ext2": ["cold_snap", "wildfire_psps", "high_wind", "solar_drop", "demand_response", "market_power_review",
                   "geomagnetic_storm", "cyber_physical"]}
SHORT = {"planned_outage": "outage", "thunderstorm": "storm", "heatwave": "heat", "gas_price": "gas"}


def suffix(size: str) -> str:
    """Output tag of a fine-tuned model: small_ft -> _ft, small_ft2 -> _ft2."""
    return "_" + size.removeprefix("small_")


def regen(cfg, src: str, size: str):
    from genco_market.env.env import MarketEnv
    from genco_market.env.grid import GridInfo
    from genco_market.semantics.world import WorldGenerator
    from genco_market.surrogate.genco_service import GencoService
    from genco_market.surrogate.profit_models import DCModel, GencoModel

    info = GridInfo(cfg)
    env = MarketEnv(cfg, info, WorldGenerator(cfg, info), None)
    K = cfg.agent.markups
    gm = GencoModel(GencoService(size=size), info=info)
    dc = DCModel(None, "highs", info=info)
    out_dir = scratch_dir(f"exp2_{src}{suffix(size)}")
    for f in sorted(scratch_dir(f"exp2_{src}").glob("state*.npz")):
        path = out_dir / f.name
        if path.exists():
            continue
        z = np.load(f)
        s, n = int(f.stem[5:]), z["ac"].shape[0]
        _, state, _, scns = state_scenarios(cfg, info, env, s, n, EVENTS[src])
        gm.rules = dc.rules = state.market_rules
        chk = dc.evaluate(scns[:3], K)
        assert np.allclose(chk, z["dc"][:3], rtol=1e-4, atol=1e-2, equal_nan=True), f"{f.name}: scenario mismatch"
        g = gm.evaluate(scns, K)
        meta = json.loads(str(z["meta"]))
        meta["cost"]["genco_frozen"] = meta["cost"]["genco"]
        meta["cost"]["genco"] = gm.cost_per_eval_s()
        meta["genco_size"] = size
        np.savez(path, ac=z["ac"], dc=z["dc"], genco=g, genco_frozen=z["genco"], meta=json.dumps(meta))
        print(f"{src} {f.name}: ok", flush=True)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mvp.yaml")
    ap.add_argument("--src", required=True, help="comma list of exp2 tags (heatv4, heatv4_hs, ext2)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", default="small_ft")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--assumed-rho", type=float, default=0.7)
    a = ap.parse_args()
    cfg = Config.load(a.config)
    dfs = []
    for src in a.src.split(","):
        d = regen(cfg, src, a.size)
        df = phase2(d, a.reps, a.assumed_rho)
        df.to_parquet(scratch_dir("results") / f"exp2_regret_{src}{suffix(a.size)}.parquet")
        dfs.append(df)
    df = pd.concat(dfs)
    pct = df.groupby(["method", "budget"]).regret_pct.mean().unstack("budget").round(2)
    print(pct)
    pct.to_csv(REPO / "results" / f"exp2_regret_pct{suffix(a.size)}_{a.name}.csv")
    df["sig"] = df.events.map(lambda e: "+".join(SHORT.get(x, x) for x in e.split("+")))
    ev = df.groupby(["sig", "budget", "method"]).regret_pct.mean().unstack("method").round(2)
    ev.to_csv(REPO / "results" / f"exp2_by_event{suffix(a.size)}_{a.name}.csv")
    print(ev.xs(2.0, level="budget"))


if __name__ == "__main__":
    main()
