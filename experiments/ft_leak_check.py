"""Leakage check for the fine-tuned GENCO: nearest training scenario (load vector + topology) for each Exp 2
test scenario (20 per state), against the 12,000 fine-tuning scenarios.

    python experiments/ft_leak_check.py
"""
import pickle, sys, glob
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from exp2_budget import state_scenarios
from exp2_genco_ft import EVENTS
from genco_market.core.config import Config, scratch_dir
from genco_market.env.env import MarketEnv
from genco_market.env.grid import GridInfo
from genco_market.semantics.world import WorldGenerator

def feat(s):
    return np.asarray(s.bus_pd, float), np.asarray(s.branch_status), np.asarray(s.gen_status), np.asarray(s.gen_cost_c1 if hasattr(s, "gen_cost_c1") else [0])

tr = []
for p in [scratch_dir("exp1") / "scenarios_ft.pkl", scratch_dir("genco_ft") / "scenarios_forced.pkl"]:
    d = pickle.load(open(p, "rb")); tr += d["scns"]
from genco_market.core.schemas import Scenario
tr = [Scenario.model_validate(x) for x in tr]
print("train scenarios", len(tr), "fields", list(tr[0].model_dump().keys()))
P = np.array([np.asarray(s.bus_pd, float) for s in tr]); B = np.array([s.branch_status for s in tr]); G = np.array([s.gen_status for s in tr])
cfg = Config.load("mvp.yaml"); info = GridInfo(cfg); env = MarketEnv(cfg, info, WorldGenerator(cfg, info), None)
rel, exact, topo_same, within = [], 0, 0, []
for src, ev in EVENTS.items():
    for f in sorted(glob.glob(str(scratch_dir(f"exp2_{src}") / "state*.npz"))):
        s = int(f.split("state")[-1][:-4])
        _, _, _, scns = state_scenarios(cfg, info, env, s, 20, ev)
        T = np.array([np.asarray(x.bus_pd, float) for x in scns])
        for k, x in enumerate(scns):
            d = np.linalg.norm(P - T[k], axis=1) / np.linalg.norm(T[k])
            j = int(np.argmin(d)); rel.append(d[j])
            exact += d[j] < 1e-9
            topo_same += np.array_equal(B[j], x.branch_status) and np.array_equal(G[j], x.gen_status)
        # spread inside one test state, for scale
        within += [np.linalg.norm(T[a] - T[b]) / np.linalg.norm(T[a]) for a in range(5) for b in range(a + 1, 5)]
rel = np.array(rel)
print(f"test scenarios {len(rel)}: exact duplicates {exact}; nearest-train rel. L2 distance of load: "
      f"min {rel.min():.4f}, 5% {np.quantile(rel, .05):.4f}, median {np.median(rel):.4f}; "
      f"same topology as nearest {topo_same}/{len(rel)}")
print(f"for scale, two scenarios of the SAME test state differ by median {np.median(within):.4f}")
