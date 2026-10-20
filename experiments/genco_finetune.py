"""Fine-tune GENCO Small on marked-up bids of the strategic firm (single firm, D).

The frozen GENCO under-reacts to large markups (TABLES T4): it was trained on permuted, never
scaled, costs. Here it is fine-tuned on AC-OPF dispatch of fresh training scenarios where firm D
bids each of the 6 markups. Scenarios come from exp1_fidelity.draw_scenarios with their own tag
("ft") and episode range, so they share no RNG stream with any test set; events are the four
original types of mvp.yaml (the extension events of T7 stay unseen).

Loss: MSE of generator dispatch against AC (normalized units, weight 4 on D's units) plus a
distillation term keeping the bus outputs (Vm, Va) close to the frozen model. Validation split by
episode (15 %); the epoch with the best validation loss is kept.

Two data versions (each gives its own model; neither overwrites the other or the original):

    v1   1,200 states drawn as in exp1 (the four original event types)       -> small_ft
    all  v1 plus 1,200 states with one forced event each, cycling all 12 event
         types (the original four and the eight extension events; cold snaps in
         winter morning/evening peaks), episodes 300000+                       -> small_ft2

Stages (outputs under <scratch>/genco_ft/, checkpoint under the GENCO checkpoint root as
case118_ieee/<size>/seed0, so GencoService("<size>") loads it):

    label  AC labels (pg, lmp) for 6 markups on the fine-tuning scenarios
    train  fine-tune GENCO (always from the original weights)
    head   price head on the fine-tuned embeddings (models/lmp_head_<size>.pt)

    CUDA_VISIBLE_DEVICES=4 python experiments/genco_finetune.py --stage label --data all --workers 32
    CUDA_VISIBLE_DEVICES=4 python experiments/genco_finetune.py --stage train --data all
    CUDA_VISIBLE_DEVICES=4 python experiments/genco_finetune.py --stage head --data all
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1_fidelity import MARKUPS, load_or_draw  # noqa: E402

from genco_market.core.config import Config, scratch_dir  # noqa: E402
from genco_market.core.seeding import rng  # noqa: E402
from genco_market.env.grid import GridInfo  # noqa: E402

OUT = scratch_dir("genco_ft")
FT_EPISODES = range(200_000, 201_200)  # 1,200 states x 5 scenarios; test sets use 1000+, 5000+, 100000+
PER_STATE = 5
FORCED_EPISODES = range(300_000, 301_200)  # "all": one forced event per state
ALL_EVENTS = ["heatwave", "planned_outage", "thunderstorm", "gas_price", "cold_snap", "wildfire_psps", "high_wind",
              "solar_drop", "demand_response", "market_power_review", "geomagnetic_storm", "cyber_physical"]
SIZE = {"v1": "small_ft", "all": "small_ft2"}
D_WEIGHT = 4.0


def draw_forced(cfg, info, episodes):
    """One state per episode with a forced event (cycling ALL_EVENTS), t inside the event window;
    cold snaps start in a winter morning/evening peak, as in exp2_budget.state_scenarios."""
    import pandas as pd

    from genco_market.env.env import MarketEnv
    from genco_market.env.physical import sample_scenario
    from genco_market.semantics.world import WorldGenerator

    env = MarketEnv(cfg, info, WorldGenerator(cfg, info), None)
    scns, meta = [], []
    for i, ep in enumerate(episodes):
        et = ALL_EVENTS[i % len(ALL_EVENTS)]
        r = rng(cfg.seed, "ft2", ep)
        if et == "cold_snap":
            start = int(r.integers(0, 59)) * 24 + int(r.choice([6, 7, 8, 17, 18, 19]))
        else:
            start = int(r.integers(0, 8268 // 24 - 1)) * 24 + int(r.integers(0, 24))
        env.reset(ep, start, event_types=[et])
        win = [t for e in env.world.events if e.occurs for t in range(e.start_t, e.end_t + 1) if t < env.world.horizon]
        t = int(r.choice(win)) if win else int(r.integers(env.world.horizon))
        state, base = env.true_state(t), env.baseline(t)
        ev = sorted({e.event_type for e in env.world.events if e.occurs and e.start_t <= t <= e.end_t})
        for j in range(PER_STATE):
            scns.append(sample_scenario(state, info, base, rng(cfg.seed, "ft2", ep, "scn", j)))
            meta.append(dict(episode=ep, t=t, start_hour=start, j=j, events="+".join(ev) or "none",
                             cap=state.market_rules.mitigation_markup_cap))
    return scns, pd.DataFrame(meta)


def _forced(cfg, info):
    import pickle

    from genco_market.core.schemas import Scenario

    p = OUT / "scenarios_forced.pkl"
    if p.exists():
        d = pickle.load(open(p, "rb"))
        return [Scenario.model_validate(x) for x in d["scns"]], d["meta"]
    scns, meta = draw_forced(cfg, info, FORCED_EPISODES)
    pickle.dump({"scns": [s.model_dump() for s in scns], "meta": meta}, open(p, "wb"))
    return scns, meta


def scenarios(cfg, info, data="v1"):
    """(scenarios, meta, labels) of a data version; meta.cap = mitigation cap of the state (or None)."""
    import pandas as pd

    scns, meta = load_or_draw(cfg, info, "ft", FT_EPISODES, PER_STATE)
    meta = meta.assign(cap=None)
    if data == "v1":
        return scns, meta
    s2, m2 = _forced(cfg, info)
    return scns + s2, pd.concat([meta, m2], ignore_index=True)


def eff_markups(meta) -> np.ndarray:
    """[n, K] markups after each state's mitigation rule (what the market clears, and what GENCO sees)."""
    from genco_market.core.schemas import MarketRules
    from genco_market.env.market import effective_markup

    return np.array([[effective_markup(m, MarketRules(mitigation_markup_cap=c if c == c else None)) for m in MARKUPS]
                     for c in meta.cap])


def split(cfg, meta):
    return np.array([rng(cfg.seed, "ft-split", e).random() for e in meta.episode]) < 0.15


def _label(cfg, info, scns, meta, path, workers):
    from genco_market.core.schemas import MarketRules
    from genco_market.env.pool import SolverPool
    from genco_market.surrogate.profit_models import ACModel

    if path.exists():
        print(f"labels exist: {path}")
        return
    print(f"{len(scns)} scenarios, events {Counter(meta.events)}", flush=True)
    n, K = len(scns), len(MARKUPS)
    lmp = np.full((n, K, info.n_bus), np.nan, np.float32)
    pg = np.full((n, K, info.n_gen), np.nan, np.float32)
    ok = np.zeros((n, K), bool)
    t0 = time.perf_counter()
    with SolverPool(workers, cfg) as pool:
        ac = ACModel(pool, info)
        caps = meta.cap.astype(object).where(meta.cap.notna(), None)
        for cap in caps.unique():  # one call per market rule
            rows = np.where(caps.map(lambda c: c == cap))[0]
            ac.rules = MarketRules(mitigation_markup_cap=cap)
            ac.evaluate([scns[i] for i in rows], MARKUPS)
            lmp[rows], pg[rows], ok[rows] = ac.last["lmp"], ac.last["pg"], ac.last["ok"]
            print(f"  cap {cap}: {len(rows)} scenarios, status {Counter(ac.last['status'].ravel())}", flush=True)
    print(f"AC labels: {n} x {K} in {time.perf_counter() - t0:.0f} s", flush=True)
    np.savez_compressed(path, lmp=lmp, pg=pg, ok=ok)


def stage_label(cfg, info, workers: int, data: str):
    scns, meta = load_or_draw(cfg, info, "ft", FT_EPISODES, PER_STATE)
    _label(cfg, info, scns, meta.assign(cap=None), OUT / "labels.npz", workers)
    if data == "all":
        s2, m2 = _forced(cfg, info)
        _label(cfg, info, s2, m2, OUT / "labels_forced.npz", workers)


def labels(data):
    z = [np.load(OUT / "labels.npz")] + ([np.load(OUT / "labels_forced.npz")] if data == "all" else [])
    return {k: np.concatenate([x[k] for x in z]) for k in ("lmp", "pg", "ok")}


def _raw(info, scns, rows, eff):
    from genco_market.env.grid import scenario_to_network
    from genco_market.surrogate.genco_adapter import raw_from_networks

    K = len(MARKUPS)
    return raw_from_networks([scenario_to_network(scns[r // K], info.bid_c1(scns[r // K], eff[r // K, r % K]))
                              for r in rows])


def stage_train(cfg, info, epochs: int, lr: float, batch: int, distill: float, data: str):
    import torch

    from genco_market.surrogate.genco_service import CKPT_ROOT, GencoService

    scns, meta = scenarios(cfg, info, data)
    eff = eff_markups(meta)
    lab = labels(data)
    K = len(MARKUPS)
    ok = lab["ok"].reshape(-1)
    pg = lab["pg"].reshape(-1, info.n_gen)
    val_scn = split(cfg, meta)
    rows = np.where(ok)[0]
    tr, va = rows[~val_scn[rows // K]], rows[val_scn[rows // K]]
    print(f"train {len(tr)}  val {len(va)} (scenario x markup)", flush=True)

    svc = GencoService("small")
    ref = GencoService("small", device=str(svc.device))
    model, base = svc.model, float(svc.normalizer.baseMVA)
    dev = svc.device
    wgen = np.ones(info.n_gen, dtype=np.float32)
    wgen[info.firm_gens[info.strategic]] = D_WEIGHT

    def loss_on(idx, train):
        raw = _raw(info, scns, idx, eff)
        b = svc.builder(raw, dev)
        keep = b["gen"].orig_index
        tgt = torch.as_tensor(pg[idx].reshape(-1) / base, device=dev)[keep]
        w = torch.as_tensor(np.tile(wgen, len(idx)), device=dev)[keep]
        with torch.no_grad():
            ro = ref.model(b.x_dict, b.edge_index_dict, b.edge_attr_dict, b.mask_dict)
        with torch.set_grad_enabled(train):
            out = model(b.x_dict, b.edge_index_dict, b.edge_attr_dict, b.mask_dict)
            lg = (w * (out["gen"][:, 0] - tgt) ** 2).sum() / w.sum()
            lb = ((out["bus"] - ro["bus"]) ** 2).mean()
            loss = lg + distill * lb
        mae_d = (out["gen"][:, 0] - tgt).abs().detach() * base
        d_mask = w > 1
        return loss, float(lg), float(lb), float(mae_d[d_mask].mean()), float(mae_d.mean())

    def evaluate(idx):
        model.eval()
        tot = np.zeros(5)
        for i in range(0, len(idx), 512):
            j = idx[i:i + 512]
            with torch.no_grad():
                loss, lg, lb, md, ma = loss_on(j, False)
            tot += np.array([float(loss), lg, lb, md, ma]) * len(j)
        return tot / len(idx)

    v0 = evaluate(va)
    print(f"frozen  val loss {v0[0]:.5f} gen {v0[1]:.5f} bus {v0[2]:.5f} | MAE D gens {v0[3]:.1f} MW, all {v0[4]:.1f} MW",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * int(np.ceil(len(tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    best, best_state, hist = v0[0], {k: v.detach().clone() for k, v in model.state_dict().items()}, []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = np.random.default_rng(ep).permutation(tr)
        for i in range(0, len(perm), batch):
            loss = loss_on(perm[i:i + batch], True)[0]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        v = evaluate(va)
        hist.append(v.tolist())
        flag = ""
        if v[0] < best:
            best, best_state, flag = v[0], {k: x.detach().clone() for k, x in model.state_dict().items()}, " *"
        print(f"epoch {ep:2d} {time.perf_counter() - t0:5.0f} s  val loss {v[0]:.5f} gen {v[1]:.5f} bus {v[2]:.5f} | "
              f"MAE D gens {v[3]:.1f} MW, all {v[4]:.1f} MW{flag}", flush=True)

    src = svc.ckpt_dir
    size = SIZE[data]
    dst = CKPT_ROOT / "case118_ieee" / size / "seed0"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "normalizer_stats.pt", dst / "normalizer_stats.pt")
    shutil.copy(src / "case118_ieee_small_seed0.yaml", dst / f"case118_ieee_{size}_seed0.yaml")
    torch.save(best_state, dst / "best_model_state_dict.pt")
    log = dict(frozen_val=v0.tolist(), best_val=best, history=hist, epochs=epochs, lr=lr, batch=batch, distill=distill,
               d_weight=D_WEIGHT, n_train=len(tr), n_val=len(va), data=data)
    (OUT / ("train_log.json" if data == "v1" else f"train_log_{data}.json")).write_text(json.dumps(log, indent=1))
    print(f"saved {dst}", flush=True)


def stage_head(cfg, info, epochs: int, data: str):
    import torch

    from genco_market.env.grid import scenario_to_network
    from genco_market.surrogate.genco_service import GencoService
    from genco_market.surrogate.lmp import head_dir, head_features, train_head

    scns, meta = scenarios(cfg, info, data)
    eff = eff_markups(meta)
    lab = labels(data)
    K = len(MARKUPS)
    ok = lab["ok"].reshape(-1)
    size = SIZE[data]
    svc = GencoService(size)
    nets = [scenario_to_network(s, info.bid_c1(s, m)) for s, row in zip(scns, eff) for m in row]
    emb = svc.predict_arrays(nets, 1024, embeddings=True)["emb_bus"].astype(np.float16)
    pd_ = np.repeat(np.array([s.bus_pd for s in scns]), K, axis=0)
    feat = head_features(pd_, eff.reshape(-1), info)
    val = np.repeat(split(cfg, meta), K)
    keep = np.where(ok)[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    h, st = train_head(emb[keep], feat[keep], lab["lmp"].reshape(-1, info.n_bus)[keep], val[keep], device=dev,
                       epochs=epochs)
    h.save(head_dir() / f"lmp_head_{size}.pt", {"best_val": st["best_val"], "strategic": info.strategic,
                                                   "n_train": int((~val[keep]).sum())})
    print({k: v for k, v in st.items() if k != "history"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["label", "train", "head"])
    ap.add_argument("--config", default="mvp.yaml")
    ap.add_argument("--data", default="v1", choices=list(SIZE))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--distill", type=float, default=1.0)
    ap.add_argument("--head-epochs", type=int, default=60)
    a = ap.parse_args()
    cfg = Config.load(a.config)
    info = GridInfo(cfg)
    if a.stage == "label":
        stage_label(cfg, info, a.workers, a.data)
    elif a.stage == "train":
        stage_train(cfg, info, a.epochs, a.lr, a.batch, a.distill, a.data)
    else:
        stage_head(cfg, info, a.head_epochs, a.data)


if __name__ == "__main__":
    main()
