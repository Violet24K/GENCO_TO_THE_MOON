"""Exp 1 + R4 (README 5.2, 6.3): fidelity of DC and GENCO profit models vs AC.

Scenario sets come from the semantic worlds: one PhysicalState per episode
(an interval with an active event when there is one, 80 %), ``--per-state``
scenarios drawn from it. A scenario set = the scenarios of one state, and a
"decision" = argmax of the set's mean profit over the 6 markups.

Stages (all outputs under <scratch>/exp1/, tables in results/):

    label  AC labels (LMP, Pg) on the training episodes, for the LMP head
    train  GENCO embeddings + LMP heads (Small/Base, and a no-GENCO ablation)
    eval   AC / DC-HiGHS / DC-IPOPT / GENCO x LMP methods on test episodes,
           metrics -> results/exp1_fidelity.csv, results/exp1_summary.md

    CUDA_VISIBLE_DEVICES=5 python experiments/exp1_fidelity.py --stage all --workers 16
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import subprocess
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from genco_market.core.config import REPO, Config, scratch_dir
from genco_market.core.schemas import Scenario
from genco_market.core.seeding import rng
from genco_market.env.grid import GridInfo
from genco_market.env.physical import baseline_zone_load, ood_flags, sample_scenario
from genco_market.estimators.multifidelity import variance_ratio

MARKUPS = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
HORIZON = 4
SUMMER = (3624, 6552)  # Jun 1 - Sep 30, hour of year
N_HOURS = 8264
SCARCITY_LMP = 500.0  # $/MWh: any AC LMP above this = VOLL scarcity (max c1 is ~125, x gas)
OUT = scratch_dir("exp1")


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------
def draw_scenarios(cfg: Config, info: GridInfo, episodes, per_state: int, tag: str):
    """-> (scenarios, meta DataFrame). Episodes start in summer half the time
    (heatwaves need a hot normal), otherwise anywhere in the year."""
    from genco_market.semantics.world import WorldGenerator

    gen = WorldGenerator(cfg, info)
    scns, meta = [], []
    for ep in episodes:
        r = rng(cfg.seed, "exp1", tag, ep)
        start = int(r.integers(*SUMMER)) if r.random() < 0.5 else int(r.integers(0, N_HOURS - HORIZON))
        w = gen.sample_world(ep, HORIZON, start)
        act = {t: [e for e in gen.labels(w, t).active_events if e.occurs] for t in range(HORIZON)}
        cand = [t for t in range(HORIZON) if act[t]]
        t = int(r.choice(cand)) if cand and r.random() < 0.8 else int(r.integers(HORIZON))
        types = sorted({e.event_type for e in act[t]})
        outage = [e.params["branch_name"] for e in act[t] if e.event_type == "planned_outage"]
        state = gen.physical_state(w, t)
        base = baseline_zone_load(info, start + t)
        for j in range(per_state):
            s = sample_scenario(state, info, base, rng(cfg.seed, "exp1", tag, ep, "scn", j))
            flags = ood_flags(s, info)
            scns.append(s)
            meta.append(dict(
                episode=ep, t=t, start_hour=start, state=f"{tag}{ep}", j=j,
                events="+".join(types) or "none", outage=",".join(outage),
                sys_load=float(np.sum(s.bus_pd) / info.base_pd.sum()),
                ood=",".join(flags), in_dist=not flags,
            ))
    return scns, pd.DataFrame(meta)


def load_or_draw(cfg, info, name, episodes, per_state):
    p = OUT / f"scenarios_{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            d = pickle.load(f)
        return [Scenario.model_validate(x) for x in d["scns"]], d["meta"]
    scns, meta = draw_scenarios(cfg, info, episodes, per_state, name)
    with open(p, "wb") as f:
        pickle.dump({"scns": [s.model_dump() for s in scns], "meta": meta}, f)
    return scns, meta


def make_pool(args, cfg):
    from genco_market.env.pool import SolverPool

    pool = SolverPool(args.workers, cfg)
    print(f"solver pool: {pool.n_workers} workers, startup {pool.startup_s:.0f} s")
    return pool


# --------------------------------------------------------------------------
# label / train
# --------------------------------------------------------------------------
def stage_label(args, cfg, info, pool):
    from genco_market.surrogate.profit_models import ACModel

    scns, meta = load_or_draw(cfg, info, "train", range(args.train_episodes), args.per_state)
    p = OUT / "labels_train.npz"
    if p.exists():
        print(f"labels exist: {p}")
        return
    ac = ACModel(pool, info)
    t0 = time.perf_counter()
    prof = ac.evaluate(scns, MARKUPS)
    print(f"AC labels: {len(scns)} x {len(MARKUPS)} in {time.perf_counter() - t0:.0f} s, "
          f"status {Counter(ac.last['status'].ravel())}")
    np.savez_compressed(p, lmp=ac.last["lmp"].astype(np.float32), pg=ac.last["pg"].astype(np.float32),
                        ok=ac.last["ok"], status=ac.last["status"].astype(str), profit=prof)


def genco_embeddings(svc, info, scns):
    from genco_market.env.grid import scenario_to_network

    nets = [scenario_to_network(s, info.bid_c1(s, m)) for s in scns for m in MARKUPS]
    out = svc.predict_arrays(nets, 1024, embeddings=True)
    return out["emb_bus"].astype(np.float16)


def stage_train(args, cfg, info):
    import torch

    from genco_market.surrogate.genco_service import GencoService
    from genco_market.surrogate.lmp import head_dir, head_features, train_head

    scns, meta = load_or_draw(cfg, info, "train", range(args.train_episodes), args.per_state)
    lab = np.load(OUT / "labels_train.npz")
    K = len(MARKUPS)
    ok = lab["ok"].reshape(-1)
    lmp = lab["lmp"].reshape(-1, info.n_bus)
    pd_ = np.repeat(np.array([s.bus_pd for s in scns]), K, axis=0)
    mk = np.tile(MARKUPS, len(scns))
    feat = head_features(pd_, mk, info)
    ep = np.repeat(meta.episode.to_numpy(), K)
    val = (np.array([rng(cfg.seed, "exp1-split", e).random() for e in ep]) < 0.15)
    keep = np.where(ok)[0]
    log = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # ablation: same head without GENCO embeddings
    h, st = train_head(None, feat[keep], lmp[keep], val[keep], device=dev, epochs=args.epochs)
    h.save(head_dir() / "lmp_head_noemb.pt", {"best_val": st["best_val"], "strategic": info.strategic})
    log["noemb"] = {k: v for k, v in st.items() if k != "history"}
    for size in ("small", "base"):
        svc = GencoService(size)
        emb = genco_embeddings(svc, info, scns)
        h, st = train_head(emb[keep], feat[keep], lmp[keep], val[keep], device=dev, epochs=args.epochs)
        h.save(head_dir() / f"lmp_head_{size}.pt", {"best_val": st["best_val"], "n_train": int((~val[keep]).sum()), "strategic": info.strategic})
        log[size] = {k: v for k, v in st.items() if k != "history"}
        del svc, emb
        torch.cuda.empty_cache()
    print(json.dumps(log, indent=1))
    (OUT / "train_log.json").write_text(json.dumps(log, indent=1))


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------
def run_models(args, cfg, info, pool, scns):
    import torch

    from genco_market.surrogate.genco_service import GencoService
    from genco_market.surrogate.lmp import LMPHead, head_dir
    from genco_market.surrogate.profit_models import ACModel, DCModel, GencoModel

    models = [ACModel(pool, info), DCModel(pool, "highs", info=info), DCModel(None, "highs", info=info),
              DCModel(pool, "ipopt", info=info)]
    svcs = {size: GencoService(size) for size in ("small", "base")}
    dc1 = models[2]
    for size, svc in svcs.items():
        models += [GencoModel(svc, "dc_hybrid", info=info, dc=dc1), GencoModel(svc, "autodiff", info=info),
                   GencoModel(svc, "head", info=info)]
    # GENCO Small dispatch with the no-GENCO head (ablation: are embeddings needed?)
    m = GencoModel(svcs["small"], "head", info=info, head=LMPHead.load(head_dir() / "lmp_head_noemb.pt", svcs["small"].device))
    m.name = "GENCO-small+head_noemb"
    models.append(m)

    raw, timing = {}, {}
    n_small = min(50, len(scns))
    for mdl in models:
        if isinstance(mdl, GencoModel):  # warm-up (CUDA kernels, cuDNN autotune)
            mdl.evaluate(scns[:8], MARKUPS)
        # agent-sized call (50 scenarios x 6) for the latency-relevant cost
        t0 = time.perf_counter()
        mdl.evaluate(scns[:n_small], MARKUPS)
        t_small = (time.perf_counter() - t0) / (n_small * len(MARKUPS))
        t0 = time.perf_counter()
        prof = mdl.evaluate(scns, MARKUPS)
        wall = time.perf_counter() - t0
        timing[mdl.name] = dict(cost_s=wall / (len(scns) * len(MARKUPS)), cost_s_n50=t_small)
        raw[mdl.name] = dict(profit=prof, lmp=mdl.last["lmp"].astype(np.float32), pg=mdl.last["pg"].astype(np.float32),
                             ok=mdl.last["ok"], status=mdl.last.get("status"), res_p=mdl.last.get("res_p_mw"))
        print(f"{mdl.name:28s} {timing[mdl.name]['cost_s'] * 1e3:8.3f} ms/eval (n50: {t_small * 1e3:.3f})"
              f"  NaN rows {np.isnan(prof).any(1).sum()}")
    del svcs
    torch.cuda.empty_cache()
    return raw, timing


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _demean(P, groups):
    out = P.copy()
    for g in np.unique(groups):
        i = groups == g
        out[i] -= np.nanmean(P[i], axis=0)
    return out


def metrics(P, Q, L_ac, L_m, meta, info, sel):
    """P, Q: (n, K) AC / model profit; L_*: (n, K, nb) LMP; sel: bool rows."""
    v = sel & np.isfinite(P).all(1) & np.isfinite(Q).all(1)
    p, q = P[v], Q[v]
    states = meta.state.to_numpy()[v]
    out = dict(n_scn=int(v.sum()), n_states=int(len(np.unique(states))))
    if v.sum() < 3:
        return out
    out["rho_pooled"] = _corr(p.ravel(), q.ravel())
    out["rho_per_markup"] = float(np.nanmean([_corr(p[:, k], q[:, k]) for k in range(p.shape[1])]))
    pw, qw = _demean(p, states), _demean(q, states)
    out["rho_within_state"] = _corr(pw.ravel(), qw.ravel())
    # response to the markup within each scenario (what ranking depends on)
    out["rho_markup_response"] = _corr((p - p[:, :1]).ravel(), (q - q[:, :1]).ravel())
    out["profit_mae"] = float(np.mean(np.abs(q - p)))
    out["profit_bias"] = float(np.mean(q - p))
    taus, regs, regp, taus1, regs1 = [], [], [], [], []
    for s in np.unique(states):
        i = states == s
        mu, nu = p[i].mean(0), q[i].mean(0)
        k = int(np.argmax(nu))
        taus.append(kendalltau(mu, nu).statistic if np.ptp(mu) > 0 and np.ptp(nu) > 0 else np.nan)
        regs.append(mu.max() - mu[k])
        regp.append(100 * (mu.max() - mu[k]) / mu.max() if mu.max() > 0 else np.nan)
    for a, b in zip(p, q):
        taus1.append(kendalltau(a, b).statistic if np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan)
        regs1.append(a.max() - a[int(np.argmax(b))])
    out.update(
        kendall_tau_set=float(np.nanmean(taus)), regret_set=float(np.mean(regs)),
        regret_set_pct=float(np.nanmean(regp)), argmax_hit_set=float(np.mean(np.array(regs) <= 1e-9)),
        kendall_tau_scn=float(np.nanmean(taus1)), regret_scn=float(np.mean(regs1)),
    )
    gb = info.net.gens[info.firm_gens[info.strategic], 0].astype(int)
    la, lm = L_ac[v][..., gb], L_m[v][..., gb]
    scar = np.nanmax(L_ac[v], axis=-1) >= SCARCITY_LMP  # (n, K)
    e = np.abs(lm - la)
    out["lmp_firm_mae"] = float(np.nanmean(e))
    out["lmp_firm_mae_normal"] = float(np.nanmean(e[~scar])) if (~scar).any() else np.nan
    out["lmp_firm_mae_scarcity"] = float(np.nanmean(e[scar])) if scar.any() else np.nan
    out["lmp_firm_medae"] = float(np.nanmedian(e))
    out["lmp_all_mae_normal"] = float(np.nanmean(np.abs(L_m[v] - L_ac[v])[~scar]))
    out["n_scarcity_evals"] = int(scar.sum())
    return out


def stage_eval(args, cfg, info, pool):
    scns, meta = load_or_draw(cfg, info, "test", range(100_000, 100_000 + args.test_episodes), args.per_state)
    p = OUT / "eval_raw.pkl"
    if p.exists() and not args.rerun:
        with open(p, "rb") as f:
            raw, timing = pickle.load(f)
    else:
        raw, timing = run_models(args, cfg, info, pool, scns)
        with open(p, "wb") as f:
            pickle.dump((raw, timing), f)
    ac = raw["AC"]
    P, L_ac = ac["profit"], ac["lmp"].astype(float)
    scar_row = (np.nanmax(L_ac, axis=(1, 2)) >= SCARCITY_LMP)
    meta = meta.assign(scarcity=scar_row)
    status = Counter(ac["status"].ravel())
    print(f"AC status: {dict(status)}")

    subsets = {"all": np.ones(len(meta), bool)}
    for ev in ["none", "heatwave", "planned_outage", "thunderstorm", "gas_price"]:
        subsets[f"event:{ev}"] = meta.events.str.contains(ev).to_numpy() if ev != "none" else (meta.events == "none").to_numpy()
    for br in ["38-37", "49-66"]:
        subsets[f"outage:{br}"] = meta.outage.str.contains(br).to_numpy()
    subsets["in_dist"] = meta.in_dist.to_numpy()
    subsets["ood"] = ~meta.in_dist.to_numpy()
    subsets["scarcity"] = scar_row
    subsets["no_scarcity"] = ~scar_row

    c_ac = timing["AC"]["cost_s"]
    rows = []
    for name, d in raw.items():
        if name == "AC":
            continue
        t = timing[name]  # wall clock of the whole evaluate call (hybrid: incl. its DC LP)
        for sub, sel in subsets.items():
            m = metrics(P, d["profit"], L_ac, d["lmp"].astype(float), meta, info, sel)
            row = dict(model=name, subset=sub, cost_s=t["cost_s"], cost_s_n50=t["cost_s_n50"],
                       speedup_vs_ac=c_ac / t["cost_s"], nan_rows=int(np.isnan(d["profit"][sel]).any(1).sum()), **m)
            if "rho_within_state" in m and np.isfinite(m["rho_within_state"]):
                vr = variance_ratio(m["rho_within_state"], c_ac / t["cost_s"])
                row["mf_ac_equiv_mult"] = 1 / vr  # README result 3: x more AC samples at equal budget
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "exp1_fidelity.csv", index=False)
    write_summary(df, meta, timing, status, raw)
    return df


def _hw() -> str:
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        gpu = "?"
    cpu = [l for l in open("/proc/cpuinfo") if l.startswith("model name")][0].split(":")[1].strip()
    return f"CPU {cpu}, GPU {gpu}"


def write_summary(df, meta, timing, status, raw):
    a = df[df.subset == "all"].set_index("model")
    lines = ["# Exp 1 / R4: profit-model fidelity vs AC-OPF (generated by experiments/exp1_fidelity.py)", ""]
    ev = Counter()
    for e in meta.drop_duplicates("state").events:
        for x in e.split("+"):
            ev[x] += 1
    lines += [
        f"- Test set: {meta.state.nunique()} states (episodes) x {meta.groupby('state').size().iloc[0]} scenarios = "
        f"{len(meta)} scenarios x markups {MARKUPS}; events per state: {dict(ev)}; "
        f"outages: {dict(Counter(x for x in meta.drop_duplicates('state').outage if x))}; "
        f"in-distribution scenarios {int(meta.in_dist.sum())}/{len(meta)}; scenarios with AC scarcity (LMP >= {SCARCITY_LMP:g}) "
        f"at some markup: {int(meta.scarcity.sum())}.",
        f"- AC status counts: {dict(status)}.",
        f"- Hardware: {_hw()}. AC / DC-IPOPT / DC-HiGHS(pool) use the 16-worker CPU solver pool (1 thread each); "
        "DC-HiGHS-1core runs in-process on one core; GENCO runs on one A100 (shared node) + 1 CPU core, "
        "all 6 markups of all scenarios in one batch. GENCO+dc_hybrid cost includes the 1-core DC LP.",
        "- cost_s = wall s per (scenario, markup) over the whole test set; n50 = same for an agent-sized call "
        "(50 scenarios x 6 markups). rho_within = correlation after removing each state's mean (what the "
        "multi-fidelity estimator inside one decision sees); MF x = 1 / variance ratio of README result 3 "
        "with rho_within and r = c_AC / c_model.",
        "",
        "## Cost vs fidelity (all scenarios)",
        "",
        "| model | ms/eval | ms/eval n50 | r = c_AC/c | rho pooled | rho within | rho markup-resp | Kendall tau (set) | regret $/h (set) | regret % | argmax hit | LMP firm MAE normal | LMP firm MAE scarcity | MF x |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| AC (truth) | {timing['AC']['cost_s'] * 1e3:.2f} | {timing['AC']['cost_s_n50'] * 1e3:.2f} | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |",
    ]
    for name, r in a.iterrows():
        lines.append(
            f"| {name} | {r.cost_s * 1e3:.3f} | {r.cost_s_n50 * 1e3:.3f} | {r.speedup_vs_ac:.0f} | {r.rho_pooled:.3f} | "
            f"{r.rho_within_state:.3f} | {r.rho_markup_response:.3f} | {r.kendall_tau_set:.2f} | {r.regret_set:.1f} | "
            f"{r.regret_set_pct:.2f} | {r.argmax_hit_set:.2f} | {r.lmp_firm_mae_normal:.2f} | {r.lmp_firm_mae_scarcity:.0f} | "
            f"{r.get('mf_ac_equiv_mult', np.nan):.1f} |"
        )
    lines += ["", "## rho within state / regret % by subset", ""]
    piv = df.pivot_table(index="model", columns="subset", values="rho_within_state")
    reg = df.pivot_table(index="model", columns="subset", values="regret_set_pct")
    nn = df[df.model == df.model.iloc[0]].set_index("subset").n_states
    cols = [c for c in piv.columns if c != "all"]
    lines.append("| model | " + " | ".join(f"{c} (n={nn.get(c, 0)})" for c in cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for name in piv.index:
        lines.append(f"| {name} | " + " | ".join(
            f"{piv.loc[name, c]:.2f} / {reg.loc[name, c]:.1f}%" if np.isfinite(piv.loc[name, c]) else "-" for c in cols) + " |")
    (REPO / "results" / "exp1_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["label", "train", "eval", "all"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--train-episodes", type=int, default=800)
    ap.add_argument("--test-episodes", type=int, default=200)
    ap.add_argument("--per-state", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--rerun", action="store_true", help="recompute eval_raw.pkl")
    args = ap.parse_args()
    cfg = Config.load()
    info = GridInfo(cfg)
    pool = None
    try:
        if args.stage in ("label", "all"):
            pool = make_pool(args, cfg)
            stage_label(args, cfg, info, pool)
        if args.stage in ("train", "all"):
            stage_train(args, cfg, info)
        if args.stage in ("eval", "all"):
            if pool is None and (args.rerun or not (OUT / "eval_raw.pkl").exists()):
                pool = make_pool(args, cfg)
            stage_eval(args, cfg, info, pool)
    finally:
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    main()
