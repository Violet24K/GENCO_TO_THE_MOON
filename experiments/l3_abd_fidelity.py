"""A+B+D played by main-line agents: what GENCO / DC / multi-fidelity buy when every firm must
estimate the 3-firm payoff table under a time budget.

Each firm is the main-line agent minus perception (scenarios from the true distribution, as in
Exp 2): within a wall-clock budget T it estimates expected profits of A, B, D on the 216 bid
profiles with AC only / DC / GENCO / AC+DC / AC+GENCO, then decides. Firms estimate
independently (own scenario draws). Truth = AC tensors of l3_abd (held-out half of the 80
scenarios).

Stages:
  head  - AC LMP labels for random (A, B, D) profiles on the Exp 1 training scenarios and a
          GENCO price head with per-firm markup features (models/lmp_head_small_ABD.pt).
  cheap - DC and GENCO profits [N, 216, 3] for every l3_abd state on N scenarios (the first 80
          are the l3_abd AC scenarios).
  play  - offline. Two ways to decide:
            nash - solve the game on its own estimated table, play its own part;
            br   - repeated hours: best response to the rivals' markups of the previous hour
                   (estimates only its own 6 markups, so 36x fewer evaluations), R rounds.
          Score the realised profile on the truth: Nash gap, match with the true Nash, consumer
          payment and firm profits vs. the true Nash.

    python experiments/l3_abd_fidelity.py --stage head
    python experiments/l3_abd_fidelity.py --stage cheap --n 400
    python experiments/l3_abd_fidelity.py --stage play
"""

from __future__ import annotations

import argparse
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import l3_abd as L  # noqa: E402

from genco_market.core.config import REPO, Config, scratch_dir  # noqa: E402
from genco_market.core.seeding import rng  # noqa: E402

FIRMS = L.STRAT  # A, B, D
SI = [L.FIRMS.index(f) for f in FIRMS]
OUT = scratch_dir("l3abd_fid")
HEAD_NAME = "lmp_head_small_ABD.pt"
N_NORMAL = 27
BUDGETS = [1, 2, 5, 10, 30, 60]
METHODS = ["ac", "dc", "genco", "mf_dc", "mf_genco"]


# --------------------------------------------------------------------------
# stage head (as l2_multifirm.stage_head, three firms)
# --------------------------------------------------------------------------
def stage_head(cfg, info, workers: int, n_scn: int, per: int, epochs: int):
    import torch

    from exp1_fidelity import load_or_draw
    from genco_market.env.grid import scenario_to_network
    from genco_market.env.pool import SolverPool
    from genco_market.surrogate.genco_service import GencoService
    from genco_market.surrogate.lmp import head_dir, head_features_profile, train_head
    from genco_market.surrogate.profit_models import ACModel

    scns, meta = load_or_draw(cfg, info, "train", range(800), 5)
    scns, meta = scns[:n_scn], meta.iloc[:n_scn].reset_index(drop=True)
    lab = OUT / "head_labels.npz"
    G = 50
    groups = [list(range(i, min(i + G, len(scns)))) for i in range(0, len(scns), G)]
    prof_idx = [rng(cfg.seed, "l3abd_head", g).choice(len(L.PROFILES), per, replace=False) for g in range(len(groups))]
    if not lab.exists():
        lmps, oks, P = [], [], []
        t0 = time.perf_counter()
        with SolverPool(workers, cfg) as pool:
            ac = ACModel(pool, info)
            for g, rows in enumerate(groups):
                profs = [L.PROFILES[k] for k in prof_idx[g]]
                ac.evaluate([scns[i] for i in rows], profs)
                lmps.append(ac.last["lmp"].reshape(-1, info.n_bus))
                oks.append(ac.last["ok"].reshape(-1))
                P.append(np.array([[p[f] for f in FIRMS] for _ in rows for p in profs]))
                if g % 10 == 0:
                    print(f"labels group {g}/{len(groups)} {time.perf_counter() - t0:.0f} s", flush=True)
        np.savez_compressed(lab, lmp=np.concatenate(lmps).astype(np.float32), ok=np.concatenate(oks),
                            prof=np.concatenate(P).astype(np.float32))
    d = np.load(lab)
    rows_scn = np.concatenate([np.repeat(rows, per) for rows in groups])
    pd_ = np.array([scns[i].bus_pd for i in rows_scn])
    feat = head_features_profile(pd_, d["prof"], FIRMS, info)
    svc = GencoService("small")
    nets = []
    for g, rows in enumerate(groups):
        profs = [L.PROFILES[k] for k in prof_idx[g]]
        nets += [scenario_to_network(scns[i], info.bid_c1(scns[i], p)) for i in rows for p in profs]
    emb = svc.predict_arrays(nets, 1024, embeddings=True)["emb_bus"].astype(np.float16)
    ep = meta.episode.to_numpy()[rows_scn]
    val = np.array([rng(cfg.seed, "exp1-split", e).random() for e in ep]) < 0.15
    keep = np.where(d["ok"])[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    h, st = train_head(emb[keep], feat[keep], d["lmp"][keep], val[keep], device=dev, epochs=epochs)
    h.save(head_dir() / HEAD_NAME, {"best_val": st["best_val"], "firms": FIRMS, "strategic": info.strategic,
                                     "n_train": int((~val[keep]).sum())})
    print({k: v for k, v in st.items() if k != "history"})


# --------------------------------------------------------------------------
# stage cheap: DC and GENCO tensors on the l3_abd states
# --------------------------------------------------------------------------
def stage_cheap(cfg, info, n: int, workers: int):
    from genco_market.env.env import MarketEnv
    from genco_market.env.pool import SolverPool
    from genco_market.semantics.world import WorldGenerator
    from genco_market.surrogate.genco_service import GencoService
    from genco_market.surrogate.lmp import head_dir
    from genco_market.surrogate.profit_models import DCModel, GencoModel

    env = MarketEnv(cfg, info, WorldGenerator(cfg, info), None)
    files = sorted(L.OUT.glob("state[0-9][0-9][0-9].npz"))
    with SolverPool(workers, cfg) as pool:
        dc = DCModel(pool, "highs", info=info)
        gm = GencoModel(GencoService("small"), info=info, head=head_dir() / HEAD_NAME)
        for f in files:
            s = int(f.stem[5:])
            path = OUT / f"cheap_{f.stem}.npz"
            if path.exists():
                continue
            t0 = time.perf_counter()
            scns = L._scenarios(cfg, info, env, s, N_NORMAL, n)
            ac_pd = np.load(f)["pd"]
            assert np.allclose(np.array([x.bus_pd for x in scns[:len(ac_pd)]]), ac_pd, atol=1e-3), "scenario mismatch"
            out = dict(dc=dc.evaluate_profiles(scns, L.PROFILES, FIRMS).astype(np.float32))
            out["genco"] = gm.evaluate_profiles(scns, L.PROFILES, FIRMS).astype(np.float32)
            out["cost"] = np.array([dc.cost_per_eval_s(), gm.cost_per_eval_s()])
            np.savez(path, **out)
            print(f"{f.stem}: {time.perf_counter() - t0:.0f} s, cost/eval ms DC {1e3 * out['cost'][0]:.3f} "
                  f"GENCO {1e3 * out['cost'][1]:.3f}", flush=True)


# --------------------------------------------------------------------------
# stage play
# --------------------------------------------------------------------------
AC_COST_MS = 19.65  # median AC cost per evaluation (16 solver processes) measured on our machine


def estimate(method, T, P_cols, ac, lo, pa, ex, c, rho=0.7):
    """Estimated expected profits [len(P_cols), 3] within budget T (s), on the profile columns P_cols."""
    from genco_market.estimators.multifidelity import multifidelity, single_fidelity, split_budget

    P = len(P_cols)
    if method == "ac":
        k = min(len(pa), int(T / (P * c["ac"])))
        if k < 1:  # not even one scenario of the table fits in the budget
            return None, 0, 0
        return single_fidelity(ac[pa[:k]][:, P_cols].reshape(k, -1)).mean.reshape(P, 3), k, 0
    m = method.replace("mf_", "")
    L_ = lo[m]
    if method in ("dc", "genco"):
        rows = np.concatenate([pa, ex])
        kk = max(1, min(len(rows), int(T / (P * c[m]))))
        return single_fidelity(L_[rows[:kk]][:, P_cols].reshape(kk, -1)).mean.reshape(P, 3), 0, kk
    if T < 2 * P * c["ac"]:  # the control variate needs >= 2 paired AC scenarios: use the cheap model alone
        return estimate(m, T, P_cols, ac, lo, pa, ex, c)
    n_hi, n_lo = split_budget(T, c["ac"], c[m], rho, P)
    n_hi = min(n_hi, len(pa))
    lo_ids = np.concatenate([pa[:n_hi], ex])[:max(n_lo, n_hi)]
    hi = ac[pa[:n_hi]][:, P_cols].reshape(n_hi, -1)
    est = multifidelity(hi, L_[lo_ids][:, P_cols].reshape(len(lo_ids), -1)).mean.reshape(P, 3)
    return est, n_hi, len(lo_ids)


def nash_gap(U, k) -> float:
    a = L.PROF_IDX[k]
    g = 0.0
    for f in range(3):
        q = [L.pidx(*[x if j != f else y for j, x in enumerate(a)]) for y in range(L.NK)]
        g = max(g, U[q, f].max() - U[k, f])
    return float(g)


def stage_play(reps: int, rounds: int, ac_cost_ms: float = AC_COST_MS):
    c = {"ac": ac_cost_ms / 1e3}
    files = [f for f in sorted(L.OUT.glob("state[0-9][0-9][0-9].npz")) if (OUT / f"cheap_{f.stem}.npz").exists()]
    cheap = {f.stem: np.load(OUT / f"cheap_{f.stem}.npz") for f in files}
    cc = np.array([cheap[k]["cost"] for k in cheap])
    c["dc"], c["genco"] = float(np.median(cc[:, 0])), float(np.median(cc[:, 1]))
    print("cost per eval (ms):", {k: round(1e3 * v, 3) for k, v in c.items()},
          "| evals for one full 216 table in 1 s: AC", round(1 / (216 * c["ac"]), 2))
    rows = []
    for f in files:
        z, meta, prof, W, ok = L.load_state(f)
        ac_all = prof[..., SI]  # [n_ok, 216, 3]
        okrows = np.where(ok)[0]
        lo = {m: cheap[f.stem][m] for m in ("dc", "genco")}
        lo_ok = ~(np.isnan(lo["dc"]).any((1, 2)) | np.isnan(lo["genco"]).any((1, 2)))
        n_ac = len(ok)
        # AC rows are indexed in the full scenario numbering below
        ac_full = np.full((n_ac,) + ac_all.shape[1:], np.nan)
        ac_full[okrows] = ac_all
        good = okrows[lo_ok[okrows]]
        half = len(good) // 2
        truth = ac_full[good[half:]].mean(0)
        pay = np.nanmean(z["welfare"][good[half:]][:, :, 1].astype(float), 0)  # consumer payment [216]
        ne, _ = L.nash(truth)
        ne = ne or L.ibr(truth)
        k_true = L.pidx(*ne)
        paired = good[:half]
        extra = np.where(lo_ok)[0]
        extra = extra[extra >= n_ac]
        scarce = bool(np.mean((z["welfare"][okrows, 0, 2] > 1.0) |
                              (z["lmp"][okrows, 0].astype(float).max(1) >= L.SCARCITY_LMP)) >= 0.10)
        r = np.random.default_rng(zlib.crc32(f.name.encode()))
        allP = np.arange(len(L.PROFILES))
        for T in BUDGETS:
            for method in METHODS:
                for rep in range(reps):
                    # --- nash: each firm estimates the full table on its own draws, plays its own part
                    a = []
                    for fi in range(3):
                        pa, ex = r.permutation(paired), r.permutation(extra)
                        U, n_hi, n_lo = estimate(method, T, allP, ac_full, lo, pa, ex, c)
                        if U is None:
                            break
                        p, _ = L.nash(U)
                        p = p or L.ibr(U)
                        a.append(p[fi])
                    if len(a) == 3:
                        rows.append(_score("nash", f.stem, scarce, T, method, rep, truth, pay, k_true, a, n_hi, n_lo))
                    # --- br: repeated hours, best response to the rivals' previous markups
                    a = [0, 0, 0]
                    for _ in range(rounds):
                        new = []
                        for fi in range(3):
                            cols = np.array([L.pidx(*[a[j] if j != fi else y for j in range(3)]) for y in range(L.NK)])
                            pa, ex = r.permutation(paired), r.permutation(extra)
                            U, n_hi, n_lo = estimate(method, T, cols, ac_full, lo, pa, ex, c)
                            if U is None:  # cannot evaluate its 6 markups in time: keep the previous markup
                                new.append(a[fi])
                                continue
                            new.append(int(np.argmax(U[:, fi])))
                        a = new
                    rows.append(_score(f"br{rounds}", f.stem, scarce, T, method, rep, truth, pay, k_true, a, n_hi, n_lo))
        print(f"play {f.stem} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(scratch_dir("results") / "l3abd_fid_play.parquet")
    for mode in ("nash", f"br{rounds}"):
        for lab, sub in (("normal", d[(d["mode"] == mode) & ~d.scarce]), ("scarcity", d[(d["mode"] == mode) & d.scarce])):
            for col in ("nash_gap", "match", "d_payment_vs_nash"):
                t = sub.pivot_table(index="method", columns="budget", values=col).reindex(METHODS)
                t = t.round(3 if col == "match" else 0)
                print(f"\n== {mode} / {lab}: {col}\n{t.to_string()}")
                t.to_csv(REPO / "results" / f"l3abd_fid_{mode}_{lab}_{col}.csv")


def _score(mode, state, scarce, T, method, rep, truth, pay, k_true, a, n_hi, n_lo):
    k = L.pidx(*a)
    return dict(mode=mode, state=state, scarce=scarce, budget=T, method=method, rep=rep,
                mA=L.K[a[0]], mB=L.K[a[1]], mD=L.K[a[2]], nash_gap=nash_gap(truth, k), match=k == k_true,
                d_payment_vs_nash=pay[k] - pay[k_true], loss_A=truth[k_true, 0] - truth[k, 0],
                loss_B=truth[k_true, 1] - truth[k, 1], loss_D=truth[k_true, 2] - truth[k, 2], n_hi=n_hi, n_lo=n_lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["head", "cheap", "play"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--head-scenarios", type=int, default=2000)
    ap.add_argument("--head-per", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--ac-cost-ms", type=float, default=AC_COST_MS, help="AC cost per evaluation used for the budgets")
    a = ap.parse_args()
    if a.stage == "play":
        stage_play(a.reps, a.rounds, a.ac_cost_ms)
        return
    from genco_market.env.grid import GridInfo

    cfg = Config.load("mvp.yaml")
    info = GridInfo(cfg)
    if a.stage == "head":
        stage_head(cfg, info, a.workers, a.head_scenarios, a.head_per, a.epochs)
    else:
        stage_cheap(cfg, info, a.n, a.workers)


if __name__ == "__main__":
    main()
