"""Three strategic firms A, B, D: payoff tensors and game outcomes.

Stages:
  tensor - for S world states: n scenarios x all 216 (mA, mB, mD) profiles, AC
           clearing (genco_market.env.ac_detail:ac_detail, with load shedding at VOLL).
           Saves per (scenario, profile): profit of every firm, true generation
           cost, consumer payment, shed MW, and the bus LMPs (float16, for
           impact tests / price caps). The first n_normal states reuse the
           Layer 2 worlds and scenario seeds (so the A = 1.0 slice matches the
           B x D tensors); the rest force a scarcity event (cold snap in winter
           or a heatwave on a summer afternoon) at its first active interval.
  game   - offline: per state, expected payoffs, the pure Nash equilibria of
           the 3-firm game (the one with the largest joint profit), L1, and
           restricted games (B + D with A at cost = the old Layer 2 setting;
           B + D with A fixed at 3.0, its dominant strategy). States are
           labelled "scarcity" when the all-at-cost clearing sheds load or
           reaches an LMP >= SCARCITY_LMP in at least 10% of scenarios.

  pivotal - for each scenario and each of A, B, D: clear with that firm's units removed (others at
           cost), to tell whether it is a pivotal supplier (rule R3).
  rules  - offline: regulation rules R0-R5 applied on the tensors;
           firms naive (keep the unregulated Nash profile) or anticipating (Nash of the
           regulated game); metrics of 12.5 split by scarcity / normal states.

    python experiments/l3_abd.py --stage tensor --normal 27 --scarce 13 --n 80 [--reverse]
    python experiments/l3_abd.py --stage game
    python experiments/l3_abd.py --stage pivotal
    python experiments/l3_abd.py --stage rules
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd

from genco_market.core.config import REPO, Config, scratch_dir
from genco_market.core.seeding import rng, sample_start_hour


STRAT = ["A", "B", "D"]
FIRMS = ["A", "B", "C", "D", "E"]
K = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
NK = len(K)
PROFILES = [dict(zip(STRAT, p)) for p in itertools.product(K, K, K)]  # index = iA*36 + iB*6 + iD
OUT = scratch_dir("l3abd")
SCARCITY_LMP = 500.0
TERMS = ["true_cost", "payment", "shed"]


def pidx(iA: int, iB: int, iD: int) -> int:
    return (iA * NK + iB) * NK + iD


# --------------------------------------------------------------------------
# stage tensor
# --------------------------------------------------------------------------
def scarce_world(cfg, s: int):
    """(episode id, start hour, event type) of the s-th forced scarcity state."""
    r = rng(cfg.seed, "l3abd_scarce", s)
    if s % 2 == 0:  # cold snap: winter morning / evening peak
        return 7000 + s, int(r.integers(0, 59)) * 24 + int(r.choice([6, 7, 8, 17, 18, 19])), "cold_snap"
    return 7000 + s, int(r.integers(182, 243)) * 24 + int(r.integers(13, 17)), "heatwave"


def stage_tensor(cfg, info, n_normal: int, n_scarce: int, n: int, workers: int, reverse: bool = False):
    from genco_market.env.env import MarketEnv
    from genco_market.env.physical import sample_scenario
    from genco_market.env.pool import SolverPool
    from genco_market.env.welfare import welfare_terms
    from genco_market.semantics.world import WorldGenerator

    gen = WorldGenerator(cfg, info)
    env = MarketEnv(cfg, info, gen, None)
    with SolverPool(workers, cfg) as pool:
        order = list(range(n_normal + n_scarce))
        for s in (order[::-1] if reverse else order):  # a second process can work from the end
            path = OUT / f"state{s:03d}.npz"
            if path.exists() or (OUT / f"state{s:03d}.lock").exists():
                continue
            (OUT / f"state{s:03d}.lock").touch()
            t0 = time.perf_counter()
            if s < n_normal:  # Layer 2 / Exp 2 worlds and scenario seeds
                env.reset(1000 + s, sample_start_hour(cfg.seed + 7, s))
                t = int(rng(cfg.seed, "exp2_t", s).integers(env.world.horizon))
                seed_key, kind = ("l2", s), "normal"
            else:
                ep, start, et = scarce_world(cfg, s - n_normal)
                env.reset(ep, start, event_types=[et])
                ev = [e for e in env.world.events if e.event_type == et][0]
                t = min(max(ev.start_t, 0), env.world.horizon - 1)
                seed_key, kind = ("l3abd", s), f"forced_{et}"
            state, base = env.true_state(t), env.baseline(t)
            scns = [sample_scenario(state, info, base, rng(cfg.seed, *seed_key, i)) for i in range(n)]
            res = pool.map("genco_market.env.ac_detail:ac_detail", [(x, p) for x in scns for p in PROFILES])
            P = len(PROFILES)
            prof = np.full((n, P, len(FIRMS)), np.nan)
            W = np.full((n, P, len(TERMS)), np.nan)
            lmp = np.zeros((n, P, info.n_bus), dtype=np.float16)
            gens = sorted(g for g in info.gen_firm if info.base_pmax[g] > 0)  # firm units with capacity
            pg = np.zeros((n, P, len(gens)), dtype=np.float32)
            shed_b = np.zeros((n, P, info.n_bus), dtype=np.float16)
            status = np.empty((n, P), dtype=object)
            for j, r in enumerate(res):
                i, p = divmod(j, P)
                if not isinstance(r, dict) or not np.all(np.isfinite(r["lmp"])):
                    status[i, p] = "FAILED"
                    continue
                status[i, p] = r["status"]
                w = welfare_terms(r["lmp"], r["pg"], scns[i], info, shed_mw=np.maximum(r.get("shed", 0.0), 0.0))
                prof[i, p] = [r["profit"][f] for f in FIRMS]
                W[i, p] = (w["true_cost"], w["consumer_payment"], w["shed_mw"])
                lmp[i, p] = np.clip(r["lmp"], -1e4, 6e4)
                pg[i, p] = np.asarray(r["pg"])[gens]
                shed_b[i, p] = np.maximum(np.asarray(r.get("shed", np.zeros(info.n_bus)), float), 0.0)
            pd_ = np.array([x.bus_pd for x in scns], dtype=np.float32)
            evs = sorted({e.event_type for e in env.world.events if e.occurs and e.start_t <= t <= e.end_t})
            meta = dict(kind=kind, events=evs, t=t, start_hour=env.world.start_hour, load=float(sum(base.values())),
                        n_failed=int((status == "FAILED").sum()), n_dc_fallback=int((status == "DC_FALLBACK").sum()))
            np.savez(path.with_suffix(".tmp.npz"), profit=prof.astype(np.float32), welfare=W.astype(np.float32), lmp=lmp, pd=pd_, pg=pg, shed=shed_b, gens=np.array(gens),
                     c1=np.array([x.gen_c1 for x in scns], dtype=np.float32)[:, gens],
                     meta=json.dumps(meta))
            path.with_suffix(".tmp.npz").rename(path)
            (OUT / f"state{s:03d}.lock").unlink(missing_ok=True)
            print(f"state {s} ({kind}): events={evs} {time.perf_counter() - t0:.0f} s, failed {meta['n_failed']}, "
                  f"DC fallback {meta['n_dc_fallback']}", flush=True)


# --------------------------------------------------------------------------
# game solving on U [216, 3] (expected profits of A, B, D)
# --------------------------------------------------------------------------
def br(U, firm: int, prof: tuple[int, int, int], allowed=None) -> int:
    """Best markup index of ``firm`` (0=A, 1=B, 2=D) given the others in ``prof``."""
    opts = allowed if allowed is not None else range(NK)
    vals = []
    for k in opts:
        q = list(prof)
        q[firm] = k
        vals.append(U[pidx(*q), firm])
    return list(opts)[int(np.argmax(vals))]


def nash(U, free=(0, 1, 2), fixed=None) -> tuple[tuple[int, int, int] | None, int]:
    """Pure Nash equilibria among the ``free`` firms (others fixed at ``fixed`` indices).
    Returns (equilibrium with the largest joint profit of the free firms, count)."""
    fixed = fixed or {}
    grids = [range(NK) if f in free else [fixed.get(f, 0)] for f in range(3)]
    eq = [p for p in itertools.product(*grids) if all(br(U, f, p) == p[f] for f in free)]
    if not eq:
        return None, 0
    return max(eq, key=lambda p: sum(U[pidx(*p), f] for f in free)), len(eq)


def ibr(U, start=(0, 0, 0), free=(0, 1, 2), iters=30) -> tuple[int, int, int]:
    p = list(start)
    for _ in range(iters):
        for f in free:
            p[f] = br(U, f, tuple(p))
    return tuple(p)


def load_state(f):
    z = np.load(f)
    meta = json.loads(str(z["meta"]))
    prof, W = z["profit"].astype(float), z["welfare"].astype(float)
    ok = ~(np.isnan(prof).any((1, 2)) | np.isnan(W).any((1, 2)))
    return z, meta, prof[ok], W[ok], ok


def stage_game():
    rows = []
    sA, sB, sD = FIRMS.index("A"), FIRMS.index("B"), FIRMS.index("D")
    for f in sorted(OUT.glob("state[0-9][0-9][0-9].npz")):
        z, meta, prof, W, ok = load_state(f)
        U = prof[..., [sA, sB, sD]].mean(0)  # [216, 3]
        Wm, Pm = W.mean(0), prof.mean(0)
        lmp0 = z["lmp"][ok][:, 0].astype(float)
        scarce = float(np.mean((W[:, 0, 2] > 1.0) | (lmp0.max(1) >= SCARCITY_LMP))) >= 0.10
        c0 = pidx(0, 0, 0)
        L1 = (br(U, 0, (0, 0, 0)), br(U, 1, (0, 0, 0)), br(U, 2, (0, 0, 0)))
        ne, n_ne = nash(U)
        ne_bd, n_bd = nash(U, free=(1, 2), fixed={0: 0})
        ne_bd3, n_bd3 = nash(U, free=(1, 2), fixed={0: NK - 1})
        behaviours = {
            "cost": ((0, 0, 0), 1), "A_only": ((br(U, 0, (0, 0, 0)), 0, 0), 1), "L1": (L1, 1),
            "Nash_ABD": (ne if ne else ibr(U, L1), n_ne),
            "Nash_BD_Acost": (ne_bd if ne_bd else ibr(U, (0, L1[1], L1[2]), free=(1, 2)), n_bd),
            "Nash_BD_A3": (ne_bd3 if ne_bd3 else ibr(U, (NK - 1, L1[1], L1[2]), free=(1, 2)), n_bd3),
        }
        for name, (p, n_eq) in behaviours.items():
            k = pidx(*p)
            rows.append(dict(state=f.stem, kind=meta["kind"], scarce=scarce, events="+".join(meta["events"]) or "none",
                             behaviour=name, mA=K[p[0]], mB=K[p[1]], mD=K[p[2]], n_eq=n_eq,
                             d_payment=Wm[k, 1] - Wm[c0, 1], d_true_cost=Wm[k, 0] - Wm[c0, 0],
                             d_shed=Wm[k, 2] - Wm[c0, 2], payment_cost=Wm[c0, 1],
                             **{f"profit_{x}": Pm[k, i] for i, x in enumerate(FIRMS)},
                             **{f"d_profit_{x}": Pm[k, i] - Pm[c0, i] for i, x in enumerate(FIRMS)}))
    d = pd.DataFrame(rows)
    d.to_parquet(scratch_dir("results") / "l3abd_game.parquet")
    cols = ["mA", "mB", "mD", "d_payment", "d_true_cost", "d_shed"] + [f"d_profit_{x}" for x in FIRMS]
    order = ["cost", "A_only", "Nash_BD_Acost", "Nash_BD_A3", "L1", "Nash_ABD"]
    for lab, sub in (("all", d), ("normal", d[~d.scarce]), ("scarcity", d[d.scarce])):
        s = sub.groupby("behaviour")[cols].mean().reindex(order).round(1)
        print(f"== {lab}: {sub.state.nunique()} states\n{s.to_string()}")
        s.to_csv(REPO / "results" / f"l3abd_game_{lab}.csv")
    ne = d[d.behaviour == "Nash_ABD"]
    print("pure Nash (3 firms): states with >=1:", int((ne.n_eq > 0).sum()), "of", len(ne),
          "| count distribution:", ne.n_eq.value_counts().sort_index().to_dict())
    print("Nash profiles:", ne.groupby(["mA", "mB", "mD"]).size().sort_values(ascending=False).head(8).to_dict())


# --------------------------------------------------------------------------
# stage pivotal (rule R3): clear without each strategic firm, others at cost
# --------------------------------------------------------------------------
def _scenarios(cfg, info, env, s: int, n_normal: int, n: int):
    from genco_market.env.physical import sample_scenario

    if s < n_normal:
        env.reset(1000 + s, sample_start_hour(cfg.seed + 7, s))
        t = int(rng(cfg.seed, "exp2_t", s).integers(env.world.horizon))
        key = ("l2", s)
    else:
        ep, start, et = scarce_world(cfg, s - n_normal)
        env.reset(ep, start, event_types=[et])
        ev = [e for e in env.world.events if e.event_type == et][0]
        t = min(max(ev.start_t, 0), env.world.horizon - 1)
        key = ("l3abd", s)
    state, base = env.true_state(t), env.baseline(t)
    return [sample_scenario(state, info, base, rng(cfg.seed, *key, i)) for i in range(n)]


def stage_pivotal(cfg, info, n_normal: int, workers: int):
    from genco_market.env.env import MarketEnv
    from genco_market.env.pool import SolverPool
    from genco_market.semantics.world import WorldGenerator

    env = MarketEnv(cfg, info, WorldGenerator(cfg, info), None)
    files = sorted(OUT.glob("state[0-9][0-9][0-9].npz"))
    with SolverPool(workers, cfg) as pool:
        for f in files:
            path = OUT / f"pivotal_{f.stem}.npz"
            if path.exists():
                continue
            s = int(f.stem[5:])
            z = np.load(f)
            n = z["profit"].shape[0]
            scns = _scenarios(cfg, info, env, s, n_normal, n)
            assert np.allclose(np.array([x.bus_pd for x in scns], dtype=np.float32), z["pd"]), "scenario mismatch"
            jobs = []
            for x in scns:
                for firm in STRAT:
                    st = list(x.gen_status)
                    for g in info.firm_gens[firm]:
                        st[g] = 0
                    jobs.append((x.model_copy(update={"gen_status": st}), 1.0))
            res = pool.map("genco_market.env.ac_detail:ac_detail", jobs)
            shed = np.full((n, 3), np.nan)
            avg = np.full((n, 3), np.nan)
            for j, r in enumerate(res):
                i, k = divmod(j, 3)
                if isinstance(r, dict) and np.all(np.isfinite(r["lmp"])):
                    shed[i, k] = float(np.maximum(r.get("shed", 0.0), 0.0).sum())
                    avg[i, k] = float(np.dot(r["lmp"], scns[i].bus_pd) / max(sum(scns[i].bus_pd), 1e-9))
            np.savez(path, shed=shed, avg_lmp=avg)
            print(f"pivotal {f.stem}: shed>1MW share A/B/D = {np.round(np.nanmean(shed > 1, 0), 2).tolist()}", flush=True)


# --------------------------------------------------------------------------
# stage rules
# --------------------------------------------------------------------------
VOLL = 2000.0
RULES = ([("R0", {})]
         + [(f"R1_cap{c:g}", {"cap": c}) for c in (1.5, 2.0, 2.5)]
         + [(f"R2_c{tc:g}_i{ti:g}", {"tc": tc, "ti": ti}) for tc in (0.1, 0.5, 1.0) for ti in (2.0, 5.0, 10.0, 25.0)]
         + [("R3_shed", {"piv": "shed"}), ("R3_price20", {"piv": 20.0}), ("R3_price50", {"piv": 50.0})]
         + [("R4_A_only", {"only_A": True})]
         + [(f"R5_price{c:g}", {"price_cap": c}) for c in (300.0, 500.0, 1000.0)])
MARK = np.array(K)
PROF_IDX = np.array(list(itertools.product(range(NK), range(NK), range(NK))))  # [216, 3] markup indices of A, B, D


def _avg_lmp(lmp, pd_):
    w = pd_ / np.maximum(pd_.sum(1, keepdims=True), 1e-9)
    return np.einsum("npb,nb->np", lmp, w)


def _flat(idx3):
    return (idx3[..., 0] * NK + idx3[..., 1]) * NK + idx3[..., 2]


def effective(rule: dict, avg, piv):
    """Effective profile index [n, 216] after bid mitigation, and the mitigation flags [n, 216, 3]."""
    n = avg.shape[0]
    idx = np.broadcast_to(PROF_IDX, (n, len(PROFILES), 3)).copy()
    flag = np.zeros((n, len(PROFILES), 3), dtype=bool)
    m = MARK[PROF_IDX]  # [216, 3]
    if "cap" in rule:
        flag[:] = m > rule["cap"] + 1e-9
    elif "tc" in rule:
        for f in range(3):
            ref = PROF_IDX.copy()
            ref[:, f] = 0
            drop = avg - avg[:, _flat(ref)]  # LMP drop if f alone is reset to cost
            flag[..., f] = (m[:, f] > 1 + rule["tc"] + 1e-9) & (drop > rule["ti"])
    elif "piv" in rule:
        flag[:] = (m > 1.0)[None] & piv[:, None, :]
    elif rule.get("only_A"):
        flag[..., 0] = m[:, 0] > 1.0
    idx[flag] = 0
    return _flat(idx), flag


def settle(z, ok, eff, price_cap=None):
    """Per-scenario profits of all firms [n, 216, 5] and (true cost, payment, shed) [n, 216, 3] at the
    effective profiles; with a price cap, settlement LMPs are clipped (dispatch unchanged)."""
    prof, W = z["profit"][ok].astype(float), z["welfare"][ok].astype(float)
    n = prof.shape[0]
    rows = np.arange(n)[:, None]
    P, Wx = prof[rows, eff], W[rows, eff]
    if price_cap is not None:
        lmp = z["lmp"][ok].astype(float)[rows, eff]
        pg = z["pg"][ok].astype(float)[rows, eff]
        shed = z["shed"][ok].astype(float)[rows, eff]
        pd_ = z["pd"][ok].astype(float)
        gens = z["gens"]
        info = _info()
        gbus = info.net.gens[gens, 0].astype(int)
        cut = np.maximum(lmp - price_cap, 0.0)  # $/MWh removed by the cap at each bus
        P = P.copy()
        for k, firm in enumerate(FIRMS):
            sel = np.array([info.gen_firm.get(int(g)) == firm for g in gens])
            if sel.any():
                P[..., k] -= (cut[..., gbus[sel]] * pg[..., sel]).sum(-1)
        Wx = Wx.copy()
        Wx[..., 1] -= (cut * (pd_[:, None, :] - shed)).sum(-1)
    return P, Wx


_INFO = None


def _info():
    global _INFO
    if _INFO is None:
        from genco_market.env.grid import GridInfo

        _INFO = GridInfo(Config.load("mvp.yaml"))
    return _INFO


def stage_rules():
    sA, sB, sD = (FIRMS.index(x) for x in STRAT)
    rows = []
    for f in sorted(OUT.glob("state[0-9][0-9][0-9].npz")):
        z, meta, prof, W, ok = load_state(f)
        pv = OUT / f"pivotal_{f.stem}.npz"
        piv_z = np.load(pv) if pv.exists() else None
        lmp = z["lmp"][ok].astype(float)
        pd_ = z["pd"][ok].astype(float)
        avg = _avg_lmp(lmp, pd_)
        c0 = 0
        scarce = float(np.mean((W[:, c0, 2] > 1.0) | (lmp[:, c0].max(1) >= SCARCITY_LMP))) >= 0.10
        U0 = prof[..., [sA, sB, sD]].mean(0)
        ne0, _ = nash(U0)
        ne0 = ne0 or ibr(U0)
        p_ref_normal = 36.0  # $/MWh, typical competitive load-weighted LMP of normal states (all at cost)
        pay_comp = W[:, c0, 1].mean()
        scar_prem = max(pay_comp - p_ref_normal * pd_.sum(1).mean(), 0.0)
        for name, rule in RULES:
            if "piv" in rule:
                if piv_z is None:
                    continue
                if rule["piv"] == "shed":
                    piv = piv_z["shed"][ok] > 1.0
                else:
                    piv = (piv_z["avg_lmp"][ok] - avg[:, [c0]]) > rule["piv"]
                piv = np.nan_to_num(piv, nan=0).astype(bool)
            else:
                piv = None
            eff, flag = effective(rule, avg, piv)
            Pm, Wm = settle(z, ok, eff, rule.get("price_cap"))
            Um = Pm[..., [sA, sB, sD]].mean(0)
            ne, n_eq = nash(Um)
            ne = ne or ibr(Um)
            _, Wc = settle(z, ok, np.zeros_like(eff), rule.get("price_cap"))  # competitive outcome under the rule
            for beh, p in (("naive", ne0), ("anticipating", ne)):
                k = pidx(*p)
                fl = flag[:, k]  # [n, 3] who got mitigated at the played profile
                # impact of each mitigation: avg LMP drop when that firm alone is reset
                imp = np.zeros_like(fl, dtype=float)
                for fi in range(3):
                    q = list(p)
                    q[fi] = 0
                    imp[:, fi] = avg[:, k] - avg[:, pidx(*q)]
                n_mit = int(fl.sum())
                pay = Wm[:, k, 1].mean()
                rows.append(dict(
                    state=f.stem, kind=meta["kind"], scarce=scarce, rule=name, behaviour=beh,
                    mA=K[p[0]], mB=K[p[1]], mD=K[p[2]], n_eq=n_eq,
                    d_payment=pay - pay_comp, d_social_cost=(Wm[:, k, 0] + VOLL * Wm[:, k, 2]).mean()
                    - (W[:, c0, 0] + VOLL * W[:, c0, 2]).mean(),
                    markup_premium=pay - Wc[:, c0, 1].mean(), scarcity_removed=pay_comp - Wc[:, c0, 1].mean(),
                    scarcity_premium=scar_prem, mitig_rate=fl.any(1).mean(),
                    false_mitig=float((fl & (imp < 1.0)).sum() / n_mit) if n_mit else np.nan, n_mitig=n_mit,
                    **{f"profit_{x}": Pm[:, k, i].mean() for i, x in enumerate(FIRMS)}))
        print(f"rules {f.stem} ({'scarce' if scarce else 'normal'}) done", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(scratch_dir("results") / "l3abd_rules.parquet")
    r0 = d[(d.rule == "R0")].set_index(["state", "behaviour"])
    d["premium_removed"] = [1 - row.markup_premium / r0.loc[(row.state, row.behaviour), "markup_premium"]
                            if r0.loc[(row.state, row.behaviour), "markup_premium"] > 1 else np.nan for row in d.itertuples()]
    d["scarcity_removed_frac"] = d.scarcity_removed / d.scarcity_premium.where(d.scarcity_premium > 1)
    cols = ["mA", "mB", "mD", "d_payment", "markup_premium", "premium_removed", "scarcity_removed_frac", "d_social_cost",
            "mitig_rate", "false_mitig"] + [f"profit_{x}" for x in FIRMS]
    for lab, sub in (("normal", d[~d.scarce]), ("scarcity", d[d.scarce])):
        s = sub.groupby(["behaviour", "rule"])[cols].mean().round(3)
        print(f"== {lab}: {sub.state.nunique()} states\n{s.to_string()}")
        s.to_csv(REPO / "results" / f"l3abd_rules_{lab}.csv")
    d.to_parquet(scratch_dir("results") / "l3abd_rules.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tensor", "game", "pivotal", "rules"])
    ap.add_argument("--normal", type=int, default=27)
    ap.add_argument("--scarce", type=int, default=13)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--reverse", action="store_true", help="process states from the last one (run a second copy)")
    a = ap.parse_args()
    if a.stage == "tensor":
        from genco_market.env.grid import GridInfo

        cfg = Config.load("mvp.yaml")  # the config the Layer 2 tensors were built with (same worlds and scenarios)
        stage_tensor(cfg, GridInfo(cfg), a.normal, a.scarce, a.n, a.workers, a.reverse)
    elif a.stage == "pivotal":
        from genco_market.env.grid import GridInfo

        cfg = Config.load("mvp.yaml")
        stage_pivotal(cfg, GridInfo(cfg), a.normal, a.workers)
    elif a.stage == "game":
        stage_game()
    else:
        stage_rules()


if __name__ == "__main__":
    main()
