"""Locational prices from GENCO, which only predicts dispatch.

Three ways to get LMPs [$/MWh, per bus] for a batch of (scenario, bid) networks:

- ``autodiff_lmp``: LMP_i = d(sum_g c1_g * Pg_hat_g) / d Pd_i through the
  frozen GENCO graph (bid costs, since the ISO minimizes the bid cost).
- ``dc_lmp``: LMPs from a DC-OPF (HiGHS LP) of the same scenario and bid; the
  "hybrid" pairs these prices with GENCO's AC dispatch.
- ``LMPHead``: a small MLP on frozen GENCO bus embeddings (+ load, bid and
  scenario-level features), trained on AC-OPF LMP labels. Prices are learned
  in the space y = asinh(lmp / LMP_SCALE): VOLL scarcity prices (up to
  $2000/MWh) would otherwise dominate a squared loss.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from genco_market.core.config import scratch_dir
from genco_market.core.schemas import MarketRules, Scenario
from genco_market.env.grid import GridInfo

LMP_SCALE = 50.0  # $/MWh; asinh is ~linear below this, ~log above


def to_y(lmp):
    return np.arcsinh(np.asarray(lmp) / LMP_SCALE)


def from_y(y):
    return LMP_SCALE * np.sinh(np.asarray(y))


def head_dir() -> Path:
    return scratch_dir("models")


# --------------------------------------------------------------------------
# (a) autodiff through GENCO
# --------------------------------------------------------------------------
def autodiff_lmp(service, nets, batch_size: int = 256) -> dict:
    """GENCO prediction plus d(bid cost)/dPd per bus. Returns ``pg`` [B, n_gen]
    MW, ``cost`` [B] $/h at the bid costs, ``lmp`` [B, n_bus] $/MWh. Needs one
    forward + backward pass per batch."""
    from gridfm_graphkit.datasets.globals import PD_H, PG_OUT_GEN

    from genco_market.surrogate.genco_adapter import raw_from_networks

    raw_all = raw_from_networks(nets)
    model, base = service.model, float(service.normalizer.baseMVA)
    n_gen = raw_all.gen.shape[1]
    pgs, costs, lmps = [], [], []
    for i in range(0, len(raw_all), batch_size):
        raw = raw_all[i : i + batch_size]
        B, nb = raw.bus.shape[:2]
        batch = service.builder(raw, service.device)
        x_bus = batch["bus"].x.detach().clone().requires_grad_(True)
        x_dict = {"bus": x_bus, "gen": batch["gen"].x}
        with torch.enable_grad():
            out = model(x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.mask_dict)
            pg_kept = out["gen"][:, PG_OUT_GEN] * base  # MW, in-service gens only
            c1 = torch.as_tensor(raw.gen[..., 4].reshape(-1), dtype=pg_kept.dtype, device=pg_kept.device)
            c1 = c1[batch["gen"].orig_index]
            cost = c1 * pg_kept
            (grad,) = torch.autograd.grad(cost.sum(), x_bus)
        pg = torch.zeros(B * n_gen, device=pg_kept.device)
        pg[batch["gen"].orig_index] = pg_kept.detach()
        pgs.append(pg.view(B, n_gen).cpu().numpy())
        cost_s = torch.zeros(B * n_gen, device=pg_kept.device)
        cost_s[batch["gen"].orig_index] = cost.detach()
        costs.append(cost_s.view(B, n_gen).sum(1).cpu().numpy())
        # x Pd is Pd_MW / baseMVA, so d cost / d Pd_MW = grad / baseMVA
        lmps.append((grad[:, PD_H] / base).view(B, nb).cpu().numpy())
    return {"pg": np.concatenate(pgs), "cost": np.concatenate(costs), "lmp": np.concatenate(lmps).astype(float)}


# --------------------------------------------------------------------------
# (b) DC-OPF prices of the same scenario and bid
# --------------------------------------------------------------------------
def dc_lmp(
    scenarios: list[Scenario],
    markups: list[float],
    info: GridInfo,
    pool=None,
    rules: MarketRules | None = None,
    task: str = "dc_highs",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lmp [n, K, n_bus], pg [n, K, n_gen], ok [n, K]) from DC-OPF clearing,
    in-process (HiGHS) or on ``pool`` with ``task`` ("dc_highs" | "dc")."""
    from genco_market.env.market import clear_dc

    n, K = len(scenarios), len(markups)
    if pool is None:
        assert task == "dc_highs", "DC-Ipopt needs Julia: pass a SolverPool"
        res = [clear_dc(s, m, info, rules, backend="highs") for s in scenarios for m in markups]
    else:
        res = pool.map(task, [(s, m, rules) for s in scenarios for m in markups])
    lmp = np.full((n * K, info.n_bus), np.nan)
    pg = np.full((n * K, info.n_gen), np.nan)
    ok = np.zeros(n * K, dtype=bool)
    for j, r in enumerate(res):
        if r and r.status != "FAILED":
            lmp[j], pg[j], ok[j] = r.lmp, r.pg, True
    return lmp.reshape(n, K, -1), pg.reshape(n, K, -1), ok.reshape(n, K)


# --------------------------------------------------------------------------
# (c) LMP head on frozen GENCO embeddings
# --------------------------------------------------------------------------
def head_features(scenarios_pd: np.ndarray, markup: np.ndarray, info: GridInfo) -> np.ndarray:
    """Hand features per (scenario-bid, bus): [Pd_b / 100, system load factor,
    markup, is a strategic-firm gen bus] -> [B, n_bus, 4]."""
    pd = np.asarray(scenarios_pd, dtype=np.float32)
    B, nb = pd.shape
    sys_f = pd.sum(1, keepdims=True) / info.base_pd.sum()
    cbus = np.zeros(nb, dtype=np.float32)
    cbus[info.net.gens[info.firm_gens[info.strategic], 0].astype(int)] = 1
    return np.stack(
        [pd / 100.0, np.broadcast_to(sys_f, (B, nb)), np.broadcast_to(np.asarray(markup, np.float32)[:, None], (B, nb)),
         np.broadcast_to(cbus, (B, nb))],
        axis=-1,
    ).astype(np.float32)


def head_features_profile(scenarios_pd: np.ndarray, profiles: np.ndarray, firms: list[str], info: GridInfo) -> np.ndarray:
    """Hand features for multi-firm bids: [Pd_b / 100, system load factor,
    markup of the firm owning gens at bus b (1 if none), 1 if bus b has a gen of
    any firm in ``firms`` (feature 3 = loss-weight indicator, as in head_features),
    markup of each firm in ``firms``] -> [B, n_bus, 4 + len(firms)]. ``profiles`` [B, len(firms)]."""
    pd = np.asarray(scenarios_pd, dtype=np.float32)
    prof = np.asarray(profiles, dtype=np.float32)
    B, nb = pd.shape
    sys_f = pd.sum(1, keepdims=True) / info.base_pd.sum()
    bus_m = np.ones((B, nb), dtype=np.float32)
    ind = np.zeros(nb, dtype=np.float32)
    for j, f in enumerate(firms):
        buses = info.net.gens[info.firm_gens[f], 0].astype(int)
        bus_m[:, buses] = prof[:, j:j + 1]
        ind[buses] = 1
    cols = [pd / 100.0, np.broadcast_to(sys_f, (B, nb)), bus_m, np.broadcast_to(ind, (B, nb))]
    cols += [np.broadcast_to(prof[:, j:j + 1], (B, nb)) for j in range(len(firms))]
    return np.stack(cols, axis=-1).astype(np.float32)


class LMPHead(nn.Module):
    """Per-bus MLP on [emb_b, mean_b' emb, hand features, bus id embedding]
    -> asinh(lmp / LMP_SCALE). ``emb_dim`` = 0 gives the no-GENCO ablation."""

    def __init__(self, emb_dim: int, n_feat: int = 4, n_bus: int = 118, hidden: int = 256, bus_dim: int = 16):
        super().__init__()
        self.emb_dim, self.n_feat, self.n_bus, self.hidden, self.bus_dim = emb_dim, n_feat, n_bus, hidden, bus_dim
        self.bus_id = nn.Embedding(n_bus, bus_dim)
        d = 2 * emb_dim + n_feat + bus_dim
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.register_buffer("emb_mu", torch.zeros(max(emb_dim, 1)))
        self.register_buffer("emb_sd", torch.ones(max(emb_dim, 1)))

    def forward(self, emb: torch.Tensor | None, feat: torch.Tensor) -> torch.Tensor:
        B, nb, _ = feat.shape
        parts = []
        if self.emb_dim:
            e = (emb - self.emb_mu) / self.emb_sd
            parts += [e, e.mean(1, keepdim=True).expand(-1, nb, -1)]
        parts += [feat, self.bus_id.weight[None].expand(B, -1, -1)]
        return self.net(torch.cat(parts, -1)).squeeze(-1)

    @torch.no_grad()
    def predict_lmp(self, emb: np.ndarray | None, feat: np.ndarray, batch: int = 2048) -> np.ndarray:
        dev = next(self.parameters()).device
        out = []
        for i in range(0, len(feat), batch):
            e = None if emb is None or not self.emb_dim else torch.as_tensor(emb[i : i + batch], device=dev).float()
            out.append(self(e, torch.as_tensor(feat[i : i + batch], device=dev)).cpu().numpy())
        return from_y(np.concatenate(out)).astype(float)

    def save(self, path: Path, meta: dict | None = None) -> None:
        cfg = dict(emb_dim=self.emb_dim, n_feat=self.n_feat, n_bus=self.n_bus, hidden=self.hidden, bus_dim=self.bus_dim)
        torch.save({"cfg": cfg, "state": self.state_dict(), "meta": meta or {}}, path)

    @staticmethod
    def load(path: Path, device="cpu") -> "LMPHead":
        d = torch.load(path, map_location="cpu", weights_only=False)
        h = LMPHead(**d["cfg"])
        h.load_state_dict(d["state"])
        h.meta = d.get("meta", {})
        return h.to(device).eval()


def train_head(
    emb: np.ndarray | None,
    feat: np.ndarray,
    lmp: np.ndarray,
    val: np.ndarray,
    device: str = "cpu",
    epochs: int = 60,
    lr: float = 2e-3,
    batch: int = 256,
    c_weight: float = 4.0,
    seed: int = 0,
    log=print,
) -> tuple[LMPHead, dict]:
    """Fit an LMPHead. emb [N, nb, E] (or None), feat [N, nb, F], lmp [N, nb]
    AC labels, val: boolean mask of validation rows (split by episode by the
    caller). Huber loss in asinh space, extra weight ``c_weight`` on the strategic firm's
    gen buses (feature 3). Keeps the epoch with the best validation loss."""
    torch.manual_seed(seed)
    N, nb, F = feat.shape
    E = 0 if emb is None else emb.shape[-1]
    h = LMPHead(E, F, nb).to(device)
    tr = np.where(~val)[0]
    va = np.where(val)[0]
    if E:
        sub = emb[tr[:: max(1, len(tr) // 2000)]].reshape(-1, E)
        h.emb_mu.copy_(torch.as_tensor(sub.mean(0)))
        h.emb_sd.copy_(torch.as_tensor(sub.std(0) + 1e-3))
    y = torch.as_tensor(to_y(lmp), dtype=torch.float32)
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, hist = np.inf, None, []

    def run(idx, train):
        h.train(train)
        tot, cnt = 0.0, 0
        perm = np.random.default_rng(seed + len(hist)).permutation(idx) if train else idx
        for i in range(0, len(perm), batch):
            j = perm[i : i + batch]
            e = torch.as_tensor(emb[j], device=device).float() if E else None
            f = torch.as_tensor(feat[j], device=device)
            t = y[j].to(device)
            with torch.set_grad_enabled(train):
                w = 1 + (c_weight - 1) * f[..., 3]
                loss = (nn.functional.huber_loss(h(e, f), t, reduction="none", delta=0.5) * w).sum() / w.sum()
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += float(loss.detach()) * len(j)
            cnt += len(j)
        return tot / max(cnt, 1)

    t0 = time.perf_counter()
    for ep in range(epochs):
        ltr = run(tr, True)
        sched.step()
        lva = run(va, False)
        hist.append((ltr, lva))
        if lva < best:
            best, best_state = lva, {k: v.detach().clone() for k, v in h.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  head E={E} epoch {ep:3d} train {ltr:.4f} val {lva:.4f}")
    h.load_state_dict(best_state)
    h.eval()
    return h, {"best_val": best, "train_s": time.perf_counter() - t0, "history": hist}
