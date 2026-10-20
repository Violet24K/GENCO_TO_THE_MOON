"""Operator GENCO: social cost (true generation cost + VOLL * shed) from GENCO.

GENCO is trained on feasible OPF solutions only, so it never predicts load
shedding, which dominates the social cost of risky outages (Layer 4). The
``SocialHead`` is a small per-bus MLP on frozen GENCO bus embeddings (+ load
and outage-endpoint features, bus id embedding); per-bus outputs are summed
into y, and the residual over GENCO's own generation cost is
r = SC_SCALE * sinh(y), so shed costs up to ~$10^5/h stay in a well-scaled
range. Trained on AC labels with a loss on both the per-network residual and
the outage delta (line out minus line in on the same scenario), since the
delta is what the operator's estimator (and its control variate) uses.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from genco_market.env.grid import GridInfo
from genco_market.surrogate.lmp import head_dir

SC_SCALE = 1000.0  # $/h


def to_y(x):
    return np.arcsinh(np.asarray(x) / SC_SCALE)


def head_features(bus_pd: np.ndarray, off: np.ndarray, info: GridInfo) -> np.ndarray:
    """[B, n_bus, 3]: Pd_b / 100, system load factor, bus is an endpoint of an out-of-service branch."""
    pd = np.asarray(bus_pd, np.float32)
    B, nb = pd.shape
    fb, tb = info.net.branches[:, 0].astype(int), info.net.branches[:, 1].astype(int)
    end = np.zeros((B, nb), np.float32)
    rows, br = np.nonzero(np.asarray(off))
    end[rows, fb[br]] = 1
    end[rows, tb[br]] = 1
    sys_f = pd.sum(1, keepdims=True) / info.base_pd.sum()
    return np.stack([pd / 100.0, np.broadcast_to(sys_f, (B, nb)), end], -1).astype(np.float32)


class SocialHead(nn.Module):
    def __init__(self, emb_dim: int, n_feat: int = 3, n_bus: int = 118, hidden: int = 256, bus_dim: int = 16):
        super().__init__()
        self.emb_dim, self.n_feat, self.n_bus, self.hidden, self.bus_dim = emb_dim, n_feat, n_bus, hidden, bus_dim
        self.bus_id = nn.Embedding(n_bus, bus_dim)
        d = 2 * emb_dim + n_feat + bus_dim
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer("emb_mu", torch.zeros(emb_dim))
        self.register_buffer("emb_sd", torch.ones(emb_dim))

    def forward(self, emb: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """y [B] (asinh space)."""
        B, nb, _ = feat.shape
        e = (emb - self.emb_mu) / self.emb_sd
        x = torch.cat([e, e.mean(1, keepdim=True).expand(-1, nb, -1), feat, self.bus_id.weight[None].expand(B, -1, -1)], -1)
        return self.net(x).squeeze(-1).sum(1)

    @torch.no_grad()
    def predict_residual(self, emb: np.ndarray, feat: np.ndarray, batch: int = 2048) -> np.ndarray:
        """r = SC - GENCO generation cost, $/h."""
        dev = next(self.parameters()).device
        out = [self(torch.as_tensor(emb[i : i + batch], device=dev).float(), torch.as_tensor(feat[i : i + batch], device=dev))
               for i in range(0, len(feat), batch)]
        return SC_SCALE * np.sinh(torch.cat(out).cpu().numpy().astype(float))

    def save(self, path: Path, meta: dict | None = None) -> None:
        cfg = dict(emb_dim=self.emb_dim, n_feat=self.n_feat, n_bus=self.n_bus, hidden=self.hidden, bus_dim=self.bus_dim)
        torch.save({"cfg": cfg, "state": self.state_dict(), "meta": meta or {}}, path)

    @staticmethod
    def load(path: Path, device="cpu") -> "SocialHead":
        d = torch.load(path, map_location="cpu", weights_only=False)
        h = SocialHead(**d["cfg"])
        h.load_state_dict(d["state"])
        h.meta = d.get("meta", {})
        return h.to(device).eval()


def head_path(genco_size: str) -> Path:
    return head_dir() / f"social_head_{genco_size}.pt"


def train_head(emb, feat, resid, val, device="cpu", epochs=40, lr=2e-3, batch=128, w_delta=2.0, seed=0, log=print):
    """emb [N, 3, nb, E], feat [N, 3, nb, F], resid [N, 3] AC SC minus GENCO generation cost, per scenario
    triplet (line in, line out, other line out); val: boolean mask over N (split by episode by the caller).
    Huber loss in asinh space on each network's residual and on both outage deltas (out - in).
    Keeps the epoch with the best validation loss."""
    torch.manual_seed(seed)
    N, K, nb, E = emb.shape
    h = SocialHead(E, feat.shape[-1], nb).to(device)
    tr, va = np.where(~val)[0], np.where(val)[0]
    sub = emb[tr[:: max(1, len(tr) // 1000)]].reshape(-1, E).astype(np.float32)
    h.emb_mu.copy_(torch.as_tensor(sub.mean(0)))
    h.emb_sd.copy_(torch.as_tensor(sub.std(0) + 1e-3))
    y = torch.as_tensor(to_y(resid), dtype=torch.float32)
    # delta targets: SC(out) - SC(in) = resid(out) - resid(in) + GENCO cost delta; the head models the residual part
    yd = torch.as_tensor(to_y(resid[:, 1:] - resid[:, :1]), dtype=torch.float32)
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, hist = np.inf, None, []

    def run(idx, train):
        h.train(train)
        tot, cnt = 0.0, 0
        perm = np.random.default_rng(seed + len(hist)).permutation(idx) if train else idx
        for i in range(0, len(perm), batch):
            j = perm[i : i + batch]
            e = torch.as_tensor(emb[j], device=device).float().reshape(-1, nb, E)
            f = torch.as_tensor(feat[j], device=device).reshape(-1, nb, feat.shape[-1])
            with torch.set_grad_enabled(train):
                yh = h(e, f).view(len(j), K)
                r = SC_SCALE * torch.sinh(yh)
                dh = torch.asinh((r[:, 1:] - r[:, :1]) / SC_SCALE)
                loss = (nn.functional.huber_loss(yh, y[j].to(device), delta=0.5)
                        + w_delta * nn.functional.huber_loss(dh, yd[j].to(device), delta=0.5))
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(h.parameters(), 5.0)
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
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"  social head epoch {ep:3d} train {ltr:.4f} val {lva:.4f}")
    h.load_state_dict(best_state)
    h.eval()
    return h, {"best_val": best, "train_s": time.perf_counter() - t0, "history": hist}
