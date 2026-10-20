"""Train the operator GENCO's social-cost head (genco_market/surrogate/social_head.py) on the
Layer 4 training episodes from experiments/l4_op_data.py; validation = the last 10% of episodes.
Reports the validation correlation of the outage delta (head vs. AC), overall and per line.

    CUDA_VISIBLE_DEVICES=6 python experiments/l4_op_train.py --genco-size small_ft2
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from genco_market.core.config import REPO, scratch_dir


def load(genco_size: str):
    fs = sorted(scratch_dir("l4", f"op_train_{genco_size}").glob("ep*.npz"))
    parts = [np.load(f) for f in fs]
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    return {k: cat(k) for k in ("meta", "ac_cost", "ac_shed", "emb", "genco_cost", "bus_pd", "off")}, len(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="l4.yaml")
    ap.add_argument("--genco-size", default="small_ft2")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()

    import torch

    from genco_market.env.grid import GridInfo
    from genco_market.env.outage import load_l4
    from genco_market.surrogate.social_head import head_features, head_path, train_head

    cfg, l4 = load_l4(a.config)
    info = GridInfo(cfg)
    d, nf = load(a.genco_size)
    N = len(d["meta"])  # triplets: line in, requested line out, other line out
    ok = np.isfinite(d["ac_cost"].reshape(N, 3)).all(1)
    nb, E = d["emb"].shape[1:]
    emb = d["emb"].reshape(N, 3, nb, E)[ok]
    feat = head_features(d["bus_pd"], d["off"], info).reshape(N, 3, nb, -1)[ok]
    resid = (d["ac_cost"] - d["genco_cost"]).reshape(N, 3)[ok]
    meta = d["meta"][ok]
    eps = meta[:, 0]
    val = eps >= np.quantile(np.unique(eps), 0.9)
    print(f"{nf} files, {N} triplets ({ok.mean():.3f} AC ok), train {int((~val).sum())}, val {int(val.sum())}; "
          f"shed>1MW share {np.nanmean(d['ac_shed'] > 1):.3f}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    h, st = train_head(emb, feat, resid, val, device=dev, epochs=a.epochs)

    # validation: outage-delta correlation with AC, overall and per line (requested line and the other line)
    r = h.predict_residual(emb[val].reshape(-1, nb, E), feat[val].reshape(-1, nb, feat.shape[-1])).reshape(-1, 3)
    gc = d["genco_cost"].reshape(N, 3)[ok][val]
    ac = d["ac_cost"].reshape(N, 3)[ok][val]
    rows = []
    for k in (1, 2):
        br = meta[val][:, 2 + k]
        for name, sel in [("all", np.ones(len(br), bool))] + [(info.branch_name[b], br == b) for b in np.unique(br)]:
            if sel.sum() < 5:
                continue
            da = ac[sel, k] - ac[sel, 0]
            dh = (gc[sel, k] + r[sel, k]) - (gc[sel, 0] + r[sel, 0])
            dg = gc[sel, k] - gc[sel, 0]
            cr = lambda x: float(np.corrcoef(da, x)[0, 1]) if da.std() > 0 and x.std() > 0 else np.nan  # noqa: E731
            rows.append(dict(which="requested" if k == 1 else "other", line=name, n=int(sel.sum()), ac_delta=da.mean(),
                             op_delta=dh.mean(), genco_delta=dg.mean(), rho_op=cr(dh), rho_genco=cr(dg)))
    rep = pd.DataFrame(rows).round(3)
    print(rep.to_string(index=False))
    rep.to_csv(REPO / "results" / f"l4_op_head_val_{a.genco_size}.csv", index=False)
    h.save(head_path(a.genco_size), {"genco_size": a.genco_size, "best_val": st["best_val"], "train_files": nf,
                                     "episodes": [int(eps.min()), int(eps.max())]})
    print(json.dumps({"best_val": st["best_val"], "train_s": round(st["train_s"])}))


if __name__ == "__main__":
    main()
