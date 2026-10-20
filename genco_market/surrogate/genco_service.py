"""GENCO-OPF inference service.

Loads one official checkpoint + normalizer once onto a GPU and predicts
batches of datakit ``Network``s (bid costs already in ``gencosts``):

    svc = GencoService("small")            # or "base"
    out = svc.predict(nets, batch_size=512)  # list of dicts, one per network
    arr = svc.predict_arrays(nets)           # same, stacked [B, ...] arrays

Per scenario: ``pg`` [n_gen] MW (original gen order, 0 for out-of-service),
``vm`` [n_bus] p.u., ``va`` [n_bus] rad, ``pg_bus``/``qg_bus`` [n_bus] MW/MVar,
``cost`` $/h at the input costs, and the label-free power-balance residual of
the predicted state (same formulas as graphkit ``OptimalPowerFlowTask``):
``res_p_mw`` / ``res_q_mvar`` = mean |residual| over buses, ``pbe_mean`` /
``pbe_max`` = mean / max |S residual| in MVA.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from gridfm_datakit.network import Network
from gridfm_graphkit.datasets.globals import PG_OUT, QG_OUT, VA_OUT, VM_OUT
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task_transforms, load_model, load_normalizer
from gridfm_graphkit.models.utils import ComputeBranchFlow, ComputeNodeInjection, ComputeNodeResiduals
from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model

from genco_market.surrogate.genco_adapter import BB, GB, OPFInputBuilder, RawBatch, raw_from_networks

CKPT_ROOT = Path(os.environ.get("GENCO_OPF_CKPT", Path(__file__).resolve().parents[2] / "checkpoints" / "genco-opf-datakit"))


def pick_gpu() -> str:
    """Least-memory-used visible GPU (respects CUDA_VISIBLE_DEVICES), else cpu."""
    if not torch.cuda.is_available():
        return "cpu"
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda:0"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        load = [tuple(map(float, line.split(","))) for line in out.strip().splitlines()]
        return f"cuda:{min(range(len(load)), key=lambda i: load[i])}"
    except Exception:
        return "cuda:0"


class GencoService:
    def __init__(
        self,
        size: str = "small",
        grid: str = "case118_ieee",
        seed: int = 0,
        weights: str = "best_model_state_dict.pt",  # what produced metrics.csv; "last.pt" also works
        device: str | None = None,
        deterministic: bool = False,  # bit-reproducible GPU runs, ~4x slower
        ckpt_root: Path = CKPT_ROOT,
    ):
        d = Path(ckpt_root) / grid / size / f"seed{seed}"
        self.ckpt_dir = d
        self.args = NestedNamespace(**yaml.safe_load((d / f"{grid}_{size}_seed{seed}.yaml").read_text()))
        assert self.args.task.task_name == "OptimalPowerFlow"
        self.device = torch.device(device or pick_gpu())
        self.deterministic = deterministic
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        self.normalizer = load_normalizer(self.args)
        stats = torch.load(d / "normalizer_stats.pt", map_location="cpu", weights_only=True)
        self.normalizer.fit_from_dict(stats[grid])
        self.builder = OPFInputBuilder(self.normalizer, get_task_transforms(self.args))

        self.model = load_model(self.args)
        sd = torch.load(d / weights, map_location="cpu", weights_only=True)
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        self.model.load_state_dict(adapt_state_dict_for_model(sd, self.model))
        self.model.to(self.device).eval()
        self.branch_flow = ComputeBranchFlow()
        self.node_injection = ComputeNodeInjection()
        self.node_residuals = ComputeNodeResiduals()
        self._emb = None
        self.timing: dict[str, float] = {}

    # ------------------------------------------------------------------
    def predict(self, networks: list[Network], batch_size: int = 512, **kw) -> list[dict]:
        arr = self.predict_arrays(networks, batch_size, **kw)
        return [{k: v[i] for k, v in arr.items()} for i in range(len(networks))]

    def predict_arrays(self, networks: list[Network], batch_size: int = 512, embeddings: bool = False) -> dict:
        t0 = time.perf_counter()
        parts = [raw_from_networks(networks[i : i + batch_size]) for i in range(0, len(networks), batch_size)]
        self.timing = {"raw_s": time.perf_counter() - t0}
        return self.predict_raw(RawBatch.concat(parts), batch_size, embeddings)

    def predict_raw(self, raw: RawBatch, batch_size: int = 512, embeddings: bool = False) -> dict:
        prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(self.deterministic or prev, warn_only=True)
        try:
            return self._predict_raw(raw, batch_size, embeddings)
        finally:
            torch.use_deterministic_algorithms(prev)

    @torch.no_grad()
    def _predict_raw(self, raw: RawBatch, batch_size: int, embeddings: bool) -> dict:
        tm = {"build_s": 0.0, "model_s": 0.0, "post_s": 0.0}
        outs = []
        for i in range(0, len(raw), batch_size):
            t0 = time.perf_counter()
            batch = self.builder(raw[i : i + batch_size], self.device)
            self._sync()
            t1 = time.perf_counter()
            out = self._forward(batch, embeddings)
            self._sync()
            t2 = time.perf_counter()
            o = self._postprocess(batch, out, raw.gen.shape[1])
            g = raw.gen[i : i + batch_size]  # raw costs; off gens have pg = 0
            o["cost"] = (g[..., 6] * g[..., 3] + g[..., 4] * o["pg"] + g[..., 5] * o["pg"] ** 2).sum(1)
            outs.append(o)
            tm["build_s"] += t1 - t0
            tm["model_s"] += t2 - t1
            tm["post_s"] += time.perf_counter() - t2
        self.timing.update(tm)
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------
    def _forward(self, batch, embeddings: bool):
        hook = None
        if embeddings:
            def grab(_m, inp):
                self._emb = inp[0]
            hook = self.model.mlp_bus.register_forward_pre_hook(grab)  # last call = final layer
        try:
            out = self.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.mask_dict)
        finally:
            if hook is not None:
                hook.remove()
        if embeddings:
            out["emb_bus"] = self._emb
        return out

    def _postprocess(self, batch, out: dict, n_gen: int) -> dict:
        """Same steps as OptimalPowerFlowTask.predict_step, vectorized per scenario."""
        B = batch.num_scenarios
        self.normalizer.inverse_transform(batch)
        self.normalizer.inverse_output(out, batch)
        bus_out = out["bus"]
        nb = bus_out.shape[0] // B
        ei, ea = batch[BB].edge_index, batch[BB].edge_attr
        pft, qft = self.branch_flow(bus_out, ei, ea)
        p_in, q_in = self.node_injection(pft, qft, ei, bus_out.shape[0])
        rp, rq = self.node_residuals(p_in, q_in, bus_out, batch["bus"].x)
        rp, rq = rp.abs().view(B, nb), rq.abs().view(B, nb)
        pbe = torch.sqrt(rp**2 + rq**2)

        pg = torch.zeros(B * n_gen, device=bus_out.device)
        pg[batch["gen"].orig_index] = out["gen"].view(-1)
        res = {
            "pg": pg.view(B, n_gen),
            "vm": bus_out[:, VM_OUT].view(B, nb),
            "va": bus_out[:, VA_OUT].view(B, nb),
            "pg_bus": bus_out[:, PG_OUT].view(B, nb),
            "qg_bus": bus_out[:, QG_OUT].view(B, nb),
            "res_p_mw": rp.mean(1),
            "res_q_mvar": rq.mean(1),
            "pbe_mean": pbe.mean(1),
            "pbe_max": pbe.max(1).values,
        }
        if "emb_bus" in out:
            res["emb_bus"] = out["emb_bus"].view(B, nb, -1)
        return {k: v.float().cpu().numpy() for k, v in res.items()}
