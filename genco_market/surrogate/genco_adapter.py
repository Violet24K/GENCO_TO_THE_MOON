"""In-memory adapter: datakit ``Network`` -> GENCO-OPF input graph.

Reproduces, without parquet round-trips, what the official pipeline feeds the
model for one scenario:

  datakit ``pf_post_processing`` (raw rows)
  -> graphkit ``merge_agg_gen_into_bus`` + ``build_hetero_data_for_scenario``
  -> ``HeteroDataMVANormalizer.transform``
  -> ``OptimalPowerFlowTransforms`` (drop inactive branches/gens, OPF mask, mask fill)
  -> PyG batching.

Raw features are built vectorized for a whole batch (``RawBatch``, float64 like
the parquet columns). The batch is then laid out as one disjoint "union" graph in
exactly the order PyG's ``Batch.from_data_list`` produces, and graphkit's own
normalizer and transforms are applied to it; all of them act row-wise, so this
equals transforming each scenario separately and collating.

Solution columns are unknown at inference and set to 0 (bus Qg/Vm/Va, gen Pg,
branch flows). Under the OPF mask they are either masked (filled with
``mask_value``) or are exactly 0 in the official data (REF Va, PQ-bus Qg).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from gridfm_datakit.network import Network, branch_vectors
from gridfm_datakit.utils.idx_brch import ANGMAX, ANGMIN, BR_STATUS, F_BUS, RATE_A, T_BUS, TAP
from gridfm_datakit.utils.idx_bus import BASE_KV, BS, BUS_I, BUS_TYPE, GS, PD, PQ, PV, QD, REF, VMAX, VMIN
from gridfm_datakit.utils.idx_cost import COST, NCOST
from gridfm_datakit.utils.idx_gen import GEN_BUS, GEN_STATUS, PMAX, PMIN, QMAX, QMIN
from gridfm_graphkit.datasets.globals import PG_H, VA_H
from gridfm_graphkit.datasets.hetero_preprocess import (
    BUS_FEATURES,
    FORWARD_BRANCH_FEATURES,
    GEN_FEATURES,
    REVERSE_BRANCH_FEATURES,
)
from torch_geometric.data import HeteroData

BB = ("bus", "connects", "bus")
GB = ("gen", "connected_to", "bus")
BG = ("bus", "connected_to", "gen")


@dataclass
class RawBatch:
    """Unnormalized per-scenario features (graphkit column order), B scenarios
    of one grid. Shapes: bus [B,nb,15], gen [B,ng,7], fwd/rev [B,nl,11],
    from_bus/to_bus [B,nl], gen_bus [B,ng]."""

    bus: np.ndarray
    gen: np.ndarray
    fwd: np.ndarray
    rev: np.ndarray
    from_bus: np.ndarray
    to_bus: np.ndarray
    gen_bus: np.ndarray

    def __len__(self) -> int:
        return self.bus.shape[0]

    def __getitem__(self, s) -> "RawBatch":
        s = slice(s, s + 1) if isinstance(s, (int, np.integer)) else s
        return RawBatch(*(getattr(self, f)[s] for f in self.__dataclass_fields__))

    @staticmethod
    def concat(parts: list["RawBatch"]) -> "RawBatch":
        return RawBatch(
            *(np.concatenate([getattr(p, f) for p in parts]) for f in RawBatch.__dataclass_fields__)
        )


# ---------------------------------------------------------------------------
# Network(s) -> RawBatch (mirrors datakit pf_post_processing + graphkit merge)
# ---------------------------------------------------------------------------
def _gen_costs(gencosts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(c0, c1, c2) from MATPOWER gencost rows, as datakit writes them."""
    n_cost = int(gencosts.reshape(-1, gencosts.shape[-1])[0, NCOST])
    assert np.all(gencosts[..., NCOST] == n_cost), "NCOST must be uniform"
    z = np.zeros(gencosts.shape[:-1])
    if n_cost == 3:
        return gencosts[..., COST + 2], gencosts[..., COST + 1], gencosts[..., COST]
    if n_cost == 2:
        return gencosts[..., COST + 1], gencosts[..., COST], z
    if n_cost == 1:
        return gencosts[..., COST], z, z
    raise ValueError(f"unsupported NCOST={n_cost}")


def raw_from_networks(nets: list[Network]) -> RawBatch:
    """Vectorized raw features for networks sharing one grid (same sizes)."""
    buses = np.stack([n.buses for n in nets])
    gens = np.stack([n.gens for n in nets])
    branches = np.stack([n.branches for n in nets])
    gencosts = np.stack([n.gencosts for n in nets])
    base_mva = np.array([float(n.baseMVA) for n in nets])[:, None]
    B, nb, _ = buses.shape
    ng, nl = gens.shape[1], branches.shape[1]
    assert np.all(buses[:, :, BUS_I] == np.arange(nb)), "buses must be in index order"

    gen_on = gens[:, :, GEN_STATUS] == 1
    gen_bus = gens[:, :, GEN_BUS].astype(np.int64)
    rows = np.repeat(np.arange(B), ng)

    # PV buses without an in-service gen are PQ (datakit Network.deactivate_gens);
    # a no-op for networks built through deactivate_gens.
    n_on = np.zeros((B, nb))
    np.add.at(n_on, (rows, gen_bus.ravel()), gen_on.ravel())
    btype = buses[:, :, BUS_TYPE].copy()
    btype[(btype == PV) & (n_on == 0)] = PQ

    # bus Q limits: sum over *all* gen rows at the bus, in service or not
    # (graphkit merge_agg_gen_into_bus groups gen_data without filtering)
    qmin = np.zeros((B, nb))
    qmax = np.zeros((B, nb))
    np.add.at(qmin, (rows, gen_bus.ravel()), gens[:, :, QMIN].ravel())
    np.add.at(qmax, (rows, gen_bus.ravel()), gens[:, :, QMAX].ravel())

    bus = np.zeros((B, nb, len(BUS_FEATURES)))
    bus[..., 0] = buses[..., PD]
    bus[..., 1] = buses[..., QD]
    # 2..4: Qg, Vm, Va unknown -> 0
    bus[..., 5] = btype == PQ
    bus[..., 6] = btype == PV
    bus[..., 7] = btype == REF
    bus[..., 8] = buses[..., VMIN]
    bus[..., 9] = buses[..., VMAX]
    bus[..., 10] = qmin
    bus[..., 11] = qmax
    bus[..., 12] = buses[..., GS] / base_mva
    bus[..., 13] = buses[..., BS] / base_mva
    bus[..., 14] = buses[..., BASE_KV]

    c0, c1, c2 = _gen_costs(gencosts)
    gen = np.zeros((B, ng, len(GEN_FEATURES)))
    # 0: Pg unknown -> 0
    gen[..., 1] = gens[..., PMIN]
    gen[..., 2] = gens[..., PMAX]
    gen[..., 3] = c0
    gen[..., 4] = c1
    gen[..., 5] = c2
    gen[..., 6] = gen_on

    Ytt, Yff, Yft, Ytf = branch_vectors(branches.reshape(B * nl, -1), B * nl)
    tap = branches[..., TAP].copy()
    tap[tap == 0] = 1
    common = np.stack(
        [tap, branches[..., ANGMIN], branches[..., ANGMAX], branches[..., RATE_A], branches[..., BR_STATUS]],
        axis=-1,
    )
    zero2 = np.zeros((B, nl, 2))  # pf, qf / pt, qt unknown

    def y(*ys):
        return np.stack([f(v) for v in ys for f in (np.real, np.imag)], axis=-1).reshape(B, nl, -1)

    fwd = np.concatenate([zero2, y(Yff, Yft), common], axis=-1)
    rev = np.concatenate([zero2, y(Ytt, Ytf), common], axis=-1)
    return RawBatch(
        bus=bus,
        gen=gen,
        fwd=fwd,
        rev=rev,
        from_bus=branches[..., F_BUS].astype(np.int64),
        to_bus=branches[..., T_BUS].astype(np.int64),
        gen_bus=gen_bus,
    )


def raw_from_rows(
    bus_df: pd.DataFrame, gen_df: pd.DataFrame, branch_df: pd.DataFrame, keep_targets: bool = False
) -> RawBatch:
    """RawBatch from datakit parquet rows (several scenarios, sorted by scenario).
    Solution columns are zeroed unless ``keep_targets`` (as the official path)."""
    bus_df = bus_df.sort_values(["scenario", "bus"], kind="stable")
    gen_df = gen_df.sort_values(["scenario", "idx"], kind="stable")
    branch_df = branch_df.sort_values(["scenario", "idx"], kind="stable")
    B = bus_df["scenario"].nunique()
    agg = gen_df.groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
    bus_df = bus_df.merge(agg, on=["scenario", "bus"], how="left").fillna(0)
    bus = bus_df[BUS_FEATURES].to_numpy(np.float64).reshape(B, -1, len(BUS_FEATURES))
    gen = gen_df[GEN_FEATURES].to_numpy(np.float64).reshape(B, -1, len(GEN_FEATURES))
    fwd = branch_df[FORWARD_BRANCH_FEATURES].to_numpy(np.float64).reshape(B, -1, 11)
    rev = branch_df[REVERSE_BRANCH_FEATURES].to_numpy(np.float64).reshape(B, -1, 11)
    if not keep_targets:
        bus[..., 2:5] = 0
        gen[..., 0] = 0
        fwd[..., :2] = 0
        rev[..., :2] = 0
    nl = fwd.shape[1]
    return RawBatch(
        bus=bus,
        gen=gen,
        fwd=fwd,
        rev=rev,
        from_bus=branch_df["from_bus"].to_numpy(np.int64).reshape(B, nl),
        to_bus=branch_df["to_bus"].to_numpy(np.int64).reshape(B, nl),
        gen_bus=gen_df["bus"].to_numpy(np.int64).reshape(B, -1),
    )


# ---------------------------------------------------------------------------
# RawBatch -> model-ready batched HeteroData
# ---------------------------------------------------------------------------
def scenario_heterodata(raw: RawBatch, i: int, scenario_id: int = 0) -> HeteroData:
    """Single-scenario unnormalized HeteroData, field-for-field what graphkit's
    ``build_hetero_data_for_scenario`` stores in processed/*.pt (for tests)."""
    r = raw[i]
    return _union(r, torch.tensor([scenario_id]), torch.device("cpu"))


def _union(raw: RawBatch, scenario_id: torch.Tensor, device: torch.device) -> HeteroData:
    B, nb, _ = raw.bus.shape
    ng, nl = raw.gen.shape[1], raw.fwd.shape[1]
    f32 = torch.float32

    def f32c(a):  # C-contiguous like torch.tensor(df.values); layout affects fp32 bits
        return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=f32)

    data = HeteroData()
    bus_x = f32c(raw.bus.reshape(B * nb, -1))
    gen_x = f32c(raw.gen.reshape(B * ng, -1))
    data["bus"].x = bus_x
    data["gen"].x = gen_x
    data["bus"].y = bus_x[:, : VA_H + 1].clone()
    data["gen"].y = gen_x[:, : PG_H + 1].clone()

    # per graph: forward edges then reverse edges, node ids offset per graph
    off = (np.arange(B) * nb)[:, None]
    f, t = raw.from_bus + off, raw.to_bus + off
    ei = np.stack([np.concatenate([f, t], 1), np.concatenate([t, f], 1)])  # [2,B,2nl]
    data[BB].edge_index = torch.from_numpy(ei.reshape(2, -1)).to(device)
    ea = np.concatenate([raw.fwd, raw.rev], axis=1).reshape(B * 2 * nl, -1)
    data[BB].edge_attr = f32c(ea)
    data[BB].y = data[BB].edge_attr[:, :2].clone()

    gidx = np.arange(B * ng)
    gbus = (raw.gen_bus + off).ravel()
    data[GB].edge_index = torch.from_numpy(np.stack([gidx, gbus])).to(device)
    data[BG].edge_index = torch.from_numpy(np.stack([gbus, gidx])).to(device)
    data["scenario_id"] = scenario_id.to(device)
    return data


class OPFInputBuilder:
    """Applies graphkit's fitted normalizer + OPF task transforms to a RawBatch."""

    def __init__(self, normalizer, transforms):
        self.normalizer = normalizer
        self.transforms = transforms

    def __call__(self, raw: RawBatch, device="cpu", scenario_ids=None) -> HeteroData:
        device = torch.device(device)
        B, nb, _ = raw.bus.shape
        ng = raw.gen.shape[1]
        sid = torch.arange(B) if scenario_ids is None else torch.as_tensor(scenario_ids)
        data = _union(raw, sid, device)
        # gen ids that survive RemoveInactiveGenerators (to map outputs back)
        gen_keep = data["gen"].x[:, -1] == 1
        self.normalizer.transform(data=data)
        data = self.transforms(data)
        # PyG collation copies each sliced tensor into a contiguous one
        for store in (data["gen"], data[BB]):
            for k in ("x", "y", "edge_attr"):
                if k in store:
                    store[k] = store[k].contiguous()
        data["bus"].batch = torch.arange(B, device=device).repeat_interleave(nb)
        data["gen"].batch = torch.arange(B, device=device).repeat_interleave(ng)[gen_keep]
        data["gen"].orig_index = torch.arange(B * ng, device=device)[gen_keep]
        data.num_scenarios = B
        return data
