from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from anndata import AnnData

from scvelo.tools.utils import get_iterative_indices
from tide.velocity_graph_pairwise2d import VelocityGraph


@dataclass
class PairwiseCellContext:
    """Pairwise 2D context for one source cell."""

    vg: VelocityGraph
    cell_idx: int
    neighs_idx: np.ndarray
    sim: np.ndarray
    cos_ijg: np.ndarray
    idx_cand: np.ndarray



def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)



def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)



def build_velocity_graph_object(
    adata: AnnData,
    *,
    vkey: str = "velocity",
    xkey: str = "Ms",
    graph_mode: str = "pairwise2d_geneTopK",
    ukey: str = "Mu",
    skey: str | None = None,
    vkey_u: str | None = None,
    pairwise_topk_genes: int = 100,
    pairwise_candidate_genes: int = 300,
    pairwise_use_signed_cos: bool = True,
    store_k: int = 100,
    store_edge_topm: int = 30,
    store_edge_weight: str = "sim_pos",
    mode_neighbors: str = "distances",
    n_recurse_neighbors: int | None = None,
    random_neighbors_at_max: int | None = None,
    approx: bool | None = None,
) -> VelocityGraph:
    """Build a configured :class:`VelocityGraph` object without running the full graph pass."""

    vg = VelocityGraph(
        adata,
        vkey=vkey,
        xkey=xkey,
        tkey=None,
        basis=None,
        n_neighbors=None,
        sqrt_transform=None,
        n_recurse_neighbors=n_recurse_neighbors,
        random_neighbors_at_max=random_neighbors_at_max,
        gene_subset=None,
        approx=approx,
        report=False,
        compute_uncertainties=False,
        mode_neighbors=mode_neighbors,
        graph_mode=graph_mode,
        ukey=ukey,
        skey=skey,
        vkey_u=vkey_u,
        eps=1e-8,
        weight_mode="speed",
        pairwise_topk_genes=pairwise_topk_genes,
        pairwise_candidate_genes=pairwise_candidate_genes,
        pairwise_use_signed_cos=pairwise_use_signed_cos,
        store_gene_stats=True,
        store_k=store_k,
        store_edge_topm=store_edge_topm,
        store_edge_weight=store_edge_weight,
    )
    return vg



def build_velocity_graph_object_from_adata_params(
    adata: AnnData,
    *,
    vkey: str = "velocity",
    default_xkey: str = "Ms",
    default_ukey: str = "Mu",
) -> VelocityGraph:
    """Build a :class:`VelocityGraph` using parameters previously written to ``adata.uns[f"{vkey}_params"]``."""

    p = dict(adata.uns.get(f"{vkey}_params", {}))
    return build_velocity_graph_object(
        adata,
        vkey=vkey,
        xkey=str(p.get("skey", default_xkey)),
        graph_mode=str(p.get("graph_mode", "pairwise2d_geneTopK")),
        ukey=str(p.get("ukey", default_ukey)),
        skey=str(p.get("skey", default_xkey)),
        vkey_u=p.get("vkey_u", None),
        pairwise_topk_genes=_safe_int(p.get("pairwise_topk_genes", 100), 100),
        pairwise_candidate_genes=_safe_int(p.get("pairwise_candidate_genes", 300), 300),
        pairwise_use_signed_cos=_safe_bool(p.get("pairwise_use_signed_cos", True), True),
        store_k=_safe_int(p.get("store_k", 100), 100),
        store_edge_topm=_safe_int(p.get("store_edge_topm", 30), 30),
        store_edge_weight=str(p.get("store_edge_weight", "sim_pos")),
        mode_neighbors=str(p.get("mode_neighbors", "distances")),
        n_recurse_neighbors=_safe_int(p.get("n_recurse_neighbors", 2), 2),
        random_neighbors_at_max=None,
        approx=None,
    )



def compute_cell_pairwise_context(vg: VelocityGraph, cell_idx: int) -> PairwiseCellContext:
    """Recompute pairwise 2D edge scores and per-edge gene cosines for one cell, matching TIDE internals."""

    neighs_idx = get_iterative_indices(vg.indices, int(cell_idx), vg.n_recurse_neighbors, vg.max_neighs)
    eps = float(vg.eps)
    U, S, Vu, Vs = vg.U, vg.S, vg.Vu, vg.Vs

    if U is None or S is None or Vu is None or Vs is None or len(neighs_idx) == 0:
        return PairwiseCellContext(
            vg=vg,
            cell_idx=int(cell_idx),
            neighs_idx=np.asarray(neighs_idx, dtype=np.int32),
            sim=np.zeros((0,), dtype=np.float32),
            cos_ijg=np.zeros((0, 0), dtype=np.float32),
            idx_cand=np.zeros((0,), dtype=np.int32),
        )

    vu = Vu[cell_idx].astype(np.float32)
    vs = Vs[cell_idx].astype(np.float32)
    speed = np.sqrt(vu * vu + vs * vs).astype(np.float32)

    inv = np.divide(1.0, speed, out=np.zeros_like(speed), where=(speed > eps))
    ux = vu * inv
    uy = vs * inv

    # TIDE fixes gene weighting mode to speed.
    w = speed

    g = ux.shape[0]
    c = vg.pairwise_candidate_genes if vg.pairwise_candidate_genes is not None else g
    c = max(1, min(int(c), g))

    if c < g:
        idx_cand = np.argpartition(w, -c)[-c:]
    else:
        idx_cand = np.arange(g, dtype=np.int32)

    k = vg.pairwise_topk_genes if vg.pairwise_topk_genes is not None else min(50, c)
    k = max(1, min(int(k), c))

    ux_c = ux[idx_cand]
    uy_c = uy[idx_cand]
    w_c = w[idx_cand].astype(np.float32)

    ui = U[cell_idx, idx_cand].astype(np.float32)
    si = S[cell_idx, idx_cand].astype(np.float32)

    un = U[neighs_idx][:, idx_cand].astype(np.float32)
    sn = S[neighs_idx][:, idx_cand].astype(np.float32)

    d_u = un - ui[None, :]
    d_s = sn - si[None, :]

    if vg.sqrt_transform:
        d_u = np.sqrt(np.abs(d_u)) * np.sign(d_u)
        d_s = np.sqrt(np.abs(d_s)) * np.sign(d_s)

    dn = np.sqrt(d_u * d_u + d_s * d_s).astype(np.float32)
    invd = np.divide(1.0, dn, out=np.zeros_like(dn), where=(dn > eps))
    dxu = d_u * invd
    dxs = d_s * invd

    cos_ijg = dxu * ux_c[None, :] + dxs * uy_c[None, :]
    cos_ijg = np.nan_to_num(cos_ijg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if not vg.pairwise_use_signed_cos:
        cos_ijg = np.clip(cos_ijg, 0.0, 1.0)

    if k < c:
        idx_top = np.argpartition(cos_ijg, -k, axis=1)[:, -k:]
        cos_sel = np.take_along_axis(cos_ijg, idx_top, axis=1).astype(np.float32)
        w_sel = w_c[idx_top].astype(np.float32)
    else:
        cos_sel = cos_ijg.astype(np.float32)
        w_sel = np.broadcast_to(w_c[None, :], cos_sel.shape).astype(np.float32)

    num = (w_sel * cos_sel).sum(axis=1).astype(np.float32)
    den = (w_sel).sum(axis=1).astype(np.float32) + eps
    sim = num / den
    sim[~np.isfinite(sim)] = 0.0

    return PairwiseCellContext(
        vg=vg,
        cell_idx=int(cell_idx),
        neighs_idx=np.asarray(neighs_idx, dtype=np.int32),
        sim=sim.astype(np.float32),
        cos_ijg=cos_ijg.astype(np.float32),
        idx_cand=np.asarray(idx_cand, dtype=np.int32),
    )



def select_positive_edges(
    sim: np.ndarray,
    *,
    store_edge_topm: int | None,
    store_edge_weight: str,
    edge_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Select outgoing positive edges and optionally top-M by edge weight."""

    if sim.size == 0:
        return None, None

    if store_edge_weight == "sim_pos":
        edge_w = np.clip(sim, 0.0, 1.0).astype(np.float32)
        pos_mask = edge_w > 0
    elif store_edge_weight == "uniform":
        edge_w = np.ones_like(sim, dtype=np.float32)
        pos_mask = sim > 0
    else:
        raise ValueError("store_edge_weight must be 'sim_pos' or 'uniform'.")

    if edge_mask is not None:
        pos_mask = pos_mask & edge_mask.astype(bool)

    if not np.any(pos_mask):
        return None, None

    cand_edges = np.where(pos_mask)[0]
    if cand_edges.size == 0:
        return None, None

    m = store_edge_topm
    if m is None or m <= 0 or cand_edges.size <= m:
        keep = cand_edges
    else:
        ww = edge_w[cand_edges]
        sel = np.argpartition(ww, -m)[-m:]
        keep = cand_edges[sel]
        keep = keep[np.argsort(edge_w[keep])[::-1]]

    ew = edge_w[keep].astype(np.float32)
    if not np.isfinite(ew).all():
        ew = np.nan_to_num(ew, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if ew.sum() <= 1e-12:
        return None, None
    return keep, ew



def aggregate_high_genes_from_edges(
    *,
    cos_ijg: np.ndarray,
    idx_cand: np.ndarray,
    edge_indices: np.ndarray,
    edge_weights: np.ndarray,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate gene support over chosen edges and return top-N candidate-gene indices + scores."""

    if edge_indices.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    sub = cos_ijg[edge_indices]  # (m, C)
    num = edge_weights @ sub
    den = float(edge_weights.sum()) + 1e-8
    cos_g = (num / den).astype(np.float32)
    cos_g = np.nan_to_num(cos_g, nan=0.0, posinf=0.0, neginf=0.0)

    c = cos_g.shape[0]
    if c == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    n = max(1, min(int(top_n), c))
    if n < c:
        top_local = np.argpartition(cos_g, -n)[-n:]
        top_local = top_local[np.argsort(cos_g[top_local])[::-1]]
    else:
        top_local = np.argsort(cos_g)[::-1]

    top_gene_idx = idx_cand[top_local].astype(np.int32)
    top_scores = cos_g[top_local].astype(np.float32)
    return top_gene_idx, top_scores
