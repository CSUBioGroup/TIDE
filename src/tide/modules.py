from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from anndata import AnnData

from ._pairwise import (
    aggregate_high_genes_from_edges,
    build_velocity_graph_object_from_adata_params,
    compute_cell_pairwise_context,
    select_positive_edges,
)



def _require_gene_stats(adata: AnnData, vkey: str) -> dict:
    key = f"{vkey}_gene_cosine_stats"
    if key not in adata.uns:
        raise KeyError(
            f"Missing {key}. Run TIDE graph with store_gene_stats=True before extracting high-cos modules."
        )
    stats = adata.uns[key]
    if "genes" not in stats or "max_idx" not in stats or "max_cos" not in stats:
        raise ValueError(f"{key} is malformed; required fields: genes/max_idx/max_cos")
    return stats



def _check_celltype_key(adata: AnnData, celltype_key: str) -> np.ndarray:
    if celltype_key not in adata.obs:
        raise KeyError(f"celltype_key='{celltype_key}' not found in adata.obs")
    return adata.obs[celltype_key].astype(str).to_numpy()



def extract_high_cos_by_celltype(
    adata: AnnData,
    *,
    celltype_key: str,
    vkey: str = "velocity",
    top_n: int = 50,
    min_cells: int = 1,
) -> pd.DataFrame:
    """Extract HIGH-cos gene module for each cell type (MID/LOW are excluded by design)."""

    stats = _require_gene_stats(adata, vkey=vkey)
    labels = _check_celltype_key(adata, celltype_key)

    genes = np.asarray(stats["genes"], dtype=str)
    max_idx = np.asarray(stats["max_idx"], dtype=np.int32)
    max_cos = np.asarray(stats["max_cos"], dtype=np.float32)

    if max_idx.shape != max_cos.shape:
        raise ValueError("max_idx and max_cos shape mismatch")
    if max_idx.shape[0] != adata.n_obs:
        raise ValueError("gene stats and adata.n_obs mismatch")

    rows: list[tuple[str, str, float, int]] = []
    for cell_i in range(adata.n_obs):
        ct = labels[cell_i]
        idx_row = max_idx[cell_i]
        cos_row = max_cos[cell_i]
        valid = (idx_row >= 0) & np.isfinite(cos_row)
        if not np.any(valid):
            continue
        for gi, cv in zip(idx_row[valid], cos_row[valid]):
            if gi < 0 or gi >= genes.shape[0]:
                continue
            rows.append((ct, genes[gi], float(cv), int(cell_i)))

    if len(rows) == 0:
        return pd.DataFrame(
            columns=[
                "celltype",
                "gene",
                "high_cos_score_mean",
                "high_cos_score_median",
                "high_hit_count",
                "high_cell_coverage",
                "n_cells_in_type",
                "rank",
            ]
        )

    df = pd.DataFrame(rows, columns=["celltype", "gene", "high_cos", "cell_idx"])
    out_frames = []
    for ct, sub in df.groupby("celltype", sort=True):
        n_cells = int(np.sum(labels == ct))
        if n_cells < int(min_cells):
            continue
        agg = (
            sub.groupby("gene", as_index=False)
            .agg(
                high_cos_score_mean=("high_cos", "mean"),
                high_cos_score_median=("high_cos", "median"),
                high_hit_count=("cell_idx", "count"),
                high_cell_coverage=("cell_idx", lambda x: x.nunique() / max(1, n_cells)),
            )
            .sort_values(["high_cos_score_mean", "high_hit_count", "gene"], ascending=[False, False, True])
            .head(int(top_n))
            .reset_index(drop=True)
        )
        agg.insert(0, "celltype", ct)
        agg["n_cells_in_type"] = n_cells
        agg["rank"] = np.arange(1, agg.shape[0] + 1)
        out_frames.append(agg)

    if len(out_frames) == 0:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)



def extract_cell_to_target_high_cos(
    adata: AnnData,
    *,
    cell_idx: int,
    celltype_key: str,
    target_types: Iterable[str],
    vkey: str = "velocity",
    top_n: int = 50,
) -> pd.DataFrame:
    """For one source cell, extract HIGH-cos modules toward specified target cell types in its neighborhood."""

    labels = _check_celltype_key(adata, celltype_key)
    vg = build_velocity_graph_object_from_adata_params(adata, vkey=vkey)
    ctx = compute_cell_pairwise_context(vg, int(cell_idx))

    if vg.gene_names_pairwise is None:
        raise RuntimeError("gene_names_pairwise is missing; ensure pairwise graph context was initialized correctly.")
    gene_names = np.asarray(vg.gene_names_pairwise, dtype=str)

    target_set = [str(t) for t in target_types]
    neigh_labels = labels[ctx.neighs_idx] if ctx.neighs_idx.size > 0 else np.array([], dtype=str)

    rows: list[dict] = []
    for target in target_set:
        mask = neigh_labels == target
        keep, ew = select_positive_edges(
            ctx.sim,
            store_edge_topm=vg.store_edge_topm,
            store_edge_weight=vg.store_edge_weight,
            edge_mask=mask,
        )
        if keep is None:
            continue

        top_idx, top_scores = aggregate_high_genes_from_edges(
            cos_ijg=ctx.cos_ijg,
            idx_cand=ctx.idx_cand,
            edge_indices=keep,
            edge_weights=ew,
            top_n=int(top_n),
        )
        for rank, (gi, sc) in enumerate(zip(top_idx, top_scores), start=1):
            if gi < 0 or gi >= gene_names.shape[0]:
                continue
            rows.append(
                {
                    "source_cell_idx": int(cell_idx),
                    "target_type": target,
                    "gene": gene_names[gi],
                    "high_cos_score": float(sc),
                    "rank": int(rank),
                    "n_target_neighbors": int(np.sum(mask)),
                    "n_used_edges": int(keep.size),
                }
            )

    return pd.DataFrame(rows).sort_values(["target_type", "high_cos_score"], ascending=[True, False]).reset_index(drop=True)



def extract_cell_all_direction_high_cos(
    adata: AnnData,
    *,
    cell_idx: int,
    celltype_key: str,
    vkey: str = "velocity",
    top_n: int = 50,
    include_self_type: bool = False,
) -> pd.DataFrame:
    """For one source cell, export HIGH-cos modules for all neighbor direction types."""

    labels = _check_celltype_key(adata, celltype_key)
    source_type = str(labels[int(cell_idx)])

    vg = build_velocity_graph_object_from_adata_params(adata, vkey=vkey)
    ctx = compute_cell_pairwise_context(vg, int(cell_idx))
    neigh_labels = labels[ctx.neighs_idx] if ctx.neighs_idx.size > 0 else np.array([], dtype=str)

    targets = sorted(np.unique(neigh_labels).tolist())
    if not include_self_type:
        targets = [t for t in targets if t != source_type]

    return extract_cell_to_target_high_cos(
        adata,
        cell_idx=int(cell_idx),
        celltype_key=celltype_key,
        target_types=targets,
        vkey=vkey,
        top_n=top_n,
    )



def export_cell_direction_modules(
    df: pd.DataFrame,
    *,
    outdir: str | Path,
    prefix: str,
) -> list[Path]:
    """Export combined and per-target module tables."""

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    combined = out / f"{prefix}_combined.csv"
    df.to_csv(combined, index=False)
    written.append(combined)

    if "target_type" in df.columns and not df.empty:
        for target, sub in df.groupby("target_type", sort=True):
            safe_target = str(target).replace("/", "_").replace(" ", "_")
            p = out / f"{prefix}_target_{safe_target}.csv"
            sub.to_csv(p, index=False)
            written.append(p)

    return written
