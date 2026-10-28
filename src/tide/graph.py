from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from anndata import AnnData

from tide.velocity_graph_pairwise2d import velocity_graph as velocity_graph_pairwise2d


@dataclass
class TIDEGraphParams:
    """Parameters for TIDE velocity-graph construction."""

    vkey: str = "velocity"
    xkey: str = "Ms"
    graph_mode: str = "pairwise2d_geneTopK"
    ukey: str = "Mu"
    skey: str | None = None
    vkey_u: str | None = "velocity_u"
    pairwise_topk_genes: int = 100
    pairwise_candidate_genes: int = 300
    pairwise_use_signed_cos: bool = True
    store_gene_stats: bool = True
    store_k: int = 100
    store_edge_topm: int = 30
    store_edge_weight: str = "sim_pos"
    n_recurse_neighbors: int | None = None
    n_jobs: int = 8
    show_progress_bar: bool = False
    copy: bool = False

    def to_kwargs(self) -> dict[str, Any]:
        d = asdict(self)
        return d



def params_from_adata(adata: AnnData, *, vkey: str = "velocity") -> TIDEGraphParams:
    """Build params from ``adata.uns[f"{vkey}_params"]`` with sane defaults."""

    p = dict(adata.uns.get(f"{vkey}_params", {}))
    return TIDEGraphParams(
        vkey=vkey,
        xkey=str(p.get("skey", "Ms")),
        graph_mode=str(p.get("graph_mode", "pairwise2d_geneTopK")),
        ukey=str(p.get("ukey", "Mu")),
        skey=str(p.get("skey", "Ms")),
        vkey_u=p.get("vkey_u", "velocity_u"),
        pairwise_topk_genes=int(p.get("pairwise_topk_genes", 100)),
        pairwise_candidate_genes=int(p.get("pairwise_candidate_genes", 300)),
        pairwise_use_signed_cos=bool(p.get("pairwise_use_signed_cos", True)),
        store_gene_stats=bool(p.get("store_gene_stats", True)),
        store_k=int(p.get("store_k", 100)),
        store_edge_topm=int(p.get("store_edge_topm", 30)),
        store_edge_weight=str(p.get("store_edge_weight", "sim_pos")),
        n_recurse_neighbors=(
            int(p["n_recurse_neighbors"]) if p.get("n_recurse_neighbors") is not None else None
        ),
        n_jobs=8,
        show_progress_bar=False,
        copy=False,
    )



def compute_velocity_graph(
    adata: AnnData,
    *,
    params: TIDEGraphParams | None = None,
    **overrides: Any,
) -> AnnData | None:
    """Compute TIDE velocity graph on an AnnData object.

    Notes
    -----
    This function is intentionally pluggable: users provide velocity-ready ``adata`` and call this API directly,
    rather than replacing imports in upstream scripts.
    """

    cfg = params if params is not None else TIDEGraphParams()
    kwargs = cfg.to_kwargs()

    # In this TIDE release, weighting is fixed to speed for stability.
    if "weight_mode" in overrides:
        wm = str(overrides.pop("weight_mode"))
        if wm != "speed":
            raise ValueError("TIDE fixes weight_mode='speed'; other values are not supported.")
    kwargs["weight_mode"] = "speed"
    kwargs.update(overrides)
    return velocity_graph_pairwise2d(adata, **kwargs)
