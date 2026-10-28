from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import scanpy as sc

from tide.velocity_graph_pairwise2d import velocity_graph as core_velocity_graph
from tide import (
    TIDEGraphParams,
    compute_velocity_graph,
    export_cell_direction_modules,
    extract_cell_all_direction_high_cos,
    extract_cell_to_target_high_cos,
    extract_high_cos_by_celltype,
)


def _array_equal(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _resolve_test_h5ad() -> Path:
    p = os.environ.get("TIDE_TEST_H5AD", "")
    return Path(p)


@pytest.fixture()
def sample_adata() -> sc.AnnData:
    fp = _resolve_test_h5ad()
    if not os.environ.get("TIDE_TEST_H5AD") or not fp.exists():
        pytest.skip("set TIDE_TEST_H5AD to run the data-backed integration tests")

    adata = sc.read_h5ad(fp)
    n = min(adata.n_obs, 2000)
    adata = adata[:n].copy()

    # Velocity-ready sample should already have these.
    for key in ["velocity", "Ms", "Mu", "velocity_u"]:
        if key not in adata.layers:
            raise KeyError(f"missing required layer: {key}")

    if "clusters" not in adata.obs:
        if "cell_type" in adata.obs:
            adata.obs["clusters"] = adata.obs["cell_type"].astype(str)
        elif "Annotation" in adata.obs:
            adata.obs["clusters"] = adata.obs["Annotation"].astype(str)
        else:
            adata.obs["clusters"] = "unknown"
    else:
        adata.obs["clusters"] = adata.obs["clusters"].astype(str)
    return adata


def test_params_defaults():
    p = TIDEGraphParams()
    assert p.graph_mode == "pairwise2d_geneTopK"
    assert p.store_gene_stats is True


def test_wrapper_matches_core(sample_adata: sc.AnnData):
    kwargs = dict(
        vkey="velocity",
        xkey="Ms",
        graph_mode="pairwise2d_geneTopK",
        ukey="Mu",
        vkey_u="velocity_u",
        pairwise_topk_genes=20,
        pairwise_candidate_genes=60,
        store_gene_stats=True,
        store_k=20,
        n_jobs=4,
        show_progress_bar=False,
    )

    ad_core = sample_adata.copy()
    core_velocity_graph(ad_core, weight_mode="speed", **kwargs)

    ad_tide = sample_adata.copy()
    compute_velocity_graph(ad_tide, params=TIDEGraphParams(**kwargs))

    g1 = ad_core.uns["velocity_graph"]
    g2 = ad_tide.uns["velocity_graph"]
    assert _array_equal(g1.toarray(), g2.toarray())

    s1 = ad_core.uns["velocity_gene_cosine_stats"]
    s2 = ad_tide.uns["velocity_gene_cosine_stats"]
    assert set(s1.keys()) == set(s2.keys())
    for k in s1:
        v1, v2 = s1[k], s2[k]
        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            assert _array_equal(v1, v2), f"mismatch in stats key {k}"
        else:
            assert v1 == v2, f"mismatch in stats key {k}"


def test_module_extraction_and_export(sample_adata: sc.AnnData, tmp_path: Path):
    compute_velocity_graph(
        sample_adata,
        params=TIDEGraphParams(
            vkey="velocity",
            xkey="Ms",
            graph_mode="pairwise2d_geneTopK",
            ukey="Mu",
            vkey_u="velocity_u",
            pairwise_topk_genes=20,
            pairwise_candidate_genes=60,
            store_gene_stats=True,
            store_k=20,
            n_jobs=4,
            show_progress_bar=False,
        ),
    )

    by_type = extract_high_cos_by_celltype(sample_adata, celltype_key="clusters", vkey="velocity", top_n=5)
    assert not by_type.empty
    assert set(["celltype", "gene", "high_cos_score_mean", "rank"]).issubset(by_type.columns)

    source_cell = 0
    target_types = sample_adata.obs["clusters"].astype(str).unique().tolist()[:2]
    pair_df = extract_cell_to_target_high_cos(
        sample_adata,
        cell_idx=source_cell,
        celltype_key="clusters",
        target_types=target_types,
        vkey="velocity",
        top_n=5,
    )
    assert set(["source_cell_idx", "target_type", "gene", "high_cos_score", "rank"]).issubset(pair_df.columns)

    all_df = extract_cell_all_direction_high_cos(
        sample_adata,
        cell_idx=source_cell,
        celltype_key="clusters",
        vkey="velocity",
        top_n=5,
    )
    assert set(["target_type", "gene", "high_cos_score"]).issubset(all_df.columns)

    out_paths = export_cell_direction_modules(all_df, outdir=tmp_path, prefix="cell0")
    assert len(out_paths) >= 1
    assert out_paths[0].exists()
