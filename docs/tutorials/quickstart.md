# Quickstart

## 1) Install

```bash
pip install -e .
```

## 2) Compute TIDE graph after velocity inference

```python
import scanpy as sc
from tide import TIDEGraphParams, compute_velocity_graph

adata = sc.read_h5ad("/path/to/velocity_ready.h5ad")

params = TIDEGraphParams(
    vkey="velocity",
    xkey="Ms",
    graph_mode="pairwise2d_geneTopK",
    ukey="Mu",
    vkey_u="velocity_u",
    pairwise_topk_genes=100,
    pairwise_candidate_genes=300,
    pairwise_use_signed_cos=True,
    store_gene_stats=True,
    store_k=100,
)
compute_velocity_graph(adata, params=params)
```

## 3) Visualize with scVelo

```python
import scvelo as scv

scv.tl.velocity_embedding(adata, basis="umap", vkey="velocity")
scv.pl.velocity_embedding_stream(adata, basis="umap", vkey="velocity")
```
