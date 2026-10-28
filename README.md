TIDE: Transition-specific trajectory refinement and Interpretable Driver Extraction via gene module ensembles
==============================================================================================================

This repository provides the refactored, modular implementation of **TIDE**, designed as a projection-stage
refinement layer for RNA velocity analysis. The graph implementation used in this project is included in
`tide.velocity_graph_pairwise2d`, so installation does not require a modified scVelo checkout.

TIDE works **downstream of existing velocity models** (e.g., scVelo, UniTVelo, tiVelo, CellDancer):
users first compute velocity with their preferred method, then call TIDE to refine local transition scoring
and extract transition-supportive gene modules.

Key features
------------
- Pluggable graph construction API without monkeypatching import paths.
- Transition-specific edge re-scoring in paired unspliced-spliced space.
- Interpretable high-cos directional gene modules:
  - per-celltype high-cos modules,
  - one-cell -> specified target-type modules,
  - one-cell -> all outgoing direction modules.

Installation
------------
We recommend Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Quickstart
----------
```python
import scanpy as sc
from tide import TIDEGraphParams, compute_velocity_graph, extract_high_cos_by_celltype

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
    store_edge_topm=30,
    store_edge_weight="sim_pos",
    n_recurse_neighbors=2,
)
compute_velocity_graph(adata, params=params)

mods = extract_high_cos_by_celltype(adata, celltype_key="clusters", vkey="velocity", top_n=50)
print(mods.head())
```

Then use standard scVelo plotting:
```python
import scvelo as scv
scv.tl.velocity_embedding(adata, basis="umap", vkey="velocity")
scv.pl.velocity_embedding_stream(adata, basis="umap", vkey="velocity")
```

Modular usage
-------------
```python
from tide import (
    extract_cell_to_target_high_cos,
    extract_cell_all_direction_high_cos,
    export_cell_direction_modules,
)

cell_df = extract_cell_to_target_high_cos(
    adata,
    cell_idx=2163,
    celltype_key="clusters",
    target_types=["CA", "ImmGranule1"],
    vkey="velocity",
    top_n=50,
)

all_df = extract_cell_all_direction_high_cos(
    adata,
    cell_idx=2163,
    celltype_key="clusters",
    vkey="velocity",
    top_n=50,
)

export_cell_direction_modules(all_df, outdir="./modules", prefix="cell2163")
```

Case-study notebook
-------------------
An end-to-end notebook is provided at:

- `docs/tutorials/notebooks/tide_case_mouse_embryonic_brain.ipynb`

Contributing and license
------------------------
Issues and pull requests are welcome. Please include reproducible commands and expected outputs.

The TIDE package is released under the MIT license. The graph implementation adapts code from scVelo;
its BSD 3-Clause notice is retained in `NOTICE`.

Chinese README
--------------
A Chinese version is available in `README_zh.md`.
