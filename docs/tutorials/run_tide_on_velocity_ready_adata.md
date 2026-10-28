# Run TIDE on velocity-ready AnnData

TIDE is designed to run on outputs from upstream velocity methods.

## Required data fields

For `graph_mode='pairwise2d_geneTopK'`, ensure:

- `adata.layers['velocity']`
- `adata.layers['Ms']`
- `adata.layers['Mu']`
- `adata.layers['velocity_u']`
- neighbor graph (`adata.obsp['distances']` / connectivities)

## Minimal run example

```python
from tide import TIDEGraphParams, compute_velocity_graph

params = TIDEGraphParams(
    graph_mode="pairwise2d_geneTopK",
    pairwise_topk_genes=100,
    pairwise_candidate_genes=300,
    store_gene_stats=True,
    store_k=100,
    store_edge_topm=30,
    store_edge_weight="sim_pos",
)

compute_velocity_graph(adata, params=params)
```

## Output contract

After running:

- `adata.uns['velocity_graph']`
- `adata.uns['velocity_gene_cosine_stats']` (when `store_gene_stats=True`)
- `adata.uns['velocity_params']` updated with graph-related settings
