# Extract high-cos modules

TIDE exports **high-cos** gene evidence for interpretability.

## A) Celltype-level high-cos modules

```python
from tide import extract_high_cos_by_celltype

by_type = extract_high_cos_by_celltype(
    adata,
    celltype_key="clusters",
    vkey="velocity",
    top_n=50,
)
```

Main columns:

- `celltype`
- `gene`
- `high_cos_score_mean`
- `high_cos_score_median`
- `high_hit_count`
- `high_cell_coverage`
- `rank`

## B) One cell -> specified target types

```python
from tide import extract_cell_to_target_high_cos

pair_df = extract_cell_to_target_high_cos(
    adata,
    cell_idx=2163,
    celltype_key="clusters",
    target_types=["CA", "ImmGranule1"],
    vkey="velocity",
    top_n=50,
)
```

## C) One cell -> all local directions + export

```python
from tide import extract_cell_all_direction_high_cos, export_cell_direction_modules

all_df = extract_cell_all_direction_high_cos(
    adata,
    cell_idx=2163,
    celltype_key="clusters",
    vkey="velocity",
    top_n=50,
)

export_cell_direction_modules(all_df, outdir="./modules", prefix="cell2163")
```

This produces:

- `cell2163_combined.csv`
- `cell2163_target_<target>.csv`
