TIDE：Transition-specific trajectory refinement and Interpretable Driver Extraction via gene module ensembles
==============================================================================================================

本仓库提供 **TIDE** 的模块化实现。TIDE 的定位是 RNA velocity 的**投影阶段精炼层**。
所需的图计算实现在 `tide.velocity_graph_pairwise2d` 中，不需要安装改动过的 scVelo 仓库。

1. 先用任意 velocity 方法（如 scVelo / UniTVelo / tiVelo / CellDancer）计算 velocity；
2. 再调用 TIDE 重算局部转移边权重并提取可解释的方向支持基因模块；
3. 最后继续用 scVelo/Scanpy 做可视化与下游分析。

核心特性
--------
- 可插拔接口：不再需要替换上游 import。
- transition-specific 边评分：在成对的 unspliced-spliced 空间中逐候选转移建模。
- high-cos 基因模块分析（仅 high，不含 mid/low）：
  - 按细胞类型汇总 high-cos 模块；
  - 指定细胞 -> 邻域指定目标类型的 high-cos 模块；
  - 指定细胞 -> 所有转移方向的 high-cos 模块导出。

安装
----
建议 Python 3.9+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

快速开始
--------
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
    store_gene_stats=True,
    store_k=100,
)
compute_velocity_graph(adata, params=params)

df = extract_high_cos_by_celltype(adata, celltype_key="clusters", top_n=50)
print(df.head())
```

继续使用 scVelo 绘图：
```python
import scvelo as scv
scv.tl.velocity_embedding(adata, basis="umap", vkey="velocity")
scv.pl.velocity_embedding_stream(adata, basis="umap", vkey="velocity")
```

模块化接口示例
--------------
```python
from tide import extract_cell_to_target_high_cos, extract_cell_all_direction_high_cos, export_cell_direction_modules

pair_df = extract_cell_to_target_high_cos(
    adata,
    cell_idx=2163,
    celltype_key="clusters",
    target_types=["CA", "ImmGranule1"],
    top_n=50,
)

all_df = extract_cell_all_direction_high_cos(
    adata,
    cell_idx=2163,
    celltype_key="clusters",
    top_n=50,
)

export_cell_direction_modules(all_df, outdir="./modules", prefix="cell2163")
```

单案例 Notebook 教程
-------------------
提供了一个端到端 notebook：

- `docs/tutorials/notebooks/tide_case_mouse_embryonic_brain.ipynb`

贡献与许可
----------
欢迎提交 issue/PR（请附可复现命令与预期结果）。

TIDE 包采用 MIT 许可证。图计算实现改写自 scVelo，其 BSD 3-Clause 许可证见
根目录的 `NOTICE`。
