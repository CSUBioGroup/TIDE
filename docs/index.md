# TIDE Documentation

Welcome to the documentation site for **TIDE: Transition-specific trajectory refinement and Interpretable Driver Extraction via gene module ensembles**.

TIDE is a projection-stage refinement framework for RNA velocity analysis. It takes velocity-ready AnnData as input, refines local transition scoring in a transition-specific manner, and exports interpretable directional gene modules.

```{toctree}
:maxdepth: 2
:hidden:

tutorials/index
```

## Getting started

- Install TIDE in a fresh Python ≥3.9 environment: `pip install -e .`
- Prepare velocity-ready AnnData (`velocity`, `Ms`, `Mu`, `velocity_u`, neighbors)
- Run TIDE graph construction and keep using scVelo/Scanpy plotting APIs

## Tutorials

Go to [Tutorials](tutorials/index.md) for:

- running TIDE on existing velocity outputs,
- extracting high-cos modules at celltype and transition levels,
- an end-to-end case notebook for mouse embryonic brain.
