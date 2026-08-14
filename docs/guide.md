# Guide

This package is a **git-versioned library**. LaminDB Transforms are thin scripts that import it.

```{toctree}
:maxdepth: 1

quickstart
```

## Why a library (not one Transform per helper)

The [exemplar instance](https://lamin.ai/laminlabs/perturbation-modeling) originally registered full pipeline scripts as Transforms (`harmonize_and_append_datasets.py`, `train_feature_selection_model.py`, …). That records lineage, but it also freezes _reusable_ science inside one-off runs.

Here the split is:

- **Library** (`perturbation_modeling/`) — what an ML scientist keeps in git and an agent treats as skills.
- **Transforms** — thin scripts that live on the LaminDB instance: `ln.track()`, artifact I/O, hyperparameters for that run.

When the agent needs a new capability, it extends the library, then writes (or edits) a transform on the instance that calls it. See [AGENTS.md](https://github.com/laminlabs/perturbation-modeling/blob/main/AGENTS.md).
