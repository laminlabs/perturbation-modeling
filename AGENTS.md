# Agent instructions

This repository is a **reusable perturbation-modeling library**, not a dump of LaminDB Transforms. Use it the way an ML scientist uses a private `src/` package: import it, run it, and extend it when the API is missing a piece.

## Contract

1. **`perturbation_modeling/` is the skill set.** Prefer importing an existing function over rewriting the same logic inside a new script.
2. **Transforms are thin.** A tracked script should `ln.track()`, load artifacts with `ln.Artifact.get` / `ln.Collection.get`, call the library, `ln.Artifact(...).save()`, `ln.finish()`. Do not paste training or harmonization code into the transform.
3. **Extend the library when the task is reusable.** If you need a new gene-panel rule, a different model head, or another enrichment backend, add it under `perturbation_modeling/` and then call it from the transform. That is the intended modification path.
4. **Never save a `.py` / `.ipynb` as a plain Artifact.** Scripts that produce data must self-track with `ln.track()` / `ln.finish()`.
5. **Do not read local paths for data that already lives in LaminDB.** Load the previous step with `ln.Artifact.get(key=...)` so lineage is recorded.

## Skill map

| Task                                          | Import                                           |
| --------------------------------------------- | ------------------------------------------------ |
| Normalize compound names                      | `perturbation_modeling.normalize_compound`       |
| Align one AnnData to the collection schema    | `perturbation_modeling.harmonize_anndata`        |
| Build Tahoe+LINCS (or any overlap collection) | `build_overlap_collection`, `tahoe_lincs_specs`  |
| Append DRUG-seq / in-house                    | `DatasetSpec`, `append_dataset`                  |
| Train / retrain feature-selection model       | `train_feature_selection`                        |
| Top genes from weights                        | `top_genes_from_weights`                         |
| Enrichr on those genes                        | `enrich_top_genes`, `best_term_per_perturbation` |
| Evidence markdown                             | `build_evidence_report`                          |

Default artifact keys for the exemplar instance are in `perturbation_modeling.keys`. Override them in the transform if the instance uses a different prefix.

## Writing a new transform

Write the tracked script **on the LaminDB instance** (not in this repo). Keep it thin:

```python
import lamindb as ln
from perturbation_modeling import train_feature_selection
from perturbation_modeling.keys import COLLECTION_KEY, WEIGHTS_KEY, TRAIN_SUMMARY_KEY

ln.track()
collection = ln.Collection.get(key=COLLECTION_KEY)
weights, summary = train_feature_selection(collection)
ln.Artifact.from_dataframe(weights.reset_index(), key=WEIGHTS_KEY).save()
ln.Artifact.from_dataframe(summary, key=TRAIN_SUMMARY_KEY).save()
ln.finish()
```

Put **run-specific** choices (which collection, how many steps, extra interpretation paragraphs) in that script. Put **reusable** behavior in the library.

Mechanistic narrative for a report belongs in the instance transform so it is attributed to that Run, not baked into the package.

## What not to do

- Do not re-register `perturbation_modeling/*.py` as Transforms.
- Do not duplicate `harmonize_anndata` inside an agent script "just this once."
- Do not hardcode `boehringer-demo/` keys; use `perturbation_modeling.keys` or pass keys in.
