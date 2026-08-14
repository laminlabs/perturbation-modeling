# perturbation-modeling

Python library for cross-study **perturbation transcriptomics**: harmonize compound screens into a LaminDB `Collection`, train a linear feature-selection model, rank genes, and enrich.

Import it like any other scientific package. Tracked scripts that call it (written by you or by an agent) live on the LaminDB instance — not in this repo.

**Exemplar instance:** [laminlabs/perturbation-modeling](https://lamin.ai/laminlabs/perturbation-modeling)

## Install

```bash
pip install -e ".[train,enrich]"
```

Connect LaminDB to `laminlabs/perturbation-modeling` or your own instance.

## Usage

Reusable logic stays in `perturbation_modeling/`. Each analysis run is a thin script with `ln.track()` / `ln.finish()` that loads and saves LaminDB artifacts:

```python
import lamindb as ln
from perturbation_modeling import train_feature_selection
from perturbation_modeling.keys import COLLECTION_KEY, WEIGHTS_KEY

ln.track()
collection = ln.Collection.get(key=COLLECTION_KEY)
weights, summary = train_feature_selection(collection)
ln.Artifact.from_dataframe(weights.reset_index(), key=WEIGHTS_KEY).save()
ln.finish()
```

Extend the library when a step is reusable (a new gene-panel rule, another model head). Keep run-specific choices (which collection, how many steps, interpretation notes) in the tracked script.

## Pipeline

1. **Harmonize** Tahoe / LINCS / DRUG-seq (or any new study) onto a shared `perturbation` label and gene panel → `Collection`
2. **Train** a [modlyn](https://modlyn.lamin.ai/quickstart) `SimpleLogReg` on `MappedCollection`
3. **Rank genes** per perturbation from classifier weights
4. **Enrich** top genes (Enrichr) and write an interpretation report

Append a study with `append_dataset(DatasetSpec(...))` and retrain — no rebuild of upstream modules.

## API

| Task                                          | Import                                           |
| --------------------------------------------- | ------------------------------------------------ |
| Normalize compound names                      | `normalize_compound`                             |
| Align one AnnData to the collection schema    | `harmonize_anndata`                              |
| Build Tahoe+LINCS (or any overlap collection) | `build_overlap_collection`, `tahoe_lincs_specs`  |
| Append DRUG-seq / in-house                    | `DatasetSpec`, `append_dataset`                  |
| Train / retrain feature-selection model       | `train_feature_selection`                        |
| Top genes from weights                        | `top_genes_from_weights`                         |
| Enrichr on those genes                        | `enrich_top_genes`, `best_term_per_perturbation` |
| Evidence markdown                             | `build_evidence_report`                          |

Default artifact keys are in `perturbation_modeling.keys`. Override them in your script if the instance uses a different prefix.

Coding agents should follow [AGENTS.md](AGENTS.md) so they import this library instead of copying pipeline logic into one-off scripts.
