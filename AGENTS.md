# Agent instructions

This is a Python library for ML scientists. Import it, call it from tracked scripts, and extend it when the API is missing a piece. Do not treat the repo as a dump of LaminDB Transforms.

## Contract

1. Prefer importing an existing function over rewriting the same logic in a new script.
2. Transforms are thin. A tracked script should `ln.track()`, load artifacts with `ln.Artifact.get` / `ln.Collection.get`, call the library, `ln.Artifact(...).save()`, `ln.finish()`. Do not paste training or harmonization code into the transform.
3. If you need a new gene-panel rule, a different model head, or another enrichment backend, add it under `perturbation_modeling/` and then call it from the transform.
4. Never save a `.py` / `.ipynb` as a plain Artifact. Scripts that produce data must self-track with `ln.track()` / `ln.finish()`.
5. Do not read local paths for data that already lives in LaminDB. Load the previous step with `ln.Artifact.get(key=...)` so lineage is recorded.

See the README for the API map and usage pattern.

## Writing a new transform

Write the tracked script on the LaminDB instance (not in this repo). Keep it thin:

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

Put run-specific choices (which collection, how many steps, extra interpretation paragraphs) in that script. Put reusable behavior in the library.

## What not to do

- Do not re-register `perturbation_modeling/*.py` as Transforms.
- Do not duplicate `harmonize_anndata` inside a script "just this once."
- Do not hardcode `boehringer-demo/` keys; use `perturbation_modeling.keys` or pass keys in.
