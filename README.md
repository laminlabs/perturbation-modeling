# perturbation-modeling

Reusable toolkit for cross-study **perturbation transcriptomics**, paired with [LaminDB](https://docs.lamin.ai) so agent runs stay organized.

The git repo is the ML scientist's library. LaminDB Transforms are the _runs_ of that library — not a second copy of the same scripts.

**Exemplar instance:** [laminlabs/perturbation-modeling](https://lamin.ai/laminlabs/perturbation-modeling)

## Library vs transforms

| Layer         | Where                            | Versioned by         | What it is                                                                                    |
| ------------- | -------------------------------- | -------------------- | --------------------------------------------------------------------------------------------- |
| Reusable code | `perturbation_modeling/`         | git                  | Harmonize, train, select features, enrich, report helpers. **Not** registered as Transforms.  |
| A task run    | a script on the LaminDB instance | LaminDB `ln.track()` | Thin script that _imports_ the library, loads/saves artifacts, and becomes a Transform + Run. |

Agents should treat `perturbation_modeling/` like skills: import the matching module, write a short tracked script **on the instance**, and only edit the library when the reusable API itself needs to change.

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

## Pipeline

The library covers the workflow already exercised on the exemplar instance:

1. **Harmonize** Tahoe / LINCS / DRUG-seq (or any new study) onto a shared `perturbation` label and gene panel → `Collection`
2. **Train** a [modlyn](https://modlyn.lamin.ai/quickstart) `SimpleLogReg` on `MappedCollection` (feature selection)
3. **Rank genes** per perturbation from classifier weights
4. **Enrich** top genes (Enrichr) and write an interpretation report

Appending a study is `append_dataset(DatasetSpec(...))` then retrain — no rebuild of upstream modules.

## Install

```bash
pip install -e ".[train,enrich]"
```

Connect LaminDB to `laminlabs/perturbation-modeling` (or your own instance) before running a tracked script.

## Layout

```
perturbation_modeling/   # importable library (skills)
tests/
AGENTS.md                # how an agent should use this repo
```

See [AGENTS.md](AGENTS.md) for the agent contract.
