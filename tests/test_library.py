"""Library tests that do not need a LaminDB instance."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from perturbation_modeling import (
    DatasetSpec,
    build_evidence_report,
    harmonize_anndata,
    lincs_spec,
    lincs_specs,
    normalize_compound,
    tahoe_artifact_key,
    tahoe_spec,
    top_genes_from_weights,
)
from perturbation_modeling.collection import overlap_compounds
from perturbation_modeling.compounds import normalize_compound as _norm
from perturbation_modeling.enrichment import best_term_per_perturbation, short_term
from perturbation_modeling.features import rank_perturbations, recurrent_genes
from perturbation_modeling.harmonize import align_to_gene_panel
from perturbation_modeling.keys import LABEL_COL, LINCS_UIDS, TAHOE_TEST_UID


def test_normalize_compound():
    assert normalize_compound("Imatinib (mesylate)") == "imatinib"
    assert normalize_compound("DMSO") == "dmso"
    assert normalize_compound("foo-bar_tf") == "foo bar"
    assert normalize_compound(None) == ""
    assert _norm(float("nan")) == ""


def test_top_genes_and_ranking():
    weights = pd.DataFrame(
        {"EGFR": [3.0, 0.1], "GAPDH": [0.2, 4.0], "TP53": [1.0, 0.5]},
        index=["imatinib", "dmso"],
    )
    top = top_genes_from_weights(weights, n=2)
    assert list(top.columns) == [LABEL_COL, "rank", "gene", "weight"]
    imatinib = top.loc[top[LABEL_COL] == "imatinib"].sort_values("rank")
    assert list(imatinib["gene"]) == ["EGFR", "TP53"]
    assert rank_perturbations(top, n=1) == ["dmso"]
    assert recurrent_genes(top, ["imatinib", "dmso"], n=2)


def test_harmonize_filters_and_gene_panel():
    adata = ad.AnnData(
        X=np.arange(12, dtype=np.float32).reshape(3, 4),
        obs=pd.DataFrame({"drug": ["Imatinib (mesylate)", "DMSO", "vehicle"]}),
        var=pd.DataFrame(index=["EGFR", "GAPDH", "ACTB", "TP53"]),
    )
    out = harmonize_anndata(
        adata,
        source="toy",
        pert_col="drug",
        gene_panel=pd.Index(["TP53", "EGFR", "MISSING"]),
        allowed_compounds={"imatinib", "dmso"},
        log1p=False,
    )
    assert list(out.obs[LABEL_COL]) == ["imatinib", "dmso"]
    assert list(out.obs["source"]) == ["toy", "toy"]
    assert list(out.var_names) == ["TP53", "EGFR"]


def test_align_duplicate_var_names():
    adata = ad.AnnData(
        X=np.array([[1.0, 9.0, 2.0]], dtype=np.float32),
        obs=pd.DataFrame({"drug": ["a"]}),
        var=pd.DataFrame(index=["EGFR", "EGFR", "TP53"]),
    )
    out = align_to_gene_panel(adata, pd.Index(["EGFR", "TP53"]))
    assert list(out.var_names) == ["EGFR", "TP53"]
    assert float(out.X[0, 0]) == 1.0


def test_enrichment_helpers_and_report():
    assert short_term("apoptosis (GO:0006915)").startswith("apoptosis")
    enrichment = pd.DataFrame(
        {
            LABEL_COL: ["imatinib", "imatinib", "dmso"],
            "Gene_set": ["Hallmark", "Hallmark", "Hallmark"],
            "Term": ["KRAS", "APOPTOSIS", "MITOSIS"],
            "Adjusted P-value": [0.02, 0.001, 0.05],
        }
    )
    best = best_term_per_perturbation(enrichment)
    imatinib = best.loc[best[LABEL_COL] == "imatinib"].iloc[0]
    assert imatinib["Term"] == "APOPTOSIS"

    summary = pd.DataFrame(
        {"n_obs": [10], "n_vars": [3], "n_classes": [2], "max_steps": [50]}
    )
    top = pd.DataFrame(
        {
            LABEL_COL: ["imatinib", "imatinib", "dmso"],
            "rank": [1, 2, 1],
            "gene": ["EGFR", "TP53", "GAPDH"],
            "weight": [3.0, 1.0, 4.0],
        }
    )
    text = build_evidence_report(
        summary=summary, best_terms=best, top_genes=top, n_focus=2
    )
    assert "imatinib" in text
    assert "Caveats" in text


def test_dataset_specs_are_independent():
    t = tahoe_spec(max_obs=1000)
    assert t.source == "tahoe"
    assert t.uid_or_key == TAHOE_TEST_UID
    assert t.pert_col == "drug"
    assert t.symbol_col is None
    assert t.max_obs == 1000

    plate = tahoe_spec(
        uid_or_key=tahoe_artifact_key(14),
        max_obs=None,
    )
    assert plate.uid_or_key == (
        "tahoe100m/2025-02-25/"
        "plate14_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab/obs.parquet"
    )
    assert plate.max_obs is None

    one = lincs_spec("lincs_phase2")
    assert one.source == "lincs_phase2"
    assert one.uid_or_key == LINCS_UIDS["lincs_phase2"]
    assert one.pert_col == "pert_compound"
    assert one.symbol_col == "pr_gene_symbol"

    all_lincs = lincs_specs()
    assert [s.source for s in all_lincs] == list(LINCS_UIDS)
    subset = lincs_specs(sources=["lincs_phase1_delta"])
    assert len(subset) == 1
    assert subset[0].source == "lincs_phase1_delta"

    custom = DatasetSpec(uid_or_key="my.h5ad", source="inhouse", pert_col="drug")
    mixed = [t, one, custom]
    assert [s.source for s in mixed] == ["tahoe", "lincs_phase2", "inhouse"]


def test_overlap_compounds_one_or_many():
    a = pd.Series(["Imatinib (mesylate)", "DMSO"])
    b = pd.Series(["imatinib", "vehicle"])
    assert overlap_compounds(a) == ["dmso", "imatinib"]
    assert overlap_compounds(a, b) == ["imatinib"]


class _FakeMembers:
    def __init__(self, names):
        self._names = list(names)

    def values_list(self, field, flat=True):
        return list(self._names)

    def __iter__(self):
        return iter(self._names)


class _FakeSchema:
    def __init__(self, names):
        self.members = _FakeMembers(names)


class _FakeArtifact:
    def __init__(self, key, schema=None, suffix=None, features=None, frame=None):
        self.key = key
        self.schema = schema
        self.suffix = suffix
        self.features = features or {}
        self._frame = frame

    def load(self):
        return self._frame

    def open(self):
        return self._frame


def test_resolve_pert_col_from_schema():
    from perturbation_modeling.schema import (
        load_pert_series,
        related_obs_artifact,
        related_x_artifact,
        resolve_pert_col,
        sibling_key,
    )

    uncurated = _FakeArtifact(
        "dataloader_v2/tahoe100M_sharded/shard_0.h5ad", suffix=".h5ad"
    )
    assert resolve_pert_col(uncurated, "drug") == "drug"
    assert resolve_pert_col(uncurated) == "drug"

    composite = _FakeArtifact(
        "study.h5ad",
        schema=type(
            "S", (), {"members": None, "slots": {"obs": _FakeSchema(["pert_compound"])}}
        )(),
        suffix=".h5ad",
    )
    assert resolve_pert_col(composite, "drug") == "pert_compound"

    curated = _FakeArtifact(
        "tahoe100m/plate14/obs.parquet",
        schema=_FakeSchema(["pert_compound", "cell_line"]),
        suffix=".parquet",
    )
    assert resolve_pert_col(curated, "drug") == "pert_compound"
    assert resolve_pert_col(curated) == "pert_compound"
    assert resolve_pert_col(curated, "custom_col") == "custom_col"

    x_art = _FakeArtifact(
        "tahoe100m/plate14/X.h5ad",
        suffix=".h5ad",
        features={"obs": curated},
    )
    obs_with_x = _FakeArtifact(
        "tahoe100m/plate14/obs.parquet",
        schema=_FakeSchema(["pert_compound"]),
        suffix=".parquet",
        features={"X": x_art},
    )
    assert resolve_pert_col(x_art, "drug") == "pert_compound"
    assert related_obs_artifact(curated) is curated
    assert related_x_artifact(x_art) is x_art
    assert related_obs_artifact(x_art) is curated
    assert related_x_artifact(obs_with_x) is x_art

    assert (
        sibling_key("a/X.h5ad", from_name="X.h5ad", to_name="obs.parquet")
        == "a/obs.parquet"
    )
    assert (
        sibling_key("a/obs.parquet", from_name="obs.parquet", to_name="X.h5ad")
        == "a/X.h5ad"
    )

    adata = ad.AnnData(
        X=np.zeros((2, 1), dtype=np.float32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["EGFR"]),
    )
    obs = _FakeArtifact(
        "obs.parquet",
        suffix=".parquet",
        frame=pd.DataFrame({"pert_compound": ["Imatinib", "DMSO"]}, index=["c1", "c2"]),
    )
    series = load_pert_series(adata, "pert_compound", obs)
    assert list(series) == ["Imatinib", "DMSO"]


def test_harmonize_external_pert_series():
    adata = ad.AnnData(
        X=np.arange(8, dtype=np.float32).reshape(2, 4),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["EGFR", "GAPDH", "ACTB", "TP53"]),
    )
    pert = pd.Series(["Imatinib (mesylate)", "DMSO"], index=["c1", "c2"])
    out = harmonize_anndata(
        adata,
        source="pertdata",
        pert_col="pert_compound",
        gene_panel=pd.Index(["EGFR", "TP53"]),
        log1p=False,
        pert=pert,
    )
    assert list(out.obs[LABEL_COL]) == ["imatinib", "dmso"]
    assert list(out.var_names) == ["EGFR", "TP53"]
