"""Default LaminDB artifact keys for the perturbation-modeling instance.

Pass a prefix to ``output_keys`` (or ``build_collection(..., prefix=...)``) when
a run should write under a different folder than ``pert_modeling/``.

Tahoe: TAHOE_TEST_* is the small shard_0.h5ad used in tests and the exemplar
(uncurated; obs column is drug). Curated plates on laminlabs/pertdata use
PertSchema obs (pert_compound). Production keys look like
tahoe100m/2025-02-25/plate14_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab/obs.parquet
(see tahoe_artifact_key).
"""

from dataclasses import dataclass

PREFIX = "pert_modeling"


@dataclass(frozen=True)
class OutputKeys:
    """Artifact / collection keys for one run, all under ``prefix/``."""

    prefix: str
    collection: str
    overlap: str
    tahoe_harmonized: str
    weights: str
    train_summary: str
    top_genes: str
    enrichment: str
    top_term: str
    report: str


def output_keys(prefix: str = PREFIX) -> OutputKeys:
    """Keys for output artifacts and the Collection under ``prefix/``."""
    prefix = prefix.strip().strip("/")
    if not prefix:
        raise ValueError("prefix must be a non-empty path, e.g. 'pert_modeling'")
    return OutputKeys(
        prefix=prefix,
        collection=f"{prefix}/harmonized",
        overlap=f"{prefix}/compounds.csv",
        tahoe_harmonized=f"{prefix}/tahoe_harmonized.h5ad",
        weights=f"{prefix}/modlyn_perturbation_weights.parquet",
        train_summary=f"{prefix}/modlyn_train_summary.csv",
        top_genes=f"{prefix}/modlyn_top_genes.csv",
        enrichment=f"{prefix}/modlyn_gene_module_enrichment.csv",
        top_term=f"{prefix}/modlyn_enrichment_top_term.csv",
        report=f"{prefix}/modlyn_interpretation_report.md",
    )


_DEFAULT = output_keys()
COLLECTION_KEY = _DEFAULT.collection
OVERLAP_KEY = _DEFAULT.overlap
TAHOE_HARMONIZED_KEY = _DEFAULT.tahoe_harmonized
WEIGHTS_KEY = _DEFAULT.weights
TRAIN_SUMMARY_KEY = _DEFAULT.train_summary
TOP_GENES_KEY = _DEFAULT.top_genes
ENRICHMENT_KEY = _DEFAULT.enrichment
TOP_TERM_KEY = _DEFAULT.top_term
REPORT_KEY = _DEFAULT.report

# Small Tahoe-100M shard for tests / the exemplar instance (~7 GB, shard_0).
TAHOE_TEST_UID = "ipI9MJQ5Jn6URPQv0000"
TAHOE_TEST_KEY = "dataloader_v2/tahoe100M_sharded/shard_0.h5ad"
# Backward-compatible alias; prefer TAHOE_TEST_UID.
TAHOE_UID = TAHOE_TEST_UID

# Production Tahoe-100M plate artifacts (folder with obs.parquet, X, var, ...).
TAHOE_KEY_PREFIX = "tahoe100m/2025-02-25"
TAHOE_PLATE_SUFFIX = "filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab"

LINCS_UIDS = {
    "lincs_phase2": "1MvhipblenpfE9ol0000",
    "lincs_phase1_epsilon": "nUIput1HM5of6JKn0000",
    "lincs_phase1_delta": "QlcIPRMMk667dGwS0000",
}

LABEL_COL = "pert_compound"
SOURCE_COL = "source"

DEFAULT_GENE_SETS = [
    "MSigDB_Hallmark_2020",
    "GO_Biological_Process_2023",
]


def tahoe_artifact_key(
    plate: str | int,
    *,
    prefix: str = TAHOE_KEY_PREFIX,
    component: str = "obs.parquet",
) -> str:
    """Key of a production Tahoe-100M plate artifact.

    >>> tahoe_artifact_key(14)
    'tahoe100m/2025-02-25/plate14_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab/obs.parquet'
    """
    if isinstance(plate, int):
        stem = f"plate{plate}_{TAHOE_PLATE_SUFFIX}"
    else:
        stem = plate
    return f"{prefix}/{stem}/{component}"
