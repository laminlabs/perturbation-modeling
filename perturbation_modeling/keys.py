"""Default LaminDB artifact keys for the perturbation-modeling instance.

Override these in a transform when working on a different instance or prefix.

Tahoe: TAHOE_TEST_* is the small shard_0.h5ad used in tests and the exemplar
(uncurated; obs column is drug). Curated plates on laminlabs/pertdata use
PertSchema obs (pert_compound). Production keys look like
tahoe100m/2025-02-25/plate14_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab/obs.parquet
(see tahoe_artifact_key).
"""

PREFIX = "pert_modeling"

COLLECTION_KEY = f"{PREFIX}/harmonized"
OVERLAP_KEY = f"{PREFIX}/compounds.csv"
TAHOE_HARMONIZED_KEY = f"{PREFIX}/tahoe_harmonized.h5ad"

WEIGHTS_KEY = f"{PREFIX}/modlyn_perturbation_weights.parquet"
TRAIN_SUMMARY_KEY = f"{PREFIX}/modlyn_train_summary.csv"
TOP_GENES_KEY = f"{PREFIX}/modlyn_top_genes.csv"
ENRICHMENT_KEY = f"{PREFIX}/modlyn_gene_module_enrichment.csv"
TOP_TERM_KEY = f"{PREFIX}/modlyn_enrichment_top_term.csv"
REPORT_KEY = f"{PREFIX}/modlyn_interpretation_report.md"

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
