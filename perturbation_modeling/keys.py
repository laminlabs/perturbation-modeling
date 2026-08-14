"""Default LaminDB artifact keys for the perturbation-modeling instance.

Override these in a transform when working on a different instance or prefix.
"""

PREFIX = "pert_modeling"

COLLECTION_KEY = f"{PREFIX}/tahoe-lincs-harmonized"
OVERLAP_KEY = f"{PREFIX}/compound_overlap.csv"
TAHOE_HARMONIZED_KEY = f"{PREFIX}/tahoe_harmonized.h5ad"

WEIGHTS_KEY = f"{PREFIX}/modlyn_perturbation_weights.parquet"
TRAIN_SUMMARY_KEY = f"{PREFIX}/modlyn_train_summary.csv"
TOP_GENES_KEY = f"{PREFIX}/modlyn_top_genes.csv"
ENRICHMENT_KEY = f"{PREFIX}/modlyn_gene_module_enrichment.csv"
TOP_TERM_KEY = f"{PREFIX}/modlyn_enrichment_top_term.csv"
REPORT_KEY = f"{PREFIX}/modlyn_interpretation_report.md"

TAHOE_UID = "ipI9MJQ5Jn6URPQv0000"
LINCS_UIDS = {
    "lincs_phase2": "1MvhipblenpfE9ol0000",
    "lincs_phase1_epsilon": "nUIput1HM5of6JKn0000",
    "lincs_phase1_delta": "QlcIPRMMk667dGwS0000",
}

LABEL_COL = "perturbation"
SOURCE_COL = "source"

DEFAULT_GENE_SETS = [
    "MSigDB_Hallmark_2020",
    "GO_Biological_Process_2023",
]
