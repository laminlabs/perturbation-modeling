"""Reusable perturbation-modeling library.

This package is the git-versioned toolkit an ML scientist (or agent) imports.
It is **not** a LaminDB Transform. Tracked runs live as thin scripts on the
LaminDB instance that call these functions and save artifacts.

.. autosummary::
   :toctree: .

   DatasetSpec
   append_dataset
   best_term_per_perturbation
   build_evidence_report
   build_overlap_collection
   enrich_top_genes
   harmonize_anndata
   normalize_compound
   tahoe_lincs_specs
   top_genes_from_weights
   train_feature_selection
"""

__version__ = "0.0.1"

from .collection import (
    DatasetSpec,
    append_dataset,
    build_overlap_collection,
    tahoe_lincs_specs,
)
from .compounds import normalize_compound
from .enrichment import best_term_per_perturbation, enrich_top_genes
from .features import top_genes_from_weights
from .harmonize import harmonize_anndata
from .report import build_evidence_report
from .train import train_feature_selection
