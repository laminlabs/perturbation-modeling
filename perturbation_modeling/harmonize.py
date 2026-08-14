"""Align a perturbation AnnData to a shared schema (label, source, gene panel)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from .compounds import normalize_compound
from .io import to_memory_adata

if TYPE_CHECKING:
    import anndata as ad


def first_index_per_symbol(var_names: pd.Index | list[str]) -> dict[str, int]:
    """Map each gene symbol to the first column index it appears at."""
    first_idx: dict[str, int] = {}
    for i, name in enumerate(pd.Index(var_names).astype(str)):
        if name not in first_idx:
            first_idx[name] = i
    return first_idx


def align_to_gene_panel(adata: ad.AnnData, gene_panel: pd.Index) -> ad.AnnData:
    """Subset columns to gene_panel by integer index (first hit per symbol).

    Duplicate var_names (common in DRUG-seq) are kept as the first occurrence
    rather than calling var_names_make_unique.
    """
    first_idx = first_index_per_symbol(adata.var_names)
    panel_genes = [g for g in gene_panel.astype(str) if g in first_idx]
    if not panel_genes:
        raise RuntimeError("No shared genes with the requested gene panel")
    out = adata[:, [first_idx[g] for g in panel_genes]].copy()
    out.var_names = pd.Index(panel_genes)
    return out


def _obs_column(adata: Any, col: str) -> pd.Series:
    if col not in adata.obs.columns:
        raise KeyError(f"{col!r} not in obs columns: {list(adata.obs.columns)}")
    return adata.obs[col]


def harmonize_anndata(
    adata: Any,
    *,
    source: str,
    pert_col: str,
    gene_panel: pd.Index | None = None,
    symbol_col: str | None = None,
    allowed_compounds: set[str] | None = None,
    max_obs: int | None = None,
    log1p: bool = True,
    label_col: str = "perturbation",
    source_col: str = "source",
) -> ad.AnnData:
    """Align one study to the collection schema.

    Steps: filter to labeled / overlapping compounds, optional obs cap, map
    symbol_col onto var_names, subset to gene_panel, write
    perturbation + source, optional log1p.
    """
    pert = _obs_column(adata, pert_col).map(normalize_compound)
    if allowed_compounds is not None:
        mask = pert.isin(allowed_compounds).to_numpy()
    else:
        mask = (pert != "").to_numpy()

    subset = adata[mask]
    pert = pert.loc[subset.obs_names]
    if max_obs is not None and subset.n_obs > max_obs:
        subset = subset[:max_obs]
        pert = pert.loc[subset.obs_names]
        print(f"capping obs at {max_obs}")

    if symbol_col is None and gene_panel is not None:
        first_idx = first_index_per_symbol(subset.var_names)
        panel_genes = [g for g in gene_panel.astype(str) if g in first_idx]
        if not panel_genes:
            raise RuntimeError(f"No shared genes between {source} and the gene panel")
        subset = subset[:, [first_idx[g] for g in panel_genes]]

    out = to_memory_adata(subset)
    if symbol_col is not None:
        if symbol_col not in out.var.columns:
            raise KeyError(
                f"symbol_col={symbol_col!r} not in var columns: {list(out.var.columns)}"
            )
        out.var_names = out.var[symbol_col].astype(str).values
        if gene_panel is not None:
            out = align_to_gene_panel(out, gene_panel)
    elif gene_panel is not None:
        out.var_names = pd.Index(out.var_names.astype(str))

    out.obs[label_col] = pert.astype(str).values
    out.obs[source_col] = source
    print(f"{source}: {out.n_obs} obs, {out.n_vars} genes")

    if log1p:
        import scanpy as sc

        sc.pp.log1p(out)
    return out


def gene_symbols(adata: Any, symbol_col: str | None = None) -> pd.Index:
    """Return unique gene symbols from var_names or a var column."""
    if symbol_col is None:
        return pd.Index(adata.var_names.astype(str)).unique()
    if symbol_col not in adata.var.columns:
        raise KeyError(
            f"symbol_col={symbol_col!r} not in var columns: {list(adata.var.columns)}"
        )
    return pd.Index(adata.var[symbol_col].astype(str)).unique()
