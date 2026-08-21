"""Align a perturbation AnnData to a shared schema (label, source, gene panel)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from .compounds import normalize_compound
from .io import n_obs, to_memory_adata
from .keys import LABEL_COL, SOURCE_COL

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


def _align_pert(pert: pd.Series, obs_names: pd.Index) -> pd.Series:
    names = pd.Index(obs_names).astype(str)
    s = pert.copy()
    s.index = s.index.astype(str)
    if s.index.equals(names) or bool(s.index.isin(names).all()):
        return s.reindex(names)
    if len(s) == len(names):
        s.index = names
        return s
    raise ValueError("Cannot align perturbation labels to obs_names")


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
    label_col: str = LABEL_COL,
    source_col: str = SOURCE_COL,
    pert: pd.Series | None = None,
) -> ad.AnnData:
    """Align one study to the collection schema.

    Steps: filter to labeled / overlapping compounds, optional obs cap, map
    symbol_col onto var_names, subset to gene_panel, write
    pert_compound + source, optional log1p.

    pert can be passed from a curated obs.parquet when adata.obs has no
    pert_compound column.
    """
    if pert is None:
        raw = _obs_column(adata, pert_col)
    else:
        raw = pert
    raw = _align_pert(raw, adata.obs_names)
    pert_norm = raw.map(normalize_compound)
    if allowed_compounds is not None:
        mask = pert_norm.isin(allowed_compounds).to_numpy()
    else:
        mask = (pert_norm != "").to_numpy()

    subset = adata[mask]
    pert_norm = pert_norm.loc[pd.Index(subset.obs_names).astype(str)]
    if max_obs is not None and n_obs(subset) > max_obs:
        subset = subset[:max_obs]
        pert_norm = pert_norm.loc[pd.Index(subset.obs_names).astype(str)]
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

    out.obs[label_col] = _align_pert(pert_norm, out.obs_names).astype(str).values
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
