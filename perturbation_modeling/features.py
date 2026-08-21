"""Rank genes from a trained perturbation classifier."""

from __future__ import annotations

import pandas as pd

from .keys import LABEL_COL


def prepare_weights(weights: pd.DataFrame, *, index_col: str = "index") -> pd.DataFrame:
    """Ensure a classes × genes weight matrix with pert_compound as the index."""
    out = weights.copy()
    if index_col in out.columns:
        out = out.set_index(index_col)
    out.index = out.index.astype(str)
    out.index.name = LABEL_COL
    return out


def top_genes_from_weights(weights: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Long table of the top-n genes (highest weight) per perturbation."""
    W = prepare_weights(weights)
    rows: list[dict[str, object]] = []
    for pert, row in W.iterrows():
        top = row.sort_values(ascending=False).head(n)
        for rank, (gene, score) in enumerate(top.items(), start=1):
            rows.append(
                {
                    LABEL_COL: pert,
                    "rank": rank,
                    "gene": gene,
                    "weight": float(score),
                }
            )
    return pd.DataFrame(rows)


def rank_perturbations(top_genes: pd.DataFrame, n: int | None = None) -> list[str]:
    """Perturbations ordered by mean |weight| of their top genes."""
    scores = (
        top_genes.groupby(LABEL_COL)["weight"]
        .apply(lambda s: s.abs().mean())
        .sort_values(ascending=False)
    )
    names = scores.index.astype(str).tolist()
    return names[:n] if n is not None else names


def recurrent_genes(
    top_genes: pd.DataFrame,
    perturbations: list[str],
    n: int = 40,
) -> list[str]:
    """Genes that appear most often in the top-N lists of perturbations."""
    return (
        top_genes.loc[top_genes[LABEL_COL].isin(perturbations), "gene"]
        .value_counts()
        .head(n)
        .index.astype(str)
        .tolist()
    )
