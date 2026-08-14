"""Gene-module enrichment from top genes per perturbation."""

from __future__ import annotations

import pandas as pd

from .keys import DEFAULT_GENE_SETS

_KEEP_COLS = (
    "perturbation",
    "Gene_set",
    "Term",
    "Overlap",
    "P-value",
    "Adjusted P-value",
    "Odds Ratio",
    "Combined Score",
    "Genes",
)


def short_term(term: str, max_len: int = 55) -> str:
    """Shorten GO-style term labels for plots."""
    s = str(term)
    if " (GO:" in s:
        name, go = s.rsplit(" (GO:", 1)
        s = (
            f"{name} ({go}"
            if len(name) <= max_len - 12
            else f"{name[: max_len - 15]}… (GO:{go}"
        )
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def enrich_top_genes(
    top_genes: pd.DataFrame,
    perturbations: list[str] | None = None,
    *,
    gene_sets: list[str] | None = None,
    organism: str = "human",
) -> pd.DataFrame:
    """Run Enrichr on the top-gene list of each perturbation."""
    import gseapy as gp

    gene_sets = gene_sets or list(DEFAULT_GENE_SETS)
    perts = perturbations or top_genes["perturbation"].astype(str).unique().tolist()
    rows: list[pd.DataFrame] = []
    for pert in perts:
        genes = (
            top_genes.loc[top_genes["perturbation"] == pert, "gene"]
            .astype(str)
            .tolist()
        )
        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=gene_sets,
                organism=organism,
                outdir=None,
                cutoff=1.0,
            )
            res = enr.results.copy()
        except Exception as e:
            print("Enrichr failed for", pert, type(e).__name__, e)
            continue
        if res is None or res.empty:
            continue
        res.insert(0, "perturbation", pert)
        rows.append(res)
        print(pert, "hits", len(res))

    if not rows:
        raise RuntimeError("No enrichment results — check gseapy/network or gene lists")

    enrichment = pd.concat(rows, ignore_index=True)
    keep = [c for c in _KEEP_COLS if c in enrichment.columns]
    return enrichment[keep].sort_values(
        ["perturbation", "Adjusted P-value"], ascending=[True, True]
    )


def best_term_per_perturbation(enrichment: pd.DataFrame) -> pd.DataFrame:
    """Lowest adjusted-p term for each perturbation."""
    return (
        enrichment.sort_values("Adjusted P-value")
        .groupby("perturbation", as_index=False)
        .first()
    )
