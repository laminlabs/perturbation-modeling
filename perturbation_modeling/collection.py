"""Build or append a multi-study perturbation Collection."""

from __future__ import annotations

from dataclasses import dataclass

import lamindb as ln
import pandas as pd

from .compounds import normalize_compound
from .harmonize import gene_symbols, harmonize_anndata
from .io import close_backed, open_backed
from .keys import (
    COLLECTION_KEY,
    LINCS_UIDS,
    OVERLAP_KEY,
    PREFIX,
    TAHOE_HARMONIZED_KEY,
    TAHOE_UID,
)


@dataclass(frozen=True)
class DatasetSpec:
    """One study to harmonize into (or onto) a Collection."""

    uid_or_key: str
    source: str
    pert_col: str
    symbol_col: str | None = None
    max_obs: int | None = None
    artifact_key: str | None = None
    description: str | None = None

    def output_key(self, prefix: str = PREFIX) -> str:
        if self.artifact_key is not None:
            return self.artifact_key
        return f"{prefix}/{self.source}_harmonized.h5ad"


def tahoe_lincs_specs(*, tahoe_max_obs: int | None = 200000) -> list[DatasetSpec]:
    """Default Tahoe shard + LINCS Level 2 studies used in the exemplar instance."""
    specs = [
        DatasetSpec(
            uid_or_key=TAHOE_UID,
            source="tahoe",
            pert_col="drug",
            max_obs=tahoe_max_obs,
            artifact_key=TAHOE_HARMONIZED_KEY,
            description=(
                "Tahoe subset: overlapping compounds, landmark genes, "
                "perturbation from drug, log1p"
            ),
        )
    ]
    for source, uid in LINCS_UIDS.items():
        specs.append(
            DatasetSpec(
                uid_or_key=uid,
                source=source,
                pert_col="pert_compound",
                symbol_col="pr_gene_symbol",
                description=(
                    f"{source} subset: overlapping compounds, "
                    "var_names=pr_gene_symbol, perturbation from pert_compound, log1p"
                ),
            )
        )
    return specs


def overlap_compounds(*series: pd.Series) -> list[str]:
    """Sorted intersection of normalized compound names, dropping empties."""
    sets = [set(s.map(normalize_compound)) - {""} for s in series]
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def load_gene_panel(key: str) -> pd.Index:
    """Gene symbols from a previously saved harmonized AnnData artifact."""
    art = ln.Artifact.filter(key=key, is_latest=True).one_or_none()
    if art is None:
        raise RuntimeError(f"Missing gene-panel artifact {key}")
    return art.load().var_names.copy()


def load_overlap_compounds(key: str = OVERLAP_KEY) -> set[str] | None:
    art = ln.Artifact.filter(key=key, is_latest=True).one_or_none()
    if art is None:
        return None
    df = art.load()
    col = "perturbation" if "perturbation" in df.columns else df.columns[0]
    return set(df[col].astype(str)) - {""}


def save_harmonized(
    adata,
    *,
    key: str,
    description: str,
) -> ln.Artifact:
    return ln.Artifact.from_anndata(adata, key=key, description=description).save()


def build_overlap_collection(
    specs: list[DatasetSpec],
    *,
    collection_key: str = COLLECTION_KEY,
    overlap_key: str = OVERLAP_KEY,
    prefix: str = PREFIX,
    log1p: bool = True,
    collection_description: str | None = None,
) -> ln.Collection:
    """Intersect perturbation labels across studies, harmonize, save a Collection.

    Gene panel is the intersection of symbols (var_names or symbol_col)
    across every spec. Each spec is subset to the compound overlap before save.
    """
    if len(specs) < 2:
        raise ValueError("Need at least two DatasetSpec entries to compute overlap")

    loaded: list[tuple[DatasetSpec, object, object, pd.Series]] = []
    symbol_sets: list[pd.Index] = []
    for spec in specs:
        art, adata = open_backed(spec.uid_or_key)
        pert = adata.obs[spec.pert_col]  # type: ignore[union-attr]
        loaded.append((spec, art, adata, pert))
        symbol_sets.append(gene_symbols(adata, spec.symbol_col))
        print(spec.source, art.key, adata.shape)  # type: ignore[union-attr]

    overlap = overlap_compounds(*[pert for _, _, _, pert in loaded])
    print("normalized compound overlap:", len(overlap))

    gene_panel = symbol_sets[0]
    for symbols in symbol_sets[1:]:
        gene_panel = gene_panel.intersection(symbols)
    print("common genes:", len(gene_panel))
    if len(gene_panel) == 0:
        raise RuntimeError("No shared gene symbols across DatasetSpec entries")

    artifacts: list[ln.Artifact] = []
    try:
        for spec, _art, adata, _pert in loaded:
            harmonized = harmonize_anndata(
                adata,
                source=spec.source,
                pert_col=spec.pert_col,
                gene_panel=gene_panel,
                symbol_col=spec.symbol_col,
                allowed_compounds=set(overlap),
                max_obs=spec.max_obs,
                log1p=log1p,
            )
            artifacts.append(
                save_harmonized(
                    harmonized,
                    key=spec.output_key(prefix),
                    description=spec.description
                    or (
                        f"{spec.source} subset: overlapping compounds, "
                        f"pert_col={spec.pert_col}, log1p={log1p}"
                    ),
                )
            )
    finally:
        for _spec, _art, adata, _pert in loaded:
            close_backed(adata)

    ln.Artifact.from_dataframe(
        pd.DataFrame({"perturbation": overlap}),
        key=overlap_key,
        description=f"Normalized compound overlap ({len(overlap)} names)",
    ).save()

    collection = ln.Collection(
        artifacts,
        key=collection_key,
        description=collection_description
        or f"Harmonized perturbation collection ({len(specs)} studies, log1p={log1p})",
    ).save()
    print("collection", collection.uid, collection.key)
    return collection


def append_dataset(
    spec: DatasetSpec,
    *,
    collection_key: str = COLLECTION_KEY,
    gene_panel_key: str | None = None,
    overlap_key: str | None = OVERLAP_KEY,
    filter_to_overlap: bool = True,
    log1p: bool = True,
    prefix: str = PREFIX,
) -> ln.Collection:
    """Harmonize one more study onto an existing Collection and version it."""
    collection = ln.Collection.get(key=collection_key)
    print("appending to", collection.uid, collection.key)

    if gene_panel_key is None:
        gene_panel_key = f"{prefix}/tahoe_harmonized.h5ad"
    gene_panel = load_gene_panel(gene_panel_key)
    allowed = (
        load_overlap_compounds(overlap_key)
        if filter_to_overlap and overlap_key
        else None
    )
    if filter_to_overlap and allowed is None:
        print("warning: no overlap artifact; keeping all non-empty perturbations")

    src_art, src_adata = open_backed(spec.uid_or_key)
    print("source artifact", src_art.uid, src_art.key, src_adata.shape)
    try:
        harmonized = harmonize_anndata(
            src_adata,
            source=spec.source,
            pert_col=spec.pert_col,
            gene_panel=gene_panel,
            symbol_col=spec.symbol_col,
            allowed_compounds=allowed,
            max_obs=spec.max_obs,
            log1p=log1p,
        )
    finally:
        close_backed(src_adata)

    new_art = save_harmonized(
        harmonized,
        key=spec.output_key(prefix),
        description=spec.description
        or (
            f"{spec.source} appended to {collection_key}: "
            f"pert_col={spec.pert_col}, log1p={log1p}"
        ),
    )
    collection_new = collection.append(new_art).save()
    print("collection new version", collection_new.uid, collection_new.key)
    return collection_new
