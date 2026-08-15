"""Build or append a perturbation Collection from any set of studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    TAHOE_TEST_UID,
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


def tahoe_spec(
    *,
    uid_or_key: str = TAHOE_TEST_UID,
    pert_col: str = "drug",
    max_obs: int | None = 200000,
    artifact_key: str | None = TAHOE_HARMONIZED_KEY,
    description: str | None = None,
) -> DatasetSpec:
    """DatasetSpec for a Tahoe AnnData artifact.

    Default uid_or_key is the small shard_0.h5ad used in tests
    (TAHOE_TEST_UID / TAHOE_TEST_KEY). For production plates pass a key from
    tahoe_artifact_key, for example
    tahoe_spec(uid_or_key=tahoe_artifact_key(14), max_obs=None).
    """
    return DatasetSpec(
        uid_or_key=uid_or_key,
        source="tahoe",
        pert_col=pert_col,
        max_obs=max_obs,
        artifact_key=artifact_key,
        description=description
        or "Tahoe: perturbation from drug, gene symbols from var_names, log1p",
    )


def lincs_spec(
    source: str = "lincs_phase2",
    *,
    uid_or_key: str | None = None,
    pert_col: str = "pert_compound",
    symbol_col: str = "pr_gene_symbol",
    max_obs: int | None = None,
    artifact_key: str | None = None,
    description: str | None = None,
) -> DatasetSpec:
    """DatasetSpec for one LINCS Level 2 artifact.

    source should be a key in LINCS_UIDS (lincs_phase2, lincs_phase1_epsilon,
    lincs_phase1_delta) unless uid_or_key is passed explicitly.
    """
    if uid_or_key is None:
        if source not in LINCS_UIDS:
            raise KeyError(
                f"Unknown LINCS source {source!r}. Pass uid_or_key or use one of "
                f"{sorted(LINCS_UIDS)}"
            )
        uid_or_key = LINCS_UIDS[source]
    return DatasetSpec(
        uid_or_key=uid_or_key,
        source=source,
        pert_col=pert_col,
        symbol_col=symbol_col,
        max_obs=max_obs,
        artifact_key=artifact_key,
        description=description
        or (
            f"{source}: perturbation from {pert_col}, "
            f"var_names from {symbol_col}, log1p"
        ),
    )


def lincs_specs(
    sources: list[str] | None = None,
    **kwargs: Any,
) -> list[DatasetSpec]:
    """DatasetSpecs for one or more LINCS Level 2 artifacts.

    Default is all entries in LINCS_UIDS. kwargs are forwarded to lincs_spec.
    """
    names = sources if sources is not None else list(LINCS_UIDS)
    return [lincs_spec(name, **kwargs) for name in names]


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


def _gene_panel_from_collection(collection: ln.Collection) -> pd.Index:
    art = collection.artifacts.order_by("created_at").first()
    if art is None:
        raise RuntimeError(f"Collection {collection.key} has no artifacts")
    return art.load().var_names.copy()


def build_collection(
    specs: list[DatasetSpec],
    *,
    collection_key: str = COLLECTION_KEY,
    overlap_key: str | None = OVERLAP_KEY,
    prefix: str = PREFIX,
    log1p: bool = True,
    intersect_compounds: bool = True,
    collection_description: str | None = None,
) -> ln.Collection:
    """Harmonize one or more studies and save them as a Collection.

    Pass any mix of DatasetSpec values — Tahoe, LINCS, in-house, or all of them.
    Gene panel is the intersection of symbols across specs (or that one study
    if specs has a single entry).

    If intersect_compounds is True and there are multiple specs, each study is
    subset to the shared compound names. With one spec, all non-empty labels
    are kept.
    """
    if not specs:
        raise ValueError("Need at least one DatasetSpec")

    loaded: list[tuple[DatasetSpec, Any, Any, pd.Series]] = []
    symbol_sets: list[pd.Index] = []
    for spec in specs:
        art, adata = open_backed(spec.uid_or_key)
        pert = adata.obs[spec.pert_col]
        loaded.append((spec, art, adata, pert))
        symbol_sets.append(gene_symbols(adata, spec.symbol_col))
        print(spec.source, art.key, adata.shape)

    if intersect_compounds and len(specs) > 1:
        allowed = set(overlap_compounds(*[pert for _, _, _, pert in loaded]))
        print("normalized compound overlap:", len(allowed))
    else:
        allowed = None

    gene_panel = symbol_sets[0]
    for symbols in symbol_sets[1:]:
        gene_panel = gene_panel.intersection(symbols)
    print("common genes:", len(gene_panel))
    if len(gene_panel) == 0:
        raise RuntimeError("No shared gene symbols across DatasetSpec entries")

    artifacts: list[ln.Artifact] = []
    kept_compounds: set[str] = set()
    try:
        for spec, _art, adata, _pert in loaded:
            harmonized = harmonize_anndata(
                adata,
                source=spec.source,
                pert_col=spec.pert_col,
                gene_panel=gene_panel,
                symbol_col=spec.symbol_col,
                allowed_compounds=allowed,
                max_obs=spec.max_obs,
                log1p=log1p,
            )
            kept_compounds.update(harmonized.obs["perturbation"].astype(str))
            artifacts.append(
                save_harmonized(
                    harmonized,
                    key=spec.output_key(prefix),
                    description=spec.description
                    or (f"{spec.source}: pert_col={spec.pert_col}, log1p={log1p}"),
                )
            )
    finally:
        for _spec, _art, adata, _pert in loaded:
            close_backed(adata)

    if overlap_key is not None:
        compounds = sorted(kept_compounds - {""})
        ln.Artifact.from_dataframe(
            pd.DataFrame({"perturbation": compounds}),
            key=overlap_key,
            description=f"Compounds in {collection_key} ({len(compounds)} names)",
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
    filter_to_overlap: bool = False,
    log1p: bool = True,
    prefix: str = PREFIX,
) -> ln.Collection:
    """Harmonize one more study onto an existing Collection and version it.

    Gene panel is taken from gene_panel_key if given, otherwise from the first
    artifact already in the collection.
    """
    collection = ln.Collection.get(key=collection_key)
    print("appending to", collection.uid, collection.key)

    if gene_panel_key is not None:
        gene_panel = load_gene_panel(gene_panel_key)
    else:
        gene_panel = _gene_panel_from_collection(collection)

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
