"""Build or append a perturbation Collection from any set of studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lamindb as ln
import pandas as pd

from .compounds import normalize_compound
from .harmonize import gene_symbols, harmonize_anndata
from .io import close_backed, open_study
from .keys import (
    LABEL_COL,
    LINCS_UIDS,
    PREFIX,
    TAHOE_TEST_UID,
    output_keys,
)
from .schema import load_pert_series, resolve_pert_col


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
        prefix = prefix.strip().strip("/")
        return f"{prefix}/{self.source}_harmonized.h5ad"


def tahoe_spec(
    *,
    uid_or_key: str = TAHOE_TEST_UID,
    pert_col: str = "drug",
    max_obs: int | None = 200000,
    artifact_key: str | None = None,
    description: str | None = None,
) -> DatasetSpec:
    """DatasetSpec for a Tahoe AnnData artifact.

    Default uid_or_key is the small shard_0.h5ad used in tests
    (TAHOE_TEST_UID / TAHOE_TEST_KEY), which has no PertSchema and uses drug.
    Curated plates on laminlabs/pertdata (PertSchema obs) expose pert_compound;
    build_collection / append_dataset switch to that column when the artifact
    schema includes it. For production plates pass a key from
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


def overlap_compounds(*series: pd.Series, how: str = "first") -> list[str]:
    """Normalized compound overlap, dropping empties.

    how="first" (default): first series ∩ union of the rest. That is the
    original Tahoe ∩ (all LINCS phases) join — a name only has to appear in
    one LINCS file, not every phase.
    how="all": intersection of every series (often just DMSO across phases).
    """
    sets = [set(s.map(normalize_compound)) - {""} for s in series]
    if not sets:
        return []
    if how == "all":
        return sorted(set.intersection(*sets))
    if how != "first":
        raise ValueError(f"how must be 'first' or 'all', got {how!r}")
    if len(sets) == 1:
        return sorted(sets[0])
    return sorted(sets[0] & set.union(*sets[1:]))


def load_gene_panel(key: str) -> pd.Index:
    """Gene symbols from a previously saved harmonized AnnData artifact."""
    art = ln.Artifact.filter(key=key, is_latest=True).one_or_none()
    if art is None:
        raise RuntimeError(f"Missing gene-panel artifact {key}")
    return art.load().var_names.copy()


def _open_spec(spec: DatasetSpec) -> tuple[Any, Any, str, pd.Series]:
    """Open expression data and resolve the perturbation column from schema."""
    x_art, adata, obs_art = open_study(spec.uid_or_key)
    pert_col = resolve_pert_col(obs_art or x_art, spec.pert_col)
    if pert_col != spec.pert_col:
        print(f"{spec.source}: using {pert_col} from schema instead of {spec.pert_col}")
    pert = load_pert_series(adata, pert_col, obs_art)
    return x_art, adata, pert_col, pert


def load_overlap_compounds(key: str | None = None) -> set[str] | None:
    key = key or output_keys().overlap
    art = ln.Artifact.filter(key=key, is_latest=True).one_or_none()
    if art is None:
        return None
    df = art.load()
    col = LABEL_COL if LABEL_COL in df.columns else df.columns[0]
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
    prefix: str = PREFIX,
    collection_key: str | None = None,
    overlap_key: str | None = None,
    save_overlap: bool = True,
    log1p: bool = True,
    intersect_compounds: bool = True,
    collection_description: str | None = None,
) -> ln.Collection:
    """Harmonize one or more studies and save them as a Collection.

    Pass any mix of DatasetSpec values — Tahoe, LINCS, in-house, or all of them.
    Gene panel is the intersection of symbols across specs (or that one study
    if specs has a single entry).

    Output keys default to ``output_keys(prefix)`` (Collection, compound table,
    and each study's ``{prefix}/{source}_harmonized.h5ad``). Pass collection_key
    / overlap_key to override; ``save_overlap=False`` skips the compound table.

    If intersect_compounds is True and there are multiple specs, each study is
    subset to names in the first spec that also appear in at least one other
    spec (Tahoe ∩ union of LINCS, not the intersection of every phase).
    """
    if not specs:
        raise ValueError("Need at least one DatasetSpec")
    keys = output_keys(prefix)
    collection_key = collection_key or keys.collection
    if save_overlap:
        overlap_key = overlap_key or keys.overlap
    else:
        overlap_key = None

    loaded: list[tuple[DatasetSpec, Any, Any, str, pd.Series]] = []
    symbol_sets: list[pd.Index] = []
    for spec in specs:
        art, adata, pert_col, pert = _open_spec(spec)
        loaded.append((spec, art, adata, pert_col, pert))
        symbol_sets.append(gene_symbols(adata, spec.symbol_col))
        print(spec.source, art.key, adata.shape)

    if intersect_compounds and len(specs) > 1:
        allowed = set(overlap_compounds(*[pert for _, _, _, _, pert in loaded]))
        print(
            "normalized compound overlap "
            f"({loaded[0][0].source} ∩ others): {len(allowed)}"
        )
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
        for spec, _art, adata, pert_col, pert in loaded:
            harmonized = harmonize_anndata(
                adata,
                source=spec.source,
                pert_col=pert_col,
                gene_panel=gene_panel,
                symbol_col=spec.symbol_col,
                allowed_compounds=allowed,
                max_obs=spec.max_obs,
                log1p=log1p,
                pert=pert,
            )
            kept_compounds.update(harmonized.obs[LABEL_COL].astype(str))
            artifacts.append(
                save_harmonized(
                    harmonized,
                    key=spec.output_key(keys.prefix),
                    description=spec.description
                    or (f"{spec.source}: pert_col={pert_col}, log1p={log1p}"),
                )
            )
    finally:
        for _spec, _art, adata, _pert_col, _pert in loaded:
            close_backed(adata)

    if overlap_key is not None:
        compounds = sorted(kept_compounds - {""})
        ln.Artifact.from_dataframe(
            pd.DataFrame({LABEL_COL: compounds}),
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
    prefix: str = PREFIX,
    collection_key: str | None = None,
    gene_panel_key: str | None = None,
    overlap_key: str | None = None,
    filter_to_overlap: bool = False,
    log1p: bool = True,
) -> ln.Collection:
    """Harmonize one more study onto an existing Collection and version it.

    Collection / overlap / output artifact keys default to ``output_keys(prefix)``.
    Gene panel is taken from gene_panel_key if given, otherwise from the first
    artifact already in the collection.
    """
    keys = output_keys(prefix)
    collection_key = collection_key or keys.collection
    overlap_key = overlap_key or keys.overlap
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

    src_art, src_adata, pert_col, pert = _open_spec(spec)
    print("source artifact", src_art.uid, src_art.key, src_adata.shape)
    try:
        harmonized = harmonize_anndata(
            src_adata,
            source=spec.source,
            pert_col=pert_col,
            gene_panel=gene_panel,
            symbol_col=spec.symbol_col,
            allowed_compounds=allowed,
            max_obs=spec.max_obs,
            log1p=log1p,
            pert=pert,
        )
    finally:
        close_backed(src_adata)

    new_art = save_harmonized(
        harmonized,
        key=spec.output_key(keys.prefix),
        description=spec.description
        or (
            f"{spec.source} appended to {collection_key}: "
            f"pert_col={pert_col}, log1p={log1p}"
        ),
    )
    collection_new = collection.append(new_art).save()
    print("collection new version", collection_new.uid, collection_new.key)
    return collection_new
