"""Read perturbation labels from an artifact schema (PertSchema obs)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

CURATED_PERT_COL = "pert_compound"
FALLBACK_PERT_COL = "drug"


def _member_names(schema: Any) -> set[str]:
    if schema is None:
        return set()
    members = getattr(schema, "members", None)
    if members is None:
        members = getattr(schema, "features", None)
    if members is None:
        return set()
    names: set[str] = set()
    values_list = getattr(members, "values_list", None)
    if callable(values_list):
        names = set(values_list("name", flat=True))
        if names:
            return names
    for member in members:
        if isinstance(member, str):
            names.add(member)
            continue
        name = getattr(member, "name", None)
        if name:
            names.add(str(name))
    return names


def schema_feature_names(artifact: Any) -> set[str]:
    """Feature names on artifact.schema, or empty if the artifact has no schema."""
    schema = getattr(artifact, "schema", None)
    names = _member_names(schema)
    slots = getattr(schema, "slots", None) if schema is not None else None
    if isinstance(slots, dict):
        for slot_schema in slots.values():
            names |= _member_names(slot_schema)
    return names


def schema_has_feature(artifact: Any, name: str) -> bool:
    return name in schema_feature_names(artifact)


def sibling_key(key: str, *, from_name: str, to_name: str) -> str | None:
    """Replace a trailing component (obs.parquet <-> X.h5ad) on an artifact key."""
    key = str(key)
    if key.endswith(from_name):
        return key[: -len(from_name)] + to_name
    return None


def _linked_artifact(value: Any) -> Any | None:
    """FeatureManager[slot] is an Artifact only when it has key + load/open."""
    if value is None or value == {} or value == []:
        return None
    if hasattr(value, "key") and (hasattr(value, "open") or hasattr(value, "load")):
        return value
    return None


def _features_get(artifact: Any, slot: str) -> Any | None:
    features = getattr(artifact, "features", None)
    if features is None:
        return None
    try:
        return _linked_artifact(features[slot])
    except Exception:
        return None


def _get_by_key(key: str) -> Any | None:
    from .io import get_artifact

    try:
        return get_artifact(key)
    except Exception:
        return None


def related_obs_artifact(artifact: Any) -> Any | None:
    """obs.parquet linked to this artifact, if any.

    Curated Tahoe on laminlabs/pertdata stores PertSchema obs on the parquet
    sidecar; X.h5ad itself has schema=None.
    """
    key = getattr(artifact, "key", None) or ""
    suffix = getattr(artifact, "suffix", None) or ""
    if suffix == ".parquet" or str(key).endswith("obs.parquet"):
        return artifact
    obs = _features_get(artifact, "obs")
    if obs is not None:
        return obs
    sibling = sibling_key(str(key), from_name="X.h5ad", to_name="obs.parquet")
    if sibling is not None:
        return _get_by_key(sibling)
    return None


def related_x_artifact(artifact: Any) -> Any | None:
    """AnnData X artifact linked from a curated obs.parquet."""
    key = getattr(artifact, "key", None) or ""
    suffix = getattr(artifact, "suffix", None) or ""
    if suffix == ".h5ad" or str(key).endswith(".h5ad"):
        return artifact
    x = _features_get(artifact, "X")
    if x is not None:
        return x
    sibling = sibling_key(str(key), from_name="obs.parquet", to_name="X.h5ad")
    if sibling is not None:
        return _get_by_key(sibling)
    return None


def resolve_pert_col(
    artifact: Any,
    requested: str | None = None,
    *,
    default: str = FALLBACK_PERT_COL,
) -> str:
    """Choose the perturbation column from schema, else the requested/default name.

    If artifact.schema (or the linked obs.parquet schema) includes
    pert_compound, that column is used instead of drug. An explicit requested
    column other than drug is left unchanged.
    """
    want_curated = requested is None or requested == default
    if want_curated and schema_has_feature(artifact, CURATED_PERT_COL):
        return CURATED_PERT_COL
    if want_curated:
        obs_art = related_obs_artifact(artifact)
        if (
            obs_art is not None
            and obs_art is not artifact
            and schema_has_feature(obs_art, CURATED_PERT_COL)
        ):
            return CURATED_PERT_COL
    return requested or default


def load_pert_series(
    adata: Any,
    pert_col: str,
    obs_artifact: Any | None = None,
) -> pd.Series:
    """Perturbation labels from adata.obs, or from a curated obs.parquet."""
    if pert_col in adata.obs.columns:
        return adata.obs[pert_col]
    if obs_artifact is not None:
        loaded = obs_artifact.load()
        frame = loaded.obs if hasattr(loaded, "obs") else loaded
        columns = getattr(frame, "columns", [])
        if pert_col in columns:
            return frame[pert_col]
    extra = ""
    if obs_artifact is not None:
        extra = f" or curated obs {getattr(obs_artifact, 'key', obs_artifact)!r}"
    raise KeyError(f"{pert_col!r} not in obs columns {list(adata.obs.columns)}{extra}")
