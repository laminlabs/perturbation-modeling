"""Load and materialize AnnData artifacts without assuming a local file path."""

from __future__ import annotations

from typing import Any

import anndata as ad
import lamindb as ln


def get_artifact(uid_or_key: str) -> ln.Artifact:
    """Resolve an artifact by uid, falling back to key."""
    try:
        return ln.Artifact.get(uid_or_key)
    except Exception:
        return ln.Artifact.get(key=uid_or_key)


def open_backed(uid_or_key: str) -> tuple[ln.Artifact, Any]:
    """Open an AnnData artifact in backed mode (or an AnnDataAccessor)."""
    art = get_artifact(uid_or_key)
    return art, art.open()


def open_study(uid_or_key: str) -> tuple[ln.Artifact, Any, Any | None]:
    """Open expression data plus the curated obs sidecar when present.

    Returns (expression_artifact, adata_or_accessor, obs_artifact_or_none).
    If uid_or_key points at a curated obs.parquet (PertSchema obs), the linked
    X.h5ad is opened for counts.
    """
    from .schema import related_obs_artifact, related_x_artifact

    art = get_artifact(uid_or_key)
    obs_art = related_obs_artifact(art)
    x_art = related_x_artifact(art)
    if x_art is None:
        raise RuntimeError(
            f"No AnnData artifact linked to {getattr(art, 'key', uid_or_key)!r}. "
            "Pass an .h5ad key or a curated obs.parquet with a linked X.h5ad."
        )
    obs_out = obs_art if obs_art is not None and obs_art is not x_art else None
    return x_art, x_art.open(), obs_out


def close_backed(adata: Any) -> None:
    """Close a backed AnnData / accessor if it still holds a file handle."""
    if getattr(adata, "isbacked", False) and getattr(adata, "file", None) is not None:
        adata.file.close()
    elif hasattr(adata, "close"):
        adata.close()


def to_memory_adata(adata: Any) -> ad.AnnData:
    """Materialize AnnData or a lamindb AnnDataAccessor into memory."""
    if isinstance(adata, ad.AnnData):
        return adata.to_memory() if adata.isbacked else adata.copy()
    if hasattr(adata, "to_memory"):
        return adata.to_memory()  # type: ignore[no-any-return]
    raise TypeError(f"Cannot materialize {type(adata)!r} to AnnData")
