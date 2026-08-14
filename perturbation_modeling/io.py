"""Load and materialize AnnData artifacts without assuming a local file path."""

from __future__ import annotations

from typing import Any

import anndata as ad
import lamindb as ln


def get_artifact(uid_or_key: str) -> ln.Artifact:
    """Resolve an artifact by uid, falling back to ``key``."""
    try:
        return ln.Artifact.get(uid_or_key)
    except Exception:
        return ln.Artifact.get(key=uid_or_key)


def open_backed(uid_or_key: str) -> tuple[ln.Artifact, Any]:
    """Open an AnnData artifact in backed mode (or an AnnDataAccessor)."""
    art = get_artifact(uid_or_key)
    return art, art.open()


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
