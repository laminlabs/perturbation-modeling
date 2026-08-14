"""Compound-name normalization for cross-study perturbation labels."""

from __future__ import annotations

import re

import pandas as pd

_EMPTY = {"", "nan", "none", "null"}


def normalize_compound(name: object) -> str:
    """Lowercase, strip salt/form in parentheses, collapse whitespace.

    Used to join Tahoe ``drug``, LINCS ``pert_compound``, and in-house labels
    onto a shared ``perturbation`` column.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).strip().lower()
    if s in _EMPTY:
        return ""
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = s.replace("_tf", "").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()
