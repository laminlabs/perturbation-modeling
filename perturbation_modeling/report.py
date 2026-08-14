"""Compose a biologist-facing report from enrichment / weight tables.

The library builds an *evidence* report from artifacts. Compound-specific
mechanistic notes belong in the tracked transform on the LaminDB instance
so they stay tied to a Run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def focus_perturbations(best_terms: pd.DataFrame, n: int = 8) -> list[str]:
    """Select the n perturbations with the strongest top Enrichr term."""
    ranked = best_terms.sort_values("Adjusted P-value").reset_index(drop=True)
    return ranked.head(n)["perturbation"].astype(str).tolist()


def train_line(summary: pd.DataFrame) -> str:
    s = summary.iloc[0]
    return (
        f"SimpleLogReg over a MappedCollection: {int(s['n_classes'])} classes, "
        f"{int(s['n_vars'])} genes, {int(s['n_obs']):,} cells, "
        f"{int(s['max_steps'])} training steps."
    )


def evidence_sections(
    *,
    summary: pd.DataFrame,
    best_terms: pd.DataFrame,
    top_genes: pd.DataFrame,
    focus: list[str],
    top_n_genes: int = 12,
) -> list[str]:
    """Markdown sections derived only from tables (no free-text biology)."""
    sections = [
        f"*{train_line(summary)} Focus = top {len(focus)} perturbations by "
        f"Enrichr adjusted p.*\n",
        "## Best Enrichr term per focus perturbation\n",
    ]
    best = best_terms.drop(
        columns=[c for c in best_terms.columns if str(c).startswith("Unnamed")]
    )
    for pert in focus:
        hit = best.loc[best["perturbation"] == pert]
        if hit.empty:
            continue
        r = hit.iloc[0]
        genes = (
            top_genes.loc[top_genes["perturbation"] == pert]
            .sort_values("rank")
            .head(top_n_genes)["gene"]
            .astype(str)
            .tolist()
        )
        adj = r["Adjusted P-value"]
        sections.append(
            f"### {pert} — *{r['Term']} ({r['Gene_set']})* — adjP = {adj:.1e}\n"
            f"Top genes: {', '.join(genes)}\n"
        )
    sections.append(
        """---

## Caveats

- Linear classifier weights are not a controlled DE contrast.
- Enrichment on high-weight genes is biased toward genes that help many classes.
- Landmark / shared gene panels are non-random; absent terms may be unmeasured.
- Retrain after appending a study; do not reuse weights from a previous collection version.
"""
    )
    return sections


def build_evidence_report(
    *,
    summary: pd.DataFrame,
    best_terms: pd.DataFrame,
    top_genes: pd.DataFrame,
    n_focus: int = 8,
    title: str = "Perturbation feature-selection interpretation",
    extra_sections: list[str] | None = None,
) -> str:
    """Assemble a markdown report; extra_sections is where an agent adds notes."""
    best = best_terms.drop(
        columns=[c for c in best_terms.columns if str(c).startswith("Unnamed")]
    )
    focus = focus_perturbations(best, n=n_focus)
    parts = [
        f"# {title}\n",
        *evidence_sections(
            summary=summary, best_terms=best, top_genes=top_genes, focus=focus
        ),
    ]
    if extra_sections:
        parts.extend(extra_sections)
    return "".join(parts)


def write_report(text: str, path: str | Path) -> Path:
    out = Path(path)
    out.write_text(text)
    return out
