"""Train a linear feature-selection model on a MappedCollection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd

from .keys import LABEL_COL, SOURCE_COL

if TYPE_CHECKING:
    import lamindb as ln
    import torch


def collate_mapped(
    batch: list[dict[str, Any]],
    *,
    label_col: str = LABEL_COL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MappedCollection yields dicts; SimpleLogReg.training_step expects (x, y)."""
    import torch

    xs = [torch.as_tensor(sample["X"], dtype=torch.float32) for sample in batch]
    ys = [int(sample[label_col]) for sample in batch]
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def weights_dataframe(
    logreg: Any,
    dataset: Any,
    *,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Per-class gene weights from SimpleLogReg, without requiring AnnData.fit()."""
    weight = logreg.linear.weight.detach().cpu().numpy()
    encoder = dataset.encoders[label_col]
    classes = [cat for cat, _ in sorted(encoder.items(), key=lambda kv: kv[1])]
    df = pd.DataFrame(weight, index=classes, columns=dataset.var_joint)
    df.attrs["method_name"] = "modlyn_logreg"
    return df


def train_feature_selection(
    collection: ln.Collection,
    *,
    label_col: str = LABEL_COL,
    obs_keys: list[str] | None = None,
    batch_size: int = 256,
    max_steps: int = 50,
    max_epochs: int = 5,
    learning_rate: float = 1e-1,
    weight_decay: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit modlyn SimpleLogReg on collection.mapped().

    Returns (weights, summary). weights is classes × genes; summary
    is a one-row frame of training metadata.
    """
    import lightning as L
    import modlyn as mn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    obs_keys = obs_keys or [label_col, SOURCE_COL]

    with collection.mapped(
        obs_keys=obs_keys,
        join="inner",
        encode_labels=[label_col],
    ) as dataset:
        print("#observations", dataset.shape[0])
        print("#variables", len(dataset.var_joint))

        n_classes = len(dataset.encoders[label_col])
        n_genes = len(dataset.var_joint)
        classes = [
            cat
            for cat, _ in sorted(
                dataset.encoders[label_col].items(), key=lambda kv: kv[1]
            )
        ]
        stub = ad.AnnData(
            X=np.zeros((n_classes, n_genes), dtype=np.float32),
            obs=pd.DataFrame({label_col: classes}),
            var=pd.DataFrame(index=dataset.var_joint),
        )
        logreg = mn.models.SimpleLogReg(
            adata=stub,
            label_column=label_col,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        sampler = WeightedRandomSampler(
            weights=dataset.get_label_weights(label_col),
            num_samples=len(dataset),
        )
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=lambda batch: collate_mapped(batch, label_col=label_col),
        )
        trainer = L.Trainer(
            max_epochs=max_epochs,
            max_steps=max_steps,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
        )
        trainer.fit(model=logreg, train_dataloaders=train_loader)

        weights = weights_dataframe(logreg, dataset, label_col=label_col)
        summary = pd.DataFrame(
            {
                "n_obs": [dataset.shape[0]],
                "n_vars": [n_genes],
                "n_classes": [n_classes],
                "max_steps": [max_steps],
                "label_column": [label_col],
                "loader": ["MappedCollection"],
            }
        )
        return weights, summary
