#!/usr/bin/env python3
"""
Read the output of generate_synthetic_latents.py and show a UMAP projection.

Unlabelled records are drawn in gray, labelled records are colour-coded by class.
UMAP is fitted on unlabelled + labelled data together, so both share one embedding.

Requires: numpy, matplotlib, umap-learn
    pip install numpy matplotlib umap-learn

Usage:
    python plot_umap.py synthetic_latents.npz                 # show window
    python plot_umap.py synthetic_latents.npz --output umap.png  # save to file
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import umap

def embed(x_unlabelled: np.ndarray, x_labelled: np.ndarray, *,
          n_neighbors: int = 30, min_dist: float = 0.1,
          metric: str = "euclidean", seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fit one UMAP on all points; return 2-D coordinates for each set."""
    X = np.vstack([x_unlabelled, x_labelled])
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, metric=metric, random_state=seed)
    Z = reducer.fit_transform(X)
    n_u = len(x_unlabelled)
    return Z[:n_u], Z[n_u:]


def plot_umap(z_unlabelled: np.ndarray, z_labelled: np.ndarray, y_labelled: np.ndarray,
              *, title: str = "UMAP of latent space", ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Gray background cloud for unlabelled data, coloured points for labelled data."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(z_unlabelled[:, 0], z_unlabelled[:, 1],
               s=3, c="lightgray", alpha=0.5, linewidths=0,
               label=f"unlabelled (n={len(z_unlabelled):,})", rasterized=True)

    classes = np.unique(y_labelled)
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(classes):
        m = y_labelled == c
        ax.scatter(z_labelled[m, 0], z_labelled[m, 1],
                   s=22, color=cmap(i % 10), edgecolors="black", linewidths=0.4,
                   label=f"class {c} (n={m.sum()})")

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", markerscale=1.5, framealpha=0.9)

    plt.tight_layout()

    return ax

def make_plot(x_unlabelled: np.ndarray, x_labelled: np.ndarray, y_labelled: np.ndarray) -> plt.Axes:
    n_neighbors = 30
    min_dist = 0.1
    metric = "euclidean"
    seed = 0
    z_unlabelled, z_labelled = embed(x_unlabelled, x_labelled, n_neighbors=n_neighbors,
                     min_dist=min_dist, metric=metric, seed=seed)
    axes = plot_umap(z_unlabelled, z_labelled, y_labelled)

    return axes
