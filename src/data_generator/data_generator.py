"""
Synthetic data for testing semi-supervised clustering.

Model
-----
Every record is a point in a `latent_dim`-dimensional space:
    * the first d dims follow a Gaussian mixture, the remaining dims are N(0, I) noise
      (collapsed VAE dimensions);
    * unlabelled records are drawn from their own mixture;
    * labelled records of class c are drawn from a separate mixture belonging to class c.

All points, labelled or not, are made by the single function `generate_points`.
Points without a label carry the label UNLABELLED (-1), the same convention
that sklearn uses for semi-supervised estimators.

Usage
-----
    python main.py --out synthetic_latents.npz --seed 42
"""

from __future__ import annotations

import numpy as np

from .mixture_spec import Mixture, load_model
from .MixtureConfig import MixtureConfig

UNLABELLED = -1


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def generate_points(mixture: Mixture, n: int, rng: np.random.Generator,
                    label: int = UNLABELLED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw points from one mixture and tag them all with the same label.

    This is the only place where points are sampled, for labelled and unlabelled data
    alike. For each point a component is chosen according to the mixture weights, then
    the first d coordinates are drawn from that component's Gaussian. Any remaining
    ``latent_dim - d`` coordinates are independent N(0, 1) noise (collapsed VAE dimensions).

    Parameters
    ----------
    mixture : Mixture
        The distribution to example from.
    n : int
        Number of points to draw (0 is allowed).
    rng : numpy.random.Generator
        Source of randomness; advancing it makes successive calls independent.
    label : int, default UNLABELLED (-1)
        Label assigned to every point. Leave at the default for unlabelled points.

    Returns
    -------
    X : numpy.ndarray, shape (n, mixture.latent_dim)
        The sampled points.
    y : numpy.ndarray of int, shape (n,)
        `label`, repeated for every point (``-1`` if the points are unlabelled).
    z : numpy.ndarray of int, shape (n,)
        Index of the mixture component each point was drawn from (ground truth).
    """
    d = mixture.dimensionality
    z = rng.choice(mixture.n_components, size=n, p=mixture.weights)

    X = np.empty((n, mixture.latent_dim))
    X[:, d:] = rng.standard_normal((n, mixture.latent_dim - d))
    for k in range(mixture.n_components):
        idx = np.flatnonzero(z == k)
        if idx.size:
            X[idx, :d] = rng.multivariate_normal(
                mixture.means[k], mixture.covs[k], size=idx.size, method="cholesky")

    y = np.full(n, label, dtype=int)
    return X, y, z


def generate(cfg: MixtureConfig) -> dict:
    """
    Generate the unlabelled and labelled data sets described by a JSON model.

    The model file (``cfg.mixture_json``) is loaded, then

    1. ``cfg.n_unlabelled`` points are drawn from the unlabelled mixture, without label;
    2. the numbers of labelled points per class are drawn from the class proportions
       (multinomial), and each class's points are drawn from that class's own mixture
       and tagged with its integer label (its index in the file's ``classes`` list);
    3. the labelled points are shuffled so that they are not ordered by class.

    Parameters
    ----------
    cfg : MixtureConfig
        Needs the attributes ``mixture_json``, ``n_unlabelled``, ``n_labelled``,
        ``labelled_class_probs`` (None to use the proportions from the file; otherwise
        one non-negative weight per class, normalised here) and ``seed``.

    Returns
    -------
    dict
        Arrays ready for ``np.savez``:

        * ``x_unlabelled`` (n_unlabelled, latent_dim), ``x_labelled`` (n_labelled, latent_dim)
        * ``y_labelled`` integer class labels
        * ground truth, for evaluation only: ``y_unlabelled`` (all -1), ``z_unlabelled_true``
          and ``z_labelled_true`` (component index within the mixture the point came from;
          for labelled points it is relative to the class's mixture)
        * the model parameters, see `LatentModel.to_arrays`

    Raises
    ------
    ValueError
        If ``labelled_class_probs`` has a different length than the number of classes.
    FileNotFoundError, ValueError, KeyError
        If the model file is missing or invalid (see `load_model`).
    """
    rng = np.random.default_rng(cfg.seed)

    model = load_model(cfg.mixture_json)

    # Class proportions of the labelled set: the config overrides those in the JSON file.
    class_probs = model.class_probs
    if cfg.labelled_class_probs is not None:
        class_probs = np.asarray(cfg.labelled_class_probs, dtype=float)
        if len(class_probs) != model.n_classes:
            raise ValueError(f"labelled_class_probs has {len(class_probs)} entries, "
                             f"the model has {model.n_classes} classes")
        class_probs = class_probs / class_probs.sum()

    # ---- unlabelled set: its own mixture, label undefined
    X_u, y_u, z_u = generate_points(model.unlabelled, cfg.n_unlabelled, rng)

    # ---- labelled set: one mixture per class
    counts = rng.multinomial(cfg.n_labelled, class_probs)
    parts = [generate_points(mixture, n_c, rng, label=c)
             for c, (mixture, n_c) in enumerate(zip(model.class_mixtures, counts))]
    X_l, y_l, z_l = (np.concatenate(arrays) for arrays in zip(*parts))

    order = rng.permutation(len(y_l))  # don't leave the labelled set sorted by class
    X_l, y_l, z_l = X_l[order], y_l[order], z_l[order]

    return dict(
        x_unlabelled=X_u, x_labelled=X_l, y_labelled=y_l,
        # ground truth, for evaluation only:
        # z_* is the component index within the mixture the point was drawn from
        # (the unlabelled mixture for z_unlabelled_true, the class's mixture for z_labelled_true)
        y_unlabelled=y_u, z_unlabelled_true=z_u, z_labelled_true=z_l,
        **model.to_arrays(),
    )
