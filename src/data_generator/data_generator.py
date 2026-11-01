"""
Synthetic data for testing semi-supervised clustering.

Mimics the latent space of a VAE:
  * ~30,000 unlabelled records
  * ~600 labelled records (4 classes)

Model
-----
Informative block (n_informative dims):
    z ~ Categorical(weights)                       (K mixture components)
    x_inf | z ~ N(mu_z, Sigma_z),  Sigma_z = A_z A_z^T + diag(psi_z)   (low rank + diag)
    The mixture is then standardised analytically so the aggregate
    distribution has mean 0 and covariance I (what the KL term pushes towards).
Inactive block (latent_dim - n_informative dims):
    x_noise ~ N(0, I)                              (collapsed / prior-like dimensions)

Labels:
    y | z ~ Categorical(label_given_component[z])  (many-to-many with components)
    Unlabelled set: y is generated but hidden (saved only as ground truth).
    Labelled set:   y ~ labelled_class_probs, then z | y ∝ weights[z] * P(y|z), then x.
                    (Leave labelled_class_probs=None for no selection bias.)

Usage
-----
    python generate_synthetic_latents.py --output synthetic_latents.npz --seed 42
"""

import numpy as np

from .MixtureConfig import MixtureConfig
from .mixture_spec import load_mixture, generate_mixture


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_given_components(z: np.ndarray, mixture: dict, cfg: MixtureConfig,
                            rng: np.random.Generator) -> np.ndarray:
    n = len(z)
    X = np.empty((n, cfg.latent_dim))
    X[:, cfg.n_informative:] = rng.standard_normal((n, cfg.latent_dim - cfg.n_informative))
    for k in range(cfg.n_components):
        idx = np.where(z == k)[0]
        if len(idx):
            X[idx, :cfg.n_informative] = rng.multivariate_normal(
                mixture["means"][k], mixture["covs"][k], size=len(idx), method="cholesky")
    return X


def generate(cfg: MixtureConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)

    mixture = load_mixture(cfg.mixture_json) if cfg.mixture_json else generate_mixture(cfg, rng)

    w, Pyz = mixture["weights"], mixture["label_given_component"]
    K, C = cfg.n_components, cfg.n_classes

    # ---- unlabelled set (labels exist only as hidden ground truth)
    z_u = rng.choice(K, size=cfg.n_unlabelled, p=w)
    y_u = np.array([rng.choice(C, p=Pyz[k]) for k in z_u])
    X_u = sample_given_components(z_u, mixture, cfg, rng)

    # ---- labelled set
    joint = w[:, None] * Pyz  # P(z, y)
    class_marginal = joint.sum(axis=0)  # P(y)
    class_probs = (np.asarray(cfg.labelled_class_probs, dtype=float)
                   if cfg.labelled_class_probs is not None else class_marginal)
    class_probs = class_probs / class_probs.sum()
    y_l = rng.choice(C, size=cfg.n_labelled, p=class_probs)
    z_l = np.empty(cfg.n_labelled, dtype=int)
    for c in range(C):
        idx = np.where(y_l == c)[0]
        pz = joint[:, c] / joint[:, c].sum()  # P(z | y=specification)
        z_l[idx] = rng.choice(K, size=len(idx), p=pz)
    X_l = sample_given_components(z_l, mixture, cfg, rng)

    return dict(
        X_unlabelled=X_u, X_labelled=X_l, y_labelled=y_l,
        # ground truth, for evaluation only
        z_unlabelled_true=z_u, y_unlabelled_true=y_u, z_labelled_true=z_l,
        mixture_weights=w, mixture_means=mixture["means"], mixture_covs=mixture["covs"],
        label_given_component=Pyz,
    )
