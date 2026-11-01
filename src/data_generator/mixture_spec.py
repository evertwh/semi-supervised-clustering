"""
Load / save a Gaussian mixture specification as JSON.

JSON format (see example_mixture.json)
--------------------------------------
{
  "latent_dim": 100,                 # total output dimensionality
  "classes": ["A", "B", "C", "D"],   # label names (order = column order of label_probs)
  "components": [
    {
      "name": "optional free text",
      "weight": 0.25,                # relative; normalised to sum to 1
      "mean": [ ... dimensionality numbers ... ], # dimensionality <= latent_dim
      "covariance": { ... },         # one of the types below
      "label_probs": [0.8, 0.1, 0.05, 0.05]   # P(class | component)
    },
    ...
  ]
}

All components must use the same dimensionality (= length of "mean"). If dimensionality < latent_dim,
the remaining latent_dim - dimensionality dimensions are filled with independent N(0, 1)
noise (e.g. collapsed VAE dimensions).

Covariance types ("type" field)
-------------------------------
scalar    {"type": "scalar",   "variance": 1.5}                      Sigma = v * I
diagonal  {"type": "diagonal", "variances": [ ... dimensionality ... ]}           Sigma = diag(v)
low_rank  {"type": "low_rank", "factor": [[...]] (dimensionality x r),
           "psi": [ ... dimensionality ... ]}                                     Sigma = F F^T + diag(psi)
std_corr  {"type": "std_corr", "std": [ ... dimensionality ... ],
           "correlation": [[...]] (dimensionality x dimensionality)}                           Sigma = D R D
full      {"type": "full",     "matrix": [[...]] (dimensionality x dimensionality)}            Sigma as given

Any of "matrix", "factor", "correlation" may instead be given as
"<name>_file": "path/to/file.npy" (relative to the JSON file), which keeps the JSON small
for large dimensions.
"""
import json
from enum import Enum
from pathlib import Path
from typing import Optional, Iterable

import numpy as np

from .MixtureConfig import MixtureConfig


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _read_array(c: dict, key: str, base_dir: Path, ndim: int, where: str) -> np.ndarray:
    """Read array `key` inline, or from `<key>_file` (.npy)."""
    if key in c:
        a = np.asarray(c[key], dtype=float)
    elif f"{key}_file" in c:
        a = np.load(base_dir / c[f"{key}_file"]).astype(float)
    else:
        raise ValueError(f"{where}: need '{key}' or '{key}_file'")
    if a.ndim != ndim:
        raise ValueError(f"{where}: '{key}' must be {ndim}-dimensional, got shape {a.shape}")
    return a


class CovarianceType(Enum):
    SCALAR = 'scalar'
    DIAGONAL = 'diagonal'
    LOW_RANK = 'low_rank'
    STD_CORR = 'std_corr'
    FULL = 'full'


def read_scalar_covariance(specification: dict, dimensionality: int, base_dir: Path, where: str) -> np.ndarray:
    return float(specification["variance"]) * np.eye(dimensionality)


def read_diagonal_covariance(specification: dict, dimensionality: int, base_dir: Path, where: str) -> np.ndarray:
    return np.diag(_read_array(specification, "variances", base_dir, 1, where))


def read_low_rank_covariance(specification: dict, dimensionality: int, base_dir: Path, where: str) -> np.ndarray:
    factor = _read_array(specification, "factor", base_dir, 2, where)
    psi = _read_array(specification, "psi", base_dir, 1, where)
    if factor.shape[0] != dimensionality:
        raise ValueError(f"{where}: factor must have {dimensionality} rows, got {factor.shape[0]}")
    return factor @ factor.T + np.diag(psi)


def read_std_corr_covariance(specification: dict, dimensionality: int, base_dir: Path, where: str) -> np.ndarray:
    std = _read_array(specification, "std", base_dir, 1, where)
    correlations = _read_array(specification, "correlation", base_dir, 2, where)
    return correlations * np.outer(std, std)


def read_full_covariance(specification: dict, dimensionality: int, base_dir: Path, where: str) -> np.ndarray:
    return _read_array(specification, "matrix", base_dir, 2, where)


COVARIANCE_BUILDERS = {
    CovarianceType.SCALAR: read_scalar_covariance,
    CovarianceType.DIAGONAL: read_diagonal_covariance,
    CovarianceType.LOW_RANK: read_low_rank_covariance,
    CovarianceType.STD_CORR: read_std_corr_covariance,
    CovarianceType.FULL: read_full_covariance
}


def covariance_from_spec(specification: dict, dimensionality: int, base_dir: Path, where: str) -> np.ndarray:
    """Turn one covariance spec into a (dimensionality, dimensionality) matrix and check it is a valid covariance."""
    covariance_type = CovarianceType(specification.get("type"))
    reader = COVARIANCE_BUILDERS[covariance_type]

    covariance = reader(specification, dimensionality, base_dir, where)

    if covariance.shape != (dimensionality, dimensionality):
        raise ValueError(
            f"{where}: covariance has shape {covariance.shape}, expected ({dimensionality}, {dimensionality})")
    if not np.allclose(covariance, covariance.T, atol=1e-8 * max(1.0, np.abs(covariance).max())):
        raise ValueError(f"{where}: covariance is not symmetric")

    covariance = (covariance + covariance.T) / 2
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        raise ValueError(f"{where}: covariance is not positive definite "
                         f"(min eigenvalue {np.linalg.eigvalsh(covariance).min():.3g})") from None
    return covariance


def parse_mixture(spec: dict, base_dir: str | Path = ".") -> dict:
    """Validate a spec dict and return the arrays used by the generator."""
    base_dir = Path(base_dir)
    comps: Iterable = spec.get("components")
    if not comps:
        raise ValueError("'components' must be a non-empty list")
    latent_dim = int(spec["latent_dim"])

    means = np.array([np.asarray(c["mean"], dtype=float) for c in comps])
    if means.ndim != 2:
        raise ValueError("all component means must have the same length")
    K, d = means.shape
    if d > latent_dim:
        raise ValueError(f"mean length {d} exceeds latent_dim {latent_dim}")

    weights = np.array([float(c["weight"]) for c in comps])
    if np.any(weights <= 0):
        raise ValueError("component weights must be > 0")
    weights /= weights.sum()

    covs = np.array([covariance_from_spec(c["covariance"], d, base_dir,
                                          where=f"component {k} ({c.get('name', '')})")
                     for k, c in enumerate(comps)])

    classes = spec.get("classes")
    try:
        label_probs = np.array([c["label_probs"] for c in comps], dtype=float)
    except KeyError:
        raise ValueError("every component needs 'label_probs'") from None
    if label_probs.ndim != 2:
        raise ValueError("all label_probs must have the same length")
    if classes is not None and label_probs.shape[1] != len(classes):
        raise ValueError(f"label_probs length {label_probs.shape[1]} != number of classes {len(classes)}")
    if np.any(label_probs < 0):
        raise ValueError("label_probs must be non-negative")
    row_sums = label_probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(f"label_probs rows must sum to 1, got {np.round(row_sums, 4)}")
    label_probs = label_probs / row_sums[:, None]  # remove rounding error

    return dict(latent_dim=latent_dim, weights=weights, means=means, covs=covs,
                label_given_component=label_probs, classes=classes)


def load_mixture(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        return parse_mixture(json.load(f), base_dir=path.parent)


def generate_mixture(cfg: MixtureConfig, rng: np.random.Generator) -> dict:
    """Generate a random mixture."""
    K, d, C = cfg.n_components, cfg.n_informative, cfg.n_classes

    weights = rng.dirichlet(np.full(K, cfg.weight_concentration))
    means = rng.normal(0.0, cfg.mean_separation, size=(K, d))

    covs = np.empty((K, d, d))
    for k in range(K):
        A = rng.normal(0.0, cfg.cov_scale, size=(d, cfg.cov_rank))
        psi = rng.uniform(*cfg.psi_range, size=d)
        covs[k] = A @ A.T + np.diag(psi)

    # P(y | z): each component has a dominant class; the rest is spread randomly.
    label_given_component = np.empty((K, C))
    dominant = np.arange(K) % C  # make sure every class dominates >= 1 component
    rng.shuffle(dominant)
    for k in range(K):
        rest = rng.dirichlet(np.ones(C - 1)) * (1.0 - cfg.class_purity)
        row = np.empty(C)
        row[dominant[k]] = cfg.class_purity
        row[np.arange(C) != dominant[k]] = rest
        label_given_component[k] = row

    mixture = dict(weights=weights, means=means, covs=covs,
                   label_given_component=label_given_component)

    if cfg.standardise:
        # Aggregate moments of the mixture
        mu = weights @ means
        second = sum(w * (S + np.outer(m, m)) for w, S, m in zip(weights, covs, means))
        total_cov = second - np.outer(mu, mu)
        L = np.linalg.cholesky(total_cov)
        W = np.linalg.inv(L)  # whitening: x' = W (x - mu)
        mixture["means"] = (means - mu) @ W.T
        mixture["covs"] = np.einsum("ij,kjl,ml->kim", W, covs, W)  # W S W^T
    return mixture


# --------------------------------------------------------------------------- #
# Writing (e.g. from a GaussianMixture fitted on the real data)
# --------------------------------------------------------------------------- #
def export_sklearn_gmm(gmm, path: str | Path, *, latent_dim: Optional[int] = None,
                       label_probs: Optional[np.ndarray] = None,
                       classes: Optional[list] = None, decimals: int = 6) -> None:
    """
    Write a fitted sklearn GaussianMixture to the JSON format.

    label_probs : (K, C) array of P(class | component), e.g. the mean of
                  gmm.predict_proba(X_labelled) per class, normalised per component.
                  If omitted, every component gets a uniform distribution over
                  `classes` (you then need to edit it by hand).
    """
    K, d = gmm.means_.shape
    if label_probs is None:
        if classes is None:
            raise ValueError("give label_probs or classes")
        label_probs = np.full((K, len(classes)), 1.0 / len(classes))
    out = dict(latent_dim=latent_dim or d,
               classes=list(classes) if classes is not None else None,
               components=[])

    for k in range(K):
        cov = gmm.covariances_[k]
        if gmm.covariance_type == "full":
            cspec = {"type": "full", "matrix": np.round(cov, decimals).tolist()}
        elif gmm.covariance_type == "diag":
            cspec = {"type": "diagonal", "variances": np.round(cov, decimals).tolist()}
        elif gmm.covariance_type == "spherical":
            cspec = {"type": "scalar", "variance": round(float(cov), decimals)}
        else:  # tied
            cspec = {"type": "full", "matrix": np.round(gmm.covariances_, decimals).tolist()}

        out["components"].append(dict(
            name=f"component_{k}", weight=round(float(gmm.weights_[k]), decimals),
            mean=np.round(gmm.means_[k], decimals).tolist(),
            covariance=cspec, label_probs=np.round(label_probs[k], decimals).tolist()))
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
