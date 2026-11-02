"""
Gaussian-mixture model of the latent space, to be loaded from a JSON specification.

The model consists of
  * one mixture for the UNLABELLED records, and
  * one separate mixture for EACH CLASS LABEL (the distribution of x given the class).

Building blocks, from small to large:
    Component    one Gaussian: weight, mean, covariance
    Mixture      a set of Components stored as arrays, plus the latent dimensionality
    LatentModel  the unlabelled Mixture plus one Mixture per class label

JSON format (see example_mixture.json)
--------------------------------------
{
  "latent_dim": 100,                          # total output dimensionality
  "unlabelled": {
    "components": [ <component>, ... ]
  },
  "classes": [                                # list order = integer label 0, 1, 2, ...
    {"name": "A", "proportion": 0.4,          # proportion is optional (all or none)
     "components": [ <component>, ... ]},
    {"name": "B", "proportion": 0.6,
     "components": [ <component>, ... ]}
  ]
}

<component> = {
  "name": "optional free text",
  "weight": 0.25,                             # relative within its mixture; normalised
  "mean": [ ... d numbers ... ],              # d <= latent_dim
  "covariance": { ... }                       # one of the types below
}

Within one mixture all means must have the same length d. If d < latent_dim, the remaining
latent_dim - d dimensions are filled with independent N(0, 1) noise (collapsed VAE dimensions).
"proportion" is the share of each class in the labelled set.

Covariance types ("type" field)
-------------------------------
scalar    {"type": "scalar",   "variance": 1.5}                          Sigma = v * I
diagonal  {"type": "diagonal", "variances": [ ... d ... ]}               Sigma = diag(v)
low_rank  {"type": "low_rank", "factor": [[...]] (d x r),
           "psi": [ ... d ... ]}                                         Sigma = F F^T + diag(psi)
std_corr  {"type": "std_corr", "std": [ ... d ... ],
           "correlation": [[...]] (d x d)}                               Sigma = D R D
full      {"type": "full",     "matrix": [[...]] (d x d)}                Sigma as given

Any of "matrix", "factor", "correlation" may instead be given as
"<name>_file": "path/to/file.npy" (relative to the JSON file).
"""
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Any

import numpy as np


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class Component:
    """
    One Gaussian component of a mixture.

    A component is valid by construction: the checks below run when it is created,
    so any `Component` has a finite positive weight and consistent shapes.
    (Whether `cov` is a valid covariance matrix is checked when it is read from a
    specification, see `covariance_from_spec`.)

    Attributes
    ----------
    weight : float
        Relative weight of the component within its mixture. It must be finite and
        strictly positive; it is normalised only when the component is combined with
        others in `Mixture.from_components`.
    mean : numpy.ndarray
        Mean vector, shape (d,).
    cov : numpy.ndarray
        Covariance matrix, shape (d, d).
    name : str
        Optional free-text label, used only in messages.

    Raises
    ------
    ValueError
        If the weight is not finite and positive, the mean is not a non-empty finite
        vector, or the covariance shape does not match the mean.
    """
    weight: float
    mean: np.ndarray
    cov: np.ndarray
    name: str = ""

    def __post_init__(self):
        """Validate the weight and the shapes of `mean` and `cov`."""
        if not (np.isfinite(self.weight) and self.weight > 0):
            raise ValueError(f"component weight must be finite and > 0, got {self.weight}")
        if self.mean.ndim != 1 or self.mean.size == 0:
            raise ValueError(f"component mean must be a non-empty vector, got shape {self.mean.shape}")
        if not np.all(np.isfinite(self.mean)):
            raise ValueError("component mean must be finite")
        d = self.mean.shape[0]
        if self.cov.shape != (d, d):
            raise ValueError(f"covariance has shape {self.cov.shape}, expected ({d}, {d})")

    @property
    def dimensionality(self) -> int:
        """Number of latent dimensions d the component covers (length of the mean)."""
        return self.mean.shape[0]


@dataclass(frozen=True, eq=False)
class Mixture:
    """
    A Gaussian mixture over the first d latent dimensions.

    The remaining `latent_dim - d` dimensions are not modelled; the sampler fills them
    with independent N(0, 1) noise (collapsed VAE dimensions). Components are stored
    as stacked arrays, which is the form the sampler needs. Use `Mixture.from_components`
    to build one from `Component` objects.

    Attributes
    ----------
    weights : numpy.ndarray
        Mixing weights, shape (K,), positive and summing to 1.
    means : numpy.ndarray
        Component means, shape (K, d).
    covs : numpy.ndarray
        Component covariance matrices, shape (K, d, d).
    latent_dim : int
        Total dimensionality of the generated points (d <= latent_dim).
    """
    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    latent_dim: int

    @property
    def n_components(self) -> int:
        """Number of mixture components K."""
        return len(self.weights)

    @property
    def dimensionality(self) -> int:
        """Number of modelled dimensions d (length of each mean vector)."""
        return self.means.shape[1]

    @classmethod
    def from_components(cls, components: Sequence[Component], latent_dim: int,
                        where: str = "mixture") -> "Mixture":
        """
        Combine components into a mixture.

        Parameters
        ----------
        components : sequence of Component
            The components; all must have the same dimensionality.
        latent_dim : int
            Total dimensionality of the generated points. Must be at least the
            dimensionality of the components.
        where : str
            Label for error messages, e.g. ``"class 'A'"``.

        Returns
        -------
        Mixture
            With weights normalised to sum to 1 and arrays stacked in the given order.

        Raises
        ------
        ValueError
            If there are no components, their dimensionalities differ, or the
            dimensionality exceeds `latent_dim`.
        """
        if not components:
            raise ValueError(f"{where}: 'components' must be a non-empty list")
        lengths = {c.dimensionality for c in components}
        if len(lengths) != 1:
            raise ValueError(f"{where}: all component means must have the same length, got {sorted(lengths)}")
        d = lengths.pop()
        if d > latent_dim:
            raise ValueError(f"{where}: mean length {d} exceeds latent_dim {latent_dim}")

        weights = np.array([c.weight for c in components], dtype=float)
        return cls(weights=weights / weights.sum(),
                   means=np.stack([c.mean for c in components]),
                   covs=np.stack([c.cov for c in components]),
                   latent_dim=latent_dim)


@dataclass(frozen=True, eq=False)
class LatentModel:
    """
    The complete generative model: one mixture for unlabelled data, one per class.

    The integer label of a class is its index in `class_mixtures`. The mixtures are
    independent; there is no shared component structure between them.

    Attributes
    ----------
    unlabelled : Mixture
        Distribution of the unlabelled records.
    class_mixtures : list of Mixture
        ``class_mixtures[c]`` is the distribution of records with label ``c``.
    class_names : list of str
        Human-readable class names, same order as `class_mixtures`; must be unique.
    class_probs : numpy.ndarray
        Proportion of each class in the labelled set, shape (C,).

    Raises
    ------
    ValueError
        On creation, if the lengths of `class_mixtures`, `class_names` and `class_probs`
        differ, the class names are not unique, or the mixtures disagree on `latent_dim`.
    """
    unlabelled: Mixture
    class_mixtures: list[Mixture]
    class_names: list[str]
    class_probs: np.ndarray

    def __post_init__(self):
        """Check that the parts of the model are consistent with each other."""
        if not (len(self.class_mixtures) == len(self.class_names) == len(self.class_probs)):
            raise ValueError("class_mixtures, class_names and class_probs must have equal length")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class names must be unique")
        if any(m.latent_dim != self.unlabelled.latent_dim for m in self.class_mixtures):
            raise ValueError("all mixtures must have the same latent_dim")

    @property
    def latent_dim(self) -> int:
        """Total dimensionality of the generated points."""
        return self.unlabelled.latent_dim

    @property
    def n_classes(self) -> int:
        """Number of class labels C."""
        return len(self.class_mixtures)

    def to_arrays(self) -> dict:
        """
        Flatten the model into a dict of arrays suitable for ``np.savez``.

        Returns
        -------
        dict
            ``class_names`` and ``class_probs``, plus ``unlabelled_{weights,means,covs}``
            and ``class{c}_{weights,means,covs}`` for every class index ``c``.
        """
        out = dict(class_names=np.array(self.class_names), class_probs=self.class_probs)
        named = [("unlabelled", self.unlabelled)] + [
            (f"class{c}", m) for c, m in enumerate(self.class_mixtures)]
        for prefix, m in named:
            out[f"{prefix}_weights"] = m.weights
            out[f"{prefix}_means"] = m.means
            out[f"{prefix}_covs"] = m.covs
        return out


# --------------------------------------------------------------------------- #
# Reading covariances
# --------------------------------------------------------------------------- #
def _read_array(c: dict, key: str, base_dir: Path, ndim: int, where: str) -> np.ndarray:
    """
    Read a numeric array from a specification, inline or from a ``.npy`` file.

    Parameters
    ----------
    c : dict
        The (covariance) specification.
    key : str
        Name of the array, e.g. ``"matrix"``. The array is taken from ``c[key]`` if
        present, otherwise from the file named in ``c[key + "_file"]``.
    base_dir : pathlib.Path
        Directory against which ``*_file`` paths are resolved.
    ndim : int
        Required number of array dimensions.
    where : str
        Label for error messages.

    Returns
    -------
    numpy.ndarray
        Float array with `ndim` dimensions.

    Raises
    ------
    ValueError
        If neither ``key`` nor ``key_file`` is given, or the array has the wrong ndim.
    """
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
    """The supported values of the ``"type"`` field of a covariance specification."""
    SCALAR = 'scalar'
    DIAGONAL = 'diagonal'
    LOW_RANK = 'low_rank'
    STD_CORR = 'std_corr'
    FULL = 'full'


def read_scalar_covariance(specification: dict, dimensionality: int, base_dir: Path, where: str) -> np.ndarray:
    """
    Build ``variance * I`` from ``{"type": "scalar", "variance": v}``.

    Parameters
    ----------
    specification : dict
        The covariance specification.
    dimensionality : int
        Size d of the matrix.
    base_dir : pathlib.Path
        Unused; present so all readers share one signature.
    where : str
        Unused; present so all readers share one signature.

    Returns
    -------
    numpy.ndarray
        Shape (d, d).
    """
    return float(specification["variance"]) * np.eye(dimensionality)


def read_diagonal_covariance(specification: dict, dimensionality: int, base_dir: Path, where: str) -> np.ndarray:
    """
    Build ``diag(v)`` from ``{"type": "diagonal", "variances": [...]}``.

    Parameters
    ----------
    specification : dict
        The covariance specification; ``variances`` may also be given as ``variances_file``.
    dimensionality : int
        Expected size d (the shape is checked by the caller).
    base_dir : pathlib.Path
        Directory for resolving ``*_file`` references.
    where : str
        Label for error messages.

    Returns
    -------
    numpy.ndarray
        Diagonal matrix, shape (len(variances), len(variances)).
    """
    return np.diag(_read_array(specification, "variances", base_dir, 1, where))


def read_low_rank_covariance(specification: dict, dimensionality: int, base_dir: Path, where: str) -> np.ndarray:
    """
    Build ``F F^T + diag(psi)`` from ``{"type": "low_rank", "factor": F, "psi": [...]}``.

    Parameters
    ----------
    specification : dict
        The covariance specification; `factor` has shape (d, r) and may also be given
        as ``factor_file``.
    dimensionality : int
        Required number of rows of the factor.
    base_dir : pathlib.Path
        Directory for resolving ``*_file`` references.
    where : str
        Label for error messages.

    Returns
    -------
    numpy.ndarray
        Shape (d, d).

    Raises
    ------
    ValueError
        If the factor does not have `dimensionality` rows.
    """
    factor = _read_array(specification, "factor", base_dir, 2, where)
    psi = _read_array(specification, "psi", base_dir, 1, where)
    if factor.shape[0] != dimensionality:
        raise ValueError(f"{where}: factor must have {dimensionality} rows, got {factor.shape[0]}")
    return factor @ factor.T + np.diag(psi)


def read_std_corr_covariance(specification: dict, dimensionality: int, base_dir: Path, where: str) -> np.ndarray:
    """
    Build ``D R D`` from ``{"type": "std_corr", "std": [...], "correlation": R}``.

    Here D is the diagonal matrix of standard deviations and R the correlation matrix.

    Parameters
    ----------
    specification : dict
        The covariance specification; ``correlation`` may also be given as ``correlation_file``.
    dimensionality : int
        Expected size d (the shape is checked by the caller).
    base_dir : pathlib.Path
        Directory for resolving ``*_file`` references.
    where : str
        Label for error messages.

    Returns
    -------
    numpy.ndarray
        Covariance matrix ``R * outer(std, std)``.
    """
    std = _read_array(specification, "std", base_dir, 1, where)
    correlations = _read_array(specification, "correlation", base_dir, 2, where)
    return correlations * np.outer(std, std)


def read_full_covariance(specification: dict, dimensionality: int, base_dir: Path, where: str) -> np.ndarray:
    """
    Read a full covariance matrix from ``{"type": "full", "matrix": [[...]]}``.

    Parameters
    ----------
    specification : dict
        The covariance specification; ``matrix`` may also be given as ``matrix_file``.
    dimensionality : int
        Expected size d (the shape is checked by the caller).
    base_dir : pathlib.Path
        Directory for resolving ``*_file`` references.
    where : str
        Label for error messages.

    Returns
    -------
    numpy.ndarray
        The matrix as given.
    """
    return _read_array(specification, "matrix", base_dir, 2, where)


COVARIANCE_BUILDERS = {
    CovarianceType.SCALAR: read_scalar_covariance,
    CovarianceType.DIAGONAL: read_diagonal_covariance,
    CovarianceType.LOW_RANK: read_low_rank_covariance,
    CovarianceType.STD_CORR: read_std_corr_covariance,
    CovarianceType.FULL: read_full_covariance
}


def covariance_from_spec(specification: dict, dimensionality: int, base_dir: Path, where: str) -> np.ndarray:
    """
    Turn one covariance specification into a validated covariance matrix.

    The matrix is built by the reader that matches ``specification["type"]`` and then
    checked: it must have shape (d, d), be symmetric, and be positive definite. Tiny
    asymmetries from rounding are removed by symmetrising.

    Parameters
    ----------
    specification : dict
        A covariance specification such as ``{"type": "diagonal", "variances": [1, 2]}``.
    dimensionality : int
        Required size d of the matrix.
    base_dir : pathlib.Path
        Directory for resolving ``*_file`` references.
    where : str
        Label for error messages, e.g. ``"class 'A', component 0 (A0)"``.

    Returns
    -------
    numpy.ndarray
        Symmetric positive-definite matrix, shape (d, d).

    Raises
    ------
    ValueError
        If the type is missing or unknown, required arrays are missing or have the wrong
        shape, or the matrix is not symmetric or not positive definite.
    """
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


# --------------------------------------------------------------------------- #
# Reading components, mixtures and the full model
# --------------------------------------------------------------------------- #
def parse_component(specification: dict, base_dir: Path, where: str) -> Component:
    """
    Validate one component specification from the JSON file and build a `Component`.

    Parameters
    ----------
    specification : dict
        ``{"name": str (optional), "weight": number, "mean": [...], "covariance": {...}}``.
        The dimensionality d is taken from the length of ``mean``.
    base_dir : pathlib.Path
        Directory for resolving ``*_file`` references in the covariance.
    where : str
        Label for error messages, e.g. ``"unlabelled, component 1"``. The component's
        ``name``, if any, is appended.

    """
    name = str(specification.get("name", ""))
    where = f"{where} ({name})" if name else where

    for key in ("weight", "mean", "covariance"):
        if key not in specification:
            raise ValueError(f"{where}: missing key {key}")

    mean = np.asarray(specification["mean"], dtype=float)
    if mean.ndim != 1:
        raise ValueError(f"{where}: 'mean' must be a list of numbers")
    covariance = covariance_from_spec(specification["covariance"], mean.shape[0], base_dir, where)

    try:
        return Component(weight=float(specification["weight"]), mean=mean, cov=covariance, name=name)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


def parse_mixture(components: list, latent_dim: int, base_dir: Path, where: str) -> Mixture:
    """
    Validate a list of component specifications and build a `Mixture`.

    Each entry is parsed with `parse_component`; the resulting components are then
    combined with `Mixture.from_components`, which checks that they are consistent with
    each other.

    Parameters
    ----------
    components : list of dict
        The ``"components"`` list of one mixture in the JSON file.
    latent_dim : int
        Total dimensionality of the generated points.
    base_dir : pathlib.Path
        Directory for resolving ``*_file`` references.
    where : str
        Name of the mixture for error messages, e.g. ``"unlabelled"`` or ``"class 'A'"``.

    Returns
    -------
    Mixture

    Raises
    ------
    ValueError
        If the list is empty or any component or the combination is invalid.
    """
    if not components:
        raise ValueError(f"{where}: 'components' must be a non-empty list")
    parsed = [parse_component(c, base_dir, f"{where}, component {k}") for k, c in enumerate(components)]
    return Mixture.from_components(parsed, latent_dim, where)


def parse_model(spec: dict, base_dir: str | Path = ".") -> LatentModel:
    """
    Validate a complete specification dict and build the `LatentModel`.

    Parameters
    ----------
    spec : dict
        The parsed JSON document: ``latent_dim``, ``unlabelled`` (with ``components``) and
        ``classes`` (a list of ``{"name", "proportion" (optional), "components"}``). The
        order of ``classes`` defines the integer labels.
    base_dir : str or pathlib.Path
        Directory for resolving ``*_file`` references.

    Returns
    -------
    LatentModel
        Class proportions are normalised; if no class gives a proportion, classes are
        equally likely.

    Raises
    ------
    ValueError
        If fewer than two classes are given, proportions are given for only some classes
        or are not positive, class names repeat, or any mixture is invalid.
    KeyError
        If a required top-level key (``latent_dim``, ``unlabelled``) is missing.
    """
    base_dir = Path(base_dir)
    latent_dim = int(spec["latent_dim"])

    unlabelled = parse_mixture(spec["unlabelled"]["components"], latent_dim, base_dir, "unlabelled")

    classes = spec.get("classes")
    if not classes or len(classes) < 2:
        raise ValueError("'classes' must list at least two classes")
    names = [str(c["name"]) for c in classes]
    mixtures = [parse_mixture(c["components"], latent_dim, base_dir, f"class {name}")
                for c, name in zip(classes, names)]

    given = ["proportion" in c for c in classes]
    if all(given):
        class_probs = np.array([float(c["proportion"]) for c in classes])
        if np.any(class_probs <= 0):
            raise ValueError("class proportions must be > 0")
        class_probs /= class_probs.sum()
    elif not any(given):
        class_probs = np.full(len(classes), 1.0 / len(classes))
    else:
        raise ValueError("give 'proportion' for all classes or for none")

    return LatentModel(unlabelled, mixtures, names, class_probs)


def load_model(path: Path) -> LatentModel:
    """
    Read a JSON specification file and build the `LatentModel`.

    Parameters
    ----------
    path : str or pathlib.Path
        The JSON file. ``*_file`` references inside it are resolved relative to its folder.
    """
    with open(path) as f:
        return parse_model(json.load(f), base_dir=path.parent)


# --------------------------------------------------------------------------- #
# Writing (e.g. from GaussianMixtures fitted on the real data)
# --------------------------------------------------------------------------- #
def _gmm_components(gmm, decimals: int) -> list[dict]:
    """
    Convert a fitted scikit-learn ``GaussianMixture`` into component specifications.

    Parameters
    ----------
    gmm : sklearn.mixture.GaussianMixture
        A fitted model with covariance type ``full``, ``diag``, ``spherical`` or ``tied``.
    decimals : int
        Number of decimals to round weights, means and covariances to.

    Returns
    -------
    list of dict
        One specification per component, named ``component_<k>``, in the JSON format
        (``full``, ``diagonal`` or ``scalar`` covariance; tied covariances are written as
        a ``full`` matrix repeated for every component).
    """
    components = []
    for k in range(gmm.means_.shape[0]):
        if gmm.covariance_type == "full":
            cspec = {"type": "full", "matrix": np.round(gmm.covariances_[k], decimals).tolist()}
        elif gmm.covariance_type == "diag":
            cspec = {"type": "diagonal", "variances": np.round(gmm.covariances_[k], decimals).tolist()}
        elif gmm.covariance_type == "spherical":
            cspec = {"type": "scalar", "variance": round(float(gmm.covariances_[k]), decimals)}
        else:  # tied: one matrix shared by all components
            cspec = {"type": "full", "matrix": np.round(gmm.covariances_, decimals).tolist()}
        components.append(dict(
            name=f"component_{k}", weight=round(float(gmm.weights_[k]), decimals),
            mean=np.round(gmm.means_[k], decimals).tolist(), covariance=cspec))
    return components


def export_sklearn_gmms(path: str | Path, *, latent_dim: int, unlabelled, classes: dict,
                        class_proportions: Optional[dict] = None, decimals: int = 6) -> None:
    """
    Write fitted scikit-learn mixtures to a JSON specification file.

    Use this to turn mixtures fitted on real data into the input format of the generator.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file.
    latent_dim : int
        Total latent dimensionality to record in the file.
    unlabelled : sklearn.mixture.GaussianMixture
        Model fitted on the unlabelled records.
    classes : dict
        ``{class name: GaussianMixture fitted on that class's labelled records}``. The
        order of the dict defines the integer labels.
    class_proportions : dict, optional
        ``{class name: share}``, e.g. the observed class frequencies. If omitted, no
        proportions are written and the classes are equally likely when loaded.
    decimals : int
        Rounding of the stored numbers.

    Returns
    -------
    None
    """
    out = dict(latent_dim=latent_dim,
               unlabelled=dict(components=_gmm_components(unlabelled, decimals)),
               classes=[])
    for name, gmm in classes.items():
        entry = dict(name=name)
        if class_proportions is not None:
            entry["proportion"] = float(class_proportions[name])
        entry["components"] = _gmm_components(gmm, decimals)
        out["classes"].append(entry)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
