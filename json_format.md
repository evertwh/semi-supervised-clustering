# Model definition file (JSON)

The generator reads its whole model from one JSON file: a Gaussian mixture for the **unlabelled** data and a
separate Gaussian mixture for **each class label**. This page is the reference for that file.

Contents: [Structure](#structure) · [Reference](#reference) · [Covariance types](#covariance-types) ·
[How the file is used](#how-the-file-is-used) · [Validation](#validation-and-errors) · [Examples](#examples) ·
[Creating a file](#creating-a-file)

## Structure

```text
{
  "latent_dim": 64,                              total number of dimensions of the generated points
  "unlabelled": {
    "components": [ <component>, ... ]           mixture for the unlabelled records
  },
  "classes": [                                   at least two; the list order defines the integer labels 0, 1, ...
    {
      "name": "A",
      "proportion": 0.6,                         optional (see below)
      "components": [ <component>, ... ]         mixture for the records with this label
    },
    ...
  ]
}
```

A `<component>` is one Gaussian:

```text
{
  "name": "optional text",
  "weight": 0.25,
  "mean": [ d numbers ],
  "covariance": { "type": "...", ... }           see "Covariance types"
}
```

## Reference

### Top level

| key | type | required | meaning |
|---|---|---|---|
| `latent_dim` | integer | yes | Total dimensionality of every generated point. |
| `unlabelled` | object | yes | Contains `components`: the mixture the unlabelled records are drawn from. |
| `classes` | list of objects | yes | One entry per label; at least 2. The position in the list is the integer label (first class = 0). |

### Class entry

| key | type | required | meaning |
|---|---|---|---|
| `name` | string | yes | Label name. Must be unique. Stored in the output as `class_names`. |
| `proportion` | number > 0 | optional | Share of this class in the labelled set. Give it for **all** classes or for **none** (then all classes are equally likely). Values are relative and get normalised to sum to 1. |
| `components` | list | yes | Non-empty list of components: the mixture that records with this label are drawn from. |

### Component

| key | type | required | meaning |
|---|---|---|---|
| `name` | string | no | Free text; only used in error messages. |
| `weight` | number > 0 | yes | Relative weight within **its own mixture**. Weights are normalised per mixture, so `[3, 1]` means 75 % / 25 %. |
| `mean` | list of numbers | yes | Mean vector of length *d*. |
| `covariance` | object | yes | Covariance specification, see below. |

Rules for `mean`:

- *d* (the length of `mean`) must be the same for all components of one mixture and at most `latent_dim`.
- The mixture describes the **first *d* coordinates**. The remaining `latent_dim − d` coordinates are filled with
  independent N(0, 1) noise (this models collapsed VAE dimensions and keeps the file small).
- Different mixtures (unlabelled, each class) may use different *d*.

Keys that are not listed here are ignored. JSON has no comments; use `name` for notes.

## Covariance types

Every covariance object has a `type`. *d* is the length of the component's `mean`.

| `type` | fields | resulting covariance Σ |
|---|---|---|
| `scalar` | `variance` (number) | `variance · I` |
| `diagonal` | `variances` (*d* numbers) | `diag(variances)` |
| `low_rank` | `factor` (*d* × *r*), `psi` (*d* numbers) | `F Fᵀ + diag(psi)` |
| `std_corr` | `std` (*d* numbers), `correlation` (*d* × *d*) | `D R D`, with D = diag(std), R = correlation |
| `full` | `matrix` (*d* × *d*) | the matrix as given |

```json
{"type": "scalar",   "variance": 0.5}
{"type": "diagonal", "variances": [0.5, 1.0, 0.3]}
{"type": "low_rank", "factor": [[1.0], [0.8], [0.0]], "psi": [0.2, 0.2, 0.4]}
{"type": "std_corr", "std": [1, 2], "correlation": [[1, 0.5], [0.5, 1]]}
{"type": "full",     "matrix": [[2, 0.5, 0], [0.5, 1, 0], [0, 0, 1]]}
```

Choosing a type: `scalar` and `diagonal` are the smallest; `low_rank` gives correlations with far fewer numbers than
`full` (about *d·r* instead of *d²/2*); `std_corr` is convenient to write by hand; `full` is what you get when fitting
mixtures to data. In 100 dimensions a `full` covariance has 5,050 distinct values per component.

### Arrays from `.npy` files

Any array field (`variances`, `factor`, `psi`, `std`, `correlation`, `matrix`) can be replaced by the same name with
a `_file` suffix, holding the path of a NumPy `.npy` file. Relative paths are resolved against the folder of the JSON
file.

```json
{"type": "full", "matrix_file": "covariances/class_A_0.npy"}
```

`variance` of the `scalar` type is a single number and cannot come from a file.

## How the file is used

1. **Unlabelled records:** for each record a component of `unlabelled` is chosen with probability `weight`; the first
   *d* coordinates are drawn from N(`mean`, Σ), the rest are N(0, 1). The records get the label `-1` (undefined).
2. **Labelled records:** the number of records per class is drawn from the class proportions (the `proportion`
   values, or `labelled_class_probs` from the configuration if set, which overrides the file). For a record of class
   *c*, a component of that class's mixture is chosen by `weight` and the record is drawn as above. The label is the
   position of the class in `classes`.
3. The unlabelled mixture and the class mixtures are **independent**: nothing links their components. If labelled
   records should lie where the unlabelled records are, put the components there.

## Validation and errors

The file is checked when it is loaded; the error message says where the problem is, for example
`class 'A', component 0 (A0): covariance is not positive definite`.

| check | error |
|---|---|
| `latent_dim` or `unlabelled` missing | `KeyError` |
| fewer than 2 classes; `proportion` given for only some classes or ≤ 0; duplicate class names | `ValueError` |
| a mixture has no components | `ValueError` |
| a component lacks `weight`, `mean` or `covariance`; `weight` ≤ 0 or not finite; `mean` not a flat list of finite numbers | `ValueError` |
| means in one mixture have different lengths, or *d* > `latent_dim` | `ValueError` |
| unknown covariance `type`; required field or `_file` missing; array of the wrong dimension | `ValueError` |
| covariance not of shape *d* × *d*; not symmetric; not positive definite | `ValueError` |

Positive definite means the matrix has strictly positive eigenvalues. A matrix such as `[[1, 1], [1, 1]]` (perfectly
correlated) is rejected, as is a negative variance. Tiny rounding asymmetries are removed automatically.

## Examples

A complete minimal file: 4 latent dimensions, of which 2 are modelled, and two classes.

```json
{
  "latent_dim": 4,
  "unlabelled": {
    "components": [
      {"name": "left",  "weight": 3, "mean": [-2, 0], "covariance": {"type": "scalar", "variance": 1.0}},
      {"name": "right", "weight": 1, "mean": [ 2, 0], "covariance": {"type": "diagonal", "variances": [0.5, 2.0]}}
    ]
  },
  "classes": [
    {"name": "A", "proportion": 0.7,
     "components": [
       {"weight": 1, "mean": [-2, 0.5],
        "covariance": {"type": "std_corr", "std": [0.4, 0.4], "correlation": [[1, 0.3], [0.3, 1]]}}
     ]},
    {"name": "B", "proportion": 0.3,
     "components": [
       {"weight": 1, "mean": [2, -0.5], "covariance": {"type": "scalar", "variance": 0.2}},
       {"weight": 1, "mean": [0,  2.0], "covariance": {"type": "full", "matrix": [[0.3, 0.1], [0.1, 0.3]]}}
     ]}
  ]
}
```

Larger examples in the repository: `data_generator/example_mixture.json` (all covariance types) and
`data_generator/example_64d.json` (64 dimensions; tight labelled clusters plus a dispersed part, with unlabelled data
overlapping the clusters).

## Creating a file

- **From your own data:** `fit_mixture_definition_databricks.py` (notebook) or `fit_mixture_definition.py` (command
  line) fit a mixture to the unlabelled data and to each label and write this format.
- **From fitted scikit-learn mixtures:** `export_sklearn_gmms(...)` in `data_generator/mixture_spec.py`.
- **By hand or by script:** see `make_example_64d.py`.

Notes for large files: numbers are plain JSON (fitted files are rounded to 6 decimals); with `full` covariances the
file grows with *components × d²*, so prefer `diagonal` or `low_rank` where correlations do not matter, or `_file`
references for the matrices.
