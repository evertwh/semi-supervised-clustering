# Synthetic latent data for semi-supervised clustering

Generates synthetic data that mimics the latent space of a VAE (e.g. 100-d embeddings), to test
semi-supervised clustering when the real data cannot be shared.

- a large **unlabelled** set (default 30,000 points) drawn from its own Gaussian mixture
- a small **labelled** set (default 600 points), where each class label has its own, separate Gaussian mixture
- optional noise dimensions, `N(0, 1)`, for latent dimensions the mixtures do not model (collapsed VAE dimensions)

The mixtures are defined in a JSON file. Nothing is generated at random except the sampled points.

## Quick start

```bash
python main.py --mixture-json test/data/example_mixture.json --out output/synthetic_latents.npz
python main.py --help        # all options
```

```python
from data_generator.MixtureConfig import MixtureConfig
from data_generator.data_generator import generate

data = generate(MixtureConfig(mixture_json="mixtures.json", n_unlabelled=30_000, n_labelled=600))
```

## Model file (JSON)

```json
{
  "latent_dim": 100,
  "unlabelled": {"components": [ <component>, ... ]},
  "classes": [
    {"name": "A", "proportion": 0.4, "components": [ <component>, ... ]},
    {"name": "B", "proportion": 0.6, "components": [ <component>, ... ]}
  ]
}
```

A component is `{"name": "...", "weight": 0.5, "mean": [...], "covariance": {...}}`.

- The order of `classes` defines the integer labels (0, 1, ...). `proportion` is optional (all or none; default: equal).
- `mean` has length d ≤ `latent_dim`; the remaining dimensions are `N(0, 1)` noise.
- Weights and proportions are relative and get normalised.

| covariance `type` | fields | Σ |
|---|---|---|
| `scalar` | `variance` | v·I |
| `diagonal` | `variances` | diag(v) |
| `low_rank` | `factor` (d×r), `psi` | F Fᵀ + diag(ψ) |
| `std_corr` | `std`, `correlation` (d×d) | D R D |
| `full` | `matrix` (d×d) | as given |

`matrix`, `factor` and `correlation` can also be given as `<name>_file`, the path of a `.npy` file relative to the JSON file.
All covariances are checked for shape, symmetry and positive definiteness. See `data_generator/example_mixture.json`.

To build the file from real data, fit one `GaussianMixture` to the unlabelled records and one per class, then call
`export_sklearn_gmms(path, latent_dim=..., unlabelled=..., classes={...})` from `mixture_spec.py`.
Check first that publishing the fitted means and covariances is acceptable.

## Configuration

`MixtureConfig` (pydantic-settings) reads keyword arguments, `SYN_*` environment variables, a `.env` file and the
defaults, in that order of priority; command-line options in `main.py` map to the same fields.

| setting | default | meaning |
|---|---|---|
| `mixture_json` | required | model file |
| `n_unlabelled` | 30000 | unlabelled points |
| `n_labelled` | 600 | labelled points |
| `labelled_class_probs` | from file | class proportions of the labelled set (overrides the file) |
| `seed` | 42 | random seed |

## Output

`generate()` returns a dict of arrays (saved by `main.py` with `np.savez_compressed`):

| key | content |
|---|---|
| `x_unlabelled`, `x_labelled` | points, shape `(n, latent_dim)` |
| `y_labelled` | class labels of the labelled points |
| `y_unlabelled` | all `-1` (label undefined; same convention as scikit-learn) |
| `z_unlabelled_true`, `z_labelled_true` | mixture component each point came from (ground truth; for labelled points relative to the class's mixture) |
| `unlabelled_*`, `class{c}_*`, `class_names`, `class_probs` | the model parameters (`weights`, `means`, `covs`) |

## Code layout

| file | role |
|---|---|
| `data_generator/data_generator.py` | `generate_points` (the single sampling function) and `generate` |
| `data_generator/mixture_spec.py` | `Component`, `Mixture`, `LatentModel`; JSON parsing and validation; sklearn export |
| `data_generator/MixtureConfig.py` | settings |
| `tests/` | unit tests |

## Tests

```bash
python -m unittest discover     # or: pytest
```

The config tests are skipped if `pydantic-settings` is not installed.

## Caveats

- The unlabelled and class mixtures are independent. Labelled points need not fall where the unlabelled mixture has
  mass unless your JSON makes it so (this is automatic when all mixtures are fitted to the same real data).
- The unlabelled points have no class, so use `z_unlabelled_true` to evaluate clusterings of them.
- Within-component distributions are Gaussian; real VAE latents may have heavier tails.
