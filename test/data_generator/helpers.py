"""Shared helpers for the test-suite."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_generator.MixtureConfig import MixtureConfig


class EnvIsolatedTestCase(unittest.TestCase):
    """Removes SYN_* variables from the environment so a developer's shell
    cannot leak into the tests."""

    def setUp(self):
        super().setUp()
        env = {k: v for k, v in os.environ.items() if not k.startswith("SYN_")}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tmp_dir(self) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return Path(td.name)


def make_cfg(**overrides) -> MixtureConfig:
    """A small, fast configuration (ignores any .env file)."""
    kw = dict(latent_dim=8, n_informative=4, n_components=4, n_classes=2,
              cov_rank=2, n_unlabelled=3000, n_labelled=1500, seed=0)
    kw.update(overrides)
    return MixtureConfig(_env_file=None, **kw)


# --- a small, well-separated, hand-specified mixture (JSON format) ---------- #
SPEC_K, SPEC_D, SPEC_C, SPEC_LATENT = 3, 4, 2, 6
SPEC_WEIGHTS = (0.5, 0.3, 0.2)
SPEC_LABEL_PROBS = ((0.9, 0.1), (0.2, 0.8), (0.5, 0.5))


def spec_dict(weights=SPEC_WEIGHTS, label_probs=SPEC_LABEL_PROBS, covariances=None):
    covariances = covariances or [{"type": "scalar", "variance": 0.3 * (k + 1)}
                                  for k in range(SPEC_K)]
    return {
        "latent_dim": SPEC_LATENT,
        "classes": ["A", "B"],
        "components": [
            {"name": f"c{k}", "weight": weights[k],
             "mean": [6.0 * (k - 1)] * SPEC_D,
             "covariance": covariances[k],
             "label_probs": list(label_probs[k])}
            for k in range(SPEC_K)
        ],
    }


def write_spec(directory: Path, spec: dict, name: str = "mixture.json") -> Path:
    path = Path(directory) / name
    path.write_text(json.dumps(spec))
    return path


def spec_cfg(path: Path, **overrides) -> MixtureConfig:
    """Config consistent with spec_dict() (generate() reads the dimensions from
    the config, not from the JSON file)."""
    kw = dict(mixture_json=path, latent_dim=SPEC_LATENT, n_informative=SPEC_D,
              n_components=SPEC_K, n_classes=SPEC_C,
              n_unlabelled=3000, n_labelled=1500, seed=0)
    kw.update(overrides)
    return MixtureConfig(_env_file=None, **kw)
