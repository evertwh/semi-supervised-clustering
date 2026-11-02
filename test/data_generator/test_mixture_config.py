import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from pydantic import ValidationError
    from data_generator.MixtureConfig import MixtureConfig
except ImportError:  # pragma: no cover
    MixtureConfig = None

from helpers import make_spec


def clean_env(**extra):
    """Environment without any SYN_* variables (plus the given ones)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SYN_")}
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


@unittest.skipIf(MixtureConfig is None, "pydantic-settings not installed")
class TestMixtureConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.json"
        self.path.write_text(json.dumps(make_spec()))

    def make(self, **kwargs):
        return MixtureConfig(_env_file=None, **kwargs)   # ignore any .env file

    def test_defaults(self):
        with clean_env():
            cfg = self.make(mixture_json=self.path)
        self.assertEqual((cfg.n_unlabelled, cfg.n_labelled, cfg.seed), (30_000, 600, 42))
        self.assertIsNone(cfg.labelled_class_probs)

    def test_mixture_json_is_required(self):
        with clean_env(), self.assertRaises(ValidationError):
            self.make()

    def test_mixture_json_must_exist(self):
        with clean_env(), self.assertRaises(ValidationError):
            self.make(mixture_json=Path(self.tmp.name) / "missing.json")

    def test_environment_variables(self):
        with clean_env(SYN_MIXTURE_JSON=str(self.path), SYN_N_LABELLED="123",
                       SYN_LABELLED_CLASS_PROBS="[0.2, 0.8]"):
            cfg = self.make()
        self.assertEqual(cfg.n_labelled, 123)
        self.assertEqual(cfg.labelled_class_probs, [0.2, 0.8])

    def test_keyword_arguments_beat_environment(self):
        with clean_env(SYN_N_LABELLED="123"):
            cfg = self.make(mixture_json=self.path, n_labelled=7)
        self.assertEqual(cfg.n_labelled, 7)

    def test_unknown_settings_are_rejected(self):
        with clean_env(), self.assertRaises(ValidationError):
            self.make(mixture_json=self.path, class_purity=0.8)

    def test_non_positive_sizes_are_rejected(self):
        for field in ("n_unlabelled", "n_labelled"):
            with self.subTest(field=field), clean_env(), self.assertRaises(ValidationError):
                self.make(mixture_json=self.path, **{field: 0})

    def test_invalid_class_probabilities(self):
        for probs in ([-0.1, 1.1], [0, 0]):
            with self.subTest(probs=probs), clean_env(), self.assertRaises(ValidationError):
                self.make(mixture_json=self.path, labelled_class_probs=probs)

    def test_config_is_immutable(self):
        with clean_env():
            cfg = self.make(mixture_json=self.path)
        with self.assertRaises(ValidationError):
            cfg.seed = 1

    def test_works_with_generate(self):
        from data_generator.data_generator import generate
        with clean_env():
            cfg = self.make(mixture_json=self.path, n_unlabelled=50, n_labelled=20)
        data = generate(cfg)
        self.assertEqual(data["x_unlabelled"].shape[0], 50)
        self.assertEqual(data["x_labelled"].shape[0], 20)


if __name__ == "__main__":
    unittest.main()
