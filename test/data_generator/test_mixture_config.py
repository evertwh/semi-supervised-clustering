"""Tests for MixtureConfig (pydantic-settings).

All tests pass ``_env_file=None`` (via make_cfg) and run with SYN_* variables removed,
so neither a local .env nor your shell can influence them.
"""
import os
from unittest import mock

from pydantic import ValidationError

from data_generator.MixtureConfig import MixtureConfig
from helpers import EnvIsolatedTestCase, make_cfg


def default_cfg(**kw):
    return MixtureConfig(_env_file=None, **kw)


class TestDefaults(EnvIsolatedTestCase):
    def test_defaults(self):
        c = default_cfg()
        self.assertIsNone(c.mixture_json)
        self.assertEqual((c.latent_dim, c.n_informative, c.n_components, c.n_classes), (100, 20, 8, 4))
        self.assertEqual((c.n_unlabelled, c.n_labelled), (30_000, 600))
        self.assertEqual(c.psi_range, (0.3, 1.0))
        self.assertTrue(c.standardise)
        self.assertIsNone(c.labelled_class_probs)
        self.assertEqual(c.seed, 42)

    def test_frozen(self):
        with self.assertRaises(ValidationError):
            default_cfg().seed = 1

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValidationError):
            default_cfg(n_componets=3)

    def test_psi_range_accepts_list(self):
        self.assertEqual(default_cfg(psi_range=[0.2, 0.9]).psi_range, (0.2, 0.9))


class TestSources(EnvIsolatedTestCase):
    def test_env_variable_overrides_default(self):
        with mock.patch.dict(os.environ, {"SYN_N_COMPONENTS": "10"}):
            self.assertEqual(default_cfg().n_components, 10)

    def test_complex_env_values_are_json(self):
        env = {"SYN_LABELLED_CLASS_PROBS": "[0.4, 0.3, 0.2, 0.1]", "SYN_PSI_RANGE": "[0.2, 0.9]"}
        with mock.patch.dict(os.environ, env):
            c = default_cfg()
        self.assertEqual(c.labelled_class_probs, [0.4, 0.3, 0.2, 0.1])
        self.assertEqual(c.psi_range, (0.2, 0.9))

    def test_kwargs_beat_environment(self):
        with mock.patch.dict(os.environ, {"SYN_N_COMPONENTS": "10"}):
            self.assertEqual(default_cfg(n_components=6).n_components, 6)

    def test_env_file(self):
        env_file = self.tmp_dir() / ".env"
        env_file.write_text("SYN_N_COMPONENTS=9\nSYN_SEED=7\n")
        c = MixtureConfig(_env_file=env_file)
        self.assertEqual((c.n_components, c.seed), (9, 7))

    def test_environment_beats_env_file(self):
        env_file = self.tmp_dir() / ".env"
        env_file.write_text("SYN_N_COMPONENTS=9\n")
        with mock.patch.dict(os.environ, {"SYN_N_COMPONENTS": "12"}):
            self.assertEqual(MixtureConfig(_env_file=env_file).n_components, 12)


class TestFieldConstraints(EnvIsolatedTestCase):
    def test_invalid_values_rejected(self):
        cases = [dict(latent_dim=0), dict(n_informative=0), dict(n_components=0),
                 dict(n_classes=1), dict(n_unlabelled=0), dict(n_labelled=0),
                 dict(mean_separation=0), dict(cov_rank=0), dict(cov_scale=-1),
                 dict(weight_concentration=0), dict(class_purity=-0.1), dict(class_purity=1.1)]
        for kw in cases:
            with self.subTest(**kw), self.assertRaises(ValidationError):
                make_cfg(**kw)

    def test_boundary_values_accepted(self):
        make_cfg(class_purity=0.0)
        make_cfg(class_purity=1.0)
        make_cfg(n_classes=2)


class TestMixtureJsonField(EnvIsolatedTestCase):
    def test_empty_string_and_whitespace_become_none(self):
        for v in ("", "   ", None):
            with self.subTest(value=v):
                self.assertIsNone(default_cfg(mixture_json=v).mixture_json)

    def test_missing_file_is_rejected(self):
        with self.assertRaises(ValidationError):
            default_cfg(mixture_json=self.tmp_dir() / "missing.json")

    def test_directory_is_rejected(self):
        with self.assertRaises(ValidationError):
            default_cfg(mixture_json=self.tmp_dir())

    def test_existing_file_accepted_as_path(self):
        f = self.tmp_dir() / "m.json"
        f.write_text("{}")
        for v in (f, str(f)):
            with self.subTest(type=type(v).__name__):
                self.assertEqual(default_cfg(mixture_json=v).mixture_json, f)

    def test_with_json_file_other_consistency_checks_are_skipped(self):
        f = self.tmp_dir() / "m.json"
        f.write_text("{}")
        c = default_cfg(mixture_json=f, latent_dim=5, n_informative=10,  # would be invalid
                        n_components=2, n_classes=4, psi_range=(2.0, 1.0),
                        labelled_class_probs=[1.0])
        self.assertEqual(c.n_informative, 10)

    def test_without_json_file_checks_apply(self):
        with self.assertRaises(ValidationError):
            default_cfg(n_informative=10, latent_dim=5)


class TestConsistencyValidator(EnvIsolatedTestCase):
    def test_valid_config_passes(self):
        make_cfg()
        make_cfg(n_informative=8, latent_dim=8)               # equal is allowed
        make_cfg(n_components=2, n_classes=2)                 # K == C is allowed
        make_cfg(psi_range=(0.5, 0.5))                        # lo == hi is allowed

    def test_n_informative_must_not_exceed_latent_dim(self):
        with self.assertRaisesRegex(ValidationError, "n_informative"):
            make_cfg(n_informative=9, latent_dim=8)

    def test_psi_range_must_be_positive_and_ordered(self):
        for rng in [(0.0, 1.0), (-0.1, 1.0), (1.0, 0.5)]:
            with self.subTest(psi_range=rng), self.assertRaisesRegex(ValidationError, "psi_range"):
                make_cfg(psi_range=rng)

    def test_n_components_must_cover_classes(self):
        with self.assertRaisesRegex(ValidationError, "n_components"):
            make_cfg(n_components=3, n_classes=4)

    def test_labelled_class_probs_must_match_n_classes(self):
        with self.assertRaisesRegex(ValidationError, "length n_classes"):
            make_cfg(n_classes=2, labelled_class_probs=[0.2, 0.3, 0.5])

    def test_labelled_class_probs_must_be_valid_weights(self):
        for p in ([-0.1, 1.1], [0.0, 0.0]):
            with self.subTest(probs=p), self.assertRaisesRegex(ValidationError, "non-negative"):
                make_cfg(n_classes=2, labelled_class_probs=p)

    def test_unnormalised_labelled_class_probs_are_accepted(self):
        self.assertEqual(make_cfg(n_classes=2, labelled_class_probs=[3, 1]).labelled_class_probs,
                         [3, 1])
