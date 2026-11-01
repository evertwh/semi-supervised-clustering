"""Tests for data_generator, using the real MixtureConfig / mixture_spec.

Run from the directory *above* the package:
    python -m unittest discover -s <pkg>/tests -t . -v
or  pytest <pkg>/tests
"""
import numpy as np

from data_generator import data_generator as dg
from helpers import (EnvIsolatedTestCase, make_cfg, spec_cfg, spec_dict, write_spec,
                      SPEC_WEIGHTS, SPEC_LABEL_PROBS)

N = 20_000  # sample size for statistical tests


def hand_mixture(K=3, d=3):
    means = np.array([np.full(d, 6.0 * (k - (K - 1) / 2)) for k in range(K)])
    covs = np.array([0.3 * (k + 1) * np.eye(d) for k in range(K)])
    return dict(means=means, covs=covs)


# --------------------------------------------------------------------------- #
# sample_given_components
# --------------------------------------------------------------------------- #
class TestSampleGivenComponents(EnvIsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.cfg = make_cfg(latent_dim=5, n_informative=3, n_components=3)
        self.mix = hand_mixture()

    def sample(self, z, seed=0, cfg=None):
        return dg.sample_given_components(np.asarray(z), self.mix, cfg or self.cfg,
                                          np.random.default_rng(seed))

    def test_output_shape_and_finite(self):
        X = self.sample([0, 1, 2, 1, 0, 2, 2])
        self.assertEqual(X.shape, (7, 5))
        self.assertTrue(np.isfinite(X).all())

    def test_empty_input(self):
        self.assertEqual(self.sample(np.array([], dtype=int)).shape, (0, 5))

    def test_deterministic_for_same_seed(self):
        z = np.random.default_rng(1).integers(0, 3, size=200)
        np.testing.assert_array_equal(self.sample(z, 7), self.sample(z, 7))

    def test_different_seeds_differ(self):
        z = np.zeros(50, dtype=int)
        self.assertFalse(np.allclose(self.sample(z, 1), self.sample(z, 2)))

    def test_noise_block_is_standard_normal(self):
        noise = self.sample(np.zeros(50_000, dtype=int))[:, 3:]
        np.testing.assert_allclose(noise.mean(axis=0), 0, atol=0.02)
        np.testing.assert_allclose(np.cov(noise, rowvar=False), np.eye(2), atol=0.03)

    def test_informative_block_matches_component_moments(self):
        for k in range(3):
            with self.subTest(component=k):
                inf = self.sample(np.full(N, k), seed=k)[:, :3]
                np.testing.assert_allclose(inf.mean(axis=0), self.mix["means"][k], atol=0.05)
                np.testing.assert_allclose(np.cov(inf, rowvar=False), self.mix["covs"][k], atol=0.1)

    def test_rows_use_their_own_component(self):
        z = np.tile([0, 1, 2], 3000)
        X = self.sample(z)
        for k in range(3):
            np.testing.assert_allclose(X[z == k, :3].mean(axis=0), self.mix["means"][k], atol=0.1)

    def test_unused_components_are_skipped(self):
        X = self.sample(np.full(100, 2))
        self.assertEqual(X.shape, (100, 5))
        self.assertTrue(np.isfinite(X).all())

    def test_all_informative_means_no_noise_block(self):
        cfg = make_cfg(latent_dim=3, n_informative=3, n_components=3)
        self.assertEqual(self.sample(np.zeros(10, dtype=int), cfg=cfg).shape, (10, 3))

    def test_non_positive_definite_covariance_raises(self):
        self.mix["covs"][0] = -np.eye(3)
        with self.assertRaises(np.linalg.LinAlgError):
            self.sample(np.zeros(5, dtype=int))


# --------------------------------------------------------------------------- #
# generate: random mixture (real generate_mixture)
# --------------------------------------------------------------------------- #
class TestGenerateRandomMixture(EnvIsolatedTestCase):
    EXPECTED_KEYS = {"X_unlabelled", "X_labelled", "y_labelled", "z_unlabelled_true",
                     "y_unlabelled_true", "z_labelled_true", "mixture_weights",
                     "mixture_means", "mixture_covs", "label_given_component"}

    def setUp(self):
        super().setUp()
        self.cfg = make_cfg()
        self.out = dg.generate(self.cfg)

    def test_keys(self):
        self.assertEqual(set(self.out), self.EXPECTED_KEYS)

    def test_shapes(self):
        c, o = self.cfg, self.out
        self.assertEqual(o["X_unlabelled"].shape, (c.n_unlabelled, c.latent_dim))
        self.assertEqual(o["X_labelled"].shape, (c.n_labelled, c.latent_dim))
        for key, n in [("y_labelled", c.n_labelled), ("z_labelled_true", c.n_labelled),
                       ("z_unlabelled_true", c.n_unlabelled), ("y_unlabelled_true", c.n_unlabelled)]:
            self.assertEqual(o[key].shape, (n,), key)
        self.assertEqual(o["mixture_weights"].shape, (c.n_components,))
        self.assertEqual(o["mixture_means"].shape, (c.n_components, c.n_informative))
        self.assertEqual(o["mixture_covs"].shape, (c.n_components, c.n_informative, c.n_informative))
        self.assertEqual(o["label_given_component"].shape, (c.n_components, c.n_classes))

    def test_value_ranges(self):
        c, o = self.cfg, self.out
        for key, hi in [("y_labelled", c.n_classes), ("y_unlabelled_true", c.n_classes),
                        ("z_labelled_true", c.n_components), ("z_unlabelled_true", c.n_components)]:
            with self.subTest(key=key):
                self.assertGreaterEqual(o[key].min(), 0)
                self.assertLess(o[key].max(), hi)

    def test_features_finite(self):
        self.assertTrue(np.isfinite(self.out["X_unlabelled"]).all())
        self.assertTrue(np.isfinite(self.out["X_labelled"]).all())

    def test_reproducible_with_same_seed(self):
        again = dg.generate(make_cfg())
        for key in self.out:
            with self.subTest(key=key):
                np.testing.assert_array_equal(self.out[key], again[key])

    def test_different_seed_changes_everything(self):
        other = dg.generate(make_cfg(seed=1))
        self.assertFalse(np.allclose(self.out["mixture_means"], other["mixture_means"]))
        self.assertFalse(np.array_equal(self.out["z_unlabelled_true"], other["z_unlabelled_true"]))

    def test_every_class_present_in_labelled_set(self):
        self.assertEqual(set(np.unique(self.out["y_labelled"])), set(range(self.cfg.n_classes)))

    def test_standardised_mixture_has_zero_mean_identity_cov_analytically(self):
        o = self.out
        w, m, S = o["mixture_weights"], o["mixture_means"], o["mixture_covs"]
        mu = w @ m
        total = sum(wk * (Sk + np.outer(mk, mk)) for wk, Sk, mk in zip(w, S, m)) - np.outer(mu, mu)
        np.testing.assert_allclose(mu, 0, atol=1e-8)
        np.testing.assert_allclose(total, np.eye(self.cfg.n_informative), atol=1e-8)

    def test_standardised_samples_are_approximately_n01(self):
        cfg = make_cfg(n_unlabelled=N, n_labelled=10)
        X = dg.generate(cfg)["X_unlabelled"]
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=0.06)
        np.testing.assert_allclose(np.cov(X, rowvar=False), np.eye(cfg.latent_dim), atol=0.1)

    def test_unstandardised_mixture_is_not_whitened(self):
        o = dg.generate(make_cfg(standardise=False, n_unlabelled=10, n_labelled=10))
        w, m = o["mixture_weights"], o["mixture_means"]
        self.assertGreater(np.abs(w @ m).max(), 1e-6)


# --------------------------------------------------------------------------- #
# generate: hand-specified mixture loaded from JSON
# --------------------------------------------------------------------------- #
class TestGenerateFromJson(EnvIsolatedTestCase):
    def generate(self, spec=None, **cfg_overrides):
        path = write_spec(self.tmp_dir(), spec or spec_dict())
        return dg.generate(spec_cfg(path, **cfg_overrides))

    def test_uses_mixture_from_file(self):
        out = self.generate()
        np.testing.assert_allclose(out["mixture_weights"], SPEC_WEIGHTS)
        np.testing.assert_allclose(out["label_given_component"], SPEC_LABEL_PROBS)
        np.testing.assert_allclose(out["mixture_means"][:, 0], [-6.0, 0.0, 6.0])

    def test_file_mixture_is_not_standardised(self):
        out = self.generate(standardise=True)
        self.assertGreater(np.abs(out["mixture_means"]).max(), 5)  # untouched

    def test_json_weights_are_normalised(self):
        out = self.generate(spec_dict(weights=(5, 3, 2)))
        np.testing.assert_allclose(out["mixture_weights"], SPEC_WEIGHTS)

    def test_seed_still_controls_sampling(self):
        a, b, c = self.generate(seed=1), self.generate(seed=1), self.generate(seed=2)
        np.testing.assert_array_equal(a["X_unlabelled"], b["X_unlabelled"])
        self.assertFalse(np.array_equal(a["X_unlabelled"], c["X_unlabelled"]))

    def test_unlabelled_component_frequencies_match_weights(self):
        out = self.generate(n_unlabelled=N, n_labelled=10)
        freq = np.bincount(out["z_unlabelled_true"], minlength=3) / N
        np.testing.assert_allclose(freq, SPEC_WEIGHTS, atol=0.01)

    def test_unlabelled_labels_follow_label_given_component(self):
        out = self.generate(n_unlabelled=N, n_labelled=10)
        z, y = out["z_unlabelled_true"], out["y_unlabelled_true"]
        for k in range(3):
            with self.subTest(component=k):
                emp = np.bincount(y[z == k], minlength=2) / (z == k).sum()
                np.testing.assert_allclose(emp, SPEC_LABEL_PROBS[k], atol=0.03)

    def test_features_follow_assigned_component(self):
        out = self.generate(n_unlabelled=N, n_labelled=N)
        for X, z in [(out["X_unlabelled"], out["z_unlabelled_true"]),
                     (out["X_labelled"], out["z_labelled_true"])]:
            for k in range(3):
                with self.subTest(component=k):
                    np.testing.assert_allclose(X[z == k, :4].mean(axis=0),
                                               out["mixture_means"][k], atol=0.1)

    def test_noise_dimensions_are_standard_normal(self):
        out = self.generate(n_unlabelled=N, n_labelled=10)
        noise = out["X_unlabelled"][:, 4:]
        np.testing.assert_allclose(noise.mean(axis=0), 0, atol=0.05)
        np.testing.assert_allclose(noise.std(axis=0), 1, atol=0.05)

    # ---- labelled set ------------------------------------------------------
    def test_labelled_class_marginal_used_when_probs_none(self):
        joint = np.array(SPEC_WEIGHTS)[:, None] * np.array(SPEC_LABEL_PROBS)
        out = self.generate(n_labelled=N, n_unlabelled=10)
        freq = np.bincount(out["y_labelled"], minlength=2) / N
        np.testing.assert_allclose(freq, joint.sum(axis=0), atol=0.01)

    def test_labelled_class_probs_override(self):
        out = self.generate(n_labelled=N, n_unlabelled=10, labelled_class_probs=[0.9, 0.1])
        freq = np.bincount(out["y_labelled"], minlength=2) / N
        np.testing.assert_allclose(freq, [0.9, 0.1], atol=0.01)

    def test_labelled_class_probs_are_normalised(self):
        out = self.generate(n_labelled=N, n_unlabelled=10, labelled_class_probs=[3, 1])
        freq = np.bincount(out["y_labelled"], minlength=2) / N
        np.testing.assert_allclose(freq, [0.75, 0.25], atol=0.01)

    def test_labelled_components_follow_posterior_given_class(self):
        joint = np.array(SPEC_WEIGHTS)[:, None] * np.array(SPEC_LABEL_PROBS)
        out = self.generate(n_labelled=N, n_unlabelled=10)
        y, z = out["y_labelled"], out["z_labelled_true"]
        for c in range(2):
            with self.subTest(cls=c):
                emp = np.bincount(z[y == c], minlength=3) / (y == c).sum()
                np.testing.assert_allclose(emp, joint[:, c] / joint[:, c].sum(), atol=0.02)

    def test_selection_bias_does_not_affect_unlabelled_set(self):
        a = self.generate(labelled_class_probs=None)
        b = self.generate(labelled_class_probs=[0.99, 0.01])
        np.testing.assert_array_equal(a["z_unlabelled_true"], b["z_unlabelled_true"])
        np.testing.assert_array_equal(a["X_unlabelled"], b["X_unlabelled"])
        self.assertFalse(np.array_equal(a["y_labelled"], b["y_labelled"]))

    def test_zero_probability_pairs_never_sampled(self):
        lp = ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5))
        out = self.generate(spec_dict(label_probs=lp))
        P = np.array(lp)
        for z_key, y_key in [("z_unlabelled_true", "y_unlabelled_true"),
                             ("z_labelled_true", "y_labelled")]:
            with self.subTest(set=z_key):
                self.assertTrue(np.all(P[out[z_key], out[y_key]] > 0))

    def test_class_served_by_single_component(self):
        lp = ((1.0, 0.0), (0.0, 1.0), (0.0, 1.0))
        out = self.generate(spec_dict(label_probs=lp))
        y, z = out["y_labelled"], out["z_labelled_true"]
        self.assertTrue(np.all(z[y == 0] == 0))
        self.assertTrue(np.all(z[y == 1] != 0))


class TestGenerateConfigMustMatchJson(EnvIsolatedTestCase):
    """generate() takes K, C and n_informative from the *config*, not from the JSON
    mixture, so an inconsistent config fails (with a rather cryptic message).
    If generate() is changed to derive these from the mixture, update these tests."""

    def test_inconsistent_config_raises(self):
        path = write_spec(self.tmp_dir(), spec_dict())
        for label, kw in [("n_classes", dict(n_classes=4)),
                          ("n_components", dict(n_components=8)),
                          ("n_informative", dict(n_informative=3))]:
            with self.subTest(mismatch=label), self.assertRaises(ValueError):
                dg.generate(spec_cfg(path, **kw))
