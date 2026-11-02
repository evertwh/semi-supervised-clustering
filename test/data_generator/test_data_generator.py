import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_generator.data_generator import UNLABELLED, generate, generate_points
from data_generator.mixture_spec import Mixture, load_model
from helpers import make_spec, LATENT_DIM


def single_component(mean, cov, latent_dim=LATENT_DIM):
    return Mixture(weights=np.array([1.0]), means=np.array([mean], dtype=float),
                   covs=np.array([cov], dtype=float), latent_dim=latent_dim)


class TestGeneratePoints(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.mixture = load_model_from_spec().unlabelled

    def test_shapes(self):
        X, y, z = generate_points(self.mixture, 50, self.rng)
        self.assertEqual(X.shape, (50, LATENT_DIM))
        self.assertEqual(y.shape, (50,))
        self.assertEqual(z.shape, (50,))

    def test_label_is_undefined_by_default(self):
        _, y, _ = generate_points(self.mixture, 20, self.rng)
        self.assertTrue(np.all(y == UNLABELLED))
        self.assertEqual(UNLABELLED, -1)

    def test_label_is_attached_to_every_point(self):
        _, y, _ = generate_points(self.mixture, 20, self.rng, label=3)
        self.assertTrue(np.all(y == 3))
        self.assertTrue(np.issubdtype(y.dtype, np.integer))

    def test_component_indices_in_range(self):
        _, _, z = generate_points(self.mixture, 500, self.rng)
        self.assertTrue(set(np.unique(z)) <= {0, 1})

    def test_zero_points(self):
        X, y, z = generate_points(self.mixture, 0, self.rng, label=1)
        self.assertEqual((X.shape, y.shape, z.shape), ((0, LATENT_DIM), (0,), (0,)))

    def test_component_frequencies_follow_weights(self):
        _, _, z = generate_points(self.mixture, 20_000, self.rng)
        np.testing.assert_allclose(np.bincount(z) / len(z), self.mixture.weights, atol=0.02)

    def test_points_follow_component_mean_and_covariance(self):
        cov = np.array([[1.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 0.5]])
        mixture = single_component([1.0, -2.0, 3.0], cov)
        X, _, _ = generate_points(mixture, 50_000, self.rng)
        np.testing.assert_allclose(X[:, :3].mean(axis=0), [1, -2, 3], atol=0.05)
        np.testing.assert_allclose(np.cov(X[:, :3], rowvar=False), cov, atol=0.06)

    def test_extra_dimensions_are_standard_normal_noise(self):
        mixture = single_component([10.0, 10.0, 10.0], np.eye(3))
        X, _, _ = generate_points(mixture, 20_000, self.rng)
        noise = X[:, 3:]
        np.testing.assert_allclose(noise.mean(axis=0), 0, atol=0.05)
        np.testing.assert_allclose(noise.std(axis=0), 1, atol=0.05)
        np.testing.assert_allclose(np.corrcoef(X[:, 2], X[:, 3])[0, 1], 0, atol=0.05)

    def test_no_noise_dimensions_when_mixture_covers_latent_space(self):
        mixture = single_component([0.0, 0.0, 0.0], np.eye(3), latent_dim=3)
        X, _, _ = generate_points(mixture, 10, self.rng)
        self.assertEqual(X.shape, (10, 3))

    def test_same_seed_same_points_different_seed_different_points(self):
        a = generate_points(self.mixture, 30, np.random.default_rng(5))
        b = generate_points(self.mixture, 30, np.random.default_rng(5))
        c = generate_points(self.mixture, 30, np.random.default_rng(6))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(a[0], c[0]))


def load_model_from_spec():
    from data_generator.mixture_spec import parse_model
    return parse_model(make_spec())


class TestGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "model.json"
        cls.path.write_text(json.dumps(make_spec()))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cfg(self, **overrides):
        values = dict(mixture_json=self.path, n_unlabelled=4000, n_labelled=4000,
                      labelled_class_probs=None, seed=1)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_output_shapes(self):
        d = generate(self.cfg(n_unlabelled=300, n_labelled=40))
        self.assertEqual(d["x_unlabelled"].shape, (300, LATENT_DIM))
        self.assertEqual(d["x_labelled"].shape, (40, LATENT_DIM))
        self.assertEqual(d["y_labelled"].shape, (40,))
        self.assertEqual(d["z_unlabelled_true"].shape, (300,))
        self.assertEqual(d["z_labelled_true"].shape, (40,))

    def test_unlabelled_points_have_undefined_label(self):
        d = generate(self.cfg(n_unlabelled=200, n_labelled=20))
        self.assertTrue(np.all(d["y_unlabelled"] == UNLABELLED))

    def test_labelled_points_have_valid_labels(self):
        d = generate(self.cfg())
        self.assertTrue(set(np.unique(d["y_labelled"])) == {0, 1})

    def test_class_proportions_come_from_json(self):
        y = generate(self.cfg())["y_labelled"]
        np.testing.assert_allclose(np.bincount(y) / len(y), [0.7, 0.3], atol=0.03)

    def test_config_overrides_class_proportions(self):
        y = generate(self.cfg(labelled_class_probs=[1, 1]))["y_labelled"]
        np.testing.assert_allclose(np.bincount(y) / len(y), [0.5, 0.5], atol=0.03)

    def test_override_can_exclude_a_class(self):
        y = generate(self.cfg(labelled_class_probs=[0, 1]))["y_labelled"]
        self.assertTrue(np.all(y == 1))

    def test_override_with_wrong_number_of_classes(self):
        with self.assertRaisesRegex(ValueError, "classes"):
            generate(self.cfg(labelled_class_probs=[1, 1, 1]))

    def test_labelled_set_is_shuffled(self):
        y = generate(self.cfg())["y_labelled"]
        self.assertTrue(np.any(np.diff(y) < 0), "labelled set is sorted by class")

    def test_unlabelled_points_follow_the_unlabelled_mixture(self):
        x = generate(self.cfg(n_unlabelled=20_000))["x_unlabelled"]
        np.testing.assert_allclose(x[:, :3].mean(axis=0), [2.5, 0, 0], atol=0.15)  # .75*5 + .25*-5

    def test_each_class_follows_its_own_mixture(self):
        d = generate(self.cfg(n_labelled=20_000))
        x, y = d["x_labelled"], d["y_labelled"]
        np.testing.assert_allclose(x[y == 0][:, :3].mean(axis=0), [0, 5, 0], atol=0.1)
        np.testing.assert_allclose(x[y == 1][:, :3].mean(axis=0), [0, -2.5, 2.5], atol=0.2)

    def test_component_ground_truth(self):
        d = generate(self.cfg())
        y, z = d["y_labelled"], d["z_labelled_true"]
        self.assertTrue(np.all(z[y == 0] == 0))                  # class A has one component
        self.assertEqual(set(np.unique(z[y == 1])), {0, 1})      # class B has two
        self.assertEqual(set(np.unique(d["z_unlabelled_true"])), {0, 1})

    def test_same_seed_reproduces_everything(self):
        a, b = generate(self.cfg(n_unlabelled=100, n_labelled=50)), generate(self.cfg(n_unlabelled=100, n_labelled=50))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_different_seed_gives_different_data(self):
        a, b = generate(self.cfg(seed=1)), generate(self.cfg(seed=2))
        self.assertFalse(np.array_equal(a["x_unlabelled"], b["x_unlabelled"]))

    def test_model_parameters_are_part_of_the_output(self):
        d = generate(self.cfg(n_unlabelled=10, n_labelled=10))
        for key in ("class_names", "class_probs", "unlabelled_weights", "unlabelled_means", "unlabelled_covs",
                    "class0_means", "class1_covs"):
            self.assertIn(key, d)

    def test_output_can_be_saved_and_reloaded_without_pickle(self):
        d = generate(self.cfg(n_unlabelled=50, n_labelled=20))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.npz"
            np.savez_compressed(out, **d)
            with np.load(out, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["x_labelled"], d["x_labelled"])
                self.assertEqual(list(loaded["class_names"]), ["A", "B"])

    def test_missing_json_file(self):
        with self.assertRaises(FileNotFoundError):
            generate(self.cfg(mixture_json=Path(self.tmp.name) / "nope.json"))


if __name__ == "__main__":
    unittest.main()
