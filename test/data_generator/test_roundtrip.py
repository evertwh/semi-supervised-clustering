"""End-to-end: data -> sklearn GaussianMixtures -> JSON -> new data that resembles the original."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_generator.data_generator import UNLABELLED, generate
from helpers import make_spec, LATENT_DIM

from sklearn.mixture import GaussianMixture

def make_cfg(model_path, n_unlabelled, n_labelled, seed):
    return SimpleNamespace(mixture_json=model_path, n_unlabelled=n_unlabelled,
                           n_labelled=n_labelled, labelled_class_probs=None, seed=seed)


class TestFitExportGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)

        # 1. "Real" data: stands in for the data that cannot be published.
        truth = tmp / "truth.json"
        truth.write_text(json.dumps(make_spec()))
        cls.real = generate(make_cfg(truth, n_unlabelled=8000, n_labelled=3000, seed=1))
        x_l, y_l = cls.real["x_labelled"], cls.real["y_labelled"]

        # 2. Fit one GMM to the unlabelled data and one per class.
        gmm = lambda k, X: GaussianMixture(k, covariance_type="full", n_init=3, random_state=0).fit(X)
        g_unlabelled = gmm(2, cls.real["x_unlabelled"])
        g_a, g_b = gmm(1, x_l[y_l == 0]), gmm(2, x_l[y_l == 1])

        # 3. Export them to the JSON model format.
        from data_generator.mixture_spec import export_sklearn_gmms
        cls.fitted = tmp / "fitted.json"
        export_sklearn_gmms(cls.fitted, latent_dim=LATENT_DIM, unlabelled=g_unlabelled,
                            classes={"A": g_a, "B": g_b},
                            class_proportions={"A": (y_l == 0).mean(), "B": (y_l == 1).mean()})

        # 4. Generate a new data set from the fitted model.
        cls.new = generate(make_cfg(cls.fitted, n_unlabelled=8000, n_labelled=3000, seed=2))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_new_data_has_the_requested_structure(self):
        new = self.new
        self.assertEqual(new["x_unlabelled"].shape, (8000, LATENT_DIM))
        self.assertEqual(new["x_labelled"].shape, (3000, LATENT_DIM))
        self.assertTrue(np.all(new["y_unlabelled"] == UNLABELLED))
        self.assertEqual(set(np.unique(new["y_labelled"])), {0, 1})
        self.assertEqual(list(new["class_names"]), ["A", "B"])

    def test_new_data_is_not_a_copy_of_the_original(self):
        self.assertFalse(np.array_equal(self.new["x_unlabelled"], self.real["x_unlabelled"]))

    def test_unlabelled_distribution_is_reproduced(self):
        old, new = self.real["x_unlabelled"], self.new["x_unlabelled"]
        np.testing.assert_allclose(new.mean(axis=0), old.mean(axis=0), atol=0.2)
        np.testing.assert_allclose(np.cov(new, rowvar=False), np.cov(old, rowvar=False), atol=0.5)

    def test_class_proportions_are_reproduced(self):
        old = np.bincount(self.real["y_labelled"]) / len(self.real["y_labelled"])
        new = np.bincount(self.new["y_labelled"]) / len(self.new["y_labelled"])
        np.testing.assert_allclose(new, old, atol=0.04)

    def test_each_class_distribution_is_reproduced(self):
        for c in (0, 1):
            with self.subTest(label=c):
                old = self.real["x_labelled"][self.real["y_labelled"] == c]
                new = self.new["x_labelled"][self.new["y_labelled"] == c]
                np.testing.assert_allclose(new.mean(axis=0), old.mean(axis=0), atol=0.25)
                np.testing.assert_allclose(np.cov(new, rowvar=False), np.cov(old, rowvar=False), atol=0.6)

    def test_classes_stay_distinguishable(self):
        x, y = self.new["x_labelled"], self.new["y_labelled"]
        gap = np.linalg.norm(x[y == 0].mean(axis=0) - x[y == 1].mean(axis=0))
        self.assertGreater(gap, 3.0)   # class means are several standard deviations apart


if __name__ == "__main__":
    unittest.main()
