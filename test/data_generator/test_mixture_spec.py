import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_generator.mixture_spec import (
    Component, Mixture, LatentModel, covariance_from_spec, parse_component, parse_mixture,
    parse_model, load_model, export_sklearn_gmms,
)
from helpers import make_spec, LATENT_DIM

BASE = Path(".")


def cov(spec, d=3):
    return covariance_from_spec(spec, d, BASE, "test")


class TestCovarianceTypes(unittest.TestCase):
    def test_scalar(self):
        np.testing.assert_allclose(cov({"type": "scalar", "variance": 2.0}), 2.0 * np.eye(3))

    def test_diagonal(self):
        np.testing.assert_allclose(cov({"type": "diagonal", "variances": [1, 2, 3]}), np.diag([1, 2, 3]))

    def test_low_rank(self):
        F = np.array([[1.0], [0.0], [2.0]])
        expected = F @ F.T + np.eye(3)
        spec = {"type": "low_rank", "factor": F.tolist(), "psi": [1, 1, 1]}
        np.testing.assert_allclose(cov(spec), expected)

    def test_std_corr(self):
        spec = {"type": "std_corr", "std": [1, 2], "correlation": [[1, .5], [.5, 1]]}
        np.testing.assert_allclose(cov(spec, d=2), [[1, 1], [1, 4]])

    def test_full(self):
        M = [[2, .5, 0], [.5, 1, 0], [0, 0, 1]]
        np.testing.assert_allclose(cov({"type": "full", "matrix": M}), M)

    def test_matrix_from_npy_file_relative_to_base_dir(self):
        M = np.array([[2.0, 0.3], [0.3, 1.0]])
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / "m.npy", M)
            result = covariance_from_spec({"type": "full", "matrix_file": "m.npy"}, 2, Path(tmp), "test")
        np.testing.assert_allclose(result, M)


class TestCovarianceValidation(unittest.TestCase):
    def assert_invalid(self, spec, d=3, pattern=None):
        with self.assertRaisesRegex(ValueError, pattern or ""):
            cov(spec, d)

    def test_unknown_type(self):
        self.assert_invalid({"type": "banana"})

    def test_missing_type(self):
        self.assert_invalid({"variance": 1})

    def test_wrong_shape(self):
        self.assert_invalid({"type": "diagonal", "variances": [1, 1]}, pattern="shape")

    def test_low_rank_wrong_number_of_rows(self):
        self.assert_invalid({"type": "low_rank", "factor": [[1], [1]], "psi": [1, 1, 1]}, pattern="rows")

    def test_wrong_ndim(self):
        self.assert_invalid({"type": "diagonal", "variances": [[1, 1, 1]]}, pattern="1-dimensional")

    def test_missing_array(self):
        self.assert_invalid({"type": "diagonal"}, pattern="variances")

    def test_asymmetric(self):
        self.assert_invalid({"type": "full", "matrix": [[1, .5, 0], [0, 1, 0], [0, 0, 1]]}, pattern="symmetric")

    def test_negative_variance_not_positive_definite(self):
        self.assert_invalid({"type": "scalar", "variance": -1}, pattern="positive definite")

    def test_singular_matrix_not_positive_definite(self):
        self.assert_invalid({"type": "full", "matrix": [[1, 1], [1, 1]]}, d=2, pattern="positive definite")


class TestComponent(unittest.TestCase):
    @staticmethod
    def make(**overrides):
        values = dict(weight=2.0, mean=np.zeros(3), cov=np.eye(3), name="c")
        values.update(overrides)
        return Component(**values)

    def test_valid_component(self):
        c = self.make()
        self.assertEqual(c.dimensionality, 3)
        self.assertEqual((c.weight, c.name), (2.0, "c"))

    def test_name_is_optional(self):
        self.assertEqual(Component(1.0, np.zeros(2), np.eye(2)).name, "")

    def test_invalid_weights(self):
        for w in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(weight=w), self.assertRaisesRegex(ValueError, "weight"):
                self.make(weight=w)

    def test_mean_must_be_a_non_empty_vector(self):
        for mean in (np.zeros((2, 2)), np.zeros(0)):
            with self.subTest(shape=mean.shape), self.assertRaisesRegex(ValueError, "mean"):
                self.make(mean=mean, cov=np.eye(2))

    def test_mean_must_be_finite(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.make(mean=np.array([0.0, np.nan, 0.0]))

    def test_covariance_shape_must_match_mean(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.make(cov=np.eye(2))

    def test_is_immutable(self):
        with self.assertRaises(Exception):
            self.make().weight = 5.0


class TestParseComponent(unittest.TestCase):
    @staticmethod
    def spec(**overrides):
        values = {"name": "c0", "weight": 2, "mean": [1, 2],
                  "covariance": {"type": "scalar", "variance": 3}}
        values.update(overrides)
        return values

    def parse(self, spec):
        return parse_component(spec, BASE, "unlabelled, component 0")

    def test_builds_component(self):
        c = self.parse(self.spec())
        self.assertIsInstance(c, Component)
        self.assertEqual((c.weight, c.name), (2.0, "c0"))
        np.testing.assert_allclose(c.mean, [1, 2])
        np.testing.assert_allclose(c.cov, 3 * np.eye(2))

    def test_dimensionality_comes_from_the_mean(self):
        c = self.parse(self.spec(mean=[1, 2, 3, 4]))
        self.assertEqual(c.cov.shape, (4, 4))

    def test_name_is_optional(self):
        spec = self.spec()
        del spec["name"]
        self.assertEqual(self.parse(spec).name, "")


    def test_mean_must_be_a_flat_list(self):
        with self.assertRaisesRegex(ValueError, "list of numbers"):
            self.parse(self.spec(mean=[[1, 2]]))

    def test_errors_name_the_location_and_component(self):
        with self.assertRaisesRegex(ValueError, r"unlabelled, component 0 \(c0\)"):
            self.parse(self.spec(weight=-1))
        with self.assertRaisesRegex(ValueError, r"unlabelled, component 0 \(c0\)"):
            self.parse(self.spec(covariance={"type": "scalar", "variance": -1}))


class TestMixtureFromComponents(unittest.TestCase):
    @staticmethod
    def comp(weight, mean):
        return Component(weight, np.array(mean, dtype=float), np.eye(len(mean)))

    def test_stacks_and_normalises(self):
        m = Mixture.from_components([self.comp(1, [0, 0]), self.comp(3, [1, 1])], latent_dim=4)
        np.testing.assert_allclose(m.weights, [0.25, 0.75])
        np.testing.assert_allclose(m.means, [[0, 0], [1, 1]])
        self.assertEqual(m.covs.shape, (2, 2, 2))
        self.assertEqual((m.n_components, m.dimensionality, m.latent_dim), (2, 2, 4))

    def test_empty(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            Mixture.from_components([], latent_dim=3)

    def test_mixed_dimensionality(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            Mixture.from_components([self.comp(1, [0, 0]), self.comp(1, [0, 0, 0])], latent_dim=5)

    def test_latent_dim_too_small(self):
        with self.assertRaisesRegex(ValueError, "exceeds latent_dim"):
            Mixture.from_components([self.comp(1, [0, 0, 0])], latent_dim=2)

    def test_where_appears_in_errors(self):
        with self.assertRaisesRegex(ValueError, "class 'A'"):
            Mixture.from_components([], latent_dim=3, where="class 'A'")


class TestParseMixture(unittest.TestCase):
    @staticmethod
    def comps():
        return make_spec()["unlabelled"]["components"]

    def parse(self, components, latent_dim=LATENT_DIM):
        return parse_mixture(components, latent_dim, BASE, "unlabelled")

    def test_shapes_and_normalised_weights(self):
        m = self.parse(self.comps())
        self.assertEqual(m.means.shape, (2, 3))
        self.assertEqual(m.covs.shape, (2, 3, 3))
        self.assertEqual((m.n_components, m.dimensionality, m.latent_dim), (2, 3, LATENT_DIM))
        np.testing.assert_allclose(m.weights, [0.75, 0.25])

    def test_mean_dimension_may_equal_latent_dim(self):
        self.assertEqual(self.parse(self.comps(), latent_dim=3).dimensionality, 3)

    def test_mean_longer_than_latent_dim(self):
        with self.assertRaisesRegex(ValueError, "exceeds latent_dim"):
            self.parse(self.comps(), latent_dim=2)

    def test_means_of_different_length(self):
        comps = self.comps()
        comps[1]["mean"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "same length"):
            self.parse(comps)

    def test_non_positive_weight(self):
        for w in (0, -1):
            with self.subTest(weight=w):
                comps = self.comps()
                comps[0]["weight"] = w
                with self.assertRaisesRegex(ValueError, "weight"):
                    self.parse(comps)

    def test_empty_components(self):
        for empty in ([], None):
            with self.subTest(components=empty):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    self.parse(empty)

    def test_covariance_error_names_the_component(self):
        comps = self.comps()
        comps[1]["name"] = "broken_one"
        comps[1]["covariance"] = {"type": "scalar", "variance": -1}
        with self.assertRaisesRegex(ValueError, "broken_one"):
            self.parse(comps)


class TestParseModel(unittest.TestCase):
    def test_structure(self):
        m = parse_model(make_spec())
        self.assertEqual(m.latent_dim, LATENT_DIM)
        self.assertEqual(m.n_classes, 2)
        self.assertEqual(m.class_names, ["A", "B"])
        self.assertEqual([c.n_components for c in m.class_mixtures], [1, 2])
        self.assertEqual(m.unlabelled.n_components, 2)

    def test_class_proportions_are_normalised(self):
        np.testing.assert_allclose(parse_model(make_spec()).class_probs, [0.7, 0.3])

    def test_missing_proportions_mean_equal_classes(self):
        spec = make_spec()
        for c in spec["classes"]:
            del c["proportion"]
        np.testing.assert_allclose(parse_model(spec).class_probs, [0.5, 0.5])

    def test_proportions_for_some_classes_only(self):
        spec = make_spec()
        del spec["classes"][0]["proportion"]
        with self.assertRaisesRegex(ValueError, "all classes or for none"):
            parse_model(spec)

    def test_non_positive_proportion(self):
        spec = make_spec()
        spec["classes"][0]["proportion"] = 0
        with self.assertRaisesRegex(ValueError, "proportions"):
            parse_model(spec)

    def test_needs_at_least_two_classes(self):
        spec = make_spec()
        spec["classes"] = spec["classes"][:1]
        with self.assertRaisesRegex(ValueError, "at least two"):
            parse_model(spec)

    def test_duplicate_class_names(self):
        spec = make_spec()
        spec["classes"][1]["name"] = "A"
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_model(spec)

    def test_missing_unlabelled_section(self):
        spec = make_spec()
        del spec["unlabelled"]
        with self.assertRaises(KeyError):
            parse_model(spec)

    def test_to_arrays(self):
        arrays = parse_model(make_spec()).to_arrays()
        self.assertEqual(list(arrays["class_names"]), ["A", "B"])
        self.assertEqual(arrays["unlabelled_means"].shape, (2, 3))
        self.assertEqual(arrays["class0_covs"].shape, (1, 3, 3))
        self.assertEqual(arrays["class1_weights"].shape, (2,))
        np.testing.assert_allclose(arrays["class_probs"], [0.7, 0.3])

    def test_latent_model_rejects_mixed_latent_dims(self):
        m = parse_model(make_spec())
        other = Mixture(m.unlabelled.weights, m.unlabelled.means, m.unlabelled.covs, latent_dim=99)
        with self.assertRaisesRegex(ValueError, "latent_dim"):
            LatentModel(other, m.class_mixtures, m.class_names, m.class_probs)


class TestLoadModel(unittest.TestCase):
    def test_load_from_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(json.dumps(make_spec()))

            self.assertEqual(load_model(path).class_names, ["A", "B"])

    def test_npy_references_are_relative_to_the_json_file(self):
        spec = make_spec()
        spec["unlabelled"]["components"][0]["covariance"] = {"type": "full", "matrix_file": "cov0.npy"}
        M = np.diag([2.0, 3.0, 4.0])
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / "cov0.npy", M)
            (Path(tmp) / "model.json").write_text(json.dumps(spec))
            model = load_model(Path(tmp) / "model.json")
        np.testing.assert_allclose(model.unlabelled.covs[0], M)


try:
    from sklearn.mixture import GaussianMixture
except ImportError:  # pragma: no cover
    GaussianMixture = None


@unittest.skipIf(GaussianMixture is None, "scikit-learn not installed")
class TestExportSklearn(unittest.TestCase):
    @staticmethod
    def expected_covariances(gmm):
        kind = gmm.covariance_type
        if kind == "full":
            return gmm.covariances_
        if kind == "diag":
            return np.array([np.diag(v) for v in gmm.covariances_])
        if kind == "spherical":
            return np.array([v * np.eye(gmm.means_.shape[1]) for v in gmm.covariances_])
        return np.array([gmm.covariances_] * gmm.n_components)  # tied

    def test_round_trip_for_every_covariance_type(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 3))
        for kind in ("full", "diag", "spherical", "tied"):
            with self.subTest(covariance_type=kind), tempfile.TemporaryDirectory() as tmp:
                g_u = GaussianMixture(2, covariance_type=kind, random_state=0).fit(X)
                g_a = GaussianMixture(1, covariance_type=kind, random_state=0).fit(X[:100])
                g_b = GaussianMixture(2, covariance_type=kind, random_state=0).fit(X[100:])
                path = Path(tmp) / "fitted.json"
                export_sklearn_gmms(path, latent_dim=LATENT_DIM, unlabelled=g_u,
                                    classes={"A": g_a, "B": g_b},
                                    class_proportions={"A": 0.4, "B": 0.6})
                model = load_model(path)

                self.assertEqual(model.latent_dim, LATENT_DIM)
                self.assertEqual(model.class_names, ["A", "B"])
                np.testing.assert_allclose(model.class_probs, [0.4, 0.6])
                for mixture, gmm in [(model.unlabelled, g_u), (model.class_mixtures[0], g_a),
                                     (model.class_mixtures[1], g_b)]:
                    np.testing.assert_allclose(mixture.weights, gmm.weights_, atol=1e-5)
                    np.testing.assert_allclose(mixture.means, gmm.means_, atol=1e-5)
                    np.testing.assert_allclose(mixture.covs, self.expected_covariances(gmm), atol=1e-5)

    def test_proportions_are_optional(self):
        X = np.random.default_rng(0).normal(size=(200, 3))
        g = GaussianMixture(1, random_state=0).fit(X)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fitted.json"
            export_sklearn_gmms(path, latent_dim=3, unlabelled=g, classes={"A": g, "B": g})
            np.testing.assert_allclose(load_model(path).class_probs, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
