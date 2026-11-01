"""Tests for mixture_spec: covariance readers, parse/load, generate_mixture, sklearn export."""
import json
import unittest
from pathlib import Path

import numpy as np

from data_generator import mixture_spec as ms
from helpers import EnvIsolatedTestCase, make_cfg, spec_dict, write_spec, SPEC_WEIGHTS

try:
    from sklearn.mixture import GaussianMixture
except ImportError:  # pragma: no cover
    GaussianMixture = None

BASE = Path(".")


def cov_from(spec, d=3, base=BASE):
    return ms.covariance_from_spec(spec, d, base, "test")


# --------------------------------------------------------------------------- #
# _read_array
# --------------------------------------------------------------------------- #
class TestReadArray(EnvIsolatedTestCase):
    def test_inline(self):
        a = ms._read_array({"m": [[1, 2], [3, 4]]}, "m", BASE, 2, "w")
        np.testing.assert_array_equal(a, [[1, 2], [3, 4]])
        self.assertEqual(a.dtype, float)

    def test_from_npy_file_relative_to_base_dir(self):
        d = self.tmp_dir()
        np.save(d / "m.npy", np.arange(6).reshape(2, 3))
        a = ms._read_array({"m_file": "m.npy"}, "m", d, 2, "w")
        np.testing.assert_array_equal(a, np.arange(6).reshape(2, 3))
        self.assertEqual(a.dtype, float)

    def test_inline_takes_precedence_over_file(self):
        a = ms._read_array({"m": [1.0], "m_file": "does_not_exist.npy"}, "m", BASE, 1, "w")
        np.testing.assert_array_equal(a, [1.0])

    def test_missing_key_raises_with_location(self):
        with self.assertRaisesRegex(ValueError, r"component 3.*need 'm' or 'm_file'"):
            ms._read_array({}, "m", BASE, 1, "component 3")

    def test_wrong_ndim_raises(self):
        with self.assertRaisesRegex(ValueError, "must be 2-dimensional"):
            ms._read_array({"m": [1, 2, 3]}, "m", BASE, 2, "w")


# --------------------------------------------------------------------------- #
# covariance_from_spec
# --------------------------------------------------------------------------- #
class TestCovarianceTypes(EnvIsolatedTestCase):
    def test_enum_values(self):
        self.assertEqual({t.value for t in ms.CovarianceType},
                         {"scalar", "diagonal", "low_rank", "std_corr", "full"})

    def test_every_enum_member_has_a_reader(self):
        self.assertEqual(set(ms.COVARIANCE_BUILDERS), set(ms.CovarianceType))

    def test_scalar(self):
        np.testing.assert_allclose(cov_from({"type": "scalar", "variance": 1.5}), 1.5 * np.eye(3))

    def test_diagonal(self):
        np.testing.assert_allclose(cov_from({"type": "diagonal", "variances": [1, 2, 3]}),
                                   np.diag([1, 2, 3]))

    def test_low_rank(self):
        F = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]])
        psi = np.array([0.1, 0.2, 0.3])
        got = cov_from({"type": "low_rank", "factor": F.tolist(), "psi": psi.tolist()})
        np.testing.assert_allclose(got, F @ F.T + np.diag(psi))

    def test_std_corr(self):
        std = np.array([1.0, 2.0, 3.0])
        R = np.array([[1, .5, .2], [.5, 1, .3], [.2, .3, 1]])
        got = cov_from({"type": "std_corr", "std": std.tolist(), "correlation": R.tolist()})
        np.testing.assert_allclose(got, np.diag(std) @ R @ np.diag(std))

    def test_full(self):
        M = np.array([[2, .5, 0], [.5, 2, .5], [0, .5, 2]])
        np.testing.assert_allclose(cov_from({"type": "full", "matrix": M.tolist()}), M)

    def test_full_and_factor_from_files(self):
        d = self.tmp_dir()
        M = np.array([[2, .5, 0], [.5, 2, .5], [0, .5, 2]])
        F = np.array([[1.0], [0.5], [0.0]])
        np.save(d / "M.npy", M)
        np.save(d / "F.npy", F)
        np.testing.assert_allclose(cov_from({"type": "full", "matrix_file": "M.npy"}, base=d), M)
        got = cov_from({"type": "low_rank", "factor_file": "F.npy", "psi": [1, 1, 1]}, base=d)
        np.testing.assert_allclose(got, F @ F.T + np.eye(3))

    def test_result_is_symmetric_and_cholesky_able(self):
        got = cov_from({"type": "std_corr", "std": [1, 2, 3],
                        "correlation": [[1, .5, .2], [.5, 1, .3], [.2, .3, 1]]})
        np.testing.assert_array_equal(got, got.T)
        np.linalg.cholesky(got)

    def test_tiny_asymmetry_is_tolerated_and_removed(self):
        M = 2 * np.eye(3)
        M[0, 1] = 1e-10
        got = cov_from({"type": "full", "matrix": M.tolist()})
        np.testing.assert_array_equal(got, got.T)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            cov_from({"type": "banana"})

    def test_missing_type_raises(self):
        with self.assertRaises(ValueError):
            cov_from({"variance": 1.0})

    def test_wrong_shape_raises(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            cov_from({"type": "diagonal", "variances": [1, 2]})
        with self.assertRaisesRegex(ValueError, "expected"):
            cov_from({"type": "full", "matrix": np.eye(4).tolist()})

    def test_low_rank_factor_row_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "rows"):
            cov_from({"type": "low_rank", "factor": np.ones((2, 1)).tolist(), "psi": [1, 1, 1]})

    def test_asymmetric_matrix_raises(self):
        M = [[2, 1, 0], [0, 2, 0], [0, 0, 2]]
        with self.assertRaisesRegex(ValueError, "not symmetric"):
            cov_from({"type": "full", "matrix": M})

    def test_not_positive_definite_raises(self):
        with self.assertRaisesRegex(ValueError, "not positive definite"):
            cov_from({"type": "full", "matrix": [[1, 2, 0], [2, 1, 0], [0, 0, 1]]})

    def test_zero_or_negative_variance_raises(self):
        for v in (0.0, -1.0):
            with self.subTest(variance=v), self.assertRaisesRegex(ValueError, "positive definite"):
                cov_from({"type": "scalar", "variance": v})

    def test_error_message_names_the_component(self):
        with self.assertRaisesRegex(ValueError, "component 2 .*not positive definite"):
            ms.covariance_from_spec({"type": "scalar", "variance": -1}, 3, BASE, "component 2 (x)")


# --------------------------------------------------------------------------- #
# parse_mixture / load_mixture
# --------------------------------------------------------------------------- #
class TestParseMixture(EnvIsolatedTestCase):
    def test_returns_expected_structure(self):
        m = ms.parse_mixture(spec_dict())
        self.assertEqual(set(m), {"latent_dim", "weights", "means", "covs",
                                  "label_given_component", "classes"})
        self.assertEqual(m["latent_dim"], 6)
        self.assertEqual(m["classes"], ["A", "B"])
        self.assertEqual(m["means"].shape, (3, 4))
        self.assertEqual(m["covs"].shape, (3, 4, 4))
        self.assertEqual(m["label_given_component"].shape, (3, 2))

    def test_weights_are_normalised(self):
        m = ms.parse_mixture(spec_dict(weights=(2, 1, 1)))
        np.testing.assert_allclose(m["weights"], [0.5, 0.25, 0.25])

    def test_label_probs_rounding_error_is_removed(self):
        lp = ((0.8, 0.19999), (0.5, 0.5), (0.3, 0.7))
        m = ms.parse_mixture(spec_dict(label_probs=lp))
        np.testing.assert_allclose(m["label_given_component"].sum(axis=1), 1, atol=1e-12)

    def test_classes_are_optional(self):
        spec = spec_dict()
        del spec["classes"]
        self.assertIsNone(ms.parse_mixture(spec)["classes"])

    def test_mixed_covariance_types(self):
        covs = [{"type": "scalar", "variance": 1.0},
                {"type": "diagonal", "variances": [1, 2, 3, 4]},
                {"type": "full", "matrix": (2 * np.eye(4)).tolist()}]
        m = ms.parse_mixture(spec_dict(covariances=covs))
        np.testing.assert_allclose(m["covs"][1], np.diag([1, 2, 3, 4]))

    def test_files_resolved_relative_to_base_dir(self):
        d = self.tmp_dir()
        np.save(d / "S.npy", 3 * np.eye(4))
        covs = [{"type": "full", "matrix_file": "S.npy"}] + [{"type": "scalar", "variance": 1}] * 2
        m = ms.parse_mixture(spec_dict(covariances=covs), base_dir=d)
        np.testing.assert_allclose(m["covs"][0], 3 * np.eye(4))

    def test_empty_or_missing_components_raise(self):
        for comps in ([], None):
            with self.subTest(components=comps), self.assertRaisesRegex(ValueError, "non-empty"):
                ms.parse_mixture({"latent_dim": 4, "components": comps})

    def test_mean_longer_than_latent_dim_raises(self):
        spec = spec_dict()
        spec["latent_dim"] = 3
        with self.assertRaisesRegex(ValueError, "exceeds latent_dim"):
            ms.parse_mixture(spec)

    def test_non_positive_weight_raises(self):
        for w in (0, -1):
            with self.subTest(weight=w), self.assertRaisesRegex(ValueError, "weights"):
                ms.parse_mixture(spec_dict(weights=(w, 1, 1)))

    def test_means_of_different_length_raise(self):
        spec = spec_dict()
        spec["components"][1]["mean"] = [0.0, 0.0]
        with self.assertRaises(ValueError):
            ms.parse_mixture(spec)

    def test_missing_label_probs_raise(self):
        spec = spec_dict()
        del spec["components"][0]["label_probs"]
        with self.assertRaisesRegex(ValueError, "label_probs"):
            ms.parse_mixture(spec)

    def test_label_probs_length_must_match_classes(self):
        spec = spec_dict()
        spec["classes"] = ["A", "B", "C"]
        with self.assertRaisesRegex(ValueError, "number of classes"):
            ms.parse_mixture(spec)

    def test_negative_label_probs_raise(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ms.parse_mixture(spec_dict(label_probs=((1.2, -0.2), (.5, .5), (.5, .5))))

    def test_label_probs_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            ms.parse_mixture(spec_dict(label_probs=((0.5, 0.2), (.5, .5), (.5, .5))))

    def test_invalid_covariance_is_reported_with_component_name(self):
        covs = [{"type": "scalar", "variance": 1}] * 2 + [{"type": "scalar", "variance": -1}]
        with self.assertRaisesRegex(ValueError, r"component 2 \(c2\)"):
            ms.parse_mixture(spec_dict(covariances=covs))


class TestLoadMixture(EnvIsolatedTestCase):
    def test_load_from_json_path_and_str_path(self):
        path = write_spec(self.tmp_dir(), spec_dict())
        for p in (path, str(path)):
            with self.subTest(type=type(p).__name__):
                np.testing.assert_allclose(ms.load_mixture(p)["weights"], SPEC_WEIGHTS)

    def test_npy_files_are_relative_to_the_json_file(self):
        d = self.tmp_dir()
        np.save(d / "S.npy", 3 * np.eye(4))
        covs = [{"type": "full", "matrix_file": "S.npy"}] + [{"type": "scalar", "variance": 1}] * 2
        path = write_spec(d, spec_dict(covariances=covs))
        np.testing.assert_allclose(ms.load_mixture(path)["covs"][0], 3 * np.eye(4))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            ms.load_mixture(self.tmp_dir() / "nope.json")


# --------------------------------------------------------------------------- #
# generate_mixture
# --------------------------------------------------------------------------- #
class TestGenerateMixture(EnvIsolatedTestCase):
    def gen(self, seed=0, **kw):
        cfg = make_cfg(**kw)
        return cfg, ms.generate_mixture(cfg, np.random.default_rng(seed))

    def test_shapes_and_normalisation(self):
        for K, C in [(4, 2), (8, 4), (5, 3), (3, 3)]:
            with self.subTest(K=K, C=C):
                cfg, m = self.gen(n_components=K, n_classes=C)
                self.assertEqual(m["weights"].shape, (K,))
                self.assertEqual(m["means"].shape, (K, cfg.n_informative))
                self.assertEqual(m["covs"].shape, (K, cfg.n_informative, cfg.n_informative))
                self.assertEqual(m["label_given_component"].shape, (K, C))
                self.assertAlmostEqual(m["weights"].sum(), 1.0)
                self.assertTrue((m["weights"] > 0).all())
                np.testing.assert_allclose(m["label_given_component"].sum(axis=1), 1.0)
                self.assertTrue((m["label_given_component"] >= 0).all())

    def test_reproducible(self):
        _, a = self.gen(seed=5)
        _, b = self.gen(seed=5)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_different_rng_seed_changes_mixture(self):
        _, a = self.gen(seed=1)
        _, b = self.gen(seed=2)
        self.assertFalse(np.allclose(a["means"], b["means"]))

    def test_covariances_are_symmetric_positive_definite(self):
        for standardise in (True, False):
            with self.subTest(standardise=standardise):
                _, m = self.gen(standardise=standardise)
                for S in m["covs"]:
                    np.testing.assert_allclose(S, S.T, atol=1e-10)
                    np.linalg.cholesky(S)

    def test_unstandardised_covariance_eigenvalues_bounded_below_by_psi(self):
        _, m = self.gen(standardise=False, psi_range=(0.3, 1.0))
        for S in m["covs"]:
            self.assertGreaterEqual(np.linalg.eigvalsh(S).min(), 0.3 - 1e-9)

    def test_class_purity_and_every_class_dominates_a_component(self):
        for K, C in [(4, 2), (8, 4), (5, 3), (3, 3)]:
            with self.subTest(K=K, C=C):
                _, m = self.gen(n_components=K, n_classes=C, class_purity=0.8)
                P = m["label_given_component"]
                np.testing.assert_allclose(P.max(axis=1), 0.8)
                self.assertEqual(set(P.argmax(axis=1)), set(range(C)))

    def test_pure_components(self):
        _, m = self.gen(class_purity=1.0)
        np.testing.assert_allclose(np.sort(m["label_given_component"], axis=1)[:, -1], 1.0)
        np.testing.assert_allclose(m["label_given_component"].sum(axis=1), 1.0)

    def test_standardise_whitens_aggregate_moments(self):
        _, m = self.gen(standardise=True)
        w, mu_k, S = m["weights"], m["means"], m["covs"]
        mu = w @ mu_k
        total = sum(wk * (Sk + np.outer(mk, mk)) for wk, Sk, mk in zip(w, S, mu_k)) - np.outer(mu, mu)
        np.testing.assert_allclose(mu, 0, atol=1e-8)
        np.testing.assert_allclose(total, np.eye(mu_k.shape[1]), atol=1e-8)

    def test_standardise_does_not_change_weights_or_label_probs(self):
        _, a = self.gen(standardise=True)
        _, b = self.gen(standardise=False)
        np.testing.assert_array_equal(a["weights"], b["weights"])
        np.testing.assert_array_equal(a["label_given_component"], b["label_given_component"])

    def test_mean_separation_scales_unstandardised_means(self):
        _, small = self.gen(standardise=False, mean_separation=0.5)
        _, large = self.gen(standardise=False, mean_separation=5.0)
        np.testing.assert_allclose(large["means"], 10 * small["means"])

    def test_lower_weight_concentration_gives_more_imbalance(self):
        spread = {}
        for conc in (0.2, 50.0):
            spread[conc] = np.mean([self.gen(seed=s, weight_concentration=conc)[1]["weights"].std()
                                    for s in range(30)])
        self.assertGreater(spread[0.2], spread[50.0])


# --------------------------------------------------------------------------- #
# export_sklearn_gmm  (round trip through load_mixture)
# --------------------------------------------------------------------------- #
@unittest.skipIf(GaussianMixture is None, "scikit-learn not installed")
class TestExportSklearnGmm(EnvIsolatedTestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.X = np.vstack([rng.normal(6.0 * k, 1.0 + 0.2 * k, size=(300, 3)) for k in range(3)])

    def fit(self, covariance_type):
        return GaussianMixture(3, covariance_type=covariance_type, random_state=0).fit(self.X)

    def roundtrip(self, gmm, **kw):
        path = self.tmp_dir() / "gmm.json"
        kw.setdefault("classes", ["A", "B"])
        ms.export_sklearn_gmm(gmm, path, **kw)
        return ms.load_mixture(path)

    @staticmethod
    def expected_covs(gmm):
        K, d = gmm.means_.shape
        ct = gmm.covariance_type
        if ct == "full":
            return gmm.covariances_
        if ct == "diag":
            return np.array([np.diag(c) for c in gmm.covariances_])
        if ct == "spherical":
            return np.array([c * np.eye(d) for c in gmm.covariances_])
        return np.array([gmm.covariances_] * K)  # tied

    def test_roundtrip_for_every_covariance_type(self):
        for ct in ("full", "diag", "spherical", "tied"):
            with self.subTest(covariance_type=ct):
                gmm = self.fit(ct)
                m = self.roundtrip(gmm)
                np.testing.assert_allclose(m["weights"], gmm.weights_, atol=1e-5)
                np.testing.assert_allclose(m["means"], gmm.means_, atol=1e-5)
                np.testing.assert_allclose(m["covs"], self.expected_covs(gmm), atol=1e-5)

    def test_default_label_probs_are_uniform(self):
        m = self.roundtrip(self.fit("full"), classes=["A", "B", "C"])
        np.testing.assert_allclose(m["label_given_component"], 1 / 3)
        self.assertEqual(m["classes"], ["A", "B", "C"])

    def test_explicit_label_probs_are_preserved(self):
        lp = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
        m = self.roundtrip(self.fit("full"), label_probs=lp)
        np.testing.assert_allclose(m["label_given_component"], lp)

    def test_latent_dim_argument(self):
        self.assertEqual(self.roundtrip(self.fit("full"), latent_dim=7)["latent_dim"], 7)
        self.assertEqual(self.roundtrip(self.fit("full"))["latent_dim"], 3)

    def test_label_probs_without_classes(self):
        lp = np.full((3, 2), 0.5)
        path = self.tmp_dir() / "g.json"
        ms.export_sklearn_gmm(self.fit("full"), path, label_probs=lp)
        self.assertIsNone(json.loads(path.read_text())["classes"])
        self.assertIsNone(ms.load_mixture(path)["classes"])

    def test_requires_label_probs_or_classes(self):
        with self.assertRaises(ValueError):
            ms.export_sklearn_gmm(self.fit("full"), self.tmp_dir() / "g.json")


if __name__ == "__main__":
    unittest.main()
