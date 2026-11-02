"""Shared test fixtures: a small, well-separated two-class model."""
import copy

LATENT_DIM = 5   # 3 informative dims + 2 noise dims


def make_spec() -> dict:
    """
    Unlabelled : 2 components around (+5,0,0) [weight .75] and (-5,0,0) [weight .25]
    Class A    : 1 component around (0,5,0), proportion 0.7
    Class B    : 2 components around (0,-5,0) and (0,0,5), equal weights, proportion 0.3
    """
    return copy.deepcopy({
        "latent_dim": LATENT_DIM,
        "unlabelled": {"components": [
            {"name": "u0", "weight": 3, "mean": [5, 0, 0],
             "covariance": {"type": "diagonal", "variances": [1, 1, 1]}},
            {"name": "u1", "weight": 1, "mean": [-5, 0, 0],
             "covariance": {"type": "scalar", "variance": 0.5}},
        ]},
        "classes": [
            {"name": "A", "proportion": 7, "components": [
                {"name": "A0", "weight": 1, "mean": [0, 5, 0],
                 "covariance": {"type": "std_corr", "std": [1, 1, 1],
                                "correlation": [[1, .5, 0], [.5, 1, 0], [0, 0, 1]]}},
            ]},
            {"name": "B", "proportion": 3, "components": [
                {"name": "B0", "weight": 1, "mean": [0, -5, 0],
                 "covariance": {"type": "full",
                                "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
                {"name": "B1", "weight": 1, "mean": [0, 0, 5],
                 "covariance": {"type": "low_rank", "factor": [[.5], [0], [.5]],
                                "psi": [.5, .5, .5]}},
            ]},
        ],
    })
