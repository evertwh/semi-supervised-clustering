"""
Pydantic-based configuration for data_generator

Requires pydantic >= 2 and pydantic-settings:
    pip install pydantic pydantic-settings

Values can come from (highest priority first):
    1. keyword arguments:        MixtureConfig(n_components=10)
    2. environment variables:    SYN_N_COMPONENTS=10
    3. a .env file:              SYN_N_COMPONENTS=10
    4. the defaults below

Complex values are given as JSON in env vars / .env, e.g.
    SYN_LABELLED_CLASS_PROBS='[0.4, 0.3, 0.2, 0.1]'
    SYN_PSI_RANGE='[0.3, 1.0]'
"""
from typing import Optional

from pydantic import Field, model_validator, FilePath, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MixtureConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SYN_",
        env_file="../../.env",
        env_file_encoding="utf-8",
        extra="forbid",  # typos in kwargs / .env raise instead of being ignored
        frozen=True,  # config is immutable once created
        validate_default=True,
    )

    mixture_json: FilePath | None = Field(
        default=None,
        description="Use this mixture specification instead of generating a random one",
    )

    @field_validator("mixture_json", mode="before")
    @classmethod
    def empty_to_none(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v

    # --- dimensions and sample sizes
    latent_dim: int = Field(100, gt=0, description="Total latent dimensionality")
    n_informative: int = Field(20, gt=0, description="Dims that carry cluster structure")
    n_components: int = Field(8, gt=0, description="Number of mixture components (K)")
    n_classes: int = Field(4, gt=1, description="Number of label categories")
    n_unlabelled: int = Field(30_000, gt=0)
    n_labelled: int = Field(600, gt=0)

    # --- mixture geometry
    mean_separation: float = Field(2.5, gt=0, description="Scale of component means")
    cov_rank: int = Field(4, gt=0, description="Rank of correlated part of each covariance")
    cov_scale: float = Field(0.8, gt=0, description="Scale of low-rank factors")
    psi_range: tuple[float, float] = Field((0.3, 1.0), description="Range of diagonal variances")
    weight_concentration: float = Field(
        3.0, gt=0, description="Dirichlet alpha for component weights (lower = more imbalance)"
    )
    standardise: bool = Field(True, description="Match aggregate mean 0 / covariance I")

    # --- labels
    class_purity: float = Field(
        0.8, ge=0.0, le=1.0, description="P(dominant class | component)"
    )
    labelled_class_probs: Optional[list[float]] = Field(
        None,
        description="Class proportions of the labelled set; None = no selection bias",
    )

    seed: int = 42

    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def _check_consistency(self) -> "MixtureConfig":

        if self.mixture_json is not None:
            # File exists (guaranteed by FilePath); skip all other checks.
            return self

        if self.n_informative > self.latent_dim:
            raise ValueError("n_informative must be <= latent_dim")

        lo, hi = self.psi_range
        if not (0 < lo <= hi):
            raise ValueError("psi_range must satisfy 0 < low <= high")

        if self.n_components < self.n_classes:
            raise ValueError(
                "n_components must be >= n_classes so every class can dominate a component"
            )

        p = self.labelled_class_probs
        if p is not None:
            if len(p) != self.n_classes:
                raise ValueError("labelled_class_probs must have length n_classes")
            if any(x < 0 for x in p) or sum(p) <= 0:
                raise ValueError("labelled_class_probs must be non-negative with a positive sum")
        return self


if __name__ == "__main__":
    print(MixtureConfig().model_dump_json(indent=2))
