"""
Pydantic-based configuration for data_generator

Requires pydantic >= 2 and pydantic-settings:
    pip install pydantic pydantic-settings

Values can come from (highest priority first):
    1. keyword arguments:        MixtureConfig(n_labelled=300, mixture_json="mixtures.json")
    2. environment variables:    SYN_N_LABELLED=300
    3. a .env file:              SYN_N_LABELLED=300
    4. the defaults below

The mixtures themselves (unlabelled and one per class) are defined in the JSON file;
see mixture_spec.py for its format. Complex values are given as JSON in env vars / .env, e.g.
    SYN_LABELLED_CLASS_PROBS='[0.4, 0.3, 0.2, 0.1]'
"""
from typing import Optional

from pydantic import Field, FilePath, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MixtureConfig(BaseSettings):
    """
    Settings for `data_generator.generate`.

    Values are read from keyword arguments, environment variables (prefix ``SYN_``), a
    ``.env`` file and the defaults below, in that order of priority. Unknown settings are
    rejected and instances are immutable.

    Attributes
    ----------
    mixture_json : pathlib.Path
        Required. Existing JSON file with the unlabelled mixture and one mixture per class
        (format: see ``mixture_spec``).
    n_unlabelled : int
        Number of unlabelled points to generate (default 30,000).
    n_labelled : int
        Number of labelled points to generate (default 600).
    labelled_class_probs : list of float, optional
        Class proportions of the labelled set in the order of the classes in the JSON file;
        overrides the proportions in the file. Entries must be non-negative with a positive
        sum; they are normalised when used. The length is checked against the model in
        `generate`.
    seed : int
        Seed of the random number generator (default 42).
    """

    model_config = SettingsConfigDict(
        env_prefix="SYN_",
        env_file="../../.env",
        env_file_encoding="utf-8",
        extra="forbid",  # typos in kwargs / .env raise instead of being ignored
        frozen=True,  # config is immutable once created
        validate_default=True,
    )

    mixture_json: FilePath = Field(
        description="JSON file specifying the unlabelled mixture and one mixture per class",
    )

    n_unlabelled: int = Field(30_000, gt=0)
    n_labelled: int = Field(600, gt=0)

    labelled_class_probs: Optional[list[float]] = Field(
        None,
        description="Class proportions of the labelled set, in the order of the classes in the "
                    "JSON file; None = use the proportions from the file (equal if it has none)",
    )

    seed: int = 42

    @field_validator("labelled_class_probs")
    @classmethod
    def _check_probs(cls, p):
        """
        Validate `labelled_class_probs`.

        Parameters
        ----------
        p : list of float or None
            The proposed class proportions.

        Returns
        -------
        list of float or None
            `p`, unchanged.

        Raises
        ------
        ValueError
            If any entry is negative or the sum is not positive.
        """
        if p is not None and (any(x < 0 for x in p) or sum(p) <= 0):
            raise ValueError("labelled_class_probs must be non-negative with a positive sum")
        return p
