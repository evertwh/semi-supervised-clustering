#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic_settings import SettingsConfigDict
from pydantic import Field

from data_generator import data_generator
from data_generator.MixtureConfig import MixtureConfig


class CliMixtureConfig(MixtureConfig):
    model_config = SettingsConfigDict(
        cli_parse_args=True,  # read sys.argv
        cli_kebab_case=True,  # --n-components instead of --n_components
        cli_implicit_flags=True,  # --standardise / --no-standardise
    )
    out: Path = Field(Path("output/synthetic_latents.npz"), description="Where to save the generated data (.npz)")
    plot: bool = Field(True, description="Generate Umap validation of generated data")
    plot_out: Optional[Path] = Field(None, description="Save the UMAP plot to this file instead of showing it")


def summarise(data: dict, out: Path) -> None:
    """Print a short summary of the generated data."""
    # x_u, x_l, y_l = data["x_unlabelled"], data["x_labelled"], data["y_labelled"]
    print(f"Saved {out}")
    print(f"Unlabelled: {data['x_unlabelled'].shape}   labelled: {data['x_labelled'].shape}")
    counts = np.bincount(data["y_labelled"], minlength=len(data["class_names"]))
    print("Labelled class counts:", {str(n): int(c) for n, c in zip(data["class_names"], counts)})
    print(f"Unlabelled per-dimension mean (max |.|): {np.abs(data['x_unlabelled'].mean(0)).max():.3f}, "
          f"std range: {data['x_unlabelled'].std(0).min():.2f} - {data['x_unlabelled'].std(0).max():.2f}")


def main():
    cfg = CliMixtureConfig()

    data = data_generator.generate(cfg)

    cfg.out.parent.mkdir(parents=True, exist_ok=True)  # np.savez does not create folders
    np.savez_compressed(cfg.out, **data)

    if cfg.plot:
        # imported here so that --no-plot does not need matplotlib / umap-learn
        from matplotlib import pyplot as plt
        from plot import plot_umap

        plot_umap.make_plot(data["x_unlabelled"], data["x_labelled"], data["y_labelled"])
        if cfg.plot_out:
            cfg.plot_out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(cfg.plot_out, dpi=200)
            print(f"Saved {cfg.plot_out}")
        else:
            plt.show()

        plt.show()

if __name__ == "__main__":
    main()
