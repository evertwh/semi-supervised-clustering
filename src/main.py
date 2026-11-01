#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from pydantic_settings import SettingsConfigDict
from pydantic import Field

from data_generator import data_generator
from data_generator.MixtureConfig import MixtureConfig
from plot import plot_umap


class CliMixtureConfig(MixtureConfig):
    model_config = SettingsConfigDict(
        cli_parse_args=True,  # read sys.argv
        cli_kebab_case=True,  # --n-components instead of --n_components
        cli_implicit_flags=True,  # --standardise / --no-standardise
    )
    out: str = "output/synthetic_latents.npz"
    plot: bool = Field(True, description="Generate Umap validation of generated data")


def main():
    cfg = CliMixtureConfig()

    data = data_generator.generate(cfg)
    np.savez_compressed(cfg.out, **data)

    # Quick sanity checks
    Xu = data["x_unlabelled"]
    print(f"Saved {cfg.out}")
    print(f"Unlabelled: {Xu.shape}, labelled: {data['x_labelled'].shape}")
    print(f"Aggregate mean (max |.|): {np.abs(Xu.mean(0)).max():.3f}")
    print(f"Aggregate std range: {Xu.std(0).min():.2f} - {Xu.std(0).max():.2f}")
    print("Labelled class counts:", np.bincount(data["y_labelled"], minlength=cfg.n_classes))
    print("Component weights:", np.round(data["mixture_weights"], 3))

    if cfg.plot:
        ax = plot_umap.make_plot(data["x_unlabelled"], data["x_labelled"],data["y_labelled"])
        plt.show()

        save_plot = False
        if save_plot:
            plt.savefig(TBD, dpi=600)
            print(f"Saved {args.out}")

        plt.show()

if __name__ == "__main__":
    main()
