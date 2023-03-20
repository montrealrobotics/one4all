import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf/", config_name="run_habitat.yaml")
def main(cfg: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.run_habitat import run_habitat
    from src import utils

    # Applies optional utilities
    utils.extras(cfg)

    # Train model
    return run_habitat(cfg)


if __name__ == "__main__":
    main()
