import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf/", config_name="test_maze.yaml")
def main(cfg: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.test_maze import test_maze
    from src import utils

    # Applies optional utilities
    utils.extras(cfg)

    # Train model
    return test_maze(cfg)


if __name__ == "__main__":
    main()
