import os
import shutil
from typing import Optional

import hydra
from omegaconf import DictConfig

import numpy as np

from src import utils
from src.datamodule.maze import sample_maze_trajectories, get_maze_env
from src.datamodule.dataset import get_transforms
from src.models.policies.maze import Dijkstra
from src.utils.eval_metrics import spl
from src.utils.visualization import data_gif

log = utils.get_logger(__name__)


def run_maze(cfg: DictConfig) -> Optional[float]:
    """
    Deploy a policy in a maze environment to compute success rate and SPL.

    Please refer to the class docstring or the yaml configs for model and datamodule arguments.

    Args:
        cfg: Config with all hyperparameters for training
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        utils.seed_experiment(cfg.seed)

    # Get environment name
    environment_name = cfg.sim_env

    # Provide logging directory for frames if we need to generate a gif
    path = 'temp' if cfg.gif_params.goal else False

    # Agent benchmark
    env = get_maze_env(maze_name=environment_name)
    agent_policy = hydra.utils.instantiate(cfg.policy)
    agent_policy.transform = get_transforms(keys=agent_policy.keys if hasattr(agent_policy, "keys") else None, resolution=cfg.resize)
    if hasattr(agent_policy, "is_oracle") and agent_policy.is_oracle:
        agent_policy.setup_sim(env)


    # Run simulation
    success, steps = sample_maze_trajectories(env, agent_policy,
                                              n_trajectories=cfg.test_params.n_trajectories,
                                              max_n_steps=cfg.test_params.max_steps,
                                              sample_start=cfg.test_params.sample_start,
                                              score=cfg.gif_params.score,
                                              path=path, save_last=True, stop_on_goal=False)

    # Oracle benchmark
    env = get_maze_env(maze_name=environment_name)
    policy = Dijkstra(env)
    _, dij_steps = sample_maze_trajectories(env, policy,
                                            n_trajectories=cfg.test_params.n_trajectories,
                                            max_n_steps=cfg.test_params.max_steps,
                                            sample_start=cfg.test_params.sample_start,
                                            score=False,
                                            path=False, stop_on_goal=True)

    # Report results
    spl_ = spl(success, steps, dij_steps)
    success_rate = np.mean(success)
    log.info(f"Success rate   : {success_rate:.4f}")
    log.info(f"PL             : {spl_/success_rate if success_rate > 0 else 0.0:.4f}")
    log.info(f"SPL            : {spl_:.4f}")
    if hasattr(agent_policy, 'duration'):
        log.info(f"Plan. freq. (amortized)  : {np.sum(steps)/agent_policy.duration:.2f} Hz")
        log.info(f"Plan. freq. (worst case) : {1/agent_policy.duration_max:.2f} Hz")

    if path:
        # Generate video
        if not os.path.exists('gifs'):
            os.mkdir('gifs')

        success = success if cfg.gif_params.failures else None

        if success is None or not success.all():
            data_gif(path, os.path.join('gifs', environment_name + '.mp4'),
                     cfg.policy._target_, success=success, duration=cfg.gif_params.duration, make_top_down=False)

        # Delete generated data
        shutil.rmtree(path)

    return spl_
