import os
import shutil
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import habitat
import habitat_sim
from habitat.utils.visualizations import maps
from habitat.core.utils import try_cv2_import
from PIL import ImageOps, Image

from scipy.spatial.distance import cdist

from src import utils
from src.utils.visualization import data_gif

from src.datamodule.dataset import get_transforms

from src.utils.habitat import get_panorama_image, get_step_dict, SE2Grid, \
    get_goal_image_state, visualize_map, depth_to_range_bearing
from src.utils.split_dataset import DatasetWriter, split_data

from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode

goal_rng = np.random.RandomState(12345)  # Only used to sample goal rotations
log = utils.get_logger(__name__)

cv2 = try_cv2_import()


def inflate_agent(env, radius=0.25):
    """Change radius of agent and alter the navmesh accordingly."""
    navmesh_settings = env.sim.pathfinder.nav_mesh_settings
    navmesh_settings.agent_radius = radius  # @param {type:"slider", min:0.01, max:0.5, step:0.01}
    env.sim.recompute_navmesh(
        env.sim.pathfinder, navmesh_settings, include_static_objects=False
    )
    log.info(f"Agent radius is now {env.sim.pathfinder.nav_mesh_settings.agent_radius}")


def apply_hydra_to_habitat(hydra_config, habitat_config):
    habitat_config.defrost()
    habitat_config.SIMULATOR.FORWARD_STEP_SIZE = hydra_config.v
    habitat_config.SIMULATOR.TURN_ANGLE = hydra_config.w
    habitat_config.ENVIRONMENT.MAX_EPISODE_STEPS = hydra_config.test_params.max_steps
    habitat_config.DATASET.DATA_PATH = os.path.join(hydra_config.data_path,
                                                    hydra_config.sim_env + "_" + hydra_config.difficulty + ".json.gz")
    habitat_config.freeze()


def run_habitat(cfg: DictConfig) -> Optional[float]:
    """
    Deploy a policy in a habitat environment.

    This script is largely controlled by conf/run_habitat.yaml.

    For habitat specific parameters, have a look at conf_habitat/imagenav.yaml.

    Args:
        cfg: Config with all hyperparameters for deployment
    """
    if not habitat_sim.cuda_enabled:
        log.warning("Habitat does not have cuda enabled.")

    # Set seed for random number generators in pytorch, numpy and python.random
    n_trajectories = cfg.test_params.n_trajectories
    topdown = cfg.gif_params.topdown
    data_type = cfg.data_type
    if cfg.get("seed"):
        utils.seed_experiment(cfg.seed)

    # Metrics accumulator
    metrics = dict(success=0, softsuccess=0, spl=0, softspl=0, distance_to_goal=0, collisions=0, collisions_free=0)
    success = list()  # Raw success results for filtering the gif

    # Step counter
    step_counter = 0

    # Load config
    habitat_config = habitat.get_config(config_paths="conf_habitat/imagenav.yaml")

    # Fetch agent height
    # Set linear and angular speed as defined in hydra config
    apply_hydra_to_habitat(cfg, habitat_config)

    # Auto pick the good global head based on the current sim_env
    checkpoint_name = cfg.sim_env.lower() + '.ckpt'
    if cfg.checkpoints.get('gr_path'):
        cfg.checkpoints.gr_path = "/".join(cfg.checkpoints.gr_path.split('/')[:-1] + [checkpoint_name])
        log.info(f"Updated geodesic regressor checkpoint {cfg.checkpoints.gr_path}")

    # Dataset writer
    scene_name = cfg.sim_env
    temp_data_directory = os.path.join(os.environ.get('SLURM_TMPDIR'), 'temp') if os.environ.get('SLURM_TMPDIR') else 'temp'
    writer = DatasetWriter(base_path="./data", dataset_name=scene_name if cfg.collect_data else temp_data_directory,
                           write=cfg.collect_data or cfg.gif_params.goal or topdown)

    if topdown:
        habitat_config.defrost()
        habitat_config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        habitat_config.freeze()

    # Extract shape of depth image
    depth_shape = (habitat_config.SIMULATOR.DEPTH_SENSOR.WIDTH,
                   habitat_config.SIMULATOR.DEPTH_SENSOR.HEIGHT)
    # Render gif with high quality images
    high_quality = cfg.gif_params.high_quality

    # Uncomment this lines if using habitat EquirectangularSensor sensor
    # if cfg.panorama:
    #     habitat_config.defrost()
    #     habitat_config.SIMULATOR.AGENT_0.SENSORS = ["EQUIRECT_RGB_SENSOR"]
    #     habitat_config.freeze()

    # Create policy
    # Before simulator to have more resources
    log.info("Creating policy...")
    policy = hydra.utils.instantiate(cfg.policy)

    with habitat.Env(config=habitat_config) as env:
        log.info("Simulation environment creation successful")

        # Set radius of agent
        inflate_agent(env, cfg.agent_radius)

        # Snap position and rotation if environment is S1 or Grid
        if data_type == 'grid':
            log.info("Creating grid")
            log.info("Resampling points start and goal positions on grid")
            # Define Habitat grid SE2
            grid = SE2Grid(v=cfg.v, d2o=cfg.grid_params.d2o, pathfinder=env._sim.pathfinder,
                           visualize=cfg.grid_params.visualize, random_state=cfg.seed)
            for episode in env._dataset.episodes:
                # Positions and rotations should be sampled from grid
                episode.start_position, episode.start_rotation = grid.sample_point()
                episode.goals[0].position, _ = grid.sample_point(exclude_position=np.array(episode.start_position))

                # Hotfix for topdown map to work
                height = env._sim.pathfinder.snap_point(episode.start_position)[1]
                episode.start_position[1] = height  # 0.2
                height = env._sim.pathfinder.snap_point(episode.goals[0].position)[1]
                episode.goals[0].position[1] = height  # 0.21
        else:
            grid = None  # No grid object
            if cfg.grid_params.visualize:
                visualize_map(env._sim.pathfinder)
            # Change height of things for top down map to work properly
            for episode in env._dataset.episodes:
                # Hack?
                height = env._sim.pathfinder.snap_point(episode.start_position)[1]
                episode.start_position[1] = height + 0.05  # 0.2
                height = env._sim.pathfinder.snap_point(episode.goals[0].position)[1]
                episode.goals[0].position[1] = height + 0.05  # 0.21

        if hasattr(policy, "is_oracle") and policy.is_oracle:
            policy.setup_sim(env._sim)  # Grant policy access to simulator
        policy.transform = get_transforms(keys=policy.keys if hasattr(policy, "keys") else None, resolution=cfg.resize)

        log.info("Policy creation successful.")
        if cfg.collect_data:
            # Skip first 1000 trajectories if we collect data. We use those for testing.
            n_trajectories += 1000

        for episode in range(n_trajectories):
            observations = env.reset()  # Get first observations

            # First iteration
            first_iter = True

            # Get goal_position
            goal_position = env.current_episode.goals[0].position
            im_goal, goal_dict = get_goal_image_state(env, goal_position, grid=grid, rng=goal_rng,
                                                      panorama=cfg.panorama, delta_pan=cfg.pan_delta)
            # Get high quality image goal if needed
            im_goal_hq, _ = get_goal_image_state(env, goal_position, grid=grid, rng=goal_rng,
                                                 panorama=cfg.panorama, delta_pan=cfg.pan_delta,
                                                 use_hq=high_quality)

            if cfg.collect_data and episode < 1000:
                # Skip first 1000 trajectories if we collect data. We use those for testing.
                continue

            # Run episode
            log.info(f"Episode {episode}...")
            # Should we filter our episodes based on this criteria?
            # https://arxiv.org/pdf/2004.05155.pdf
            # https://openaccess.thecvf.com/content/ICCV2021/papers/Kwon_Visual_Graph_Memory_With_Unsupervised_Representation_for_Visual_Navigation_ICCV_2021_paper.pdf
            log.info(f"Geodesic distance is {env.current_episode.info['geodesic_distance']}")

            # Run episode
            writer.new_trajectory()  # New trajectory
            softsuccess = False
            while not env.episode_over:
                # Observations
                im = observations["rgb"]
                im_hq = observations["robot_head_rgb"]

                # Manually rescale depth measurements
                observations["depth"] *= habitat_config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                agent_state = env._sim.get_agent_state()
                scan = depth_to_range_bearing(observations['depth'], size=depth_shape, agent_state=agent_state)

                # comment these lines if using habitat EquirectangularSensor sensor
                if cfg.panorama:
                    im = get_panorama_image(env, env._sim.get_agent_state(), delta=cfg.pan_delta)
                    im_hq = get_panorama_image(env, env._sim.get_agent_state(),
                                               delta=cfg.pan_delta, sensor='robot_head_rgb')

                # Get best action
                best_action = policy.get_action(
                    x_current=im,
                    x_goal=im_goal,
                    prev_reward=None,
                    position=goal_position,  # Should only be used by oracle policies
                    scan=scan
                )

                # Step env
                step_counter += 1
                previous_state = env._sim.get_agent_state()
                prev_measures = env.task.measurements.get_metrics()
                observations = env.step(best_action)

                # Check if we collided and potentially reset
                if cfg.reset_on_collision and env._sim.previous_step_collided:
                    log.error(f'Reset agent to previous position. Min distance to grid is {min_norm}.')
                    observations = env._sim.get_observations_at(position=previous_state.position,
                                                                rotation=previous_state.rotation,
                                                                keep_agent_at_new_pose=True)
                    observations["imagegoal"] = im_goal
                    # Update metrics to previos time step if collision
                    if topdown:
                        # Update metrics to previos time step if collision
                        env.task.measurements.measures['distance_to_goal']._metric = prev_measures['distance_to_goal']
                        env.task.measurements.measures['success']._metric = prev_measures['success']
                        env.task.measurements.measures['spl']._metric = prev_measures['spl']
                        env.task.measurements.measures['softspl']._metric = prev_measures['softspl']
                        env.task.measurements.measures['top_down_map']._metric = prev_measures['top_down_map']

                    if cfg.new_goal_on_collision:
                        # Will trigger sampling of new goal
                        env._episode_over = True

                # Fetch metrics
                info = env.get_metrics()

                if not softsuccess and info['distance_to_goal'] < 1.:
                    softsuccess = True  # Did we reach the goal at some point in the traj

                # Do some grid testing
                if data_type == 'grid':
                    current_position = env._sim.get_agent_state().position[[0, 2]].reshape((1, -1))
                    norms = cdist(current_position, grid.grid[:, [0, 2]])
                    min_norm = norms.min()
                    is_in_grid = min_norm < 1e-4
                    # if not is_in_grid:
                    #     log.warning(f'Agent not on grid. Min distance to grid is {min_norm}')
                    # else:
                    #     log.info(f'Agent on grid. Min distance to grid is {min_norm}')
                # log.info(f'Best action is {best_action}')

                # Topdown map
                # Render goal manually to avoid habitat-lab bug
                if topdown:
                    env.task.measurements.measures['top_down_map']._draw_goals_positions(env.current_episode)
                    map_size = im_hq.shape[0] if high_quality else im.shape[0]
                    top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], map_size)
                else:
                    top_down_map = None

                # During first iteration only assign top_down_map. This map is one observation in the future from img
                if first_iter:
                    prev_top_down_map = top_down_map if topdown else None
                    first_iter = False

                # Save observations
                if isinstance(best_action, str):
                    # Make sure best_action is an integer
                    best_action = env.task._action_keys.index(best_action)
                # If step -1 reward, if collision -2 reward and if done 0
                reward = -2 if env._sim.previous_step_collided else (0 if env.episode_over else -1)
                done = 1 if env.episode_over else 0
                step_dict = get_step_dict(writer.obs_counter, best_action, previous_state, reward=reward, done=done)

                # If not collecting data and gif collision option is on add colored frame to images
                if not cfg.collect_data and cfg.gif_params.collisions and not cfg.gif_params.viz_depth:
                    # Only show front image for gif
                    im = Image.fromarray(im_hq[:, :256, :]) if high_quality else Image.fromarray(im[:, :128, :])
                    im = ImageOps.expand(im, border=20, fill="yellow" if policy.prev_step_collided() else "green")
                    im = np.array(im)
                    write_panorama = False
                elif not cfg.collect_data and cfg.gif_params.viz_depth:
                    # Viz depth with collisions
                    im_depth = observations['depth'].copy()
                    collision_mask = im_depth < .25001
                    im_depth = (im_depth - im_depth.min()) / im_depth.max()
                    im_depth *= 255
                    im_depth = im_depth.astype('uint8')
                    im_depth = np.concatenate([im_depth] * 3, axis=-1)  # Fake 3 channels
                    # im_depth[collision_mask[:, :, 0]] = np.array([255, 0, 0])  # Red pixels for collisions
                    im = Image.fromarray(im_depth)  # Slice front view
                    im = ImageOps.expand(im, border=20, fill="yellow" if policy.prev_step_collided() else "green")
                    im = np.array(im)
                    write_panorama = False
                else:
                    write_panorama = cfg.panorama
                writer.write_observation(image=im, meta=step_dict, scan=scan,
                                         top_down_image=prev_top_down_map, panorama=write_panorama,
                                         pan_delta=cfg.pan_delta, use_hq=high_quality)

                if env.episode_over and env._elapsed_steps < cfg.test_params.min_steps:
                    # We sample a new goal
                    log.info('Setting new random goal...')
                    if data_type == 'grid':
                        current_position = env._sim.get_agent_state().position
                        new_pos, new_rot = grid.sample_point(exclude_position=np.array(current_position))
                        env.current_episode.goals[0].position = new_pos
                        env.current_episode.goals[0].rotation = new_rot
                        goal_position = env.current_episode.goals[0].position
                    elif data_type == 'se2':
                        new_episode = list(generate_pointnav_episode(
                            env._sim, 1, is_gen_shortest_path=False,
                            geodesic_to_euclid_min_ratio=1.00,
                            shortest_path_max_steps=10,
                            # 1 will allow straight lines. Increasing it will favor hard paths
                            closest_dist_limit=0.0,  # With a step size of .25, should give us at least 20 steps
                            furthest_dist_limit=0.5,  # With a step size of .25, should give us at max 400 steps
                        ))
                        # new_pos = new_episode[0].goals[0].position
                        agent_position = env._sim.get_agent_state().position
                        new_pos = env._sim.pathfinder.get_random_navigable_point_near(
                            circle_center=agent_position[:, None],
                            radius=2.0,
                            max_tries=100)
                        angle = np.random.uniform(0, 2 * np.pi)
                        new_rot = [0.0, np.sin(angle / 2), 0, np.cos(angle / 2)]
                        env.current_episode.goals[0].position = new_pos
                        env.current_episode.goals[0].rotation = new_rot
                        goal_position = env.current_episode.goals[0].position
                    else:
                        raise NotImplementedError('Min steps is only implemented for grid.')

                    # Turn off episode over flags to keep going
                    height = env._sim.pathfinder.snap_point(env.current_episode.goals[0].position)[1]
                    env.current_episode.goals[0].position[1] = height + 0.05  # 0.21  # Hotfix for map
                    env._episode_over = False
                    env.task._is_episode_active = True
                    env.task.is_stop_called = False
                    # Update image goal and dict
                    im_goal, goal_dict = get_goal_image_state(env, goal_position, grid=grid, rng=goal_rng,
                                                              panorama=cfg.panorama, delta_pan=cfg.pan_delta,
                                                              use_hq=high_quality)

                # top_down_map is delayed one observation!
                prev_top_down_map = top_down_map

            # Write final goal image and state dict
            success.append(softsuccess)
            # Save high/low resolusiton goal image
            im_goal = im_goal_hq if high_quality else im_goal
            writer.write_goal(image=im_goal, meta=goal_dict, panorama=cfg.panorama, pan_delta=cfg.pan_delta)

            # Sum metrics over episodes
            for key in metrics.keys():
                if key == 'collisions':
                    metrics[key] += info[key]['count']
                elif key == 'collisions_free':
                    metrics[key] += 1 if not info['collisions']['count'] else 0
                elif key == 'softsuccess':
                    metrics[key] += softsuccess
                else:
                    metrics[key] += info[key]

    log.info('Done.')
    log.info(f'Difficulty : {cfg.difficulty}.')
    for key in metrics.keys():
        metrics[key] /= n_trajectories
    log.info(f"Success      : {metrics['success']:.4f}")
    log.info(f"Soft Success : {metrics['softsuccess']:.4f}")
    log.info(f"SPL          : {metrics['spl']:.4f}")
    log.info(f"Soft SPL     : {metrics['softspl']:.4f}")
    log.info(f"DTG          : {metrics['distance_to_goal']:.4f}")
    log.info(f"Collisions   : {metrics['collisions']:.4f}")
    log.info(f"CFT          : {metrics['collisions_free']:.4f}")

    # Also log index of failed trajectories
    success = np.array(success)
    log.info(f"Failed Traj. : {list(np.argwhere(~success).flatten())}")

    if not cfg.gif_params.failures:
        success = np.zeros(n_trajectories).astype(bool) #  Log all images (data_gif is coded to log failures if an array is provided)
    else:
        # Log first n trajectories only
        if cfg.gif_params.log_n_first > 0:
            success[:cfg.gif_params.log_n_first] = False
            success[cfg.gif_params.log_n_first:] = True

    # Load metrics to comet
    # Init lightning loggers
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger_ = hydra.utils.instantiate(lg_conf)
                # Add comet tag if using comet logger
                if cfg.logger.get('comet'):
                    # Always suffix the tag with -nav
                    logger_.experiment.add_tag(cfg.logger.tag + "-nav")
                    logger_.experiment.log_asset_data(OmegaConf.to_container(cfg), 'run_habitat.yaml')
                    logger_.experiment.log_metrics(metrics)

                    log.info("Logging hyperparameters!")
                    logger_.log_hyperparams(cfg)
                    logger_.log_hyperparams(habitat_config)
                    # Used to debug graph-planning methods
                    if hasattr(policy, 'load_graph_img'):
                        graph_img = policy.load_graph_img()
                        logger_.experiment.log_image(image_data=graph_img, name='computed_graph')

    # Define gif path
    if os.environ.get('SLURM_TMPDIR'):
        gif_path = os.path.join(os.environ.get('SLURM_TMPDIR'), scene_name + ".mp4")
    else:
        gif_path = os.path.join('movies', scene_name + ".mp4")

    if cfg.gif_params.goal:
        log.info('Compiling image/goal gif...')
        data_gif(agent_name=cfg.agent_name,
                 path=writer.data_path,
                 gif_path=gif_path,
                 duration=cfg.gif_params.duration, success=success)
        # Log gif
        if "logger" in cfg and cfg.logger.get('comet'):
            log.info("Logging gif...")
            logger_.experiment.log_image(gif_path, image_format="mp4")

    if cfg.gif_params.topdown:
        log.info('Compiling image/map gif...')
        data_gif(agent_name=cfg.agent_name,
                 path=writer.data_path,
                 gif_path=gif_path,
                 make_top_down=True,
                 duration=cfg.gif_params.duration, success=success)
        # Log gif
        if "logger" in cfg and cfg.logger.get('comet'):
            log.info("Logging gif to path")
            log.info(gif_path)
            logger_.experiment.log_image(gif_path, image_format="mp4")

    if cfg.collect_data:
        # Split data into train/val
        split_data(split=cfg.split, path=writer.data_path)
    elif os.path.exists(writer.data_path):
        # Remove data
        shutil.rmtree(writer.data_path)

    return metrics['spl']
