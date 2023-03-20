from typing import Union, Optional, List, Tuple
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import habitat_sim
import quaternion
from habitat_sim import Agent, AgentState, Simulator, PathFinder
from habitat.utils.visualizations import maps
from habitat_sim.utils.common import quat_rotate_vector, quat_from_angle_axis, quat_from_coeffs
from habitat.tasks.utils import cartesian_to_polar


def create_dir(out_dir: str, episode: int, top_down: Optional[bool] = False):
    """
    Create output dir for the trajectories generated
    Args:
        out_dir: Path to output directory
        episode: Current episode
        top_down: If we are going to store top-down views of the agent navigatin

    Returns:

    """
    # Creating directories
    path_img = os.path.join(out_dir, "traj_%d" % episode, "images")
    path_meta = os.path.join(out_dir, "traj_%d" % episode, "meta")
    path_top = os.path.join(out_dir, "traj_%d" % episode, "top_down")
    try:
        os.system("mkdir -p %s" % path_img)
        os.system("mkdir -p %s" % path_meta)
        if top_down:
            os.system("mkdir -p %s" % path_top)
    except:
        pass


def get_step_dict(step: int, best_action: int, agent_state: AgentState,
                  reward: Union[int, float] = 0, done: int = 0):
    """
    Create a dictionary with the information of one timestep in habitat
    Habitat reference frame is +x to the right, +y upwards and +z out of the screen
    https://github.com/facebookresearch/habitat-lab/issues/531
    https://github.com/facebookresearch/habitat-sim/issues/524
    https://aihabitat.org/docs/habitat-lab/view-transform-warp.html
    Args:
        step: Current timestep (index)
        best_action: Action taken for the agent
        agent_state: State of the agent with its pose
        reward: Used primary to detect collisions
        done: Denotes terminal state
    Returns:
        Dictionary with metadata of the agent at current timestep
    """
    step_dict = {}
    position, rot = agent_state.position, agent_state.rotation
    step_dict["time"] = step
    step_dict["actions"] = {'motion': int(best_action)}
    step_dict["reward"] = reward
    step_dict["done"] = done
    # This receives x, y, z, w - This rotation transforms world frame to x+ front, y+ left and z+ upwards
    q = quat_from_coeffs(np.asarray([-0.5, 0.5, 0.5, -0.5]))
    # Map position and rotation to new coordinate frame
    transformed_position = quat_rotate_vector(q, np.asarray(position))
    # This is hacky, but as the rotation frame of the agent is weird (the same as habitat)
    # The resulting quaternion will be also pretty odd, hence the transform
    transformed_rotation = q * rot * q.inverse()
    step_dict["pose"] = {"x": float(transformed_position[0]),
                         'y': float(transformed_position[1]),
                         'z': float(transformed_position[2]),
                         "orientation.x": transformed_rotation.x,
                         "orientation.y": transformed_rotation.y,
                         "orientation.z": transformed_rotation.z,
                         "orientation.w": transformed_rotation.w}
    return step_dict


def depth_to_point_cloud(depth: np.ndarray, hfov_deg: int = 140):
    """Convert DEPTH_SENSOR measurement to point cloud.

    x is pointing right, y is up and -z is forward (where camera is pointing).

    See https://aihabitat.org/docs/habitat-lab/view-transform-warp.html."""
    hfov = hfov_deg * np.pi / 180
    K = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0., 1, 0],
        [0., 0., 0, 1]])

    # Now get an approximation for the true world coordinates -- see if they make sense
    # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
    W = depth.shape[0]  # Assume square depth map
    xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, W))
    depth = depth[:, :, 0].reshape(1, W, W)
    xs = xs.reshape(1, W, W)
    ys = ys.reshape(1, W, W)

    # Unproject
    xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)
    xy_c0 = np.matmul(np.linalg.inv(K), xys)

    return xy_c0.T


def depth_to_range_bearing(depth: np.ndarray, agent_state, size: Tuple[int, int], window: int = 5,
                           visualize=False) -> np.ndarray:
    """
    Convert depth measurement to range and bearing measurement.
    Args:
        depth: Depth image
        size: Shape of depth image [width, height]
        window: Number of points above and below middle of image used to compute average estimate

    Returns:
        Range and bearing scan
    """
    pc = depth_to_point_cloud(depth)

    # Filter for a specific height
    # Height is y
    pc[:, 1] += agent_state.sensor_states['depth'].position[1]  # Add sensor height to put floor around 0.0
    mask = (pc[:, 1] > .10) & (pc[:, 1] < 1.00)

    if visualize:
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        hit_mask = (np.abs(x) < .1) & (z < .00) & (z > -.5)
        # hit_mask = (np.abs(x) < .1) & (z < .00)

        import open3d as o3d
        sample = pc.copy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sample[:, :3])
        color = np.array([[0, 0, 255]] * sample.shape[0])
        color[mask] = np.array([0, 255, 0])
        color[hit_mask] = np.array([255, 128, 0])
        color[mask & hit_mask] = np.array([255, 0, 0])
        pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([pcd])

        # Exit since open3D currently bricks habitat rendering
        exit()

    # Now we ignore height and convert back to polar coordinates
    pc = pc[mask]
    x, z = pc[:, 0], pc[:, 2]
    rho = np.sqrt(x ** 2 + z ** 2)
    phi = (np.arctan2(z, x) + np.pi / 2) % (2 * np.pi)  # Rotate to ensure 0 rad is front
    scan = np.vstack((phi, rho)).T

    return scan


def get_panorama_image(env, agent_state, anchor_pose=None, delta=90, sensor='rgb'):
    """
    Get a stack of panorama images around a given pose by rotating the camera 360 degrees delta degrees at a time
    Args:
        env: habitat environment
        agent_state: Habitat Agent's state 
        anchor_pose (optional): Pose around which to take the panorama stack (Anchor pose) (Yaw of the robot)
        delta: Rotation angle b/w two successive images to build 360 panorama
        sensor: Sensor used to collect image
    Returns:
        A single panorama image 
    """

    agent_position = agent_state.position
    if anchor_pose is None:
        agent_rotation = agent_state.rotation
    else:
        agent_rotation = quat_from_angle_axis(np.deg2rad(anchor_pose), habitat_sim.geo.UP)

    panorama_stack = []
    for i in range(int(360 / delta)):
        delta_rot = quat_from_angle_axis(np.deg2rad(-delta * i), habitat_sim.geo.UP)
        new_rot = agent_rotation * delta_rot
        obs = env._sim.get_observations_at(position=agent_position,
                                           rotation=new_rot,  # Provide habitat-style rotation
                                           keep_agent_at_new_pose=False)
        panorama_stack.append(obs[sensor])

    panorama = np.concatenate(panorama_stack, axis=1)

    return panorama


def save_json(dict_: dict, filename: str):
    """
    Utility to store json file
    Args:
        dict_: Dictionary with data
        filename: Name of Json file

    """
    with open(filename, "w") as f:
        json.dump(dict_, f, indent=4)


def make_cfg(settings):
    '''
    Setting up configuration to initialize the simulator and the agent. Adding sensors, specifying action space etc.
    Returns:
        cfg: Configuration for both agent and simulator
    '''
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    color_sensor_spec.hfov = settings["fov"]
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    color_sensor_spec.hfov = settings["fov"]
    sensor_specs.append(depth_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=settings["linear_speed"])
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=settings["angular_speed"])
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=settings["angular_speed"])
        ),
    }

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    return cfg




def convert_points_to_topdown(pathfinder: PathFinder, points: List[np.ndarray], meters_per_pixel: Union[int, float]):
    """
    convert 3d points to 2d topdown coordinates
    https://colab.research.google.com/github/facebookresearch/habitat-sim/blob/main/examples/tutorials/colabs/ECCV_2020_Navigation.ipynb#scrollTo=REZrxWAFid1-
    """
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


def plot_start_goal(top_down_map: np.ndarray, start_goal: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Plot start and goal position of the agent
    https://colab.research.google.com/github/facebookresearch/habitat-sim/blob/main/examples/tutorials/colabs/ECCV_2020_Navigation.ipynb#scrollTo=REZrxWAFid1-
    """
    # Plot start and goal position
    for point, c in zip(start_goal, [(0, 0, 255), (255, 0, 0)]):
        cv2.circle(top_down_map, (int(point[0]), int(point[1])), radius=6, color=c, thickness=cv2.FILLED)


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None, fig_name="topdownmap.png"):
    """
    Display a topdown map with grid points 
    https://colab.research.google.com/github/facebookresearch/habitat-sim/blob/main/examples/tutorials/colabs/ECCV_2020_Navigation.ipynb#scrollTo=REZrxWAFid1-
    """
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", color='r', markersize=10, alpha=0.8)
    topdown_map = topdown_map.astype(int)
    # plt.savefig(fig_name)
    plt.show()


def visualize_map(pathfinder: PathFinder):
    # Plot the topdown map
    height = pathfinder.get_random_navigable_point()[1]
    sim_topdown_map = pathfinder.get_topdown_view(0.05, height).astype(np.uint8)
    recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
    sim_topdown_map = recolor_map[sim_topdown_map]
    display_map(sim_topdown_map)


def make_top_down_view(sim: Simulator, agent: AgentState,
                       start_goal: Optional[List[AgentState]] = None,
                       trajectory: Optional[List[AgentState]] = None,
                       height: Optional[float] = None,
                       meters_per_pixel: Union[int, float] = 0.025):
    """
    Create a top-down image of the agent navigating a scene

    Args:
        sim: Instance of thew simulator
        agent: Current state of the agent
        start_goal: Start and goal position of the agent
        trajectory: Points navigated by the agent
        height: Optional height to produce map
        meters_per_pixel: Resolution of the grid

    Returns:
        np.ndarray with the top-down image
    """
    # Grab center and dimensions of navmesh
    scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    # Extraxt min height of the mesh
    if height is None:
        height = scene_bb.y().min
    # Fetch topdown view of current scene and change its color
    top_down_map = maps.get_topdown_map(sim.pathfinder, height, meters_per_pixel=meters_per_pixel)
    recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
    top_down_map = recolor_map[top_down_map]
    # Extract dimensions of current grid
    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
    # Convert world trajectory points to maps module grid points
    agent_world_pos = [maps.to_grid(agent.position[2], agent.position[0], grid_dimensions, pathfinder=sim.pathfinder)]
    # Draw trajectory if provided
    if trajectory is not None:
        trajectory = [
            maps.to_grid(
                path_point[2],
                path_point[0],
                grid_dimensions,
                pathfinder=sim.pathfinder,
            )
            for path_point in trajectory
        ]
        # Draw the agent and trajectory on the map
        maps.draw_path(top_down_map, trajectory)
    # Draw goal if provided
    if start_goal is not None:
        start_goal_world = convert_points_to_topdown(sim.pathfinder, start_goal, meters_per_pixel)
        plot_start_goal(top_down_map, start_goal_world)
    # Extract orientation of agent in the map!
    # https: // github.com / facebookresearch / habitat - lab / issues / 109
    heading_vector = quat_rotate_vector(agent.rotation.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    top_down_map_angle = phi - np.pi
    # Angle is somehow flipped 180 degrees
    maps.draw_agent(top_down_map, agent_world_pos[0], top_down_map_angle, agent_radius_px=12)
    return top_down_map


class SE2Grid:
    """
    Produce grid for SE2 envs in habitat

        Args:
            v: Linear speed of the agent
            pathfinder:  Instance of pathfinder
            d2o: Distance to obstacle
            visualize: Plot the grid - used for debugging
            random_state: Random state for rng
    """

    def __init__(self, v: float, pathfinder: PathFinder, d2o: int = 0.5,
                 visualize: bool = False, random_state: int = 1234):
        self.rng = np.random.default_rng(random_state)
        self.v = v
        self.pathfinder = pathfinder
        self.d2o = d2o
        self.visualize = visualize
        self.height: Union[int, float] = None
        self.grid: np.ndarray = None
        # Create grid
        self.create_grid()
        # Visualize grid
        if self.visualize:
            self.visualize_grid()

    def create_grid(self):
        ## Defining S2 grid
        start_node, end_node = self.pathfinder.get_bounds()
        # Add one extra sample to wrap the whole grid
        width, height = int(end_node[0] - start_node[0]) + 1, int(end_node[2] - start_node[2]) + 1
        self.height = self.pathfinder.get_random_navigable_point()[1]
        x = start_node[0] + np.arange(start=0, stop=width, step=self.v)
        # Number of samples in x
        points_x = x.shape[0]
        y = start_node[2] + np.arange(start=0, stop=height, step=self.v)
        # Number of samples in y
        points_y = y.shape[0]
        # Define grid
        x, y = np.meshgrid(x, y)
        # Set the height of a navigable point ow, is_navigable may fail
        grid_points = np.concatenate(
            (np.expand_dims(x, axis=1),
             np.ones((points_y, 1, points_x)) * self.height,
             np.expand_dims(y, axis=1)), axis=1)
        grid_points = np.transpose(grid_points, (0, 2, 1)).reshape(points_x * points_y, 3)
        valid_grid_points = []
        for point in grid_points:
            # Only store navigable points as possible start points
            if self.pathfinder.is_navigable(point, max_y_delta=2.0):
                if self.pathfinder.distance_to_closest_obstacle(point, 2.0) > self.d2o:
                    valid_grid_points.append(point)

        self.grid = np.array(valid_grid_points)
        # self.grid = np.asarray(grid_points)

    def visualize_grid(self):
        """
        Visualize top-down map image
        """
        # Plot the points on the topdown map
        xy_vis_points = convert_points_to_topdown(self.pathfinder, self.grid, 0.05)
        sim_topdown_map = self.pathfinder.get_topdown_view(0.05, self.height).astype(np.uint8)
        recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        sim_topdown_map = recolor_map[sim_topdown_map]
        display_map(sim_topdown_map, key_points=xy_vis_points)

    def sample_point(self, exclude_position=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a navigable point within the grid
        """
        while True:
            position = self.grid[self.rng.choice(len(self.grid))]
            if exclude_position is None or not np.all(position == exclude_position):
                break

        rotation = quat_from_angle_axis(np.deg2rad(self.rng.choice([-90, 0, 90, 180])), habitat_sim.geo.UP).components
        return position.tolist(), [rotation[1], rotation[2], rotation[3], rotation[0]]


def get_goal_image_state(env, goal_position, rng, grid=None, panorama=False, delta_pan=90, use_hq: bool = False):
    """Render goal image and state dict."""
    # Given the position, this is how Habitat generates the rotation.
    # See habitat-lab/habitat/tasks/nav.py
    # seed = abs(hash(env.current_episode.episode_id)) % (2 ** 32)
    if grid is None:
        # Sample random rotation
        angle = rng.uniform(0, 2 * np.pi)

        # This is the x, y, z, w convention of habitat-lab
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
    else:
        # Get rotation from grid object
        _, source_rotation = grid.sample_point()

    # Save goal
    state = habitat_sim.agent.AgentState(position=goal_position,
                                         rotation=quaternion.quaternion(  # Quaternion follows, w, x, y z
                                             source_rotation[3],
                                             source_rotation[0],
                                             source_rotation[1],
                                             source_rotation[2],
                                         ))
    step_dict = get_step_dict(0, 0, state, reward=0, done=1)

    # Get goal image
    sensor_type = "robot_head_rgb" if use_hq else "rgb"
    if panorama:
        im = get_panorama_image(env, state, delta=delta_pan, sensor=sensor_type)
        # Uncomment this lines if using habitat EquirectangularSensor sensor
        # obs = env._sim.get_observations_at(position=state.position,
        #                                    rotation=state.rotation,  # Provide habitat-style rotation
        #                                    keep_agent_at_new_pose=False)
        # im = obs['rgb']
    else:
        obs = env._sim.get_observations_at(position=goal_position,
                                           rotation=source_rotation,  # Provide habitat-style rotation
                                           keep_agent_at_new_pose=False)
        im = obs[sensor_type]

    return im, step_dict
