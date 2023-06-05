"""Adapted Mazelab Environment from zuoxingdong.

This code assumes you are using the following mazelab fork : https://github.com/sachaMorin/mazelab.

The goal is hidden since we use dataset images as starting points/goals.

See the original tutorial here.
https://github.com/zuoxingdong/mazelab/blob/master/examples/navigation_env.ipynb
"""
from typing import List, Tuple

import os
import shutil
import json

import matplotlib.pyplot as plt
import numpy as np

import PIL.Image as Image

import gym
from gym.spaces import Box
from gym.spaces import Discrete

from mazelab import BaseEnv
from mazelab import MooreMotion
from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color
from mazelab.generators import morris_water_maze, random_maze, random_shape_maze, u_maze, double_t_maze

from src.models.policies.base_policy import Policy
from src.utils.split_dataset import split_data


class Maze(BaseMaze):
    """Maze object from the original Maze tutorial notebook.

    Attributes:
        x (ndarray): Impassable array representing the maze.
    """

    def __init__(self, x: np.ndarray):
        self.x = x
        super().__init__()

    @property
    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.x == 1), axis=1))
        goal = Object('goal', 2, color.goal, False, [])
        agent = Object('agent', 3, color.agent, False, [])
        return free, obstacle, goal, agent


class MazeEnv(BaseEnv):
    """OpenAI-style environment for mazes.

    The focus here is on goal-conditioned RL. Goals are not rendered on the image
    and policies are expected to understand the goal from another image representing
    the goal state.

    Attributes:
        maze (MazeEnv): Actual MazeEnv object from the Mazelab repo.
        start_idx(list): List of shape [[start_x, start_y]] determining the agent starting position.
        goal_idx(list): List of shape [[goal_x, goal_y]] determining the agent's goal.
        motions(namedtuple): Named tuple representing available agent actions.
        observation_space(gym.Box): Observation space.
        action_space(gym.Discrete): Discrete action space.
        random_state(np.random.RandomState): Custom random number generator.

    """

    def __init__(self, x: np.ndarray, start_idx: List, goal_idx: List, random_state: int = 42):
        super().__init__()

        self.maze = Maze(x)
        self.start_idx = start_idx
        self.goal_idx = goal_idx
        self.motions = MooreMotion()

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

        # Own random_state
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def step(self, action: int) -> tuple:
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            # Goal reached
            reward = +1
            done = True
        elif not valid:
            # Collision
            reward = -2
            done = False
        else:
            # Movement
            reward = -1
            done = False
        return self.maze.to_value(), reward, done, {}

    def reset(self) -> np.ndarray:
        self.maze.objects.agent.positions = self.start_idx
        self.maze.objects.goal.positions = self.goal_idx
        self.random_state = np.random.default_rng(self.random_state)
        return self.maze.to_value()

    def _is_valid(self, position: List) -> bool:
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position: List) -> bool:
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def set_goal(self, x: int, y: int):
        self.goal_idx = [[x, y]]

    def set_start(self, x: int, y: int):
        self.start_idx = [[x, y]]

    def get_image(self) -> np.ndarray:
        """Render current state image."""
        img = self.maze.to_rgb()
        return img

    def get_goal_img(self) -> np.ndarray:
        """Render goal image."""
        # Back up the current position.
        # Warp the agent to goal.
        # Render image.
        # Return to current position.
        current_position = self.maze.objects.agent.positions[0]
        self.maze.objects.agent.positions[0] = self.maze.objects.goal.positions[0]
        x_goal = self.render('rgb_array')
        self.maze.objects.agent.positions[0] = current_position
        return x_goal

    def sample_passable(self) -> Tuple[int, int]:
        # Sample a passable point that is neither the current start nor the current goal
        # Make sure there's at least 10 passable points before we enter the while loop to find one.
        passable = np.logical_not(self.maze.to_impassable())
        if passable.sum() < 10:
            raise Exception('Make sure maze has at least 10 passable cells before sampling.')

        while True:
            x = self.rng.choice(passable.shape[0], size=1)[0]
            y = self.rng.choice(passable.shape[1], size=1)[0]
            is_passable = passable[x, y]
            is_not_goal = x != self.goal_idx[0][0] and y != self.goal_idx[0][1]
            is_not_start = x != self.start_idx[0][0] and y != self.start_idx[0][1]

            if is_passable and is_not_goal and is_not_start:
                break
        return x, y

    def check_collisions(self)-> List:
        """Return a list of 8 booleans checking whether actions result in a collision or not."""
        not_passable = self.maze.to_impassable()
        current_position = self.maze.objects.agent.positions[0]
        result = []
        for motion in self.motions:
            new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
            collides = not_passable[new_position[0], new_position[1]]
            result.append(collides)

        return result



    def sample_start(self):
        """Resample the starting position."""
        x, y = self.sample_passable()
        self.start_idx = [[x, y]]

    def sample_goal(self, n_goals: int):
        """Resample a goal position.

        Args:
            n_goals(int): Number of goals to sample. The furthest from the start in terms of L1 norm is retained.
        """
        coordinates = list()
        distance = list()
        s_x, s_y = self.start_idx[0]

        for i in range(n_goals):
            x, y = self.sample_passable()
            coordinates.append([[x, y]])
            distance.append(np.abs(x - s_x) + np.abs(y - s_y))

        self.goal_idx = coordinates[np.argmax(distance)]

def get_random_shape_maze(random_state, start_idx , goal_idx):
    i = 0
    while True:
        x = random_shape_maze(width=20, height=20, max_shapes=15, max_size=8, allow_overlap=False,
                              shape=None, random_state=random_state + 2 * i)

        if not x[start_idx[0][0], start_idx[0][0]] and not x[goal_idx[0][0], goal_idx[0][0]]:
            break
        i += 1
    return x


def get_maze_env(maze_name: str, random_state: int = 123456789) -> MazeEnv:
    """Build maze environment and return it.

    Args:
        maze_name(str): Maze name. Should be one of ('empty', 'double_t', 'morris_water', 'random_shape',
        'random_maze', 'umaze').
        random_state(int): Random state.

    Returns:
        MazeEnv : Maze environment.

    """
    if maze_name == 'empty':
        x = np.zeros((20, 20))
        x[0] = 1
        x[-1] = 1
        x[:, 0] = 1
        x[:, -1] = 1
        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
    elif maze_name == 'two_doors':
        x = np.zeros((20, 20))
        x[0] = 1
        x[-1] = 1
        x[:, 0] = 1
        x[:, -1] = 1
        x[:, 6] = 1
        x[:, 13] = 1
        x[[14, 15], 6] = 0
        x[[4, 5], 13] = 0
        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
    elif maze_name == 'room':
        x = np.zeros((20, 20))
        for i in range(20):
            for j in range(20):
                if i in (0, 19) or j in (0, 19):
                    x[i, j] = 1
                elif i in (4, 15) and 4 <= j <= 15:
                    x[i, j] = 1
                elif j in (4, 15) and 4 <= i <= 15:
                    x[i, j] = 1
        x[15, 9] = 0
        x[15, 10] = 0
        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
    elif maze_name == '9_rooms':
        start_idx = [[3, 3]]
        goal_idx = [[16, 16]]
        x = np.zeros((20, 20))
        x[0] = 1
        x[-1] = 1
        x[:, 0] = 1
        x[:, -1] = 1
        for i in range(0, 3):
            x[0 + i] = 1
            x[-1-i] = 1
            x[:, 0+i] = 1
            x[:, -1-i] = 1
        x[:, 7] = 1
        x[:, 12] = 1
        x[7, :] = 1
        x[12, :] = 1

        x[[4, 5], 7] = 0
        x[[4, 5], 12] = 0
        x[[9, 10], 7] = 0
        x[[9, 10], 12] = 0
        # x[[14, 15], 7] = 0
        x[[14, 15], 12] = 0
        x[7, [14, 15]] = 0
        x[12, [4, 5]] = 0
        x[12, [9, 10]] = 0
    elif maze_name == 'bugtrap':
        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
        x = np.zeros((20, 20))
        x[0] = 1
        x[-1] = 1
        x[:, 0] = 1
        x[:, -1] = 1
        x[:, [9, 10]] = 1
        x[[9, 9, 10, 10], [9, 10, 9, 10]] = 0
        x[[7, 7, 7, 7, 8, 8, 8, 8], [5, 6, 7, 8, 5, 6, 7, 8]] = 1
        x[[11, 11, 11, 11, 12, 12, 12, 12], [5, 6, 7, 8, 5, 6, 7, 8]] = 1
        x[[7, 7, 7, 7, 8, 8, 8, 8], [11, 12, 13, 14, 11, 12, 13, 14]] = 1
        x[[11, 11, 11, 11, 12, 12, 12, 12],[11, 12, 13, 14, 11, 12, 13, 14]] = 1
    elif maze_name == 'double_t':
        x = double_t_maze()
        start_idx = [[8, 6]]
        goal_idx = [[1, 1]]
    elif maze_name == 'morris_water':
        x = morris_water_maze(radius=20, platform_center=[15, 30], platform_radius=1)
        start_idx = [[3, 15]]
        goal_idx = np.stack(np.where(x == 3), axis=1)
    elif maze_name == 'random_shape':
        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
        x = get_random_shape_maze(random_state, start_idx, goal_idx)
    elif maze_name == 'random_shape1':
        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
        x = get_random_shape_maze(7 * random_state, start_idx, goal_idx)
    elif maze_name == 'random_shape2':
        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
        x = get_random_shape_maze(3 * random_state, start_idx, goal_idx)
    elif maze_name == 'random_shape3':
        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
        x = get_random_shape_maze(13 * random_state, start_idx, goal_idx)
    elif maze_name == 'random_shape4':
        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
        x = get_random_shape_maze(21 * random_state, start_idx, goal_idx)
    elif maze_name == 'random_shape5':
        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
        x = get_random_shape_maze(120 * random_state, start_idx, goal_idx)
    elif maze_name == 'random_maze':
        while True:
            x = random_maze(width=81, height=51, complexity=.75, density=.75, random_state=random_state)
            start_idx = [[1, 1]]
            goal_idx = [[49, 79]]
            if not x[start_idx[0][0], start_idx[0][0]] and not x[goal_idx[0][0], goal_idx[0][0]]:
                break
    elif maze_name == 'umaze':
        x = u_maze(width=20, height=20, obstacle_width=12, obstacle_height=4)
        start_idx = [[18, 1]]
        goal_idx = [[1, 1]]
    elif maze_name == 'omaze':
        x = np.zeros((20, 20))
        x[0] = 1
        x[-1] = 1
        x[:, 0] = 1
        x[:, -1] = 1

        x[8:12, 8:12] = 1

        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
    elif maze_name == 'hmaze':
        x = np.ones((20, 20))
        x[1:19, 1:3] = 0
        x[1:19, 17:19] = 0
        x[9:11, 1:19] = 0

        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
    elif maze_name == 'buttonmaze':
        x = np.zeros((20, 20))
        x[0] = 1
        x[-1] = 1
        x[:, 0] = 1
        x[:, -1] = 1
        x[4:8, 4:8] = 1
        x[4:8, 12:16] = 1
        x[12:16, 4:8] = 1
        x[12:16, 12:16] = 1

        start_idx = [[1, 1]]
        goal_idx = [[18, 18]]
    else:
        raise ValueError('Provided maze_name is not supported.')

    gym.envs.register(id=maze_name, entry_point=MazeEnv, max_episode_steps=10000, kwargs={'x': x,
                                                                                          'start_idx': start_idx,
                                                                                          'goal_idx': goal_idx})
    env = gym.make(maze_name)
    env.reset()

    return env


def plot_with_scores(path, dist, action_probs, image):
    # Clip action probs for nicer visualizations
    action_probs = np.clip(action_probs, .01, .99)

    vn_map = {
        0: (0, 1),  # North
        1: (2, 1),  # South
        2: (1, 0),  # West
        3: (1, 2),  # East
        4: (0, 0),  # Northwest
        5: (0, 2),  # Northeast
        6: (2, 0),  # Soutwest
        7: (2, 2),  # Southeast
    }
    dist_array = np.zeros(9).reshape((3, 3))
    dist_array[1, 1] = np.inf
    action_array = np.zeros(9).reshape((3, 3))

    for i, (d, a) in enumerate(zip(dist, action_probs)):
        dist_array[vn_map[i]] = d
        action_array[vn_map[i]] = a

    plt.subplot(233)
    action_array[1, 1] = np.nan
    plt.imshow(action_array, vmin=0, vmax=1)
    plt.title('Free Space Prob.')

    plt.subplot(236)
    dist_array[1, 1] = np.nan
    plt.imshow(-dist_array)
    plt.title('Negative Potential')

    plt.subplot(1, 3, (1, 2))
    plt.imshow(image)
    # plt.title(title)

    f = plt.gcf()
    axes = f.get_axes()

    for a in axes:
        a.axis('off')

    f.tight_layout()
    plt.savefig(path)
    plt.close(f)


def sample_maze_trajectories(env: MazeEnv, policy: Policy, n_trajectories: int, max_n_steps: int = 100,
                             path: str = None,
                             sample_start: bool = True, save_last: bool = False, score: bool = False,
                             stop_on_goal: bool = True) -> Tuple[
    np.ndarray, np.ndarray]:
    """Run the provided policy in the provided maze environment.

    Args:
        env(MazeEnv): Maze enrvironment.
        policy(Policy): Behavior policy.
        n_trajectories(int): Number of trajectories to try. A trajectory is defined as reaching the goal or
        max_n_steps.
        max_n_steps(int): Maximum number of steps in a given trajectory.
        path(str): Where to save the datamodule. If not provided, images are not rendered.
        sample_start(bool): Resample the sampling position between each trajectory. If false, the last position
        of the previous trajectory is kept.
        save_last(bool): Save the last image of the trajectory. Not recommended for data generation since we want
        transitions, but useful for nicer gifs.
        score(bool): Use to add probabilities and global distances in the images. Do not use for generating training
        images.
        stop_on_goal(bool): Stop running if goal is reached. Use for data collection to ensure all trajectories have
        the same length. This will break the success metric.

    Returns:
        ndarray of size (n_trajectories,): booleans indicating if a given trajectory was successful (goal reached).
        ndarray of size (n_trajectories,): number of steps per trajectory.

    """
    # Do some cleanup
    if path:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

    results = dict(success=[], steps=[])

    # Trajectories
    for t in range(n_trajectories):
        # Reset environment and get current goal
        env.reset()
        x_goal = env.get_goal_img()
        x_goal = np.array(Image.fromarray(x_goal).resize((256, 256)))
        current_reward = None

        # Save goal image
        if path:
            # Create a trajectory directory
            traj_path = os.path.join(path, f'traj_{t}')
            os.mkdir(traj_path)
            os.mkdir(os.path.join(traj_path, 'meta'))
            os.mkdir(os.path.join(traj_path, 'images'))
            Image.fromarray(x_goal).resize((256, 256)).save(os.path.join(traj_path, 'goal.png'))

            # Save goal pose
            # Other fields are dummy to keep the same json format
            # Cast numpy to standard int to avoid errors
            x, y = env.goal_idx[0]
            x, y = int(x), int(y)
            d = dict(
                time=0,
                actions=dict(motion=0),
                reward=0,  # Dummy reward for compatibility
                done=0,
                pose=dict(x=x, y=y)
            )
            with open(os.path.join(traj_path, f'goal.json'), 'w') as fp:
                json.dump(d, fp, indent=1)

        # Run episode
        scan = 1 - np.array(env.check_collisions())

        for n in range(max_n_steps):

            # Extract image and coordinates
            x_current = env.render('rgb_array')
            x_current = np.array(Image.fromarray(x_current).resize((256, 256)))

            # Save
            x, y = env.maze.objects.agent.positions[0]

            # Environment step
            a = policy.get_action(x_current, x_goal, current_reward, scan=scan)

            # Add optional score charts
            if score:
                scan = 1 - np.array(env.check_collisions())
                dist = np.zeros(len(scan))

            # Step environment
            _, current_reward, done, _ = env.step(a)

            # Save frame and datamodule
            if path:
                current_path = os.path.join(traj_path, 'images', f'{n}.png')
                if score:
                    # Save image with collision probabilities and global distances
                    plot_with_scores(path=current_path, dist=dist, action_probs=scan, image=x_current)
                else:
                    Image.fromarray(x_current).resize((256, 256)).save(current_path)
                x, y = int(x), int(y)
                d = dict(
                    time=int(n),
                    actions=dict(motion=int(a)),
                    reward=int(current_reward),
                    done=int(done),
                    pose=dict(x=x, y=y)
                )
                with open(os.path.join(traj_path, 'meta', f'{n}.json'), 'w') as fp:
                    json.dump(d, fp, indent=1)

            if done and stop_on_goal:
                if path and save_last:
                    # Render final image
                    current_path = os.path.join(traj_path, 'images', f'{n + 1}.png')
                    x_current = env.render('rgb_array')
                    x_current = np.array(Image.fromarray(x_current).resize((256, 256)))
                    if score:
                        scan = 1 - np.array(env.check_collisions())
                        dist = np.zeros(len(scan))
                        # Save image with probabilities and global distances
                        plot_with_scores(path=current_path, dist=dist, action_probs=scan, image=x_current)
                    else:
                        Image.fromarray(x_current).resize((256, 256)).save(current_path)
                break

        # Check if we reached goal and the number of steps
        results['success'].append(done)
        results['steps'].append(n + 1)

        if sample_start:
            env.sample_start()
        else:
            # Use current position as the starting point
            x, y = env.maze.objects.agent.positions[0]
            env.set_start(x, y)

        env.sample_goal(1)

    return np.array(results['success']), np.array(results['steps'])


if __name__ == '__main__':
    from src.models.policies.policies import RandomMaze, Dijkstra

    # Maze datamodule generation
    mazes = ['empty', 'umaze', 'omaze', 'hmaze', 'buttonmaze', 'room', 'two_doors', 'bugtrap', '9_rooms',
             'random_shape', 'random_shape1', 'random_shape2', 'random_shape3', 'random_shape4',  'random_shape5']
    agents = ['random']

    for m in mazes:
        for a in agents:
            env = get_maze_env(maze_name=m)
            policy = Dijkstra(env) if a == 'dijkstra' else RandomMaze(random_state=42)
            success, steps = sample_maze_trajectories(env, policy, 200, max_n_steps=25, sample_start=a == 'random',
                                                      path=os.path.join('data', f'{m}_{a}'), stop_on_goal=False)
            print(f'{success.sum()} goals achieved in {steps.sum()} steps in environment {m}.')
            # data_gif(os.path.join('data', f'{m}_{a}'), gif_path=os.path.join('datamodule', 'movies', f'{m}_{a}.gif'))
            path = os.path.join('./data', f'{m}_{a}')
            split_data(.5, path)
            print()
