import numpy as np

from mazelab.solvers import dijkstra_solver
from src.models.policies.base_policy import Policy


class RandomMaze(Policy):
    """Random policy for maze lab with uniform actions."""

    def __init__(self, random_state: int):
        self.rng = np.random.default_rng(random_state)

    def get_action(self, x_current: np.ndarray, x_goal: np.ndarray, prev_reward: int, scan: np.ndarray = None) -> int:
        return self.rng.choice(8, size=1)[0]


class Dijkstra(Policy):
    """Dijkstra on fully observed passable map.

    This policy has access to privileged information."""

    def __init__(self, env: object):
        # Need access to env to figure out paths
        self.env = env
        self.goal = np.array([[-1, -1]])
        self.actions = None

    def get_action(self, x_current: np.ndarray, x_goal: np.ndarray, prev_reward: int, scan: np.ndarray = None) -> int:
        goal = self.env.goal_idx

        # If goal has changed we resolve
        if not (goal == self.goal).all():
            array = self.env.unwrapped.maze.to_impassable()
            motions = self.env.unwrapped.motions
            start = self.env.unwrapped.maze.objects.agent.positions[0]
            goal = self.env.unwrapped.maze.objects.goal.positions[0]
            self.actions = dijkstra_solver(array, motions, start, goal)

        return self.actions.pop(0)
