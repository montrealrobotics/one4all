import numpy as np
import torch
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from src.models.policies.base_policy import Policy

class Random(Policy):
    """Random policy with proper collision checking and oracle stopping."""
    def __init__(self, goal_radius, limit, d_to_obstacle, random_state=42):
        self.rng = np.random.default_rng(random_state)
        self.goal_radius = goal_radius
        self.is_oracle = True

        # Parameters for scan box
        self.limit = limit
        self.d_to_obstacle = d_to_obstacle

    def setup_sim(self, sim):
        # Use shortest path follower to detect goals
        self.follower = ShortestPathFollower(sim, self.goal_radius, False)

    def reset_goal(self):
        pass

    def update_collision_prob(self):
        """Check which actions are likely to lead to collisions."""
        # Collision probs vector with zeros
        self.collision_probs = torch.zeros(3).float()

        # Fetch range and bearing measurement
        range_scan = self.current_scan[:, 1]
        bearing_scan = self.current_scan[:, 0]
        # measurement_mask = np.logical_and(bearing_scan >= -self.limit, bearing_scan <= self.limit)

        # Compute hitbox in front of agent
        x = range_scan * np.cos(bearing_scan)
        y = range_scan * np.sin(bearing_scan)
        measurement_mask = (np.abs(y) < self.limit) & (x > .00) & (x < self.d_to_obstacle)


        # Populate forward action with collision probability
        if measurement_mask.any():
            self.collision_probs[0] = 1.0

    def get_action(self, x_current: np.ndarray, x_goal: np.ndarray, prev_reward: int,
                   position: np.ndarray = None, scan: np.ndarray = None) -> int:
        expert_action = self.follower.get_next_action(position)
        self.current_scan = scan
        self.update_collision_prob()

        if expert_action == 0:
            # Calling STOP
            return 0

        if self.collision_probs[0]:
            # Forward collides, only sample rotations
            actions = [2, 3]
        else:
            # Include forward
            actions = [1, 2, 3]

        # Return random action
        return self.rng.choice(actions)
class ShortestPathPolicy(Policy):
    def __init__(self, goal_radius, random_p=0.0, random_state=42):
        self.random_p = random_p
        self.rng = np.random.default_rng(random_state)
        self.goal_radius = goal_radius
        self.is_oracle = True

    def setup_sim(self, sim):
        self.follower = ShortestPathFollower(sim, self.goal_radius, False)

    def reset_goal(self):
        pass

    def get_action(self, x_current: np.ndarray, x_goal: np.ndarray, prev_reward: int,
                   position: np.ndarray = None, scan: np.ndarray = None) -> str:
        prob = self.rng.uniform(size=1)
        if prob < self.random_p:
            # Return random action with probability self.random_p
            return self.rng.choice([1, 2, 3])
        else:
            # Return best action
            return self.follower.get_next_action(position)

