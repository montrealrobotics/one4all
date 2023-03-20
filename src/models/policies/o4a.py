"""O4A Policy.

Avoid simulator imports in this file for easier deployment on robot.

This file uses legacy names for components.
Local metric = Local Backbone + connectivity head
Global metric = Geodesic Regressor
"""
import time
from typing import Tuple

import numpy as np
import torch

from src import utils
from src.datamodule.dataset import get_aug_keys
from src.models.base_model import BaseModel
from src.models.forward_dynamics import ForwardDynamicsHead

log = utils.get_logger(__name__)


class One4All(BaseModel):
    """One-4-All Policy.

    Implemented as a potential field planner. A step can be described as :
        - Generate waypoints based on a forward dynamics model applied to current location.
        - Identify actions leading to collisions using a collision checker. Defines the collision penalty.
        - Pick the action leading to the waypoint minimizing the potential defined as :
            - The intrinsic distance to goal +
            - Collision penalty +
            - Clipped distance to the previously visited n_visited states. Only triggered if close to visited state.


        Args:
            local_metric_path: Path to local metric checkpoint.
            global_metric_path:  Path to global metric checkpoint. Trained over local codes.
            fd_path: Path to forward dynamics network. Trained over local codes.
            transform: Transforms to be applied to the images before passing them to the models.
            att_factor: Scaling factor applied to the attractive force to the goal.
            rep_factor: Scaling factor applied to the repulsive force from previously visited states.
            repulsor_radius: Radius for neighborhood computations. Used to determine if we are in the neighborhood of a
            previously visited state.
            n_visited: Number of visited states to keep in the buffer. A repulsive force to avoid visited states
            is computed.
            collision_len: Length of the collision detection box (pointing in front of the agent)
            collision_width: Half the width of the collision detection box (orthogonal to the collision_len
            collision_penalty: Collision penalty added to potentials of action/waypoint predicted as not being not free.
            stop_d_local: Agent calls stop if local distance to goal is < stop_d_local.
    """

    def __init__(self, local_metric_path:str, global_metric_path:str=None, fd_path:str=None,
                 transform=None, att_factor:float=1., rep_factor:float=1., repulsor_radius:float=1.1, n_visited:int=10,
                 collision_len: float = .26, collision_width: float = .1, collision_penalty:float=10.,
                 stop_d_local:float=0.):
        """Init."""
        super().__init__()
        self.transform = transform
        self.save_hyperparameters(logger=False)

        # State descriptors
        self.current_image = None  # Current image
        self.state_local = None  # Current location local code
        self.state_global = None  # Current location global code

        # Goal descriptors
        self.goal_local = None  # Goal local code
        self.goal_global = None  # Goal global code

        # Waypoints and potential descriptors
        self.waypoints_local = None
        self.waypoints_global = None
        self.actions = None
        self.collision_probs = None
        self.potentials = None
        self.d_to_goal_local = np.inf
        self.d_to_goal_global = np.inf

        # Attribute for repulsive force of visited states
        self.visited_states = []
        self.prev_goal = None
        self.last_waypoint = None

        # Collision detection parameters
        self.collision_len = collision_len
        self.collision_width = collision_width  # Angle limit

        # Duration (used to compute time statistics)
        self.duration = 0
        self.duration_max = 0

        # Load all neural components
        self.setup_components()
        self.eval()
        # Prevent backbone of contrasting and map head to proper device
        self.backbone.contrast = False
        self.backbone.connectivity_head = self.backbone.connectivity_head.to(self.device)

        # Extract number of 'panorama' images for the model
        self.k = self.backbone.hparams.net.k

        # Forward dynamics are not loaded by parent class
        if fd_path is not None:
            self.fd_head = ForwardDynamicsHead.load_from_checkpoint(fd_path, strict=False)
            self.fd_head.freeze()
            self.fd_head.eval()

        # Transform keys
        self.current_aug_keys, _ = get_aug_keys(k=self.k, n_positives=1)
        self.goal_aug_keys = [f"{key}_goal" for key in self.current_aug_keys]
        self.keys = self.current_aug_keys + self.goal_aug_keys
        self.prev_action = None

        # Check device
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        print(f"O4A device is {self.device}")

    def get_current_state(self) -> torch.Tensor:
        return self.state_local

    def get_goal(self) -> torch.Tensor:
        return self.goal_global

    def prev_step_collided(self) -> bool:
        """Check if a collision was predicted during previous step."""
        # We only use 0 or 1. See update_collision_prob
        return (self.collision_probs > .0).any()

    def img_to_albumentation_sample(self, x_current, x_goal) -> torch.Tensor:
        x_current = np.array_split(x_current, self.k, axis=1)
        x_current = {key: sample for key, sample in zip(self.current_aug_keys, x_current)}
        x_goal = np.array_split(x_goal, self.k, axis=1)
        x_goal = {key: sample for key, sample in zip(self.goal_aug_keys, x_goal)}
        x_current.update(x_goal)
        return x_current

    def albumentation_sample_to_tensors(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
        current = [sample[key] for key in self.current_aug_keys]
        goal = [sample[key] for key in self.goal_aug_keys]
        current = torch.stack(current, dim=0).to(self.device)
        goal = torch.stack(goal, dim=0).to(self.device)
        return current, goal

    def update_state_goal_codes(self, x_current: np.ndarray, x_goal: np.ndarray):
        """Run inference on images to state code, goal code and waypoints."""
        sample = self.img_to_albumentation_sample(x_current, x_goal)
        sample = self.transform(**sample)
        x_current, x_goal = self.albumentation_sample_to_tensors(sample)
        self.current_image = torch.clone(x_current)

        # Compute  local embeddings
        x = torch.stack((x_current, x_goal)).to(self.device)
        local_codes, _, _ = self.backbone(x.unsqueeze(0))
        self.state_local, self.goal_local = local_codes[0].unsqueeze(0), local_codes[1].unsqueeze(0)

        # Compute global embeddings if global head is detected
        if self.global_head is not None:
            global_codes = self.global_head(local_codes)
            self.state_global, self.goal_global = global_codes[0].unsqueeze(0), global_codes[1].unsqueeze(0)

        # Compute distances to goal
        if self.global_head is not None:
            self.d_to_goal_global = self.global_head.head(
                self.state_global.float(),
                self.goal_global.float(),
            ).double().squeeze()
        else:
            self.d_to_goal = np.inf
        self.d_to_goal_local = torch.linalg.norm(self.state_local - self.goal_local, ord='fro')

    def update_waypoints_actions(self):
        """Forward all actions to generate waypoints."""
        # This vector is of dim (actions - 2) since we exclude NOT CONNECTED and STOP
        # NOT CONNECTED is already excluded so we only need to exclude the first component again
        # self.waypoints_local = self.fd_head.net.forward_all_actions(self.state_local)[0][1:]
        self.waypoints_local = self.fd_head.net.forward_all_actions(self.state_local)[0]
        self.actions = torch.arange(self.waypoints_local.shape[0]) + 1

    def update_collision_prob(self):
        """Check which actions are likely to lead to collisions."""
        # Collision probs vector with zeros
        self.collision_probs = torch.zeros(3).float().to(self.state_local.device)

        # Fetch range and bearing measurement
        range_scan = self.current_scan[:, 1]
        bearing_scan = self.current_scan[:, 0]
        # measurement_mask = np.logical_and(bearing_scan >= -self.limit, bearing_scan <= self.limit)

        # Compute hitbox in front of agent
        x = range_scan * np.cos(bearing_scan)
        y = range_scan * np.sin(bearing_scan)
        measurement_mask = (np.abs(y) < self.collision_width) & (x > .00) & (x < self.collision_len)


        # Populate forward action with collision probability
        if measurement_mask.any():
            self.collision_probs[0] = 1.0

    def update_visited_states(self):
        """Update buffer of visited states."""
        self.reset_visited_states()  # Check if we need to reset visited states because of goal change
        state = self.state_local if self.last_waypoint is None else self.last_waypoint
        self.visited_states.append(state.clone().squeeze())

        # Keep buffer at most
        if len(self.visited_states) > self.hparams.n_visited:
            self.visited_states = self.visited_states[1:]


    def reset_visited_states(self):
        """Reset buffer of visited states if goal is new"""
        goal = self.goal_local

        # Check if goal is new. If so, remove previously visited states from buffer
        if self.prev_goal is not None:
            d = torch.norm(self.prev_goal - goal).item()
            if d > .01:
                self.visited_states = []

                # Also reset distance to goal
                self.d_to_goal_local = np.inf

        # Save new goal
        self.prev_goal = goal


    def update_state(self, x_current: np.ndarray, x_goal: np.ndarray, scan: np.ndarray) -> None:
        """Update all attributes given new input."""
        # Update state & goal codes
        self.update_state_goal_codes(x_current, x_goal)

        # Save scan data
        self.current_scan = scan

        # Check collisions
        self.update_collision_prob()

        # Update visited states
        self.update_visited_states()

        # Update waypoints and actions
        self.update_waypoints_actions()

        # Push waypoints in global space
        if self.global_head is not None:
            self.waypoints_global = self.global_head(self.waypoints_local)


    def compute_attractive_potentials(self) -> torch.Tensor:
        """Attractor to goal in global space."""
        # Global planning using geodesic distances
        potentials = self.global_head.head.forward_all_codes(
            self.waypoints_global.float(),
            self.get_goal().float(),
            deployment=True
        ).double().squeeze()
        if not potentials.ndim:
            # Torch item. Map to 1D vector of size 1
            potentials = potentials.reshape((1,))

        return self.hparams.att_factor * potentials

    def compute_repulsive_potentials(self) -> torch.Tensor:
        """Repulsors from visited states in local space."""
        # Only triggered if we have visited states + a positive repulsive factor
        if len(self.visited_states) and self.hparams.rep_factor > 0:
            # Only trigger repulsive force in radius
            last_anchors = torch.stack(self.visited_states, dim=0)
            rep_force = self.hparams.repulsor_radius - torch.cdist(last_anchors, self.waypoints_local)
            rep_force = torch.clip(rep_force, min=0)
            rep_force = rep_force.sum(dim=0)
        else:
            rep_force = torch.zeros(self.waypoints_local.shape[0]).to(self.waypoints_local.device)

        return self.hparams.rep_factor * rep_force

    def compute_potentials(self):
        # Attractive force
        self.potentials = self.compute_attractive_potentials()

        # Repulsive force
        self.potentials += self.compute_repulsive_potentials()

        # Collision penalty
        mask = self.collision_probs > .01
        self.potentials[mask] += self.hparams.collision_penalty

    def goal_action(self):
        """Check if goal is reachable"""
        a_to_goal = self.backbone.connectivity_head(self.state_local.float(), self.goal_local.float())

        return a_to_goal.squeeze().argmax().item()

    def check_stop(self):
        """Check if we should call STOP."""

        # Get connectivity prediction to goal
        goal_action = self.goal_action()

        if goal_action > 0:
            print("Calling STOP because goal is reachable.")
            return True


        if self.d_to_goal_local < self.hparams.stop_d_local:
            print(f"Calling STOP because local distance is under {self.hparams.stop_d_local}.")
            return True

        return False


    def get_action(self, x_current: np.ndarray, x_goal: np.ndarray, prev_reward: int,
                   position: np.ndarray = None, scan: np.ndarray = None) -> int:
        """Predict action.

        Position is only present for compatibility reasons. DO NOT USE as this describes the actual position of the
        goal.
        """
        start_time = time.time()
        with torch.no_grad():
            # Update current state
            self.update_state(x_current, x_goal, scan)

            if self.check_stop():
                # Call stop
                action = 0
            else:
                # Compute potentials
                self.compute_potentials()

                # Greedily pick action
                # Slice 1 to ignore the STOP action, this is handled by goal_action
                next_id = self.potentials.argmin().item()
                action = self.actions[next_id].item()
                waypoint = self.waypoints_local[next_id]

                # Save last waypoint
                self.last_waypoint = waypoint
                self.prev_action = action

        # Update duration for benchmarking
        duration = time.time() - start_time
        self.duration += duration
        if duration > self.duration_max:
            self.duration_max = duration

        return action

    def score(self) -> Tuple[np.ndarray, np.ndarray]:
        """Utility function to generate visualizations with collision checking and forward dynamics.

        Return collision probabilities and potentials."""
        # FIXME hot fix to remove NON_CONNECTED and STOP for maze envs
        return 1 - self.collision_probs.cpu().numpy()[1:], self.potentials.cpu().numpy()[1:]