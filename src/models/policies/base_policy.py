from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    @abstractmethod
    def get_action(self, x_current: np.ndarray, x_goal: np.ndarray,
                   prev_reward: int, scan: np.ndarray) -> int:
        """Get goal-conditioned action.

        Args:
            x_current(ndarray): Current image.
            x_goal(ndarray): Goal image.
            prev_reward(int): Reward from the previous step.
            scan(ndarray): Depth scan from current agent pose

        Returns:
            int: Action.

        """
        raise NotImplementedError

    def prev_step_collided(self):
        """Check if a collision was predicted during previous step."""
        return False
