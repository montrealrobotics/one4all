from typing import Optional
import numpy as np


class SE2:
    def __init__(self, pose: Optional[np.ndarray] = None, pose_matrix: Optional[np.ndarray] = None):
        if pose_matrix is not None:
            self.g = pose_matrix
        elif pose is not None:
            x = pose[0]
            y = pose[1]
            theta = pose[2]
            rot_matrix = np.asarray([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta), np.cos(theta)]])
            g = np.eye(3)
            g[:2, :2] = rot_matrix
            g[0, -1], g[1, -1] = x, y
            self.g = g
        else:
            raise NotImplementedError("Provide a pose in cartesian coordinates or in SE(2)")

    def __str__(self):
        msg = '[[{}  {}  {}]\n' \
              ' [{}  {}  {}]\n' \
              ' [{}  {}  {}]]'.format(self.g[0, 0].round(3), self.g[0, 1].round(3), self.g[0, 2].round(3),
                                      self.g[1, 0].round(3), self.g[1, 1].round(3), self.g[1, 2].round(3),
                                      self.g[2, 0].round(3), self.g[2, 1].round(3), self.g[2, 2].round(3))
        return msg

    def __repr__(self):
        msg = '[[{}  {}  {}]\n' \
              ' [{}  {}  {}]\n' \
              ' [{}  {}  {}]]'.format(self.g[0, 0].round(3), self.g[0, 1].round(3), self.g[0, 2].round(3),
                                      self.g[1, 0].round(3), self.g[1, 1].round(3), self.g[1, 2].round(3),
                                      self.g[2, 0].round(3), self.g[2, 1].round(3), self.g[2, 2].round(3))
        return msg

    def invert(self, update: bool = False):
        # Invert matrix
        g = np.eye(3)
        g[:2, :2] = self.g[:2, :2].T
        g[:2, -1] = (-self.g[:2, :2].T @ self.g[:2, -1][:, np.newaxis]).squeeze()
        if update:
            self.g = g
        return SE2(pose_matrix=g)

    def compose(self, g_):
        # Compound transformations and return a new one
        new_g = self.g @ g_.g
        return SE2(pose_matrix=new_g)


class ExpSE2:
    def __init__(self, pose_matrix: Optional[SE2] = None, tau: Optional[np.ndarray] = None):
        if tau is not None:
            self.tau = tau
        elif pose_matrix is not None:
            self.g = pose_matrix.g
            self.tau = self.log_map()
        else:
            raise NotImplementedError("Provide a pose in SE(2) or exp coordinates")

    @staticmethod
    def skew_symmetric_so2(theta):
        return np.asarray([[0, -theta], [theta, 0]])

    def hat_se2(self, tau: Optional[np.ndarray] = None):
        tau = tau if tau is not None else self.tau
        tau_hat = np.zeros((3, 3))
        tau_hat[:2, :2] = self.skew_symmetric_so2(tau[-1])
        tau_hat[0, -1], tau_hat[1, -1] = tau[0], tau[1]
        return tau_hat

    def vee_se2(self, tau_hat):
        return np.asarray([tau_hat[0], tau_hat[1], tau_hat[2]])

    def exp_max(self):
        theta = self.tau[-1]
        # Compute Jacobian SE(2)
        jac = (np.sin(theta) / theta) * np.eye(2) + \
              ((1 - np.cos(theta)) / theta) * self.skew_symmetric_so2(1)
        # Obtain rotation matrix
        g = np.eye(3)
        traslation = (jac @ self.tau[:2][:, np.newaxis]).squeeze()  # Eq. (6-7)
        rotation = np.asarray([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        g[:2, :2] = rotation
        g[:2, -1] = traslation

        return SE2(pose_matrix=g)

    def log_map(self):
        # Compute logmap SO(2)
        theta = np.arctan2(self.g[1, 0], self.g[0, 0])
        # Compute Jacobian SE(2)
        # Edge case when theta is zero
        if theta != 0.0:
            jac = (np.sin(theta) / theta) * np.eye(2) + \
                  ((1 - np.cos(theta)) / theta) * self.skew_symmetric_so2(1)
        else:
            jac = np.eye(2)
        # Compute translation component
        rho = (np.linalg.inv(jac) @ self.g[:2, -1]).squeeze()
        tau = np.asarray([rho[0], rho[1], theta])
        return tau

    def __str__(self):
        msg = '[{}  {}  {}]'.format(self.tau[0], self.tau[1], self.tau[2])
        return msg

    def __repr__(self):
        msg = '[{}  {}  {}]'.format(self.tau[0], self.tau[1], self.tau[2])
        return msg
