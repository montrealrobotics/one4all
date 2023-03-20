from tqdm import tqdm

import numpy as np
import scipy


def habitat_connectivity(gt_array: np.ndarray, forward: float, angle: float) -> np.ndarray:
    """
    Use habitat action model to know if poses in gt_array are connected or not.
    Note: Only used for monitoring/research - not used by the actual O4A model

    Args:
        gt_array: GT array of poses
        forward: Distance covered by forward action
        angle: Angle covered when rotating

    Returns:
        Array with GT connectivity
    """
    # Habitat motions
    FORWARD_D = forward  # Distance covered when going forward
    ANGLE = angle  # Angle covered when rotating
    STOP = np.eye(4)  # Don't move
    FORWARD = np.array([
        # Add polar coordinates (unit norm) to current position. Make sure heading and position are in the same frame tho
        [1., 0., 0., FORWARD_D],
        [0., 1., FORWARD_D, 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ])

    ROTATE_LEFT = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., np.cos(ANGLE), -np.sin(ANGLE)],
        [0., 0., np.sin(ANGLE), np.cos(ANGLE)],
    ])

    ROTATE_RIGHT = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., np.cos(-ANGLE), -np.sin(-ANGLE)],
        [0., 0., np.sin(-ANGLE), np.cos(-ANGLE)],
    ])

    ACTIONS = [STOP, FORWARD, ROTATE_LEFT, ROTATE_RIGHT]
    n_actions = len(ACTIONS)
    n_dims = gt_array.shape[1]
    n_samples = gt_array.shape[0]

    actions = np.stack(ACTIONS)
    forward_codes = actions @ gt_array.T

    # Reformat things
    forward_codes = forward_codes.reshape((n_actions, n_dims, n_samples, 1))
    test_array = gt_array.T.reshape((1, n_dims, 1, n_samples))

    # Assess code by code to prevent bottleneck in computation
    rows, cols, data = list(), list(), list()
    for code_id in tqdm(range(n_samples), desc='Forwarding codes through motion model...'):
        # Hacky way to prevent dropping dimension
        code = forward_codes[:, :, code_id:code_id + 1, :]
        # Compute distance between and all other codes
        dist = np.linalg.norm(code - test_array, axis=1)
        is_close = dist < 0.125
        is_connected = is_close.sum(axis=0)
        # Count nonzero rows-cols i.e. ground truth samples
        row, col = np.nonzero(is_connected)
        weight = is_connected[row, col]
        # Remember row is always zero as we iterate row by row, correct index with code_id
        rows.extend((np.ones_like(row) * code_id).tolist())
        cols.extend(col.tolist())
        data.extend(weight.tolist())

    result = scipy.sparse.csr_matrix((data, (rows, cols)))

    return result
