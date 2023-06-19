from typing import Dict, List, Optional, Tuple, Union

import scipy
import numpy as np
import glob
import os
import json
import re
from tqdm import tqdm
import networkx as nx

from skimage import io
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import radius_neighbors_graph

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.datamodule import save_graph, load_graph, remove_attributes, get_aug_keys

from src.models.local_backbone import LocalBackbone
from src.utils.rotations import Quaternion
from src.utils.se2 import SE2, ExpSE2
from src.utils import get_logger
from src.utils.motion_model import habitat_connectivity

log = get_logger(__name__)


def get_transforms(keys: List[str] = None, resolution: int = 256):
    """
    Return basic transforms (no augmentations). Use this for testing and inference.

        Args:
        :param keys: Keys to add to dictionary
        :param resolution: Size to resize image
    """
    if keys is None:
        anchor_keys, pos_keys = get_aug_keys(1, 1)
        keys = anchor_keys + sum(pos_keys, [])  # pos_keys is a list of list, make it flat

    center_crop_size = resolution
    t = [
        A.Resize(height=resolution, width=resolution, interpolation=cv2.INTER_AREA, always_apply=True),
        A.CenterCrop(center_crop_size, center_crop_size),
        A.ToFloat(),
        ToTensorV2()
    ]

    targets = {key: 'image' for key in keys}
    return A.Compose(t, additional_targets=targets)


def get_shift_transforms(keys: List[str], resolution: int = 256):
    """
    Return basic transforms + shift transforms with wrap borders. Used for maze.

        Args:
        :param keys: Keys to add to dictionary
        :param resolution: Size to resize image
    """
    center_crop_size = resolution
    # Transforms
    t = [
        A.Resize(height=resolution, width=resolution, interpolation=cv2.INTER_AREA, always_apply=True),
        A.CenterCrop(center_crop_size, center_crop_size),
        A.ShiftScaleRotate(shift_limit=1.00, scale_limit=0.00, rotate_limit=0, p=.5, border_mode=cv2.BORDER_WRAP),
        A.ToFloat(),
        ToTensorV2()
    ]

    targets = {key: 'image' for key in keys}

    return A.Compose(t, additional_targets=targets)


def get_augmented_transforms(keys: List[str], resolution: int):
    """
    Transforms with augmentations.

    Note: ResNet is used with CenterCrop augmentation at 224px.
    Based on https://arxiv.org/abs/1903.02531
    
        Args:
        :param keys: Keys to add to dictionary
        :param resolution: Size to resize image
        :param env: Name of the environment
    """
    center_crop_size = resolution
    # Transforms
    t = [
        A.Resize(height=resolution, width=resolution, interpolation=cv2.INTER_AREA, always_apply=True),
        A.MotionBlur(blur_limit=(3, 3), p=.5),  # Motion blur with Kernel Size 3x3
        A.Sharpen(alpha=(0.1, 0.2), lightness=(0.5, 1.0), p=0.25),
        # Maintain mostly of the original image and illuminate
        A.GaussNoise(var_limit=(10, 20), mean=0, p=0.5),  # Max and min variance range
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, p=0.5),
        A.CoarseDropout(max_holes=256, max_width=2, max_height=2, p=0.25),
        A.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.025, rotate_limit=5, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01, p=.5),
        A.CenterCrop(height=center_crop_size, width=center_crop_size, always_apply=True),
        A.ToFloat(),  # Returns float32 always between zero and one -- Use this or Normalize only
        ToTensorV2()
    ]

    targets = {key: 'image' for key in keys}

    return A.Compose(t, additional_targets=targets)


class One4AllDataset(Dataset):
    """Dataset used for O4A.

    The following datamodule structure is assumed :

    /datamodule
        /env_1
            /traj_0
                goal.png
                goal.json
                /images
                    0.png
                    1.png
                    ...
                /meta
                    0.json
                    1.json
                    ...
            /traj_1
            ...
        /env_2
        ...
        
    The graph structure is arrange as follows:
        grah.nodes[<node_id>]
        Nodes:
            gt: {'x': x, 'y': y, 'sin': sinθ, 'cos': cosθ}
            image_path: path_to_image

        graph.edges[<node1, node_2>]
        Edges:
            weight: edge_weight
            action: action_to_next_state
            reward: transition_reward

    Process the different trajectories samples from maze environments or the robot.

        Args:
            :param path: Path to root directory with data
            :param environment: Name of the environent
            :param env_type: Type of environment [maze, jackal, habitat]
            :param split: Either train or validation split
            :param k: Context length i.e. number of frames to be stacked
            :param dt: gap between two observations
            :param h: Horizon or number of time-steps in the future where a positive can be picked
            :param transform: Tensor transformations
            :param n_positives: Number of positives to sample from the updated graph to train a generator. Set to 0 to 
            use the standard geodesic regressor temporal positive
            :param any_positive: All samples can be used as positives. The graph is ignored
            :param gt_radius: Radius to connect ground truth samples. First position is position second is orientation -
            Note: Only used for monitoring/research - not used by the actual O4A model
            :param aug_anchor: flag to create a different augmentation of anchor, use to enforce distant 0
            :param negative_prob: Probability to enforce negatives to be from other chains
            :param panorama: Flag to use panoramas
    """

    def __init__(self, path: str = './datamodule', environment: str = 'umaze_random', env_type: str = 'maze',
                 split: str = 'train', k: int = 1, dt: int = 1, h: int = 1, transform: Optional[A.Compose] = None,
                 n_positives: int = 0, any_positive: bool = False, gt_radius: List[float] = [np.sqrt(2) + 0.01, 0.262],
                 aug_anchor: bool = False, negative_prob: float = .2, panorama: bool = False):

        self.split_raw = split  # Remember if we intend to use this for train, val or test

        # Condition to prevent issues with test set
        self.split = 'val' if split == 'test' else split  # Each dataset has two fold. Which one to use.
        self.path = os.path.join(path, environment, self.split)
        self.experiment = environment
        self.env_type = env_type
        self.k = k
        self.dt = dt
        self.h = h
        # Create undirected graph
        self.graph = nx.Graph()
        self.any_positive = any_positive
        # Radius to compute ground truth graph
        self.gt_radius = [gt_radius[0], gt_radius[1]]
        self.aug_anchor = aug_anchor
        self.negative_prob = negative_prob  # Ratio of samples that are converted to negatives for the classifier
        self.panorama = panorama

        if n_positives == 'temporal':
            self.n_positives = 1
            self.temporal = True
        elif isinstance(n_positives, int) and n_positives > 0:
            self.n_positives = n_positives
            self.temporal = False
        else:
            raise ValueError('n_positives should be "temporal" or a positive int')

        # List all trajectories and sort
        trajectories_path = [f.path for f in os.scandir(self.path) if f.is_dir()]
        trajectories_path.sort(key=lambda i: int(re.findall(r'\d+', os.path.basename(i))[0]))

        assert len(trajectories_path) > 0, "There are not trajectories for this environment."

        # Load datamodule of each trajectory
        for folder in trajectories_path:
            chain_graph = self.load_trajectory(folder)  # Number of edges are (states - 1)
            self.graph.update(chain_graph)

        # Backup original graph of chains as undirected - only needed to train local metric
        self.chain_graph: nx.Graph = remove_attributes(self.graph.copy(),
                                                       node_attr=['gt', 'image_path', 'action_counts'],
                                                       edge_attr=['action', 'reward'])

        # Create list of keys to pass augmentations
        self.anchor_aug_keys, self.pos_aug_keys = get_aug_keys(self.k, self.n_positives)
        # Only create ground truth graph if it is not already in memory
        if not os.path.exists(os.path.join(self.path, 'gt_graph.pkl')):
            log.info("Computing ground truth dataset for environment {}".format(self.experiment))
            gt_graph = self._build_gt_graph(self.graph)
            save_graph(os.path.join(self.path, 'gt_graph.pkl'), gt_graph)
        # Store transformations
        self.transforms = transform

    def get_ground_truth_graph(self):
        return load_graph(os.path.join(self.path, 'gt_graph.pkl'))

    def _build_gt_graph(self, graph):
        log.info('Computing ground truth graph for environment {}'.format(self.experiment))
        gt = nx.get_node_attributes(graph, "gt")
        # Connectivity radius
        r_xy, r_rot = self.gt_radius

        if self.env_type == 'maze':
            # GT of maze-like environments
            # Maze envs heading is composed of x-y coordinates only
            gt_array = np.asarray([[val['x'], val['y']] for _, val in gt.items()])
            # Compute connectivity graph of ground truth data
            r_xy += 0.1
            edges = self.euclidian_radius_neighbors(gt_array, r_xy)

            # Create final ground truth graph
            gt_graph = nx.create_empty_copy(graph)
            gt_graph.update(edges=edges)
        else:
            # Used to compute GT geodesic of SE(2) environments
            gt_array = np.asarray([[val['x'], val['y'], val['sin'], val['cos']] for _, val in gt.items()])
            mask = habitat_connectivity(gt_array, forward=self.gt_radius[0], angle=self.gt_radius[1])

            # Display some stats on connectivity
            mask_chains = nx.adjacency_matrix(self.chain_graph)
            chain_edges = mask_chains.sum()
            closure_edges = ((mask - mask_chains) == 1).sum()
            total = mask.sum()
            log.info(f"Chain (time) edges : {chain_edges} ({100 * chain_edges / total:.2f} %)")
            log.info(f"Closure edges      : {closure_edges} ({100 * closure_edges / total:.2f} %)")
            chain_not_in_gt = ((mask_chains - mask) == 1).sum()
            log.info(f"Chain edges not in gt  : {chain_not_in_gt}")

            # Compute graph edge weights
            rows, cols = mask.nonzero()
            weights = self.compute_geodesics(gt_array, mask)
            # Create edges
            edges = [(r, c, {'weight': w + 1e-6}) for r, c, w in zip(rows, cols, weights)]

            # Create undirected edges
            gt_graph = nx.create_empty_copy(graph)
            gt_graph.update(edges=edges)

        log.info(f"Is ground truth graph connected: {nx.is_connected(gt_graph)} - Environment {self.experiment}")
        return gt_graph

    def compute_geodesics(self, gt_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Compute geodesic distances for GT graph.
        Note: Only used for monitoring/research - not used by the actual O4A model

        Args:
            gt_array: Array with ground truth poses in SE(2)
            mask: Used to pick specific elements of the awway and not compute geodesics over all of them

        Returns:
            Array of geodesic distances
        """
        log.info("Computing geodesics for {} split - environment {}".format(self.split_raw, self.experiment))
        rows, cols = mask.nonzero()
        # Placeholder for geodesic distances
        geodesics = list()
        for i, j in tqdm(zip(rows, cols), total=len(rows), desc='Computing geodesics'):
            # Extract pose i
            x_i, y_i, sin_i, cos_i = gt_array[i]
            theta_i = np.arctan2(sin_i, cos_i)
            pose_i = SE2(pose=np.asarray([x_i, y_i, theta_i]))
            # Extract pose j
            x_j, y_j, sin_j, cos_j = gt_array[j]
            theta_j = np.arctan2(sin_j, cos_j)
            pose_j = SE2(pose=np.asarray([x_j, y_j, theta_j]))
            # Compute relative pose
            pose_ij = pose_i.invert().compose(pose_j)
            # Map to lie algebra and compute norm --> geodesic
            geo = np.linalg.norm(ExpSE2(pose_matrix=pose_ij).tau)
            geodesics.append(geo)
        return np.asarray(geodesics)

    def euclidian_radius_neighbors(self, position: np.ndarray,
                                   r: float,
                                   gtype: str = 'Translation') -> List[Tuple[int, int, Dict[str, float]]]:
        """
        Compute neighbors based on euclidian distance of x-y coordinates
        Note: Only used for monitoring/research - not used by the actual O4A model

        Args:
            position: X-Y coordinates as a ndarray
            r: Max allowed radius to connect neighbours
            gtype: Type of graph being computed, translation or rotation

        Returns:
            List of edges in the form (row, column {weight: value})
        """
        log.info("Computing {} graph.".format(gtype))
        A = radius_neighbors_graph(position, radius=r, mode='distance', include_self=True)
        # Connectivity matrix
        B = radius_neighbors_graph(position, radius=r, mode='connectivity', include_self=True)
        # get rows and cols of non-zero elements + weights - networkx populates zero valued edges
        rows, cols = B.nonzero()
        offset = 1e-6
        # Add extra weight for zero-valued vertex who are actually different example in graph when k!=1
        weights = np.asarray(A[rows, cols] + offset, dtype="float32").squeeze()
        # Create edges
        edges = [(r, c, {'weight': w}) for r, c, w in zip(rows, cols, weights)]
        return edges

    def load_trajectory(self, trajectory_path: str) -> nx.Graph:
        """
        Load the images within a trajectory folder and build a chain graph.

        Args:
            trajectory_path: Path to trajectory folder

        Returns:
            Chain graph of given trajectory. The structure of the chain graph is the same one as the self.G structure.
        """
        # Extract images and append subfolder
        images_path = [image for image in glob.glob(os.path.join(trajectory_path, 'images', "*.png"))]
        images_path = sorted(images_path, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        # Number of new states to be added
        number_new_states = len(images_path)
        # Count current number of nodes in Graph
        number_states = self.graph.number_of_nodes()
        # Chain graph and add edges with weight = 1
        local_graph = nx.path_graph(np.arange(number_states, number_states + number_new_states))
        nx.set_edge_attributes(local_graph, values=1, name='weight')

        for s in range(number_states, number_states + number_new_states, 1):
            # Read actions and reward
            with open(os.path.join(trajectory_path, 'meta', f'{s - number_states}.json'), 'r') as fp:
                d = json.load(fp)

            # Store ground-truth as node information
            if self.env_type in {'jackal', 'habitat'}:
                # Transform SO(3) to SO(2) by ignoring roll and pitch.
                q = Quaternion(w=d['pose']['orientation.w'], x=d['pose']['orientation.x'],
                               y=d['pose']['orientation.y'], z=d['pose']['orientation.z'])
                # Map heading to radians, note that to use Quaternions library it should be on deg
                heading = q.to_euler()[-1]
                local_graph.nodes[s]['gt'] = {'x': d['pose']['x'], 'y': d['pose']['y'],
                                              'sin': np.sin(heading), 'cos': np.cos(heading)}
            else:
                local_graph.nodes[s]['gt'] = d['pose']
            # Store path in node
            local_graph.nodes[s]['image_path'] = images_path[s - number_states]

            # Store transition in edge if not last node in chain
            if s < number_states + number_new_states - 1:
                # Actions are in a dict. We create a vector and make sure we always preserve the order.
                actions = list(d['actions'].values())
                # Add one to actions as position zero is for negatives
                # Maze doesn't currently have a stop action, need to add it to data collection
                # So we increment maze by two
                action_increment = 1 if self.env_type == 'habitat' else 2
                local_graph.edges[s, s + 1]['action'] = (np.asarray(actions) + action_increment).tolist()
                local_graph.edges[s, s + 1]['reward'] = d['reward']
        return local_graph

    def __len__(self) -> int:
        """
        Number of states (nodes) in the dataset

        Returns:
            Dataset size
        """
        return self.graph.number_of_nodes()

    def extract_sequence(self, idx: int) -> Tuple[List[List[int]], List[List[int]], int]:
        """
        Extract list of nodes or context length (k) connected to query node (idx) and spaced by gap (dt). An example
        would be the list [0, 1, 2, 3, 4, 5, 6, 7]. Given query node idx=3, k=3 and dt=2 the resulting list will be
        [1, 3, 5]. Additionally, this method returns a positive sample for the chain an in the previous example it would
        be 7.

        This method assumes that trajectory length are at least > (k + 1) * dt to extract chain and positive.

        Args:
            idx: Query node

        Returns:
            List of connected nodes in a chain given gap and context + list of positive example for the given chain.
            Also returns if the new positive is in the head (-1) or tail of the list (0). Return the target of the
            current sample.
        """
        # Extract list of connected nodes to query node and number of nodes in connected chain
        # Always use the original graph of chains for sequence computations (w/o loop closures)
        connected = list(nx.node_connected_component(self.chain_graph, idx))
        connected = sorted(connected)  # Make sure ids are sorted

        # Split connected list with dt gaps while maintaining query node within it.
        connected_spaced = connected[connected.index(idx) % self.dt::self.dt]
        # Extract chain of nodes
        i = connected_spaced.index(idx)
        chain = connected_spaced[np.maximum(i - self.k + 1, 0):i + 1]
        # Trim chain if len(chain) > context length
        while len(chain) > self.k:
            # Remove elements at left of right part of chain iteratively
            chain.pop(0)

        # Obtain index of chain with respect to original spaced chain
        mask = np.where(np.in1d(connected_spaced, chain))[0]
        max_id = mask[-1]
        # Check if there are samples ahead of anchor, otherwise repeat anchor and return [stop]
        is_future = max_id + 1 != len(connected_spaced)
        # Sample positive h steps away from anchor
        if is_future:
            # Compute valid samples in the future to sample from
            valid_samples = connected_spaced[max_id + 1:np.minimum(max_id + self.h + 1, len(connected_spaced))]
            positive = np.random.choice(valid_samples)
            # Set label to 2 which means extract true action
            label = 2
        else:
            positive = chain[-1]
            label = 1

        # Make sure the sequences are of length k
        if len(chain) < self.k:
            diff = self.k - len(chain)
            # Reconstruct anchors by repeating first element diff number of times
            chain = [chain[0]] * diff + chain

        # Flip a coin to maintain a positive or sample a negative
        non_connected = np.random.binomial(n=1, p=self.negative_prob)

        if non_connected:
            # Sample graph until a true negative (other chain) is obtained
            label = 0
            while True:
                positive = np.random.randint(len(self), size=1)[0]
                # Break loop if sample is not in chain
                if not (positive in connected):
                    break

        # For label, 0 -> NON_CONNECTED, 1 -> repeated or STOP, 2 -> placeholder to pick right ACTION
        return ([chain], [[positive]], label) if not self.panorama else ([[chain[-1]]], [[positive]], label)

    def _apply_transform(self, anchors: List[torch.Tensor],
                         positives: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transformation to tensor
        Args:
            anchors: Anchor sequence
            positives: List of positive sequences

        Returns:
            Transformed samples as a Tensor
        """
        # Add anchor sequence
        samples = {aug_name: sample for aug_name, sample in zip(self.anchor_aug_keys, anchors[0])}

        # Add sequence of positives
        for i, pos in enumerate(positives):
            samples.update({aug_name: sample for aug_name, sample in zip(self.pos_aug_keys[i], pos)})

        # Transform samples
        samples = self.transforms(**samples)

        # Get anchors
        anchors = torch.stack([samples[key] for key in self.anchor_aug_keys], dim=0)

        # Get positives - This is required since we may only have one positive image. Will pad up to k later
        actual_positives = len(positives[0])
        positives = [torch.stack([samples[key] for key in key_list[:actual_positives]], dim=0) for key_list in
                     self.pos_aug_keys]
        positives = torch.stack(positives, dim=0)

        return anchors.unsqueeze(0), positives

    def fetch_action(self, anchor_id: List[List[int]], positive_id: List[List[int]], target: int) -> torch.Tensor:
        """
        Fetch action for the anchor positive pair
            target = 0 is NON_CONNECTED
            target = 1 is STOP
            target = 2 is Retrieve action
        """
        action = list()
        # Negative
        if target == 0:
            action.append([0])
        # Repeated anchor
        elif target == 1:
            action.append([1])
        # Retrieve actual action
        else:
            # Anchor id - last position of list
            i = anchor_id[0][-1]
            # Iterate over positives to fetch action
            for j in positive_id:
                # If we select h positives ahead or chain is spaced, use action of next neighbour as target
                if self.h > 1 or self.dt > 1:
                    j = [i + 1]
                if 'action' in self.graph.edges[i, j[0]]:
                    _, a, _ = self.graph.edges[i, j[0]].values()
                else:
                    # -1 to denote unknown action
                    a = -1
                action.append(a)
        action = torch.tensor(action).type(torch.int32)
        return action

    def _load_samples(self, ids: List, prefix: str) -> Dict[str, torch.Tensor]:
        """
        Dic
        Args:
            ids: List of indices used to load samples
            prefix: Either anchores/positives

        Returns:
            Dict with sample composed of image, id and ground truth data (NOT USED).
        """
        # Temporal containers for samples and ground-truth
        samples, samples_gt = list(), list()
        for seq in ids:
            seq_sample, seq_gt = list(), list()
            for sample_id in seq:
                # For each node in the chain obtain its ground truth and respective image
                gt, sample_path = self.graph.nodes[sample_id].values()
                gt = np.array(list(gt.values()))
                sample = io.imread(sample_path)
                # Condition to split image in for chunks when using panoramas
                if self.panorama:
                    sample = np.array_split(sample, self.k, axis=1)
                    seq_sample.extend(sample)
                else:
                    seq_sample.append(sample)
                seq_gt.append(gt)
            samples.append(seq_sample)
            samples_gt.append(seq_gt)

        # Map ground truth samples and ids as ndarray for better slicing
        samples_gt = np.asarray(samples_gt)
        ids = np.asarray(ids)

        # Warning, samples are not a tensor yet, we apply transforms later
        # to enable uniform transforms across anchors and positives
        sample = {prefix: samples,
                  f'{prefix}_id': torch.tensor(ids[:, -1].squeeze()).type(torch.int64),
                  f'gt_{prefix}': torch.tensor(samples_gt[:, -1, :].squeeze())}

        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return current state.
        Args:
            idx: Index of anchor

        Returns:
            Dictionary with anchor and positive sample.

            Dimension of anchors are [n_positives, S, C, H, W]
            Dimension of positive is [n_positives, S, C, H, W]
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Variable to control how to augment sequence (positives)
        anchor_id, positive_ids, target = self.extract_sequence(idx=idx)
        sample = self._load_samples(anchor_id, 'anchors')

        # Check if we should sample positives
        if self.n_positives > 0 and not self.any_positive and not self.temporal:
            #############################################
            ## This condition is only used to train FD ##
            #############################################
            # Ignore temporal neighbor and sample from updated graph instead
            neighbors = list(self.graph.neighbors(idx))
            positive_ids = np.random.choice(neighbors, size=self.n_positives).reshape(-1, 1).tolist()
            # # Ignore temporal neighbor and sample from updated graph instead
            # successors = list(self.graph.successors(idx))
            # # Condition to prevent sampling empty list when populating list for the first time
            # successors = [idx] if len(successors) == 0 else successors
            # positive_ids = np.random.choice(successors, size=self.n_positives).reshape(-1, 1).tolist()
            # All actions for forward dynamics come from connection head - This is an instance of Co-training
            target = 0
        elif self.n_positives > 0 and self.any_positive and not self.temporal:
            ###################################################
            ## This condition is only used to train Geodesic ##
            ###################################################
            # Mask out anchors to prevent sampling same anchor as 'random goals'
            ind = np.arange(len(self))
            # mask = np.logical_not(np.in1d(ind, anchor_id))
            # ind = ind[mask]
            positive_ids = np.random.choice(ind, size=self.n_positives).reshape(-1, 1).tolist()
            # This method is only used for global, return action as NOT CONNECTED
            target = 0

        if self.n_positives != 0:
            # Load positive sample and its gt
            sample_pos = self._load_samples(positive_ids, 'positives')
            sample.update(sample_pos)

        # Apply transformations
        if self.transforms:
            # Augment anchors twice to obtain two representations of it
            if self.aug_anchor:
                sample['anchors_aug'], _ = self._apply_transform(sample['anchors'], sample['positives'])
            # Apply same augmentation to anchor and positive
            # Maze environments apply same augmentation to anchor and positive, habitat and jackal does not
            if self.env_type == 'maze':
                sample['anchors'], sample['positives'] = self._apply_transform(sample['anchors'], sample['positives'])
            # Augment anchor and positive independently
            else:
                anchors_aug, _ = self._apply_transform(sample['anchors'], sample['positives'])
                _, positives_aug = self._apply_transform(sample['anchors'], sample['positives'])
                sample['anchors'], sample['positives'] = anchors_aug, positives_aug

            # Positive sequence may be a single frame, in which case we expand the single frame to fill the sequence
            sample_i = sample['positives']
            n_pos, s, c, h, w = sample_i.size()
            sample_i = sample_i.expand((n_pos, self.k, c, h, w)) if s != self.k else sample_i
            sample['positives'] = sample_i
        # Add action to sample
        sample['action'] = self.fetch_action(anchor_id, positive_ids, target)
        # Add number of time-steps between samples - Only works with temporal sampling!
        sample['steps'] = sample['positives_id'] - sample['anchors_id']

        if hasattr(self, 'geodesics_len') and self.n_positives > 0:
            anchor_id, positive_ids = np.asarray(anchor_id), np.asarray(positive_ids)
            sample['geodesics_len'] = self.geodesics_len[anchor_id[:, -1].squeeze(), positive_ids[:, -1].squeeze()]
        return sample

    def compute_dijkstra_target(self) -> None:
        """
        Compute Dijkstra target to train geodesic regressor using updated graph.

        Returns:
            None
        """
        # Check if geodesics array already exists and use it
        geodesics_path = os.path.join(self.path, 'geodesics.npy')
        if os.path.exists(geodesics_path):
            with open(geodesics_path, 'rb') as f:
                self.geodesics_len = np.load(f)
            # If geodesics are successfully read, return
            return None
        # Precompute all the dijkstra stuff on the graph
        graph = nx.to_scipy_sparse_array(self.graph, nodelist=np.arange(len(self.graph)))

        # Check components in graph
        n_components, labels = scipy.sparse.csgraph.connected_components(graph,
                                                                         directed=False,
                                                                         return_labels=True)
        if n_components != 1:
            log.warning(f'Train graph is not connected. Has {n_components} components.')
            components, elements = np.unique(labels, return_counts=True)
            for c, e in zip(components, elements):
                log.warning("Component {} has {} elements!".format(c, e))

        log.info('Precomputing Dijkstra distances (this may take a while)...')
        # If predecessors = 9999, then there is not path between the two samples
        geodesics_len, predecessors = dijkstra(graph, return_predecessors=True, directed=False)
        self.geodesics_len = geodesics_len

        # Store newly computed geodesics in memory
        with open(geodesics_path, 'wb') as f:
            np.save(f, geodesics_len)


# k = 1
# dt = 1
# h = 1
# # env = 'castle_s1_15'
# env = 'omaze_random'
# n_positives = 'temporal'
# anchor_keys, pos_keys = get_aug_keys(k, 1)
# keys = anchor_keys + sum(pos_keys, [])  # pos_keys is a list of list, make it flat
# # t = get_augmented_transforms(keys=keys, resolution=96, env='habitat')
# t = get_transforms(keys=keys, resolution=64)
# # gt_radius = [1.0, 1.571]
# gt_radius = [1.5, 1.571]
# asd = One4AllDataset(gt_radius=gt_radius, k=k, dt=dt, n_positives=n_positives, h=h, env_type='maze', path='./data',
#                      environment=env, transform=t, aug_anchor=True, split='val', panorama=False, negative_prob=0.0)
# for idx in [0, 10, 17, 52, 63, 98, 99, 100, 247, 64, 355, 54, 25, 69, 458, 46, 299, 302]:
#     asd[idx]


class O4ADataModule(pl.LightningDataModule):
    """
    Pytorch Lighting wrapper for O4A Dataset.
    
        Args:
            :param data_dir: Path to root directory with data
            :param environment: Family of environments used ['maze', 'jackal', 'habitat']
            :param environments: List of environments for training
            :param val_environments: List of environments for validation
            :param test_environments: List of environments for testing
            :param dt: gap between two observations
            :param k: Context length i.e. number of frames to be stacked
            :param h: Horizon or number of timesteps in the future where a positive can be picked
            :param batch_size: Batch size. Recall that if using k > 1, effective batch size becomes batch_size * (k + 1)
            :param val_batch_size: Validation batch size. Keep this fixed for better model comparisons
            :param resize: Resize size for the input image
            :param num_workers: Workers used to process datamodule
            :param shuffle: Shuffle train set or not
            :param drop_last: Drop last batch if not of size batch size
            :param n_positives: Number of positives to sample. Set to 'temporal' to sample temporal neighbors from the
            original chain graph. Set to an int to sample n_positives from the updated graph. Set to 0 to only return
            anchors. Ignored if transitions=True
            :param any_positive: Ignore graph structure and sample any other sample as a positive
            :param gt_radius: Radius to connect ground truth samples. First position is position second is orientation -
            Note: Only used for monitoring/research - not used by the actual O4A model
            :param backbone_path: List of paths to trained local metrics checkpoint. If provided, used
            to update the graph during the setup phase
            :param compute_geodesics: Compute geodesic targets for global training
            :param per_env_batches: Make sure a given batch comes from a single environment. Useulf for local metric
            training since we use other samples in the batch as negatives (and want them to be from the same env)
            :param augmentations: Choice of augmentations. Set to False for no augmentations
            :param aug_anchor: flag to create a different augmentation of anchor, use to enforce distant 0
            :param panorama: Whether to use panoramas or not
            :param negative_prob: Probability to enforce negatives to be from other chains
    """

    def __init__(
            self,
            data_dir: str = "./datamodule",
            environment: str = 'maze',
            environments: List[str] = ["umaze_random", "omaze_random"],
            val_environments: List[str] = ["empty_random"],
            test_environments: List[str] = ["room_random"],
            dt: int = 1,
            k: int = 1,
            h: int = 1,
            batch_size: int = 32,
            val_batch_size: int = 64,
            resize: int = 64,
            num_workers: int = 2,
            shuffle: bool = False,
            drop_last=True,
            n_positives: Union[str, int] = 'temporal',
            any_positive: bool = False,
            gt_radius: List[float] = [np.sqrt(2) + 0.01, 0.262],
            backbone_path: List[str] = None,
            compute_geodesics: bool = False,
            per_env_batches: bool = False,
            augmentations: Union[bool, str] = False,
            aug_anchor: bool = False,
            negative_prob: float = 0.00,
            panorama: bool = False
    ):

        super().__init__()
        # This line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # Set dataset containers
        self.train_set: Optional[List[Dataset]] = None
        self.val_set: Optional[List[Dataset]] = None
        self.test_set: Optional[List[Dataset]] = None

        # Maximum frames we may need to augment
        n_positives = 1 if self.hparams.n_positives == 'temporal' else self.hparams.n_positives
        anchor_keys, pos_keys = get_aug_keys(self.hparams.k, n_positives)
        keys = anchor_keys + sum(pos_keys, [])  # pos_keys is a list of list, make it flat

        if self.hparams.augmentations == 'augmented':
            # Training augmentations
            self.train_transform = get_augmented_transforms(resolution=self.hparams.resize, keys=keys)
        elif self.hparams.augmentations == 'shift':
            # Training augmentations for Maze
            self.train_transform = get_shift_transforms(resolution=self.hparams.resize, keys=keys)
        elif self.hparams.augmentations == False:
            # Val and Test augmentations
            self.train_transform = get_transforms(resolution=self.hparams.resize, keys=keys)
        else:
            raise ValueError('Invalid choice of augmentations.')

        # Never use transforms for validation
        self.val_transform = get_transforms(resolution=self.hparams.resize, keys=keys)

    def _loop_closure(self, dataset: Dataset, dataloader: DataLoader, backbone_path):

        # Back up transforms and use no transforms for embedding
        transforms = dataset.transforms  # Backup transforms
        # Fetch gt graph number of edges
        number_gt_edges = dataset.get_ground_truth_graph().number_of_edges()
        dataset.transforms = self.val_transform

        # Inference
        model = LocalBackbone.load_from_checkpoint(backbone_path)
        model.freeze()
        model.eval()
        trainer = pl.Trainer(accelerator='auto')
        preds = trainer.predict(model, dataloader)
        z = torch.cat([d['anchors'] for d in preds])
        ids = torch.cat([d['anchors_id'] for d in preds])

        model.update_graph(anchors=z.cpu().numpy(), indices=ids.view(-1).tolist(), update=True,
                           dataset=dataset, number_gt_edges=number_gt_edges)

        # Reapply original transforms
        dataset.transforms = transforms

    def _setup_environment(self, dataset: Dataset, split: str, env: str) -> None:
        """
        Set up a single environment either for train or val

        Args:
            dataset: Dataset object
            split: Split, options are train, val and test
            env: Environment name
        """

        # Assign train datasets for use in dataloaders
        current_set = dataset(path=self.hparams.data_dir,
                              environment=env,
                              env_type=self.hparams.environment,
                              split=split, dt=self.hparams.dt, k=self.hparams.k, h=self.hparams.h,
                              transform=self.train_transform if split == 'train' else self.val_transform,
                              n_positives=self.hparams.n_positives,
                              any_positive=self.hparams.any_positive,
                              gt_radius=self.hparams.gt_radius,
                              negative_prob=self.hparams.negative_prob,
                              # Anchor is only augmented in training
                              aug_anchor=self.hparams.aug_anchor if split == 'train' else False,
                              panorama=self.hparams.panorama)

        # Loop closure in train set if backbone_path is provided
        if self.hparams.backbone_path is not None:
            self._loop_closure(current_set,
                               DataLoader(
                                   current_set,
                                   batch_size=self.hparams.batch_size if split == 'train' else self.hparams.val_batch_size,
                                   drop_last=self.hparams.drop_last,
                                   shuffle=self.hparams.shuffle,
                                   num_workers=self.hparams.num_workers),
                               self.hparams.backbone_path)

        # Compute global network target
        if self.hparams.compute_geodesics:
            current_set.compute_dijkstra_target()

        # Store train or test env in list
        if split == 'train':
            # Store validation set
            self.train_set.append(current_set)
        elif split == 'val':
            # Store validation set
            self.val_set.append(current_set)
        else:
            # Store validation set
            self.test_set.append(current_set)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup dataset
        """
        dataset = One4AllDataset

        # Create List of train and Val Envs and iterate over those
        self.train_set = list()
        self.val_set = list()
        self.test_set = list()

        # Set up all splits
        for split, envs in [
            ('train', self.hparams.environments),
            ('val', self.hparams.val_environments),
            ('test', self.hparams.test_environments)
        ]:
            for e in envs:
                self._setup_environment(dataset, split, e)

    def train_dataloader(self, test_on_train=False) -> DataLoader:
        """
        Load Train Dataloader
        """
        if test_on_train:
            log.info('Turning off augmentations on train set...')
            # Turn off augmentations in train set
            for d in self.train_set:
                d.transforms = self.val_transform
                d.aug_anchor = False

        # Concat all datasets in one big one
        if self.hparams.per_env_batches:
            # Return multiple dataloaders where each one represent an environment
            # This means training_step will receive a list of batches instead of a unique batch
            dataloaders = list()
            workers_per_env = max(1, self.hparams.num_workers // len(self.hparams.environments))

            for env, train_set in zip(self.hparams.environments, self.train_set):
                dataloaders.append(DataLoader(train_set,
                                              batch_size=self.hparams.val_batch_size,
                                              drop_last=self.hparams.drop_last,
                                              shuffle=self.hparams.shuffle,
                                              num_workers=workers_per_env))
        else:
            # Combine all environments. Batches will have samples from multiple environments
            concat_dataset = ConcatDataset(self.train_set)
            dataloaders = DataLoader(concat_dataset,
                                     batch_size=self.hparams.batch_size,
                                     drop_last=self.hparams.drop_last,
                                     shuffle=self.hparams.shuffle,
                                     num_workers=self.hparams.num_workers)
        return dataloaders

    def val_dataloader(self) -> DataLoader:
        """
        Load Validation Dataloader
        """
        # Return multiple dataloaders where each one represent an environment
        dataloaders = list()
        for env, val_dataset in zip(self.hparams.val_environments, self.val_set):
            dataloaders.append(DataLoader(val_dataset,
                                          batch_size=self.hparams.val_batch_size,
                                          drop_last=self.hparams.drop_last,
                                          shuffle=self.hparams.shuffle,
                                          num_workers=self.hparams.num_workers))

        return dataloaders

    def test_dataloader(self) -> DataLoader:
        """
        Load Validation Dataloader
        """
        # Return multiple dataloaders where each one represent an environment
        dataloaders = list()
        for env, test_dataset in zip(self.hparams.test_environments, self.test_set):
            dataloaders.append(DataLoader(test_dataset,
                                          batch_size=self.hparams.val_batch_size,
                                          drop_last=self.hparams.drop_last,
                                          shuffle=self.hparams.shuffle,
                                          num_workers=self.hparams.num_workers))

        return dataloaders
