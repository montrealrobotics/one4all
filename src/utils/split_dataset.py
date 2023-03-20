import os
import shutil

import PIL.Image as Image
import json

import numpy as np


class DatasetWriter:
    def __init__(self, base_path: str, dataset_name: str, write: bool = True):
        self.data_path = os.path.join(base_path, dataset_name)
        self.current_traj_path = None
        self.current_traj_path_im = None
        self.current_traj_path_meta = None
        self.current_traj_path_scan = None
        self.current_traj_path_tm = None
        self.traj_counter = 0
        self.obs_counter = 0
        self.write = write

        # Remove old data folder
        if os.path.exists(self.data_path):
            shutil.rmtree(self.data_path)

    def new_trajectory(self):
        if self.write:
            # Update paths
            self.current_traj_path = os.path.join(self.data_path, f"traj_{self.traj_counter}")
            self.current_traj_path_im = os.path.join(self.current_traj_path, "images")
            self.current_traj_path_meta = os.path.join(self.current_traj_path, "meta")
            self.current_traj_path_tm = os.path.join(self.current_traj_path, "top_down")
            self.current_traj_path_scan = os.path.join(self.current_traj_path, "scan")

            # Create directories
            os.makedirs(self.current_traj_path_im)
            os.makedirs(self.current_traj_path_meta)
            os.makedirs(self.current_traj_path_tm)
            os.makedirs(self.current_traj_path_scan)

        # Increment trajectory counter
        self.traj_counter += 1

        # Reset observation counter
        self.obs_counter = 0

    def write_observation(self, image: np.ndarray, meta: dict, scan: np.ndarray,
                          top_down_image: np.ndarray = None, panorama: bool = False,
                          pan_delta: int = None, use_hq: bool = False):
        if self.write:
            # Define size of image - hardcoded!
            im_size = 256 if use_hq else 128
            # Save panorama image
            if panorama:
                n_images = int(360 / pan_delta)
                im_width = im_size * n_images
                Image.fromarray(image).resize((im_width, im_size)).save(os.path.join(self.current_traj_path_im,
                                                                                     f"{self.obs_counter}.png"))
            # Save standard image
            else:
                Image.fromarray(image).resize((im_size, im_size)).save(os.path.join(self.current_traj_path_im,
                                                                                    f"{self.obs_counter}.png"))
            # Save Metadata
            with open(os.path.join(self.current_traj_path_meta, f'{self.obs_counter}.json'), 'w') as fp:
                json.dump(meta, fp, indent=1)
            # Save laser scar as npy
            with open(os.path.join(self.current_traj_path_scan, f'{self.obs_counter}.npy'), 'wb') as f:
                np.save(f, scan)

            if top_down_image is not None:
                Image.fromarray(top_down_image).resize((im_size, im_size)).save(os.path.join(self.current_traj_path_tm,
                                                                                             f"{self.obs_counter}.png"))
        self.obs_counter += 1

    def write_goal(self, image: np.ndarray, meta: dict, panorama: bool = False, pan_delta: int = None):
        if self.write:
            # Define size of image - hardcoded!
            im_size = 128
            if panorama:
                n_images = int(360 / pan_delta)
                im_width = im_size * n_images
                Image.fromarray(image).resize((im_width, im_size)).save(os.path.join(self.current_traj_path,
                                                                                     f"goal.png"))
            else:
                Image.fromarray(image).resize((im_size, im_size)).save(os.path.join(self.current_traj_path,
                                                                                    f"goal.png"))
            with open(os.path.join(self.current_traj_path, f'goal.json'), 'w') as fp:
                json.dump(meta, fp, indent=1)


def split_data(split: float = 0.2, path: str = './data/umaze_random') -> None:
    """
    Split dataset into train and validation for a given maze

    Note: To see expected datastructure see dataset.py.
    The following datastructure will be generated:

    /datamodule
        /env_1
            /train
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
            /eval
                /traj_n
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
                /traj_m
                ...
        /env_2
        ...

    Args:
        split: Percentage of trajectories used for validation
        path: Path of the dataset (maze)

    """
    # dst locations
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')

    # Check if train is already created
    if os.path.isdir(train_path):
        if os.listdir(train_path):
            print("Train and Val split already done.")
            return None
        else:
            print("Train directory is empty, creating...")
    else:
        print("Splitting dataset...")

    # List all subfolders
    trajectory_folders = [f for f in os.listdir(path)]
    # Count number of trajectories
    n_traj = len(trajectory_folders)
    # Grab the first chunk of trajectories as training
    train_trajectories = trajectory_folders[:-round(n_traj * split)]
    val_trajectories = trajectory_folders[-round(n_traj * split):]

    # Create directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # Move folders to proper directory
    for train_t in train_trajectories:
        shutil.move(os.path.join(path, train_t), train_path)

    print('Train directory created.')
    for val_t in val_trajectories:
        shutil.move(os.path.join(path, val_t), val_path)
    print('Validation directory created')
