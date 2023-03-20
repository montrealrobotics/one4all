"""Adapted from https://gist.githubusercontent.com/erikwijmans/1c596183fb8de1e9afd80088b6b5c115/raw/52e235a1e40f13a2a7d68995543667cfb4f16264/create_pointnav_dataset.py"""
import glob
import gzip
import json
import multiprocessing
import os
import os.path as osp
from itertools import product

import tqdm

import habitat
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode

num_episodes_per_scene = int(1750)

def inflate_agent(sim, radius=0.25):
    """Change radius of agent and alter the navmesh accordingly."""
    navmesh_settings = sim.pathfinder.nav_mesh_settings
    navmesh_settings.agent_radius = radius  # @param {type:"slider", min:0.01, max:0.5, step:0.01}
    sim.recompute_navmesh(
        sim.pathfinder, navmesh_settings, include_static_objects=False
    )
    print(f"Agent radius is now {sim.pathfinder.nav_mesh_settings.agent_radius}")

difficulty_setting = {
    "easy": {
        "closest_dist_limit" : 1.5,
        "furthest_dist_limit": 3.0,
    },
    "medium": {
        "closest_dist_limit": 3.0,
        "furthest_dist_limit": 5.0,
    },
    "hard": {
        "closest_dist_limit": 5.0,
        "furthest_dist_limit": 10.0,
    },
    "very_hard": {
        "closest_dist_limit": 10.0,
        "furthest_dist_limit": 100.0,
    },
    "data_collection": {
        "closest_dist_limit": 5.0,
        "furthest_dist_limit": 100.0,
    },
}


def _generate_fn(scene, difficulty):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR"]  # Useless sensor to avoid renderer issue
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    inflate_agent(sim, .1)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    bounds = difficulty_setting[difficulty]
    dset.episodes = list(
        generate_pointnav_episode(
            sim, num_episodes_per_scene, is_gen_shortest_path=False,
            geodesic_to_euclid_min_ratio=1.00,  # 1 will allow straight lines. Increasing it will favor hard paths
            # Easy -> 1.5 -- Medium -> 3 -- Hard -> 5 -- Original for data collection 5
            # closest_dist_limit=5.0,  # With a step size of .25, should give us at least 20 steps
            # Easy -> 3 -- Medium -> 5 -- Hard -> 10  -- Original for data collection 100
            # furthest_dist_limit=10.0,  # With a step size of .25, should give us at max 400 steps
            **bounds
        )
    )
    for ep in dset.episodes:
        ep.scene_id = ep.scene_id[len("./data_habitat/scene_datasets/") :]

    scene_key = osp.splitext(osp.basename(scene))[0]
    out_file = f"./data_habitat/datasets/pointnav/o4a/{scene_key}_{difficulty}.json.gz"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())


scenes = glob.glob("./data_habitat/scene_datasets/gibson/*.glb")
with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
    difficulties = list(difficulty_setting.keys())
    for _ in pool.starmap(_generate_fn, product(scenes, difficulties)):
        pbar.update()

with gzip.open(f"./data_habitat/datasets/pointnav/o4a/all.json.gz", "wt") as f:
    json.dump(dict(episodes=[]), f)
