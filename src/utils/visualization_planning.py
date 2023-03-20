import shutil

import scipy
import pytorch_lightning as pl
import torch
import networkx as nx

from src.models.policies.policies import One4All
from src.datamodule.dataset import P2VDataModule
from src.utils.eval_metrics import a_star_eval_global
from src.utils.visualization import data_gif


def gen_planning_trajectories(local_metric_path: str, global_metric_path: str, environment: str,
                              n_paths: int, eps: float = 1.1, split: str ='train', data_path: str = './datamodule',
                              resample_start: bool = True) -> None:
    """Generate gif of planning A* planning trajectories

    Args:
        local_metric_path: Path to trained local metric.
        global_metric_path: Path to trained global head.
        environment: String describing environment.
        n_paths: Number of episodes to sample.
        eps: Threshold to determine edges on the graph.
        split: Train or val.
        data_path: Data path.
        resample_start: If starting positions should be resampled. If False, the previous goal is used as the new start.

    """
    # Load model and dataset
    trainer = pl.Trainer(accelerator='gpu', devices=1, deterministic=True,
                         max_epochs=1, check_val_every_n_epoch=0)

    resize = 256 if environment == 'jackal' else 64
    p2v_dataset = P2VDataModule(data_dir=data_path, environment=environment, batch_size=64,
                                resize=resize, num_workers=4, shuffle=False, drop_last=False)
    p2v_dataset.setup()

    loader = p2v_dataset.train_dataloader() if split == 'train' else p2v_dataset.val_dataloader()
    data_module = p2v_dataset.train_set if split == 'train' else p2v_dataset.val_set

    # Models
    system = One4All(local_metric_path, global_metric_path)

    # Get model predictions and ground truth coordinates
    output = trainer.predict(system, dataloaders=loader)
    local_codes = torch.cat([x["local_code"] for x in output])
    global_codes = torch.cat([x["global_code"] for x in output])
    gt = torch.cat([x["gt_anchor"] for x in output])
    indices = torch.cat([x["anchor_id"] for x in output])

    # Build local graph
    # Build graph from local metric
    print('Building graph with local metric...')
    nx_graph = data_module.update_graph(anchors=local_codes, s_id=indices, eps=eps, batch_size=64)
    local_metric_graph = nx.to_scipy_sparse_array(nx_graph)

    # Debugging visualizations
    # from src.utils.visualization import plot_connectivity_graph, plot_2d_latent
    # target = gt.squeeze(1).cpu().numpy()
    # fig = plot_2d_latent(local_codes.cpu().numpy(),  target[:, 0] - target[:, 1])
    # fig.show()
    #
    # fig = plot_2d_latent(global_codes.cpu().numpy(),  target[:, 0] - target[:, 1])
    # fig.show()
    #
    # fig = plot_connectivity_graph(nx_graph, environment=environment)
    # fig.show()

    n_components = scipy.sparse.csgraph.connected_components(local_metric_graph, directed=False, return_labels=False)
    if n_components != 1:
        raise ValueError(f'Graph is not connected. Has {n_components} components.')

    # Compute A* trajectories
    _, _, paths = a_star_eval_global(local_metric_graph, n_paths=n_paths, global_codes=global_codes,
                                     resample_start=resample_start)

    # Save the frames and meta info in the standard dataset format
    new_env_name = environment + '_planning'
    data_path = os.path.join(data_path, new_env_name)
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)

    for i, traj in enumerate(paths):
        traj_path = os.path.join(data_path, f'traj_{i}', 'images')
        meta_path = os.path.join(data_path, f'traj_{i}', 'meta')
        os.makedirs(traj_path)
        os.makedirs(meta_path)

        # Save goal
        img_path = nx_graph.nodes[traj[-1]]['image_path']

        # Infer meta path
        meta_path_goal = img_path.split(os.sep)
        meta_path_goal[-2] = 'meta'
        meta_path_goal[-1] = meta_path_goal[-1].replace("png", "json")
        meta_path_goal = os.path.join(*meta_path_goal)

        # Move files
        shutil.copy(img_path, os.path.join(data_path, f'traj_{i}', f"goal.png"))
        shutil.copy(meta_path_goal, os.path.join(data_path, f'traj_{i}', f"goal.json"))

        for j, k in enumerate(traj):
            img_path = nx_graph.nodes[k]['image_path']

            # Infer meta path
            meta_path_k = img_path.split(os.sep)
            meta_path_k[-2] = 'meta'
            meta_path_k[-1] = meta_path_k[-1].replace("png", "json")
            meta_path_k = os.path.join(*meta_path_k)

            # Move files
            shutil.copy(img_path, os.path.join(traj_path, f"{j}.png"))
            shutil.copy(meta_path_k, os.path.join(meta_path, f"{j}.json"))

    # Generate gif
    data_gif(data_path, os.path.join('movies', new_env_name + '.gif'), 'A* with global heuristic')

    # Delete generated datamodule
    shutil.rmtree(data_path)


if __name__ == '__main__':
    import os

    local_path = os.path.join('results', 'local_omaze_random_emb_8_seed_42_adam_lr_0.0005-v1.ckpt')
    global_path = os.path.join('results', 'global_head_omaze_random_emb_8_seed_42_adam_lr_0.0005-v10.ckpt')

    gen_planning_trajectories(local_metric_path=local_path, global_metric_path=global_path, n_paths=10,
                              environment='omaze_random', split='val', resample_start=False)
