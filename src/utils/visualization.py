from typing import List, Tuple

import os
import shutil

import os.path

import networkx as nx
import imageio
import numpy as np
from PIL import Image as Image
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import kde
from tqdm import tqdm

import torch


# From https://stackoverflow.com/questions/53074947/examples-for-search-graph-using-scipy
def get_path(Pr, j):
    path = [j]
    k = j
    while Pr[k] != -9999:
        path.append(Pr[k])
        k = Pr[k]
    return path[::-1]


def visualize_spurious_edges(gt_graph: nx.Graph, graph: nx.Graph,
                             rho: int, dataset: object, n_samples: int = 10) -> Tuple[object, np.ndarray, float]:
    """
    Method to visualize spurious edges.
    Note: Only used for monitoring/research - not used by the actual O4A model.
    
    Args:
        gt_graph: Ground truth graph
        graph: Loop closure graph
        rho: threshold to obtain spurious edges
        dataset: Dataset object to fetch image samples
        n_samples: Number of samples to be plotted
    Returns:
        matplotlib fig with spurious edges, ndarray with spurious edges and worst edge distance
    """
    # Obtain adjacency matrices
    adj_graph = nx.adjacency_matrix(graph)
    adj_gt = nx.adjacency_matrix(gt_graph)
    adj_gt_bin, adj_graph_bin = (adj_gt > 0).astype(int), (adj_graph > 0).astype(int)
    # Compute rows and cols of false positives
    adj_fp = (adj_graph_bin - adj_gt_bin) == 1
    rows, cols = adj_fp.nonzero()
    # Placeholder for spourious edges
    anchor_id, positive_id, gt_weight = list(), list(), list()
    worst_edge_dist = 0
    for i, j in tqdm(zip(rows, cols), total=len(rows), desc="Fetching spurious edges"):
        gt_i, sample_path_i, = gt_graph.nodes[i].values()
        gt_j, sample_path_j, = gt_graph.nodes[j].values()
        # Compute distance only based on position
        gt_i, gt_j = np.array(list(gt_i.values()))[:2], np.array(list(gt_j.values()))[:2]
        dist = np.linalg.norm(gt_i - gt_j)
        # store only far away edges
        if dist > rho:
            anchor_id.append(i)
            positive_id.append(j)
            gt_weight.append(dist)

        # Store the worst edge distance
        worst_edge_dist = max(worst_edge_dist, dist)
    anchor_id = np.asarray(anchor_id)
    positive_id = np.asarray(positive_id)
    gt_weight = np.asarray(gt_weight)
    pred_weight = list()
    # Randomly sample n_samples from the list
    fig = None
    # Condition to prevent plotting when there are not large spurious edges
    if len(anchor_id) > 0:
        # Fetch index worst edges
        idx = np.argsort(gt_weight)[-n_samples:]
        # idx = np.random.randint(len(anchor_id), size=n_samples)
        anchor_id = anchor_id[idx]
        positive_id = positive_id[idx]
        gt_weight = gt_weight[idx]
        # Check if panoramas are being used
        panorama = dataset.panorama
        # Number of rows and cols for image and iterators for plot
        rows = n_samples
        cols = 2
        iterator = 1
        fig = plt.figure(figsize=(8, 12))
        for i, j, weight in zip(anchor_id, positive_id, gt_weight):
            # Plot anchor
            anchor = dataset[i]['anchors'][0]  # Squeeze batch dim
            # Plot positive
            positive = dataset[j]['positives'][0]  # Squeeze batch dim and assume n_positives = 1
            if panorama:
                # Extend image along width as it is split on four parts
                _, _, h, w = anchor.size()
                # Seq dim, Channel dim, Width, Height
                anchor = anchor.permute(0, 1, 3, 2)
                # First dimension are number of samples i.e. k
                k = anchor.size(0)
                anchor = torch.hstack(tuple([anchor[elem] for elem in range(k)])).permute(2, 1, 0)
                positive = positive.permute(0, 1, 3, 2)
                positive = torch.hstack(tuple([positive[elem] for elem in range(k)])).permute(2, 1, 0)
            else:
                # take only last position in the list as that is anchor
                anchor = anchor[-1, :, :, :].permute(1, 2, 0)
                positive = positive[-1, :, :, :].permute(1, 2, 0)
            # Plot anchor
            fig.add_subplot(rows, cols, iterator)
            plt.imshow(anchor)
            plt.axis('off')
            plt.title("Anchor idx {} - GT edge {:.3f}".format(i, weight))
            # Increase iterator and plot positive
            iterator += 1
            fig.add_subplot(rows, cols, iterator)
            plt.imshow(positive)
            plt.axis('off')
            plt.title("Positive idx {} - Pred edge {:.3f}".format(j, adj_graph[i, j]))
            pred_weight.append(adj_graph[i, j])
            iterator += 1

    return fig, np.vstack((anchor_id, positive_id, gt_weight, np.asarray(pred_weight))), worst_edge_dist


def plot_spurious_edges_graph(graph: nx.Graph, spurious_edges: np.ndarray, environment: str = 'maze') -> object:
    """
    Plot the connectivity graph for spurious edges
    Args:
        graph: Computed graph
        spurious_edges: ndarray with spurious edges information
        environment: Environment type
    Returns:
        Matplotlib figure of connectivity graph
    """

    # Pick parameters based on environment
    fig_size, node_size = ((16, 16), 50) if environment in {'jackal', 'habitat'} else ((8, 8), 80)
    fig = plt.figure(figsize=fig_size)

    # Extract ground truth
    ground_truth = nx.get_node_attributes(graph, 'gt')
    gt_list = np.array([[p['x'], p['y']] for p in ground_truth.values()])
    gt_list = gt_list if environment in {'jackal', 'habitat'} else np.concatenate((gt_list[:, 1].reshape(-1, 1),
                                                                                  -gt_list[:, 0].reshape(-1, 1)),
                                                                                  axis=1)
    position = dict(zip(np.arange(graph.number_of_nodes()), gt_list.tolist()))
    # Obtain min and max values from ground truth to create grid
    x_max, y_max, x_min, y_min = gt_list[:, 0].max(), gt_list[:, 1].max(), gt_list[:, 0].min(), gt_list[:, 1].min()
    # Color based on sum of x + y coordinates
    c = gt_list[:, 0] - gt_list[:, 1] if environment == 'maze' else gt_list[:, 0] + gt_list[:, 1]
    node_color = dict(zip(np.arange(graph.number_of_nodes()), c))
    # Color nodes based on weight only if there are edges in graph
    labels_node = dict()
    if len(spurious_edges) != 0:
        weights = np.round(spurious_edges[2, :], 2).tolist()
        weights_pred = np.round(spurious_edges[3, :], 2).tolist()
        edges = [(i, j) for i, j in zip(spurious_edges[0, :], spurious_edges[1, :])]
        label = {e: 'GT {} - Pred {}'.format(w_gt, w_pred) for e, w_gt, w_pred in zip(edges, weights, weights_pred)}
        labels_node = {idx: int(idx) for idx in spurious_edges[:2, :].flatten()}

        nx.draw_networkx_edges(graph, position, node_size=node_size, edgelist=list(edges), edge_color=list(weights),
                               edge_vmin=0.0, edge_cmap=plt.cm.Reds, width=2.0, alpha=0.6)
        nx.draw_networkx_edge_labels(graph, position, edge_labels=label, font_color='black')
        # Add color-bar to plot
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0.0, vmax=np.array(weights).max()))
        sm.set_array([])
        plt.colorbar(sm, label='Distance', extend='both')
    # Draw nodes in the graph
    nx.draw_networkx_nodes(
        graph,
        position,
        nodelist=list(node_color.keys()),
        node_size=node_size,
        node_color=list(node_color.values()),
        cmap=plt.cm.plasma)
    nx.draw_networkx_labels(graph, position, labels=labels_node, font_color='black')
    # Limits offset
    offset = 0.2
    plt.xlim(x_min - offset, x_max + offset)
    plt.ylim(y_min - offset, y_max + offset)
    plt.axis("off")
    plt.tight_layout()
    plt.title("Spurious edges", y=-0.01)
    return fig


def plot_2d_latent(coordinates: np.ndarray, values: np.ndarray) -> object:
    """
    Plot Embedding obtained with Local Metric of given dataset.

    Args:
        coordinates: Embedding representation of current state (image)
        values: Value use to colour embeddings

    Returns:
        Embedding plot
    """
    # Look at the first two components if we have more than 2D
    if coordinates.shape[1] > 2:
        coordinates = PCA(n_components=2).fit_transform(coordinates)

    # Plot image
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    _ = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=values, cmap="plasma")
    ax.axis('off')
    plt.tight_layout()
    return fig


def plot_connectivity_graph(graph: nx.Graph, environment: str = 'maze',
                            n_edges: int = 10000, plot_type: str = 'pos') -> object:
    """
    Plot the connectivity network of given graph

    Args:
        graph: Graph of the type of environment
        environment: Current type of environment
        n_edges: Number of edges to plot
        plot_type: Type of plot, options are 'pos' and 'rot'

    Returns:
        Matplotlib figure of connectivity graph
    """

    # Pick parameters based on environment
    fig_size, node_size = ((16, 16), 50) if environment in {'jackal', 'habitat'} else ((8, 8), 80)
    fig = plt.figure(figsize=fig_size)

    # Extract ground truth
    ground_truth = nx.get_node_attributes(graph, 'gt')
    if plot_type == 'rot':
        gt_list = np.array([[p['sin'], p['cos']] for p in ground_truth.values()])
    else:
        gt_list = np.array([[p['x'], p['y']] for p in ground_truth.values()])
    gt_list = gt_list if environment in {'jackal', 'habitat'} else np.concatenate((gt_list[:, 1].reshape(-1, 1),
                                                                                   -gt_list[:, 0].reshape(-1, 1)),
                                                                                  axis=1)
    position = dict(zip(np.arange(graph.number_of_nodes()), gt_list.tolist()))
    # Obtain min and max values from ground truth to create grid
    x_max, y_max, x_min, y_min = gt_list[:, 0].max(), gt_list[:, 1].max(), gt_list[:, 0].min(), gt_list[:, 1].min()
    # Color based on sum of x + y coordinates
    c = gt_list[:, 0] - gt_list[:, 1] if environment == 'maze' else gt_list[:, 0] + gt_list[:, 1]
    node_color = dict(zip(np.arange(graph.number_of_nodes()), c))

    # Color nodes based on weight only if there are edges in graph
    if graph.number_of_edges() != 0:
        edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
        edges, weights = list(edges), list(weights)

        # Subsample edges
        idx = np.random.randint(low=0, high=len(weights), size=n_edges)
        edges = [edges[i] for i in idx]
        weights = [weights[i] for i in idx]

        # Filter out self loops (sharing same position)
        new_edges = []
        new_weights = []
        for e, w in zip(edges, weights):
            # Do not add edges that are self loops
            if np.linalg.norm(np.array(position[e[0]]) - np.array(position[e[1]])) > 1e-1:
                new_edges.append(e)
                new_weights.append(w)

        if len(new_edges):
            e = nx.draw_networkx_edges(
                graph,
                position,
                node_size=node_size,
                edgelist=list(new_edges),
                edge_color=list(new_weights),
                edge_vmin=0.0,
                edge_cmap=plt.cm.copper,
                width=0.6,
                alpha=0.6)
        # Add color-bar to plot
        sm = plt.cm.ScalarMappable(cmap=plt.cm.copper,
                                   norm=plt.Normalize(vmin=0.0, vmax=np.array(weights).max()))
        sm.set_array([])
        plt.colorbar(sm, label='Distance', extend='both')

    # Draw nodes in the graph
    nx.draw_networkx_nodes(
        graph,
        position,
        nodelist=list(node_color.keys()),
        node_size=node_size,
        node_color=list(node_color.values()),
        cmap=plt.cm.plasma,
    )
    # Limits offset
    offset = 0.2
    plt.xlim(x_min - offset, x_max + offset)
    plt.ylim(y_min - offset, y_max + offset)
    plt.axis("off")
    plt.tight_layout()
    plt.title(f"Is {plot_type} graph weakly connected? {nx.is_weakly_connected(graph)}", fontsize=16, y=-0.01)
    return fig


def data_gif(path: str, gif_path: str = 'movie.gif', agent_name: str = 'Agent', frame_skip: int = 1,
             duration: float = .1, success: np.ndarray = None, make_top_down: bool = False):
    """Take a datamodule folder produced by sample_maze_trajectories and generate a gif with all frames.

    Args:
        path(str): Path to datamodule folder.
        gif_path(str): Path of target gif.
        agent_name(str): Agent name. Used as title for the agent plots.
        frame_skip(int): Skip frames. 1 will use every frame. 2 will use every 2 frames etc.
        duration(float): duration of each frame in the gif.
        success(np.ndarray): Boolean array describing which trajectories were not successful. If provided,
        only failures will be included in the gif.
        make_top_down(float): If create gif with top_down map, used in habitat envs
    """
    if os.environ.get('SLURM_TMPDIR'):
        temp_path = os.path.join(os.environ.get('SLURM_TMPDIR'), 'temp_gif_factory')
    else:
        temp_path = 'temp_gif_factory'
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.mkdir(temp_path)

    n_traj = len(os.listdir(path))
    frame_counter = 0
    files = list()

    for traj in range(n_traj):
        if success is not None:
            # Skip successful runs
            if success[traj]:
                continue

        # Goal image
        traj_str = f'traj_{traj}'
        img_goal = np.asarray(Image.open(os.path.join(path, traj_str, 'goal.png')).convert("RGB"))
        img_goal = Image.fromarray(np.split(img_goal, 2, 1)[0])

        n_images = len(os.listdir(os.path.join(path, traj_str, 'images')))
        for n in range(0, n_images, frame_skip):
            img_current = Image.open(os.path.join(path, traj_str, 'images', f'{n}.png')).convert("RGB")

            fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
            for a in axes:
                a.set_xticks([])
                a.set_yticks([])
            axes[0].imshow(img_current)
            axes[0].axis('off')
            axes[0].set_title(agent_name)
            top_down = Image.open(os.path.join(path, traj_str, 'top_down', f'{n}.png')).convert("RGB")
            axes[1].imshow(top_down)
            axes[1].set_title(f'Top down - Trajectory {traj + 1}')
            axes[1].axis('off')
            axes[2].imshow(img_goal)
            axes[2].set_title(f'Goal')
            axes[2].axis('off')
            plt.tight_layout()

            file_name = os.path.join(temp_path, f'{frame_counter}.png')
            files.append(file_name)
            plt.savefig(file_name)
            plt.close()

            frame_counter += 1
        # Copy final frame to better visualize if the goal was achieved
        for _ in range(10):
            files.append(files[-1])

    if os.path.exists(gif_path):
        os.remove(gif_path)

    # with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
    with imageio.get_writer(gif_path, mode='I', fps=1 / duration) as writer:
        for filename in files:
            image = imageio.imread(filename)
            writer.append_data(image)
    # Remove temp gif factory
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
