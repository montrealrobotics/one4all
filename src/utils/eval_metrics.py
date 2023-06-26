"""Evaluation Metrics."""
from typing import Tuple, List, Callable, Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import torch

import networkx as nx


def compute_graph_clf_metrics(adj_gt, adj_graph, suffix):
    n_edges = adj_graph.shape[0] ** 2  # Consider all edges

    p = adj_gt.count_nonzero()  # All actual positives
    tp = adj_graph.multiply(adj_gt).count_nonzero()  # Predicted true positives
    fn = p - tp
    fp = adj_graph.sum() - tp
    tn = n_edges - tp - fn - fp

    # Metrics
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 / (1 / precision + 1 / recall) if recall and precision else 0.0
    acc = (tp + tn) / n_edges

    return {
        f'acc{suffix}': acc,
        f'f1{suffix}': f1,
        f'precision{suffix}': precision,
        f'recall{suffix}': recall,
    }


def sp_in_range(sp_matrix, low, high):
    low_sp = (low < sp_matrix).astype(int)
    high_sp = (sp_matrix < high).astype(int)
    return low_sp.multiply(high_sp)


class GraphMetrics:
    """
    Utility class to compute accuracy, precision and recall of edge connections in graph
    Note: Only used for monitoring/research - not used by the actual O4A model
    """

    def __init__(self):
        pass

    def __call__(self, graph: nx.Graph, gt_graph: nx.Graph) -> Dict[str, float]:
        """
        Compute graph metrics (MSE, ACC, Precision and Recall) used privileged ground truth information

        Args:
            graph: Adjacency matrix of predicted graph
            gt_graph: Adjacency matrix of ground truth graph

        Returns:
            RMSE, Accuracy, Precision and Recall
        """
        adj_graph = nx.adjacency_matrix(graph)
        adj_gt = nx.adjacency_matrix(gt_graph)

        # Total number of possible edges in graph
        N = adj_gt.shape[0]
        N = N * (N - 1) / 2

        # Compute L2 Norm between ground truth and current graph (2N(N-1)/2)
        rmse = 1 / np.sqrt(2 * N) * norm(adj_gt - adj_graph, ord='fro')
        metrics = dict(rmse=rmse)

        # Cast adjacency matrices as binary matrices
        adj_gt_all, adj_graph_all = (adj_gt > 0).astype(int), (adj_graph > 0).astype(int)

        # Compute metrics for edge connectivity in general
        metrics.update(compute_graph_clf_metrics(adj_gt_all, adj_graph_all, suffix=""))

        # Note: from here on, this is habitat specific
        # STOP metrics
        adj_gt_stop = sp_in_range(adj_gt, 0, .1)
        adj_pred_stop = sp_in_range(adj_graph, 0, .1)
        metrics.update(compute_graph_clf_metrics(adj_gt_stop, adj_pred_stop, suffix="_stop"))

        # FORWARD metrics
        adj_gt_f = sp_in_range(adj_gt, .245, .255)
        adj_pred_f = sp_in_range(adj_graph, .245, .255)
        metrics.update(compute_graph_clf_metrics(adj_gt_f, adj_pred_f, suffix="_forward"))

        # ROTATE metrics
        adj_gt_r = sp_in_range(adj_gt, 0.255, 0.3)
        adj_pred_r = sp_in_range(adj_graph, 0.255, 0.3)
        metrics.update(compute_graph_clf_metrics(adj_gt_r, adj_pred_r, suffix="_rotation"))

        return metrics


def edges_minus_chains_recovered(gt_graph: nx.DiGraph, chain_graph: nx.Graph, graph: nx.Graph) -> Dict[str, float]:
    """
    Compute metrics over connectivity with different trajectories
    Args:
        gt_graph: Ground truth graph.
        chain_graph: Chain graph.
        graph: Loop closure graph with Local Metric.

    Returns:
        Dict with metrics
    """
    adj_graph = nx.adjacency_matrix(graph)
    adj_chain = nx.adjacency_matrix(chain_graph)
    adj_gt = nx.adjacency_matrix(gt_graph)

    # Cast adjacency matrices as binary matrices
    adj_gt, adj_chain, adj_graph = (adj_gt > 0).astype(int), (adj_chain > 0).astype(int), (adj_graph > 0).astype(int)

    # Remove chain edges from ground truth
    adj_closures = ((adj_gt - adj_chain) == 1).astype(int)
    adj_pred_closures = ((adj_graph - adj_chain) == 1).astype(int)

    # Compute metrics for edge connectivity in general
    metrics = compute_graph_clf_metrics(adj_closures, adj_pred_closures, suffix="_closures")
    return metrics


def aggregate_val_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Average validation metrics.

    Looks at all the keys in the metrics Dict, and averages
    all those starting with val/{metric_name}.
    Only returns the averages.
    
    Args:
        metrics: Dict with metricd
    Returns:
        Average over each metric
    """
    result = dict()

    for key, value in metrics.items():
        splits = key.split('/')
        stub = splits[0] + '/' + splits[1]  # Ignore last part

        # Check if we have more than two splits
        # This is to filter out aggregated metrics from previous epochs
        # e.g. val/loss, which would otherwise skew the new averages
        if (stub.startswith('val') or stub.startswith('test') or stub.startswith('train_final')) and len(splits) > 2:
            if stub in result:
                result[stub].append(value.item() if type(value) == torch.Tensor else value)
            else:
                result[stub] = [value.item() if type(value) == torch.Tensor else value]

    # Average all metrics
    result_avg = {key: sum(l) / len(l) for key, l in result.items()}

    return result_avg


class GlobalHeuristic:
    """Utility class to hold global codes and return geodesic length estimates"""

    def __init__(self, z_global: torch.Tensor, head: object):
        # Compute lengths all vs all
        lengths = list()
        z_global = z_global.float()
        # Set model in inference model
        with torch.inference_mode():
            for y in z_global:
                lengths.append(head.forward_all_codes(y.unsqueeze(0), z_global))
        self.lengths = torch.stack(lengths).double()

        # Track how many times the heuristic is accessed
        self.counter = 0

    def __call__(self, node_i, node_j):
        self.counter += 1
        return self.lengths[node_i, node_j]


def a_star_eval(graph: csr_matrix, n_paths: int, heuristic: Callable, dijkstra_dist: np.ndarray = None,
                random_state: int = 42, resample_start: bool = True) -> Tuple[float, List[List]]:
    """Solve n_paths planning episodes on the graph using A*

    Computes the average path length and the visited states in each path.

    Args:
        graph: Graph to run evaluation on. Must be connected.
        n_paths: Number of (start, goal) episodes to sample.
        heuristic: Callable heuristic. Must accept 2 indices and return the heuristic value.
        dijkstra_dist: Dijkstra distances. Must be an (n_nodes, n_nodes) array. If provided, the A* path length
        are normalized by the actual shortest path length.
        random_state: Random state.
        resample_start: If starting positions should be resampled. If False, the previous goal is used as the new start.

    Returns:
        avg_length: Average path length over episodes.
        paths: List of paths. Each path is a list of indices.

    """
    rng = np.random.default_rng(random_state)

    # Build Networkx graph from the csr_matrix
    nx_graph = graph

    # Sample random start and goals
    n = graph.number_of_nodes()
    obs = rng.choice(n, size=2 * n_paths, replace=False)
    start, end = obs[:n_paths], obs[n_paths:]

    if not resample_start:
        # Use previous goal as starting position for continuous trajectories
        start[1:] = end[:-1]

    # Run astar loop
    lengths = list()
    paths = list()
    for s, e in zip(start, end):
        # Check if there is no path between samples, otherwise resample start and goal
        while np.isinf(dijkstra_dist[s, e]):
            s, e = rng.choice(n, size=2, replace=False)

        path = nx.astar_path(nx_graph, s, e, heuristic=heuristic, weight='weight')

        # Sum weights to recover path length
        length = 0
        for i in range(len(path) - 1):
            length += nx_graph.edges[path[i], path[i + 1]]['weight']

        if dijkstra_dist is not None:
            # Normalize by the actual shortest path
            length /= dijkstra_dist[s, e]

        lengths.append(length)
        paths.append(path)

    # Return average path length
    return np.mean(lengths), paths


def spl(success: List, agent_path_length: List, shortest_path_length: List):
    """Compute success weighted by path length (SPL)

    Args:
        success: Binary list where 0 indicates episode failure and 1 indicates success.
        agent_path_length: Length of the episodes
        shortest_path_length: Shortest possible length of the episodes

    Returns:
        spl: average SPL over trajectories

    """
    success = np.array(success)
    agent_path_length = np.array(agent_path_length)
    shortest_path_length = np.array(shortest_path_length)
    max_ = np.max(np.vstack((agent_path_length, shortest_path_length)), axis=0)
    spl = success * shortest_path_length / max_
    return np.mean(spl)


def a_star_eval_global(graph: csr_matrix,
                       n_paths: int,
                       global_codes: torch.Tensor,
                       regression_head: object,
                       dijkstra_dist: np.ndarray = None,
                       resample_start: bool = True) -> Tuple[float, float, List[List]]:
    """Solve n_paths planning episodes on the graph using A* and a global heuristic.

    Computes the average path length, the average ratio of nodes expanded for each episode
    and the visited states in each path.

    Args:
        graph: Graph to run evaluation on. Must be connected.
        n_paths: Number of (start, goal) episodes to sample.
        global_codes: Global embedding of each observation in the graph. Used to approximate geodesic distances.
        regression_head: Head use to estimate heuristic giving two codes
        dijkstra_dist: Dijkstra distances. Must be an (n_nodes, n_nodes) array. If provided, the A* path length
        are normalized by the actual shortest path length.
        resample_start: If starting positions should be resampled. If False, the previous goal is used as the new start.

    Returns:
        avg_length: Average path length over episodes.
        avg_access_rate: Average ratio of expanded nodes over episodes.
        paths: List of paths. Each path is a list of indices.

    """
    heuristic = GlobalHeuristic(z_global=global_codes, head=regression_head)
    avg_path_len, paths = a_star_eval(graph, n_paths, heuristic=heuristic, dijkstra_dist=dijkstra_dist,
                                      resample_start=resample_start)
    avg_node_access = heuristic.counter / n_paths
    avg_access_rate = avg_node_access / graph.number_of_nodes()

    return avg_path_len, avg_access_rate, paths
