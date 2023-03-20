from typing import Union, List, Tuple

import pickle
import networkx as nx
import os

from src.utils import get_logger

log = get_logger(__name__)


def get_aug_keys(k: int, n_positives: int) -> Tuple[List[str], List[str]]:
    """
    Quick helper function to get the augmentation keys to retrieve augmented images from albumentations.
    """
    # Only one anchor sequence
    anchor_aug_keys = ['image'] + ['image_{}'.format(i) for i in range(k - 1)]

    # We may have multiple positive sequences
    # Positive keys will be nested
    pos_aug_keys = []
    for j in range(n_positives):
        pos_aug_keys.append(['image_pos_{}_{}'.format(j, i) for i in range(k)])
    return anchor_aug_keys, pos_aug_keys


def remove_attributes(graph: Union[nx.Graph, nx.DiGraph],
                      node_attr: List[str],
                      edge_attr: List[str]) -> Union[nx.Graph, nx.DiGraph]:
    """Remove edge and node attributes from graphs to save memory"""
    # Remove node attributes
    for node in range(graph.number_of_nodes()):
        for attr in node_attr:
            graph.nodes[node].pop(attr, None)

    # Clear edge attributes
    for n1, n2, edge in graph.edges(data=True):
        for attr in edge_attr:
            edge.pop(attr, None)

    return graph


def save_graph(path: str, graph: Union[nx.Graph, nx.DiGraph]) -> None:
    """
    Save a networkx graph in memory
    """
    with open(path, 'wb') as file:
        pickle.dump(graph, file, pickle.HIGHEST_PROTOCOL)


def load_graph(path: str) -> Union[nx.Graph, nx.DiGraph]:
    """
    Load a networkx graph from memory - returns None if there is not graph in givne path
    """
    log.info("Loading graph stored in: \n {}".format(path))
    if os.path.exists(path):
        with open(path, 'rb') as file:
            graph = pickle.load(file)
        return graph
    return None
