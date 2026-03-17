"""
Graph utilities and coloring verification.
"""

import numpy as np
import networkx as nx
from collections import deque


def preprocess_graph(graph, k):
    """Convert a networkx graph to optimised array structures.

    Returns a dict with all precomputed structures needed by the algorithms.
    """
    node_list = list(graph.nodes())
    node_to_idx = {v: i for i, v in enumerate(node_list)}
    n = len(node_list)

    A = nx.adjacency_matrix(graph, nodelist=node_list).astype(np.float64)

    adj = [None] * n
    for i, v in enumerate(node_list):
        adj[i] = np.array([node_to_idx[w] for w in graph.neighbors(v)],
                          dtype=np.int32)

    degrees = np.array(A.sum(axis=1)).ravel().astype(np.int32)

    edge_pairs = np.array([(node_to_idx[u], node_to_idx[v])
                           for u, v in graph.edges()], dtype=np.int32)

    return {
        'A': A, 'adj': adj, 'degrees': degrees, 'n': n, 'k': k,
        'node_list': node_list, 'node_to_idx': node_to_idx,
        'edge_pairs': edge_pairs,
    }


def verify_coloring(graph, colors):
    """Verify that a coloring is proper.

    Parameters
    ----------
    graph : nx.Graph
        The graph.
    colors : dict
        Mapping from node to color.

    Returns
    -------
    bool
        True if the coloring is proper (no adjacent vertices share a color).
    """
    for u, v in graph.edges():
        if colors[u] == colors[v]:
            return False
    return True


def has_improper_edge(graph, colors):
    """Check if any edge has same-colour endpoints (dict-based)."""
    for u, v in graph.edges():
        if colors[u] == colors[v]:
            return True
    return False


def has_improper_edge_vec(colors, edge_pairs):
    """Check if any edge has both endpoints with the same colour (array-based)."""
    if len(edge_pairs) == 0:
        return False
    return bool(np.any(colors[edge_pairs[:, 0]] == colors[edge_pairs[:, 1]]))


def connected_components_mask(adj, R_mask, n):
    """Find connected components of the subgraph induced by R_mask."""
    visited = np.zeros(n, dtype=bool)
    components = []

    for start in np.where(R_mask)[0]:
        if visited[start]:
            continue
        comp = np.zeros(n, dtype=bool)
        queue = deque([start])
        visited[start] = True
        comp[start] = True
        while queue:
            v = queue.popleft()
            for w in adj[v]:
                if R_mask[w] and not visited[w]:
                    visited[w] = True
                    comp[w] = True
                    queue.append(w)
        components.append(comp)

    return components
