"""
Core gamma-PRS algorithm for uniform graph colouring.

Implements the partial rejection sampling method from:
  "Uniform Sampling of Graph Colourings via Soft Colouring and Partial
   Rejection Sampling"
"""

import numpy as np
from collections import deque

from .utils import (
    preprocess_graph, has_improper_edge_vec, connected_components_mask,
)


# ---------------------------------------------------------------------------
# Vectorised core: n_v, bad vertices, resampling set
# ---------------------------------------------------------------------------

def compute_n_v_vec(G, colors, u_values, gamma):
    """Compute n_v(gamma, x) for all vertices using sparse matrix ops.

    n_v = number of neighbours w with c_w = c_v and u_w > gamma^{d_w}.
    """
    n, k, A, degrees = G['n'], G['k'], G['A'], G['degrees']
    gamma_pow_d = gamma ** degrees
    active = (u_values > gamma_pow_d).astype(np.float64)

    n_v = np.zeros(n, dtype=np.int32)
    for c in range(1, k + 1):
        mask_c = (colors == c)
        active_c = (mask_c & (active > 0.5)).astype(np.float64)
        counts = np.asarray(A @ active_c).ravel()
        n_v[mask_c] += counts[mask_c].astype(np.int32)

    return n_v


def find_bad_vertices_vec(G, colors, u_values, gamma, mask=None):
    """Find bad vertices: {v : u_v > gamma^{n_v}}.

    Returns boolean array. If mask is given, only vertices in mask are
    checked.
    """
    n_v = compute_n_v_vec(G, colors, u_values, gamma)
    thresholds = gamma ** n_v
    bad = u_values > thresholds
    if mask is not None:
        bad &= mask
    return bad


def find_resampling_set_vec(G, colors, u_values, gamma, mask=None):
    """Algorithm 1: Find the resampling set R via BFS from bad vertices.

    Starting from Bad(x, gamma), expand through non-passive vertices.
    Include one layer of passive boundary vertices.
    """
    bad = find_bad_vertices_vec(G, colors, u_values, gamma, mask)
    if not np.any(bad):
        return np.zeros(G['n'], dtype=bool)

    n, adj, degrees = G['n'], G['adj'], G['degrees']
    gamma_pow_d = gamma ** degrees
    passive = u_values <= gamma_pow_d

    R = bad.copy()
    visited = bad.copy()
    allowed = mask if mask is not None else np.ones(n, dtype=bool)

    boundary = np.zeros(n, dtype=bool)
    queue = deque(np.where(bad)[0])

    while queue:
        v = queue.popleft()
        for w in adj[v]:
            if allowed[w] and not visited[w]:
                visited[w] = True
                if passive[w]:
                    boundary[w] = True
                else:
                    R[w] = True
                    queue.append(w)

    R |= boundary
    return R


def resample_vertices_vec(colors, u_values, k, R_mask, rng):
    """Resample colours and u-values for vertices where R_mask is True."""
    idx = np.where(R_mask)[0]
    m = len(idx)
    if m == 0:
        return
    colors[idx] = rng.integers(1, k + 1, size=m)
    u_values[idx] = rng.random(size=m)


# ---------------------------------------------------------------------------
# Recursive gamma-PRS (Algorithm 4)
# ---------------------------------------------------------------------------

def gamma_prs_recursive(G, colors, u_values, gamma_seq, ell, mask,
                        rng, stats, depth=0, max_depth=100000):
    """Algorithm 4: gamma-PRS(G, x, ell) -- recursive implementation."""
    if depth > max_depth:
        raise RecursionError(f"gamma-PRS exceeded max depth {max_depth}")

    gamma_ell = gamma_seq[ell]
    k = G['k']

    while True:
        bad = find_bad_vertices_vec(G, colors, u_values, gamma_ell, mask)
        if not np.any(bad):
            break

        R = find_resampling_set_vec(G, colors, u_values, gamma_ell, mask)
        if not np.any(R):
            break

        resample_vertices_vec(colors, u_values, k, R, rng)
        stats['resample_count'] += 1
        stats['vertices_resampled'] += int(np.sum(R))

        components = connected_components_mask(G['adj'], R, G['n'])
        for comp in components:
            for j in range(ell + 1):
                gamma_prs_recursive(G, colors, u_values, gamma_seq, j,
                                    comp, rng, stats, depth + 1, max_depth)


# ---------------------------------------------------------------------------
# Iterative gamma-PRS at a single level
# ---------------------------------------------------------------------------

def gamma_prs_iterative(G, colors, u_values, gamma, mask,
                        rng, stats, max_iter=10**7):
    """Iterative PRS at a single gamma level.

    Repeatedly: find bad vertices -> compute R -> resample R,
    until Bad(x, gamma) is empty.
    """
    k = G['k']

    for it in range(max_iter):
        bad = find_bad_vertices_vec(G, colors, u_values, gamma, mask)
        if not np.any(bad):
            return

        R = find_resampling_set_vec(G, colors, u_values, gamma, mask)
        if not np.any(R):
            return

        resample_vertices_vec(colors, u_values, k, R, rng)
        stats['resample_count'] += 1
        stats['vertices_resampled'] += int(np.sum(R))

    raise RuntimeError(f"Iterative PRS did not converge in {max_iter} iterations")


# ---------------------------------------------------------------------------
# Main entry point: proper colouring through PRS
# ---------------------------------------------------------------------------

def prs_graph_coloring(graph, k, gamma_base=0.9, max_levels=1000,
                       seed=None, recursive=False):
    """Uniform sampling of a proper k-colouring via gamma-PRS.

    Parameters
    ----------
    graph : nx.Graph
        The input graph.
    k : int
        Number of colours (must be >= chromatic number).
    gamma_base : float
        Base for the gamma-sequence: gamma_ell = gamma_base^ell.
    max_levels : int
        Safety limit on the number of levels.
    seed : int or None
        Random seed for reproducibility.
    recursive : bool
        If True, use the full recursive Algorithm 4.
        If False, use iterative PRS at each level (faster, practical).

    Returns
    -------
    colors : dict
        A proper k-colouring mapping node -> colour (1-indexed).
    stats : dict
        Run statistics.
    """
    rng = np.random.default_rng(seed)

    G = preprocess_graph(graph, k)
    n = G['n']

    colors = rng.integers(1, k + 1, size=n)
    u_values = rng.random(size=n).astype(np.float64)

    gamma_seq = [gamma_base ** ell for ell in range(max_levels)]

    stats = {
        'levels': 0,
        'resample_count': 0,
        'vertices_resampled': 0,
    }

    ell = 0
    all_mask = np.ones(n, dtype=bool)
    while has_improper_edge_vec(colors, G['edge_pairs']):
        if ell >= max_levels:
            raise RuntimeError(f"Did not converge within {max_levels} levels")

        if recursive:
            gamma_prs_recursive(G, colors, u_values, gamma_seq, ell,
                                all_mask, rng, stats)
        else:
            gamma_prs_iterative(G, colors, u_values, gamma_seq[ell],
                                all_mask, rng, stats)
        ell += 1

    stats['levels'] = ell

    color_dict = {G['node_list'][i]: int(colors[i]) for i in range(n)}
    return color_dict, stats
