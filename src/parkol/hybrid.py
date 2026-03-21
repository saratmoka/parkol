"""
Hybrid gamma-PRS: PRS for global decomposition into independent components,
then an exact sampler on each component.

The idea: PRS at each level identifies the resampling set R and splits it
into connected components. Instead of recursive gamma-PRS on each component,
an off-the-shelf exact sampler (NRS, CFTP) produces a sample from
eta_{gamma_ell} restricted to that component.
"""

import numpy as np
import networkx as nx

from .utils import (
    preprocess_graph, has_improper_edge_vec, connected_components_mask,
)
from .prs import (
    find_bad_vertices_vec, find_resampling_set_vec, resample_vertices_vec,
)


# ===================================================================
# Component solvers
# ===================================================================

def _solve_component_nrs(G, colors, u_values, gamma, comp_mask, rng,
                         max_iter=10**6):
    """Solve a component by naive rejection sampling under gamma-soft."""
    k = G['k']
    comp_idx = np.where(comp_mask)[0]

    for it in range(max_iter):
        bad = find_bad_vertices_vec(G, colors, u_values, gamma, comp_mask)
        if not np.any(bad):
            return it

        colors[comp_idx] = rng.integers(1, k + 1, size=len(comp_idx))
        u_values[comp_idx] = rng.random(size=len(comp_idx))

    raise RuntimeError("NRS on component did not converge")


def _solve_component_cftp_huber(G, colors, u_values, gamma, comp_mask, rng,
                                max_iter=None):
    """Solve a component by Huber's bounding chain CFTP.

    Produces a uniform proper k-colouring of the component subgraph,
    independently of the configuration outside the component.
    Cross-edge conflicts are handled by the outer while loop.
    """
    from .cftp_huber import cftp_coloring

    k = G['k']
    comp_idx = np.where(comp_mask)[0]
    node_list = G['node_list']
    comp_vertices = [node_list[i] for i in comp_idx]

    # Build the subgraph induced by the component (no boundary info)
    comp_set = set(comp_vertices)
    subgraph = nx.Graph()
    subgraph.add_nodes_from(comp_vertices)
    for i in comp_idx:
        v = node_list[i]
        for w_idx in G['adj'][i]:
            w = node_list[w_idx]
            if w in comp_set:
                subgraph.add_edge(v, w)

    seed_val = int(rng.integers(0, 2**31))
    try:
        col, _ = cftp_coloring(subgraph, k, seed=seed_val)
        for i in comp_idx:
            v = node_list[i]
            if v in col:
                colors[i] = col[v]
        u_values[comp_idx] = rng.random(size=len(comp_idx))
    except (RuntimeError, RecursionError):
        # Huber didn't coalesce; try BC20 if k > 3*Delta, else NRS
        delta = max(int(G['degrees'].max()), 1)
        if k > 3 * delta:
            try:
                _solve_component_cftp_bc20(
                    G, colors, u_values, gamma, comp_mask, rng)
                return
            except (RuntimeError, ValueError):
                pass
        _solve_component_nrs(G, colors, u_values, gamma, comp_mask, rng)


def _solve_component_cftp_bc20(G, colors, u_values, gamma, comp_mask, rng,
                                max_iter=None):
    """Solve a component by BC20 CFTP (k > 3*Delta).

    Produces a uniform proper k-colouring of the component subgraph,
    independently of the configuration outside the component.
    """
    from .cftp_bc20 import cftp_bc20

    k = G['k']
    comp_idx = np.where(comp_mask)[0]
    node_list = G['node_list']
    comp_vertices = [node_list[i] for i in comp_idx]

    # Build the subgraph induced by the component (no boundary info)
    comp_set = set(comp_vertices)
    subgraph = nx.Graph()
    subgraph.add_nodes_from(comp_vertices)
    for i in comp_idx:
        v = node_list[i]
        for w_idx in G['adj'][i]:
            w = node_list[w_idx]
            if w in comp_set:
                subgraph.add_edge(v, w)

    seed_val = int(rng.integers(0, 2**31))
    try:
        col, _ = cftp_bc20(subgraph, k, seed=seed_val)
        for i in comp_idx:
            v = node_list[i]
            if v in col:
                colors[i] = col[v]
        u_values[comp_idx] = rng.random(size=len(comp_idx))
    except (RuntimeError, ValueError):
        _solve_component_nrs(G, colors, u_values, gamma, comp_mask, rng)


# ===================================================================
# Hybrid PRS algorithm
# ===================================================================

def prs_hybrid(graph, k, gamma_base=0.9, max_levels=1000, seed=None,
               component_solver='nrs'):
    """Hybrid gamma-PRS: PRS decomposition + exact sampler on components.

    Parameters
    ----------
    graph : nx.Graph
    k : int
    gamma_base : float
    max_levels : int
    seed : int or None
    component_solver : str
        'nrs'        : Naive rejection sampling on each component.
        'cftp_huber' : Huber (2004) bounding chain CFTP.
        'cftp_bc20'  : Bhandari & Chakraborty (2020) CFTP.

    Returns
    -------
    colors : dict
        Proper k-colouring mapping node -> colour (1-indexed).
    stats : dict
    """
    rng = np.random.default_rng(seed)
    G = preprocess_graph(graph, k)
    n = G['n']

    colors = rng.integers(1, k + 1, size=n)
    u_values = rng.random(size=n).astype(np.float64)
    gamma_seq = [gamma_base ** ell for ell in range(max_levels)]

    solver_fn = {
        'nrs': _solve_component_nrs,
        'cftp_huber': _solve_component_cftp_huber,
        'cftp_bc20': _solve_component_cftp_bc20,
    }[component_solver]

    stats = {
        'levels': 0,
        'resample_count': 0,
        'vertices_resampled': 0,
        'n_components': 0,
        'component_solver_calls': 0,
        'component_sizes': [],
    }

    ell = 0
    all_mask = np.ones(n, dtype=bool)

    while has_improper_edge_vec(colors, G['edge_pairs']):
        if ell >= max_levels:
            raise RuntimeError(f"Did not converge within {max_levels} levels")

        gamma = gamma_seq[ell]

        level_iter = 0
        while level_iter < 10**6:
            bad = find_bad_vertices_vec(G, colors, u_values, gamma, all_mask)
            if not np.any(bad):
                break

            R = find_resampling_set_vec(G, colors, u_values, gamma, all_mask)
            if not np.any(R):
                break

            stats['resample_count'] += 1
            stats['vertices_resampled'] += int(np.sum(R))

            components = connected_components_mask(G['adj'], R, n)
            stats['n_components'] += len(components)

            for comp in components:
                comp_size = int(np.sum(comp))
                stats['component_sizes'].append(comp_size)

                resample_vertices_vec(colors, u_values, k, comp, rng)

                solver_fn(G, colors, u_values, gamma, comp, rng)
                stats['component_solver_calls'] += 1

            level_iter += 1

        ell += 1

    stats['levels'] = ell
    if stats['component_sizes']:
        stats['mean_component_size'] = float(np.mean(stats['component_sizes']))
        stats['max_component_size'] = max(stats['component_sizes'])
    else:
        stats['mean_component_size'] = 0
        stats['max_component_size'] = 0

    color_dict = {G['node_list'][i]: int(colors[i]) for i in range(n)}
    return color_dict, stats
