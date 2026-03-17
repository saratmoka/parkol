"""
Coupling From The Past (CFTP) with bounding chains for graph colouring.

Implements Huber's (2004) bounding chain method:
  - Underlying chain: Glauber dynamics for proper k-colorings
  - Bounding chain: tracks sets Y(v) of possible colors per vertex
  - Coalescence: all Y(v) are singletons

Condition for polynomial runtime: k >= Delta*(Delta+2).
Correct (exact uniform sample) for any k > Delta.

Reference: Huber, "Perfect Sampling Using Bounding Chains",
           Annals of Applied Probability, 2004.
"""

import numpy as np


def _bounding_chain_step(Y, v, adj, k, delta, rng_state):
    """One step of the bounding chain update at vertex v."""
    nbrs = adj[v]

    B = set()
    for w in nbrs:
        if len(Y[w]) == 1:
            B.update(Y[w])

    F = set()
    for w in nbrs:
        F.update(Y[w])

    Y[v] = set()
    max_draws = 5 * k
    for _ in range(max_draws):
        c = int(rng_state.integers(1, k + 1))

        if c not in B:
            Y[v].add(c)

        if c not in F:
            break

        if len(Y[v]) > delta:
            break

    if not Y[v]:
        Y[v] = set(range(1, k + 1))


def cftp_coloring(graph, k, seed=None, max_doubling=25):
    """CFTP with bounding chains for uniform proper k-colouring.

    Parameters
    ----------
    graph : nx.Graph
    k : int
        Number of colours. Must be > max degree.
    seed : int or None
    max_doubling : int

    Returns
    -------
    colors : dict
        A uniformly selected proper k-colouring (node -> colour, 1-indexed).
    stats : dict
    """
    rng = np.random.default_rng(seed)
    node_list = list(graph.nodes())
    n = len(node_list)
    node_to_idx = {v: i for i, v in enumerate(node_list)}

    adj = []
    for v in node_list:
        adj.append([node_to_idx[w] for w in graph.neighbors(v)])
    delta = max(len(a) for a in adj) if adj else 0

    if k <= delta:
        raise ValueError(f"Need k > Delta={delta}, got k={k}")

    step_inputs = {}

    def get_step_input(t):
        if t not in step_inputs:
            v = int(rng.integers(0, n))
            s = int(rng.integers(0, 2**31))
            step_inputs[t] = (v, s)
        return step_inputs[t]

    T = 1
    for doubling in range(max_doubling):
        Y = [set(range(1, k + 1)) for _ in range(n)]

        for t in range(-T, 0):
            v, s = get_step_input(t)
            step_rng = np.random.default_rng(s)
            _bounding_chain_step(Y, v, adj, k, delta, step_rng)

        if all(len(Y[v]) == 1 for v in range(n)):
            colors = {node_list[v]: next(iter(Y[v])) for v in range(n)}
            return colors, {'T': T, 'doublings': doubling + 1}

        T *= 2

    raise RuntimeError(f"CFTP did not coalesce after T={T}")


def cftp_coloring_on_component(graph, k, component_vertices, boundary_colors,
                                seed=None, max_doubling=20):
    """CFTP on a subgraph with fixed boundary colours.

    Parameters
    ----------
    graph : nx.Graph
        The full graph.
    k : int
    component_vertices : set or list
        Vertices to recolour.
    boundary_colors : dict
        Fixed colours of vertices adjacent to the component.
    seed : int or None
    max_doubling : int

    Returns
    -------
    colors : dict
        Proper colouring of the component vertices (node -> colour, 1-indexed).
    stats : dict
    """
    rng = np.random.default_rng(seed)
    comp_list = list(component_vertices)
    n_comp = len(comp_list)
    comp_set = set(component_vertices)
    comp_to_idx = {v: i for i, v in enumerate(comp_list)}

    adj_local = []
    bdy_certain = []
    for v in comp_list:
        in_comp = []
        certain = set()
        for w in graph.neighbors(v):
            if w in comp_set:
                in_comp.append(comp_to_idx[w])
            elif w in boundary_colors:
                certain.add(boundary_colors[w])
        adj_local.append(in_comp)
        bdy_certain.append(certain)

    delta = max(graph.degree(v) for v in comp_list)
    if k <= delta:
        raise ValueError(f"Need k > Delta={delta}, got k={k}")

    step_inputs = {}

    def get_step_input(t):
        if t not in step_inputs:
            v = int(rng.integers(0, n_comp))
            s = int(rng.integers(0, 2**31))
            step_inputs[t] = (v, s)
        return step_inputs[t]

    T = 1
    for doubling in range(max_doubling):
        Y = [set(range(1, k + 1)) for _ in range(n_comp)]

        for t in range(-T, 0):
            v_local, s = get_step_input(t)
            step_rng = np.random.default_rng(s)
            nbrs = adj_local[v_local]
            bdy_c = bdy_certain[v_local]

            B = set(bdy_c)
            for w in nbrs:
                if len(Y[w]) == 1:
                    B.update(Y[w])

            F = set(bdy_c)
            for w in nbrs:
                F.update(Y[w])

            Y[v_local] = set()
            for _ in range(5 * k):
                c = int(step_rng.integers(1, k + 1))
                if c not in B:
                    Y[v_local].add(c)
                if c not in F:
                    break
                if len(Y[v_local]) > delta:
                    break

            if not Y[v_local]:
                Y[v_local] = set(range(1, k + 1))

        if all(len(Y[v]) == 1 for v in range(n_comp)):
            colors = {comp_list[v]: next(iter(Y[v])) for v in range(n_comp)}
            return colors, {'T': T, 'doublings': doubling + 1}

        T *= 2

    raise RuntimeError(f"CFTP on component did not coalesce after T={T}")
