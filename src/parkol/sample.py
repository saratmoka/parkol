"""
Main public API for parkol: sample_coloring().
"""

import networkx as nx


def sample_coloring(graph, k, method='hybrid', seed=None, adaptive=False,
                    target_components=None):
    """Draw an exact uniform sample of a proper k-coloring of a graph.

    Parameters
    ----------
    graph : nx.Graph
        The input graph.
    k : int
        Number of colours. Must be at least the chromatic number of the graph.
    method : str
        Sampling method to use:

        - ``'hybrid'`` (default): PRS decomposition with auto-selected
          component solver. Defaults to Huber CFTP (simplest, fastest in
          practice); falls back to BC20 CFTP if Huber fails to coalesce,
          and to NRS if k <= Delta.
        - ``'hybrid_gibbs'``: PRS decomposition with Gibbs sampler as
          component solver (FGY22-style). Suitable for graphs with
          sub-exponential neighbourhood growth (e.g. lattices) where
          strong spatial mixing holds.
        - ``'prs'``: Pure gamma-PRS (iterative).
        - ``'cftp_huber'``: Huber (2004) bounding-chain CFTP. Requires
          k > Delta.
        - ``'cftp_bc20'``: Bhandari & Chakraborty (2020) CFTP. Requires
          k > 3*Delta.
        - ``'nrs'``: Naive rejection sampling.
    seed : int or None
        Random seed for reproducibility.
    adaptive : bool
        If True (and method is ``'hybrid'`` or ``'hybrid_gibbs'``), use an
        adaptive gamma-sequence that decreases gamma to encourage the
        resampling set to split into multiple components, maximising
        parallelisation. The number of components may be fewer than
        ``target_components`` since the gamma-soft bad set is a subset
        of the bad set for proper colouring.
    target_components : int or None
        Maximum number of desired components for the adaptive strategy.
        Defaults to the number of available CPU cores.

    Returns
    -------
    colors : dict
        A proper k-colouring mapping node -> colour (integers in {1, ..., k}).

    Raises
    ------
    ValueError
        If the method is unknown or k does not satisfy the method's condition.
    RuntimeError
        If the algorithm does not converge.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a networkx.Graph")
    if graph.number_of_nodes() == 0:
        return {}

    degrees = [d for _, d in graph.degree()]
    delta = max(degrees) if degrees else 0

    if delta > 0 and k <= delta:
        raise ValueError(
            f"Need k > Delta for a proper colouring to be guaranteed. "
            f"Got k={k}, Delta={delta}."
        )

    if method == 'hybrid':
        from .hybrid import prs_hybrid

        # Choose component solver based on k/Delta ratio:
        # - k > 3*Delta: use Huber CFTP (fast, simple)
        # - otherwise: use NRS (CFTP may not coalesce quickly)
        if k > 3 * delta:
            solver = 'cftp_huber'
            try:
                colors, _stats = prs_hybrid(
                    graph, k, seed=seed, component_solver=solver,
                    adaptive=adaptive,
                    target_components=target_components)
                return colors
            except RuntimeError:
                pass  # fall through to NRS

        colors, _stats = prs_hybrid(
            graph, k, seed=seed, component_solver='nrs',
            adaptive=adaptive, target_components=target_components)
        return colors

    elif method == 'hybrid_gibbs':
        from .hybrid import prs_hybrid
        colors, _stats = prs_hybrid(
            graph, k, seed=seed, component_solver='gibbs',
            adaptive=adaptive, target_components=target_components)
        return colors

    elif method == 'prs':
        from .prs import prs_graph_coloring
        colors, _stats = prs_graph_coloring(graph, k, seed=seed)
        return colors

    elif method == 'cftp_huber':
        from .cftp_huber import cftp_coloring
        colors, _stats = cftp_coloring(graph, k, seed=seed)
        return colors

    elif method == 'cftp_bc20':
        from .cftp_bc20 import cftp_bc20
        colors, _stats = cftp_bc20(graph, k, seed=seed)
        return colors

    elif method == 'nrs':
        from .utils import has_improper_edge
        import numpy as np
        rng = np.random.default_rng(seed)
        nodes = list(graph.nodes())
        max_iter = 10**7
        for _ in range(max_iter):
            colors = {v: int(rng.integers(1, k + 1)) for v in nodes}
            if not has_improper_edge(graph, colors):
                return colors
        raise RuntimeError(
            f"NRS did not find a proper colouring in {max_iter} iterations"
        )

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            f"'hybrid', 'hybrid_gibbs', 'prs', 'cftp_huber', "
            f"'cftp_bc20', 'nrs'."
        )
