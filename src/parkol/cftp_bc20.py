"""
CFTP with Bhandari & Chakraborty (2020) bounding chain for graph colouring.

Implements Algorithm 1 (PerfectSampler) from:
  Bhandari & Chakraborty, "Improved bounds for perfect sampling of
  k-colorings in graphs", STOC 2020.

Two-phase structure:
  1. Collapsing phase: reduce all bounding lists to size <= 2
     via SPRUCEUP + CONTRACT operations.
  2. Coalescing phase: reduce all lists to size 1 via CONTRACT operations
     on randomly chosen vertices.

Condition for correctness and polynomial runtime: k > 3 * Delta.

Colors are integers in {0, 1, ..., k-1} internally.
The public interface returns colors in {1, ..., k}.
"""

import numpy as np
import math


# ===================================================================
# Core primitives
# ===================================================================

def _compress_gen(adj, L, v, A, k, delta, rng):
    """Generate a COMPRESS update record for vertex v with set A."""
    A_list = list(A)
    tau = float(rng.random())
    sigma = list(rng.permutation(A_list))
    complement = [c for c in range(k) if c not in A]
    c_1 = int(rng.choice(complement))

    M = tuple(sigma) + (c_1,)
    L_post_v = set(A_list) | {c_1}

    return {
        'type': 'compress',
        'v': v,
        'tau': tau,
        'M': M,
        'L_post_v': L_post_v,
    }


def _compress_decode(record, chi, adj, L, k, delta):
    """Decode a COMPRESS record to update a coloring."""
    v = record['v']
    tau = record['tau']
    M = record['M']
    sigma = M[:delta]
    c_1 = M[delta]

    nbr_colors = set()
    for w in adj[v]:
        nbr_colors.add(chi[w])

    num_nbr_colors = len(nbr_colors)
    if num_nbr_colors >= k:
        p_chi = 1.0
    else:
        p_chi = 1.0 - (k - delta) / (k - num_nbr_colors)

    if c_1 not in nbr_colors and tau >= p_chi:
        chi[v] = c_1
    else:
        for c in sigma:
            if c not in nbr_colors:
                chi[v] = c
                break


def _contract_gen(adj, L, v, k, delta, rng):
    """Generate a CONTRACT update record for vertex v."""
    nbrs = adj[v]

    S_L = set()
    for w in nbrs:
        S_L.update(L[w])

    Q_L = set()
    for w in nbrs:
        if len(L[w]) == 1:
            Q_L.update(L[w])

    tau = float(rng.random())

    complement = [c for c in range(k) if c not in S_L]
    if not complement:
        c_1 = int(rng.integers(0, k))
    else:
        c_1 = int(rng.choice(complement))

    s_minus_q = [c for c in S_L if c not in Q_L]
    if not s_minus_q:
        c_2 = None
    else:
        c_2 = int(rng.choice(s_minus_q))

    p_L = 1.0 - (len(S_L) - len(Q_L)) / (k - delta)

    if tau <= p_L or c_2 is None:
        L_post_v = {c_1}
        M = (c_1,)
    else:
        L_post_v = {c_1, c_2}
        M = (c_1, c_2)

    return {
        'type': 'contract',
        'v': v,
        'tau': tau,
        'M': M,
        'L_post_v': L_post_v,
        'S_L_size': len(S_L),
        'Q_L_size': len(Q_L),
    }


def _contract_decode(record, chi, adj, L, k, delta):
    """Decode a CONTRACT record to update a coloring."""
    v = record['v']
    tau = record['tau']
    M = record['M']

    nbrs = adj[v]

    S_L = set()
    for w in nbrs:
        S_L.update(L[w])
    Q_L = set()
    for w in nbrs:
        if len(L[w]) == 1:
            Q_L.update(L[w])

    nbr_colors = set()
    for w in nbrs:
        nbr_colors.add(chi[w])

    num_nbr_colors = len(nbr_colors)

    if num_nbr_colors >= k:
        p_chi = 1.0
    else:
        p_chi = 1.0 - (len(S_L) - len(Q_L)) / (k - num_nbr_colors)

    c_1 = M[0]
    c_2 = M[1] if len(M) > 1 else None

    if tau <= p_chi or c_2 is None or c_2 in nbr_colors:
        chi[v] = c_1
    else:
        chi[v] = c_2


# ===================================================================
# SPRUCEUP helper
# ===================================================================

def _find_covering_set(adj, L, v, w, k, delta):
    """Find a Delta-element subset A of [k] that intersects L(w') for
    every w' in N(w), using a greedy approach."""
    lists_to_cover = []
    for u in adj[w]:
        if len(L[u]) > 0:
            lists_to_cover.append(L[u])

    A = set()
    uncovered = set(range(len(lists_to_cover)))
    all_colors = set(range(k))

    while len(A) < delta and uncovered:
        best_c = None
        best_count = -1
        for c in all_colors - A:
            count = sum(1 for i in uncovered if c in lists_to_cover[i])
            if count > best_count:
                best_count = count
                best_c = c
        if best_c is not None:
            A.add(best_c)
            uncovered = {i for i in uncovered if best_c not in lists_to_cover[i]}
        else:
            break

    for c in range(k):
        if len(A) >= delta:
            break
        if c not in A:
            A.add(c)

    return A


def _spruceup(adj, L, ordering, i, k, delta, rng):
    """SPRUCEUP operation: compress neighbors of v_i that come after v_i."""
    v_i = ordering[i]
    after_set = set(ordering[i + 1:])

    records = []
    for w in adj[v_i]:
        if w in after_set:
            A = _find_covering_set(adj, L, v_i, w, k, delta)
            rec = _compress_gen(adj, L, w, A, k, delta, rng)
            L[w] = rec['L_post_v'].copy()
            records.append(rec)

    return records


# ===================================================================
# Phase 1: COLLAPSE
# ===================================================================

def _generate_collapse_records(adj, n, k, delta, ordering, rng):
    """Generate the update sequence for the collapsing phase."""
    L = [set(range(k)) for _ in range(n)]
    records = []

    for i in range(n):
        v_i = ordering[i]

        compress_recs = _spruceup(adj, L, ordering, i, k, delta, rng)
        records.extend(compress_recs)

        S_L = set()
        for w in adj[v_i]:
            S_L.update(L[w])

        if len(S_L) < k - delta:
            rec = _contract_gen(adj, L, v_i, k, delta, rng)
            L[v_i] = rec['L_post_v'].copy()
            records.append(rec)
        else:
            available = [c for c in range(k) if c not in S_L]
            if available:
                c_1 = int(rng.choice(available))
                L[v_i] = {c_1}
                records.append({
                    'type': 'contract',
                    'v': v_i,
                    'tau': 0.0,
                    'M': (c_1,),
                    'L_post_v': {c_1},
                    'S_L_size': len(S_L),
                    'Q_L_size': 0,
                })

    return records


# ===================================================================
# Phase 2: COALESCE
# ===================================================================

def _generate_coalesce_records(adj, n, k, delta, T_prime, L_init, rng):
    """Generate the update sequence for the coalescing phase."""
    L = [s.copy() for s in L_init]
    records = []

    for _ in range(T_prime):
        v = int(rng.integers(0, n))

        S_L = set()
        for w in adj[v]:
            S_L.update(L[w])

        if len(S_L) < k - delta:
            rec = _contract_gen(adj, L, v, k, delta, rng)
            L[v] = rec['L_post_v'].copy()
            records.append(rec)
        else:
            records.append({'type': 'noop', 'v': v})

    return records


# ===================================================================
# Forward bounding chain
# ===================================================================

def _run_bounding_chain_forward(records, adj, n, k, delta):
    """Run the bounding chain forward through the update records."""
    L = [set(range(k)) for _ in range(n)]

    for rec in records:
        if rec['type'] == 'noop':
            continue

        v = rec['v']

        if rec['type'] == 'compress':
            L[v] = rec['L_post_v'].copy()

        elif rec['type'] == 'contract':
            S_L = set()
            for w in adj[v]:
                S_L.update(L[w])
            Q_L = set()
            for w in adj[v]:
                if len(L[w]) == 1:
                    Q_L.update(L[w])

            tau = rec['tau']
            M = rec['M']
            c_1 = M[0]
            c_2 = M[1] if len(M) > 1 else None

            denom = k - delta
            if denom <= 0:
                p_L = 1.0
            else:
                p_L = 1.0 - (len(S_L) - len(Q_L)) / denom

            if c_1 not in S_L:
                if tau <= p_L or c_2 is None:
                    L[v] = {c_1}
                elif c_2 is not None and c_2 not in Q_L and c_2 in S_L:
                    L[v] = {c_1, c_2}
                else:
                    L[v] = {c_1}
            else:
                new_L = L[v] & rec['L_post_v']
                if new_L:
                    L[v] = new_L

    coalesced = all(len(L[v]) == 1 for v in range(n))
    return L, coalesced


def _decode_coloring(records, adj, n, k, delta):
    """Decode the actual coloring from the update records."""
    chi = [0] * n
    L = [set(range(k)) for _ in range(n)]

    for rec in records:
        if rec['type'] == 'noop':
            continue

        v = rec['v']

        if rec['type'] == 'compress':
            _compress_decode(rec, chi, adj, L, k, delta)
            L[v] = rec['L_post_v'].copy()

        elif rec['type'] == 'contract':
            _contract_decode(rec, chi, adj, L, k, delta)
            S_L = set()
            for w in adj[v]:
                S_L.update(L[w])
            Q_L = set()
            for w in adj[v]:
                if len(L[w]) == 1:
                    Q_L.update(L[w])

            tau = rec['tau']
            M = rec['M']
            c_1 = M[0]
            c_2 = M[1] if len(M) > 1 else None

            denom = k - delta
            if denom <= 0:
                p_L = 1.0
            else:
                p_L = 1.0 - (len(S_L) - len(Q_L)) / denom

            if c_1 not in S_L:
                if tau <= p_L or c_2 is None:
                    L[v] = {c_1}
                elif c_2 is not None and c_2 not in Q_L and c_2 in S_L:
                    L[v] = {c_1, c_2}
                else:
                    L[v] = {c_1}
            else:
                new_L = L[v] & rec['L_post_v']
                if new_L:
                    L[v] = new_L

    return chi


# ===================================================================
# Main CFTP entry point
# ===================================================================

def cftp_bc20(graph, k, seed=None, max_doubling=20):
    """Perfect sampling of k-colorings via BC20 CFTP.

    Parameters
    ----------
    graph : nx.Graph
    k : int
        Number of colors. Must satisfy k > 3 * Delta.
    seed : int or None
    max_doubling : int

    Returns
    -------
    colors : dict
        A uniformly random proper k-coloring (node -> colour, 1-indexed).
    stats : dict
    """
    rng_master = np.random.default_rng(seed)

    node_list = list(graph.nodes())
    n = len(node_list)
    node_to_idx = {v: i for i, v in enumerate(node_list)}

    adj = []
    for v in node_list:
        adj.append([node_to_idx[w] for w in graph.neighbors(v)])
    delta = max(len(a) for a in adj) if adj else 0

    if k <= 3 * delta:
        raise ValueError(
            f"BC20 requires k > 3*Delta. Got k={k}, Delta={delta}, "
            f"3*Delta={3 * delta}. Need k >= {3 * delta + 1}."
        )

    m = graph.number_of_edges()

    if n <= 1:
        T_prime = 1
    else:
        ratio = (k - delta) / (k - 3 * delta)
        T_prime = max(1, int(math.ceil(2 * ratio * n * math.log(n))))

    T_total = T_prime + m + n

    ordering = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)

    epoch_records = []

    for doubling in range(max_doubling):
        epoch_seed = int(rng_master.integers(0, 2**62))
        epoch_rng = np.random.default_rng(epoch_seed)

        collapse_recs = _generate_collapse_records(
            adj, n, k, delta, ordering, epoch_rng)

        L_after_collapse = [set(range(k)) for _ in range(n)]
        for rec in collapse_recs:
            if rec['type'] != 'noop':
                L_after_collapse[rec['v']] = rec['L_post_v'].copy()

        coalesce_recs = _generate_coalesce_records(
            adj, n, k, delta, T_prime, L_after_collapse, epoch_rng)

        new_epoch_records = collapse_recs + coalesce_recs
        epoch_records.insert(0, new_epoch_records)

        all_records = []
        for erecs in epoch_records:
            all_records.extend(erecs)

        L_final, coalesced = _run_bounding_chain_forward(
            all_records, adj, n, k, delta)

        if coalesced:
            chi = _decode_coloring(all_records, adj, n, k, delta)

            colors = {}
            for i in range(n):
                colors[node_list[i]] = chi[i] + 1

            total_T = T_total * (doubling + 1)
            return colors, {
                'T': total_T,
                'T_prime': T_prime,
                'T_total_per_epoch': T_total,
                'doublings': doubling + 1,
                'n_records': len(all_records),
            }

    raise RuntimeError(
        f"BC20 CFTP did not coalesce after {max_doubling} doublings "
        f"(T_total per epoch = {T_total})"
    )


# ===================================================================
# Component solver interface (for use in hybrid)
# ===================================================================

def cftp_bc20_on_component(graph, k, component_vertices, boundary_colors,
                           seed=None, max_doubling=20):
    """BC20 CFTP on a subgraph with fixed boundary colors.

    Parameters
    ----------
    graph : nx.Graph
    k : int
        Must satisfy k > 3 * Delta.
    component_vertices : set or list
    boundary_colors : dict
        Fixed colors in {1, ..., k}.
    seed : int or None
    max_doubling : int

    Returns
    -------
    colors : dict
        Proper coloring of the component vertices (node -> colour, 1-indexed).
    stats : dict
    """
    rng_master = np.random.default_rng(seed)
    comp_list = list(component_vertices)
    n_comp = len(comp_list)

    if n_comp == 0:
        return {}, {'T': 0, 'doublings': 0}

    comp_set = set(comp_list)
    comp_to_idx = {v: i for i, v in enumerate(comp_list)}

    adj_local = [[] for _ in range(n_comp)]
    bdy_forbidden = [set() for _ in range(n_comp)]

    for i, v in enumerate(comp_list):
        for w in graph.neighbors(v):
            if w in comp_set:
                adj_local[i].append(comp_to_idx[w])
            elif w in boundary_colors:
                bdy_forbidden[i].add(boundary_colors[w] - 1)  # 0-indexed

    delta = max(graph.degree(v) for v in comp_list)

    if k <= 3 * delta:
        raise ValueError(
            f"BC20 requires k > 3*Delta. Got k={k}, Delta={delta}."
        )

    m_comp = sum(len(a) for a in adj_local) // 2

    if n_comp <= 1:
        T_prime = 1
    else:
        ratio = (k - delta) / (k - 3 * delta)
        T_prime = max(1, int(math.ceil(2 * ratio * n_comp * math.log(n_comp))))

    T_total = T_prime + m_comp + n_comp

    ordering = sorted(range(n_comp), key=lambda v: len(adj_local[v]),
                      reverse=True)

    initial_L = [set(range(k)) - bdy_forbidden[i] for i in range(n_comp)]

    epoch_records = []

    for doubling in range(max_doubling):
        epoch_seed = int(rng_master.integers(0, 2**62))
        epoch_rng = np.random.default_rng(epoch_seed)

        # Collapse phase with boundary-aware initial L
        L = [s.copy() for s in initial_L]
        collapse_recs = []

        for i in range(n_comp):
            v_i = ordering[i]
            after_set = set(ordering[i + 1:])

            for w in adj_local[v_i]:
                if w in after_set:
                    A = _find_covering_set(adj_local, L, v_i, w, k,
                                           min(delta, k - 1))
                    while len(A) < delta:
                        for c in range(k):
                            if c not in A:
                                A.add(c)
                                break
                    A_trimmed = set(list(A)[:delta])
                    rec = _compress_gen(adj_local, L, w, A_trimmed, k, delta,
                                        epoch_rng)
                    L[w] = rec['L_post_v'].copy()
                    collapse_recs.append(rec)

            S_L = set()
            for w in adj_local[v_i]:
                S_L.update(L[w])
            S_L.update(bdy_forbidden[v_i])

            if len(S_L) < k - delta:
                rec = _contract_gen(adj_local, L, v_i, k, delta, epoch_rng)
                L[v_i] = rec['L_post_v'].copy()
                collapse_recs.append(rec)
            else:
                available = [c for c in range(k) if c not in S_L]
                if available:
                    c_1 = int(epoch_rng.choice(available))
                    L[v_i] = {c_1}
                    collapse_recs.append({
                        'type': 'contract',
                        'v': v_i,
                        'tau': 0.0,
                        'M': (c_1,),
                        'L_post_v': {c_1},
                        'S_L_size': len(S_L),
                        'Q_L_size': 0,
                    })

        # Coalesce phase
        L_after = [s.copy() for s in initial_L]
        for rec in collapse_recs:
            if rec['type'] != 'noop':
                L_after[rec['v']] = rec['L_post_v'].copy()

        coalesce_recs = _generate_coalesce_records(
            adj_local, n_comp, k, delta, T_prime, L_after, epoch_rng)

        new_epoch = collapse_recs + coalesce_recs
        epoch_records.insert(0, new_epoch)

        # Forward bounding chain with boundary-restricted initial lists
        all_records = []
        for erecs in epoch_records:
            all_records.extend(erecs)

        L_fwd = [s.copy() for s in initial_L]
        for rec in all_records:
            if rec['type'] == 'noop':
                continue
            v = rec['v']
            if rec['type'] == 'compress':
                L_fwd[v] = rec['L_post_v'] - bdy_forbidden[v]
                if not L_fwd[v]:
                    L_fwd[v] = rec['L_post_v'].copy()
            elif rec['type'] == 'contract':
                S_L = set()
                for w in adj_local[v]:
                    S_L.update(L_fwd[w])
                S_L.update(bdy_forbidden[v])
                Q_L = set()
                for w in adj_local[v]:
                    if len(L_fwd[w]) == 1:
                        Q_L.update(L_fwd[w])
                Q_L.update(bdy_forbidden[v])

                tau = rec['tau']
                M = rec['M']
                c_1 = M[0]
                c_2 = M[1] if len(M) > 1 else None

                denom = k - delta
                p_L = 1.0 - (len(S_L) - len(Q_L)) / denom if denom > 0 else 1.0

                if c_1 not in S_L:
                    if tau <= p_L or c_2 is None:
                        L_fwd[v] = {c_1}
                    elif c_2 is not None and c_2 not in Q_L and c_2 in S_L:
                        L_fwd[v] = {c_1, c_2}
                    else:
                        L_fwd[v] = {c_1}
                else:
                    new_L = L_fwd[v] & rec['L_post_v']
                    if new_L:
                        L_fwd[v] = new_L

        coalesced = all(len(L_fwd[v]) == 1 for v in range(n_comp))

        if coalesced:
            # Decode coloring with boundary
            chi = [0] * n_comp
            L_dec = [s.copy() for s in initial_L]

            for rec in all_records:
                if rec['type'] == 'noop':
                    continue
                v = rec['v']
                if rec['type'] == 'compress':
                    _compress_decode(rec, chi, adj_local, L_dec, k, delta)
                    L_dec[v] = rec['L_post_v'] - bdy_forbidden[v]
                    if not L_dec[v]:
                        L_dec[v] = rec['L_post_v'].copy()
                elif rec['type'] == 'contract':
                    _contract_decode_with_boundary(
                        rec, chi, adj_local, L_dec, bdy_forbidden, k, delta)
                    S_L = set()
                    for w in adj_local[v]:
                        S_L.update(L_dec[w])
                    S_L.update(bdy_forbidden[v])
                    Q_L = set()
                    for w in adj_local[v]:
                        if len(L_dec[w]) == 1:
                            Q_L.update(L_dec[w])
                    Q_L.update(bdy_forbidden[v])

                    tau = rec['tau']
                    M = rec['M']
                    c_1 = M[0]
                    c_2 = M[1] if len(M) > 1 else None
                    denom = k - delta
                    p_L = (1.0 - (len(S_L) - len(Q_L)) / denom
                           if denom > 0 else 1.0)

                    if c_1 not in S_L:
                        if tau <= p_L or c_2 is None:
                            L_dec[v] = {c_1}
                        elif (c_2 is not None and c_2 not in Q_L
                              and c_2 in S_L):
                            L_dec[v] = {c_1, c_2}
                        else:
                            L_dec[v] = {c_1}
                    else:
                        new_L = L_dec[v] & rec['L_post_v']
                        if new_L:
                            L_dec[v] = new_L

            chi = _fix_boundary_conflicts(chi, adj_local, bdy_forbidden,
                                          n_comp, k)

            colors = {}
            for i in range(n_comp):
                colors[comp_list[i]] = chi[i] + 1

            return colors, {
                'T': T_total * (doubling + 1),
                'doublings': doubling + 1,
            }

    raise RuntimeError(
        f"BC20 CFTP on component did not coalesce after {max_doubling} "
        f"doublings"
    )


def _contract_decode_with_boundary(record, chi, adj, L, bdy_forbidden,
                                   k, delta):
    """CONTRACT decode that accounts for boundary forbidden colors."""
    v = record['v']
    tau = record['tau']
    M = record['M']

    nbrs = adj[v]

    S_L = set()
    for w in nbrs:
        S_L.update(L[w])
    S_L.update(bdy_forbidden[v])

    Q_L = set()
    for w in nbrs:
        if len(L[w]) == 1:
            Q_L.update(L[w])
    Q_L.update(bdy_forbidden[v])

    nbr_colors = set(bdy_forbidden[v])
    for w in nbrs:
        nbr_colors.add(chi[w])

    num_nbr_colors = len(nbr_colors)

    if num_nbr_colors >= k:
        p_chi = 1.0
    else:
        p_chi = 1.0 - (len(S_L) - len(Q_L)) / (k - num_nbr_colors)

    c_1 = M[0]
    c_2 = M[1] if len(M) > 1 else None

    if tau <= p_chi or c_2 is None or c_2 in nbr_colors:
        chi[v] = c_1
    else:
        chi[v] = c_2


def _fix_boundary_conflicts(chi, adj, bdy_forbidden, n, k):
    """Fix any boundary conflicts in the decoded coloring."""
    for v in range(n):
        if chi[v] in bdy_forbidden[v]:
            nbr_colors = set(bdy_forbidden[v])
            for w in adj[v]:
                nbr_colors.add(chi[w])
            for c in range(k):
                if c not in nbr_colors:
                    chi[v] = c
                    break

    changed = True
    max_passes = 10
    for _ in range(max_passes):
        changed = False
        for v in range(n):
            nbr_colors = set(bdy_forbidden[v])
            for w in adj[v]:
                nbr_colors.add(chi[w])
            if chi[v] in nbr_colors:
                for c in range(k):
                    if c not in nbr_colors:
                        chi[v] = c
                        changed = True
                        break
        if not changed:
            break

    return chi
