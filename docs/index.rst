parkol — Exact Uniform Sampling of Graph Colorings
====================================================

**PaRKol** (**Pa**\ rtial **R**\ ejection sampling for **K**-c\ **ol**\ oring)
is a Python package for drawing exact uniform samples of proper
:math:`k`-colorings of a graph.

It implements the *soft coloring* framework, which decomposes the global
sampling problem into small independent subproblems via partial rejection
sampling (PRS), then solves each subproblem with coupling from the past
(CFTP) or other exact samplers. The algorithm is inherently parallelizable
and achieves near-linear runtime in the number of vertices.

Installation
------------

.. code-block:: bash

   pip install parkol

Quick Start
-----------

.. code-block:: python

   import networkx as nx
   from parkol import sample_coloring, verify_coloring

   G = nx.petersen_graph()         # 10 vertices, max degree 3
   colors = sample_coloring(G, k=5)
   print(verify_coloring(G, colors))  # True
   print(colors)                      # {0: 3, 1: 5, 2: 1, ...}

Choosing a Method
-----------------

Pass the ``method`` argument to select the sampling algorithm:

.. code-block:: python

   colors = sample_coloring(G, k=5)                          # hybrid (default)
   colors = sample_coloring(G, k=5, method='prs')            # pure gamma-PRS
   colors = sample_coloring(G, k=15, method='cftp_huber')    # Huber CFTP
   colors = sample_coloring(G, k=10, method='cftp_bc20')     # BC20 CFTP
   colors = sample_coloring(G, k=5, method='nrs')            # naive rejection

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Method
     - Description
     - Condition
   * - ``'hybrid'``
     - PRS decomposition + CFTP on components (default, recommended)
     - :math:`k > \Delta`
   * - ``'prs'``
     - Pure :math:`\gamma`-PRS (iterative)
     - :math:`k > \Delta`
   * - ``'cftp_huber'``
     - Huber (1998) bounding-chain CFTP
     - :math:`k > \Delta`
   * - ``'cftp_bc20'``
     - Bhandari & Chakraborty (2020) CFTP
     - :math:`k > 3\Delta`
   * - ``'nrs'``
     - Naive rejection sampling
     - :math:`k > \Delta`

Adaptive Gamma-Sequence
-----------------------

The ``adaptive`` option uses a gamma-sequence that encourages the
resampling set to split into multiple connected components, enabling
parallel processing:

.. code-block:: python

   # Adaptive with 8 target components
   colors = sample_coloring(G, k=20, adaptive=True, target_components=8)


Reproducibility
---------------

Pass a ``seed`` for deterministic output:

.. code-block:: python

   colors = sample_coloring(G, k=5, seed=42)  # reproducible

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api

References
----------

- S. Moka and A. Vahedi (2026).
  *Uniform Sampling of Proper Graph Colorings via Soft Coloring and
  Partial Rejection Sampling.* Preprint.

- S. Bhandari and S. Chakraborty (2020).
  *Improved Bounds for Perfect Sampling of k-Colorings in Graphs.*
  Proc. STOC 2020.

- M. Huber (1998).
  *Perfect Sampling Using Bounding Chains.*
  Ann. Appl. Probab. 14(2), 734--753.

- H. Guo, M. Jerrum, and J. Liu (2017).
  *Uniform Sampling Through the Lovasz Local Lemma.*
  Proc. STOC 2017, 342--355.
