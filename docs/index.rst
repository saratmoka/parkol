parkol — Exact Uniform Sampling of Graph Colourings
====================================================

**PaRKol** (**Pa**\ rtial **R**\ ejection sampling for **K**-c\ **ol**\ ouring)
is a Python package for drawing exact uniform samples of proper
:math:`k`-colourings of a graph.

It implements the *soft colouring* framework, which decomposes the global
sampling problem into small independent subproblems via partial rejection
sampling (PRS), then solves each subproblem with coupling from the past
(CFTP). The algorithm is inherently parallelisable and achieves
near-linear runtime in the number of vertices.

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

   colors = sample_coloring(G, k=5)                        # hybrid (default)
   colors = sample_coloring(G, k=5, method='prs')          # pure gamma-PRS
   colors = sample_coloring(G, k=15, method='cftp_huber')  # Huber CFTP
   colors = sample_coloring(G, k=10, method='cftp_bc20')   # BC20 CFTP
   colors = sample_coloring(G, k=5, method='nrs')          # naive rejection

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

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
     - Huber (2004) bounding-chain CFTP
     - :math:`k > \Delta`
   * - ``'cftp_bc20'``
     - Bhandari & Chakraborty (2020) CFTP
     - :math:`k > 3\Delta`
   * - ``'nrs'``
     - Naive rejection sampling
     - :math:`k > \Delta`

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
  *Near-Linear Time Perfect Sampling of Graph Colourings via Soft Colouring.*
  Preprint.

- S. Bhandari and S. Chakraborty (2020).
  *Improved Bounds for Perfect Sampling of k-Colorings in Graphs.*
  Proc. STOC 2020.

- M. Huber (2004).
  *Perfect Sampling Using Bounding Chains.*
  Ann. Appl. Probab. 14(2), 734–753.

- M. T. Guo, M. Jerrum, and J. Liu (2019).
  *Uniform Sampling Through the Lovász Local Lemma.*
  J. ACM 66(3), 18:1–18:31.
