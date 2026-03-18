<p align="center">
  <img src="logo-parkol.png" alt="PaRKol" width="300">
</p>

<p align="center"><strong>Pa</strong>rtial <strong>R</strong>ejection sampling for <strong>K</strong>-c<strong>ol</strong>ouring</p>

Exact uniform sampling of proper *k*-colourings of a graph, via the soft colouring framework.

To our knowledge, PaRKol is the first Python package implementing near-linear time exact uniform sampling of graph colourings via soft colouring and partial rejection sampling.

## Installation

```bash
pip install parkol
```

## Quick Start

```python
import networkx as nx
from parkol import sample_coloring, verify_coloring

G = nx.petersen_graph()
colors = sample_coloring(G, k=5)
print(verify_coloring(G, colors))  # True
```

## Methods

| Method | Description | Condition |
|--------|-------------|-----------|
| `'hybrid'` | PRS + CFTP on components (default) | k > Δ |
| `'prs'` | Pure γ-PRS | k > Δ |
| `'cftp_huber'` | Huber (2004) bounding-chain CFTP | k > Δ |
| `'cftp_bc20'` | Bhandari & Chakraborty (2020) CFTP | k > 3Δ |
| `'nrs'` | Naive rejection sampling | k > Δ |

```python
colors = sample_coloring(G, k=5, method='hybrid', seed=42)
```

## Documentation

Full documentation: [https://parkol.readthedocs.io](https://parkol.readthedocs.io)

## Reference

S. Moka and A. Vahedi (2026). *Near-Linear Time Perfect Sampling of Graph Colourings via Soft Colouring.* Preprint.

## License

MIT
