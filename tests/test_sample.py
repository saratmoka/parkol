"""
Tests for parkol.sample_coloring on small graphs.
"""

import networkx as nx
import pytest

from parkol import sample_coloring, verify_coloring


def _check_coloring(graph, colors, k):
    """Assert the coloring is proper and uses valid colors."""
    assert verify_coloring(graph, colors), "Coloring is not proper"
    for node in graph.nodes():
        assert node in colors, f"Node {node} missing from coloring"
        assert 1 <= colors[node] <= k, (
            f"Color {colors[node]} out of range [1, {k}]"
        )


class TestPRS:
    def test_petersen(self):
        G = nx.petersen_graph()
        colors = sample_coloring(G, k=4, method='prs', seed=42)
        _check_coloring(G, colors, 4)

    def test_cycle(self):
        G = nx.cycle_graph(12)
        colors = sample_coloring(G, k=3, method='prs', seed=7)
        _check_coloring(G, colors, 3)


class TestHybrid:
    def test_petersen_nrs_fallback(self):
        G = nx.petersen_graph()
        colors = sample_coloring(G, k=4, method='hybrid', seed=42)
        _check_coloring(G, colors, 4)

    def test_cycle_high_k(self):
        """k > 3*Delta should auto-select BC20."""
        G = nx.cycle_graph(10)
        colors = sample_coloring(G, k=7, method='hybrid', seed=1)
        _check_coloring(G, colors, 7)

    def test_petersen_huber_range(self):
        """k >= Delta*(Delta+2) = 15 should auto-select Huber."""
        G = nx.petersen_graph()
        colors = sample_coloring(G, k=15, method='hybrid', seed=5)
        _check_coloring(G, colors, 15)


class TestCFTPHuber:
    def test_petersen(self):
        G = nx.petersen_graph()
        colors = sample_coloring(G, k=15, method='cftp_huber', seed=42)
        _check_coloring(G, colors, 15)

    def test_cycle(self):
        G = nx.cycle_graph(10)
        colors = sample_coloring(G, k=8, method='cftp_huber', seed=42)
        _check_coloring(G, colors, 8)


class TestCFTPBC20:
    def test_cycle(self):
        G = nx.cycle_graph(10)
        colors = sample_coloring(G, k=7, method='cftp_bc20', seed=42)
        _check_coloring(G, colors, 7)

    def test_petersen(self):
        G = nx.petersen_graph()
        colors = sample_coloring(G, k=10, method='cftp_bc20', seed=42)
        _check_coloring(G, colors, 10)


class TestNRS:
    def test_small_complete(self):
        G = nx.complete_graph(4)
        colors = sample_coloring(G, k=6, method='nrs', seed=42)
        _check_coloring(G, colors, 6)

    def test_path(self):
        G = nx.path_graph(5)
        colors = sample_coloring(G, k=3, method='nrs', seed=42)
        _check_coloring(G, colors, 3)


class TestEdgeCases:
    def test_empty_graph(self):
        G = nx.Graph()
        colors = sample_coloring(G, k=3, method='prs', seed=42)
        assert colors == {}

    def test_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        colors = sample_coloring(G, k=2, method='prs', seed=42)
        assert 0 in colors
        assert 1 <= colors[0] <= 2

    def test_unknown_method(self):
        G = nx.cycle_graph(5)
        with pytest.raises(ValueError, match="Unknown method"):
            sample_coloring(G, k=3, method='bogus')

    def test_k_too_small(self):
        G = nx.petersen_graph()  # Delta = 3
        with pytest.raises(ValueError, match="Need k > Delta"):
            sample_coloring(G, k=3)

    def test_verify_bad_coloring(self):
        G = nx.complete_graph(3)
        assert not verify_coloring(G, {0: 1, 1: 1, 2: 2})
        assert verify_coloring(G, {0: 1, 1: 2, 2: 3})

    def test_reproducibility(self):
        G = nx.petersen_graph()
        c1 = sample_coloring(G, k=5, method='prs', seed=123)
        c2 = sample_coloring(G, k=5, method='prs', seed=123)
        assert c1 == c2
