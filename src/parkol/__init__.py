"""
parkol -- Exact uniform sampling of proper graph colourings.

Uses gamma-soft colouring and partial rejection sampling (PRS).
"""

from .sample import sample_coloring
from .utils import verify_coloring

__version__ = "0.1.1"
__all__ = ["sample_coloring", "verify_coloring"]
