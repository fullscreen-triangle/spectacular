"""
Visual Reasoning Framework: Extended dimensional embeddings for AI understanding.

This module implements visual reasoning capabilities that encode richer information
than traditional text embeddings, supporting 2D/3D spatial relationships and
temporal dynamics for enhanced AI comprehension.
"""

from .core.visual_embeddings import VisualEmbeddingProcessor, VisualEmbedding
from .core.spatial_reasoning import SpatialReasoningEngine, SpatialContext
from .core.mathematical_visualization import MathVisualizationEngine, MathVisualizer
from .temporal.time_series_reasoning import TimeSeriesReasoningEngine

__all__ = [
    'VisualEmbeddingProcessor',
    'VisualEmbedding', 
    'SpatialReasoningEngine',
    'SpatialContext',
    'MathVisualizationEngine',
    'MathVisualizer',
    'TimeSeriesReasoningEngine'
]

__version__ = "1.0.0"
