"""
Layer 3: Detector Layer
Applies thresholds to identify significant abundance and slope patterns
"""

from .threshold import Threshold, Slope

__all__ = ['Threshold', 'Slope']
