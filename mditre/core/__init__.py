"""
MDITRE Core Module

Contains shared utilities and base classes used across all layers:
- base_layer: Abstract base class and layer registry
- math_utils: Mathematical utility functions for differentiable operations
"""

from .base_layer import BaseLayer, LayerRegistry
from .math_utils import (
    EPSILON,
    binary_concrete,
    unitboxcar,
    transf_log,
    inv_transf_log
)

__all__ = [
    'BaseLayer',
    'LayerRegistry',
    'EPSILON',
    'binary_concrete',
    'unitboxcar',
    'transf_log',
    'inv_transf_log',
]
