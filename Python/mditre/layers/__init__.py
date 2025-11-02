"""
MDITRE Layers Module

Contains all five layers of the MDITRE architecture:
1. Phylogenetic Focus: Spatial aggregation based on phylogenetic relationships
2. Temporal Focus: Time window selection for important patterns
3. Detector: Threshold-based detection of significant signals
4. Rule: Logical combination of detectors
5. Classification: Final prediction with rule selection
"""

# Layer 1: Phylogenetic Focus
from .layer1_phylogenetic_focus import SpatialAgg, SpatialAggDynamic

# Layer 2: Temporal Focus
from .layer2_temporal_focus import TimeAgg, TimeAggAbun

# Layer 3: Detector
from .layer3_detector import Threshold, Slope

# Layer 4: Rule
from .layer4_rule import Rules

# Layer 5: Classification
from .layer5_classification import DenseLayer, DenseLayerAbun

__all__ = [
    # Layer 1
    'SpatialAgg',
    'SpatialAggDynamic',
    # Layer 2
    'TimeAgg',
    'TimeAggAbun',
    # Layer 3
    'Threshold',
    'Slope',
    # Layer 4
    'Rules',
    # Layer 5
    'DenseLayer',
    'DenseLayerAbun',
]
