"""
MDITRE: Microbiome DynamIc Time-series Rule Extraction

A deep learning framework for extracting interpretable rules from longitudinal
microbiome data for disease prediction and biological understanding.

Main Components:
- models: Original monolithic implementation (backward compatibility)
- layers: New modular architecture (recommended for new projects)
- core: Base classes and shared utilities
- seeding: Deterministic seed generation for reproducibility
- utils: Cross-platform utilities including path handling

Quick Start:
    # Using original models
    from mditre.models import MDITRE, MDITREAbun

    # Using modular layers
    from mditre.layers import (
        SpatialAggDynamic,
        TimeAgg,
        Threshold,
        Slope,
        Rules,
        DenseLayer
    )

    # Using core utilities
    from mditre.core import BaseLayer, LayerRegistry

    # Using seeding for reproducibility
    from mditre.seeding import MDITRESeedGenerator, get_mditre_seeds, set_random_seeds

    # Using cross-platform path utilities
    from mditre.utils.path_utils import (
        get_project_root,
        get_data_dir,
        normalize_path
    )

See MODULAR_ARCHITECTURE.md and CROSS_PLATFORM_PATHS.md for detailed documentation.
"""

__version__ = "1.0.1"

# Expose modular layers for new projects
from . import core, layers, seeding, utils

# Original models (backward compatibility)
from .models import MDITRE, MDITREAbun

__all__ = [
    "MDITRE",
    "MDITREAbun",
    "layers",
    "core",
    "seeding",
    "utils",
]
