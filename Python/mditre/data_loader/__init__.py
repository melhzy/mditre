"""
MDITRE Data Loader Module

Modular and extensible data loading system for microbiome time-series data.

This module provides a flexible architecture for loading and preprocessing
various microbiome data modalities (16S, WGS, Metaphlan, etc.) while maintaining
a consistent interface for the MDITRE model.

Architecture:
    - base_loader.py: Abstract base classes and registry
    - loaders/: Specific data format loaders (16S, WGS, Metaphlan)
    - transforms.py: Data transformation utilities
    - datasets.py: PyTorch Dataset implementations

Design Principles:
    1. Modular: Each data modality has its own loader
    2. Extensible: Easy to add new data formats
    3. Consistent: All loaders provide same interface
    4. Type-safe: Proper validation and error handling
    5. Documented: Clear API for researchers
"""

from .base_loader import BaseDataLoader, DataLoaderRegistry, compute_phylo_distance_matrix, get_otu_embeddings
from .datasets import TrajectoryDataset, TrajectoryDatasetWithMetadata, create_data_loader, create_stratified_loaders, create_kfold_loaders
from .transforms import (
    DataTransform,
    NormalizeTransform,
    LogTransform,
    CLRTransform,
    FilterLowAbundance,
    ZScoreTransform,
    RobustScaleTransform,
    TransformPipeline
)

# Import loaders to register them
from .loaders import PickleDataLoader, PickleTrajectoryLoader, DADA2Loader, QIIME2Loader

__all__ = [
    # Base classes
    'BaseDataLoader',
    'DataLoaderRegistry',
    'compute_phylo_distance_matrix',
    'get_otu_embeddings',
    
    # Datasets
    'TrajectoryDataset',
    'TrajectoryDatasetWithMetadata',
    'create_data_loader',
    'create_stratified_loaders',
    'create_kfold_loaders',
    
    # Transforms
    'DataTransform',
    'NormalizeTransform',
    'LogTransform',
    'CLRTransform',
    'FilterLowAbundance',
    'ZScoreTransform',
    'RobustScaleTransform',
    'TransformPipeline',
    
    # Loaders
    'PickleDataLoader',
    'PickleTrajectoryLoader',
    'DADA2Loader',
    'QIIME2Loader',
]

__version__ = '1.0.0'
