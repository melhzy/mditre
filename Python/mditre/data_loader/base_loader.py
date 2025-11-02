"""
Base Data Loader Classes and Registry

Provides abstract base class and registry for modular data loading system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
from ete3 import Tree


class BaseDataLoader(ABC):
    """
    Abstract base class for all MDITRE data loaders.
    
    All data loaders must implement methods to load and preprocess
    microbiome time-series data into the format expected by MDITRE.
    
    Expected output format:
        X: (n_subjects, n_otus, n_timepoints) - abundance data
        y: (n_subjects,) - binary labels
        times: (n_subjects, n_samples_per_subject) - time indices
        phylo_tree: ete3.Tree - phylogenetic tree
        metadata: dict - additional information
    """
    
    def __init__(self, data_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to data file or directory
            config: Optional configuration dictionary
        """
        self.data_path = data_path
        self.config = config or {}
        self.data_loaded = False
        
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """
        Load raw data from source.
        
        Returns:
            Dictionary with raw data components
        """
        pass
    
    @abstractmethod
    def preprocess(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess raw data into MDITRE format.
        
        Args:
            raw_data: Raw data dictionary from load()
            
        Returns:
            Preprocessed data dictionary with keys:
                - X: abundance matrix
                - y: labels
                - times: time indices
                - phylo_tree: phylogenetic tree
                - metadata: additional info
        """
        pass
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate data format and integrity.
        
        Args:
            data: Preprocessed data dictionary
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_keys = ['X', 'y', 'times', 'phylo_tree', 'metadata']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate shapes
        n_subjects = len(data['X'])
        if len(data['y']) != n_subjects:
            raise ValueError(f"Shape mismatch: X has {n_subjects} subjects but y has {len(data['y'])}")
        
        if len(data['times']) != n_subjects:
            raise ValueError(f"Shape mismatch: X has {n_subjects} subjects but times has {len(data['times'])}")
        
        # Validate tree
        if not isinstance(data['phylo_tree'], Tree):
            raise ValueError("phylo_tree must be an ete3.Tree instance")
        
        return True
    
    def load_and_preprocess(self) -> Dict[str, Any]:
        """
        Convenience method to load and preprocess data.
        
        Returns:
            Preprocessed and validated data dictionary
        """
        raw_data = self.load()
        processed_data = self.preprocess(raw_data)
        self.validate(processed_data)
        self.data_loaded = True
        return processed_data
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        if not self.data_loaded:
            raise RuntimeError("Data not loaded yet. Call load_and_preprocess() first.")
        
        return {
            'loader_type': self.__class__.__name__,
            'data_path': self.data_path,
            'config': self.config
        }


class DataLoaderRegistry:
    """
    Registry for managing data loader implementations.
    
    Enables dynamic selection of data loaders based on data type.
    """
    
    _registry = {}
    
    @classmethod
    def register(cls, loader_name: str):
        """
        Decorator to register a data loader.
        
        Args:
            loader_name: Unique identifier for this loader
            
        Example:
            @DataLoaderRegistry.register('16s_dada2')
            class DADA2Loader(BaseDataLoader):
                ...
        """
        def decorator(loader_class):
            if not issubclass(loader_class, BaseDataLoader):
                raise TypeError(f"{loader_class} must inherit from BaseDataLoader")
            cls._registry[loader_name] = loader_class
            return loader_class
        return decorator
    
    @classmethod
    def get_loader(cls, loader_name: str):
        """
        Get a registered loader class.
        
        Args:
            loader_name: Name of loader to retrieve
            
        Returns:
            Loader class
        """
        if loader_name not in cls._registry:
            raise ValueError(f"Unknown loader: {loader_name}. Available: {cls.list_loaders()}")
        return cls._registry[loader_name]
    
    @classmethod
    def list_loaders(cls) -> List[str]:
        """
        List all registered loaders.
        
        Returns:
            List of loader names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def create_loader(cls, loader_name: str, data_path: str, 
                     config: Optional[Dict[str, Any]] = None) -> BaseDataLoader:
        """
        Create a loader instance.
        
        Args:
            loader_name: Name of loader to create
            data_path: Path to data
            config: Optional configuration
            
        Returns:
            Initialized loader instance
        """
        loader_class = cls.get_loader(loader_name)
        return loader_class(data_path, config)


def compute_phylo_distance_matrix(phylo_tree: Tree) -> np.ndarray:
    """
    Compute pairwise phylogenetic distance matrix from tree.
    
    Args:
        phylo_tree: ete3 Tree with OTU leaves
        
    Returns:
        Distance matrix (n_otus, n_otus)
    """
    leaves = phylo_tree.get_leaves()
    n_otus = len(leaves)
    dist_matrix = np.zeros((n_otus, n_otus), dtype=np.float32)
    
    for i, src in enumerate(leaves):
        for j, dst in enumerate(leaves):
            dist_matrix[i, j] = src.get_distance(dst, topology_only=False)
    
    return dist_matrix


def get_otu_embeddings(phylo_tree: Tree, method: str = 'distance', 
                       emb_dim: int = 5) -> np.ndarray:
    """
    Generate OTU embeddings from phylogenetic tree.
    
    Args:
        phylo_tree: ete3 Tree with OTU leaves
        method: Embedding method ('distance', 'mds', 'random')
        emb_dim: Embedding dimension
        
    Returns:
        OTU embeddings (n_otus, emb_dim)
    """
    leaves = phylo_tree.get_leaves()
    n_otus = len(leaves)
    
    if method == 'distance':
        # Use first emb_dim eigenvectors of distance matrix
        dist_matrix = compute_phylo_distance_matrix(phylo_tree)
        # Center the distance matrix
        centered = dist_matrix - dist_matrix.mean(axis=0, keepdims=True)
        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        embeddings = U[:, :emb_dim] * np.sqrt(S[:emb_dim])
        
    elif method == 'mds':
        # Multidimensional scaling
        from sklearn.manifold import MDS
        dist_matrix = compute_phylo_distance_matrix(phylo_tree)
        mds = MDS(n_components=emb_dim, dissimilarity='precomputed', random_state=42)
        embeddings = mds.fit_transform(dist_matrix)
        
    elif method == 'random':
        # Random embeddings (baseline)
        embeddings = np.random.randn(n_otus, emb_dim).astype(np.float32) * 0.1
        
    else:
        raise ValueError(f"Unknown embedding method: {method}")
    
    return embeddings.astype(np.float32)
