"""
Data Transformation Utilities

Provides composable transformations for microbiome data preprocessing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import warnings


class DataTransform(ABC):
    """
    Abstract base class for data transformations.
    
    Transformations are composable and can be chained together.
    """
    
    def __init__(self, name: str):
        """
        Initialize transform.
        
        Args:
            name: Human-readable name for this transform
        """
        self.name = name
        
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply transformation to data.
        
        Args:
            X: Input data (n_subjects, n_features, n_timepoints)
            
        Returns:
            Transformed data
        """
        pass
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Allow transform to be called as a function"""
        return self.transform(X)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class NormalizeTransform(DataTransform):
    """
    Normalize abundances to sum to 1 (relative abundances).
    
    Converts absolute abundances to compositional data.
    """
    
    def __init__(self, axis: int = 1, epsilon: float = 1e-10):
        """
        Initialize normalization transform.
        
        Args:
            axis: Axis along which to normalize (1 for OTUs)
            epsilon: Small constant to avoid division by zero
        """
        super().__init__("normalize")
        self.axis = axis
        self.epsilon = epsilon
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Normalize data"""
        # Handle zero samples
        sums = X.sum(axis=self.axis, keepdims=True)
        sums = np.where(sums == 0, 1.0, sums)  # Avoid division by zero
        return X / (sums + self.epsilon)


class LogTransform(DataTransform):
    """
    Apply log transformation to handle skewed distributions.
    
    Uses log(x + pseudocount) to handle zeros.
    """
    
    def __init__(self, pseudocount: float = 1e-6, base: str = 'natural'):
        """
        Initialize log transform.
        
        Args:
            pseudocount: Value added before taking log
            base: Logarithm base ('natural', '10', '2')
        """
        super().__init__("log_transform")
        self.pseudocount = pseudocount
        self.base = base
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply log transformation"""
        X_shifted = X + self.pseudocount
        
        if self.base == 'natural':
            return np.log(X_shifted)
        elif self.base == '10':
            return np.log10(X_shifted)
        elif self.base == '2':
            return np.log2(X_shifted)
        else:
            raise ValueError(f"Unknown base: {self.base}")


class CLRTransform(DataTransform):
    """
    Centered Log-Ratio (CLR) transformation for compositional data.
    
    CLR(x) = log(x / g(x)) where g(x) is geometric mean.
    Commonly used in microbiome analysis to handle compositionality.
    """
    
    def __init__(self, pseudocount: float = 1e-6):
        """
        Initialize CLR transform.
        
        Args:
            pseudocount: Value added before taking log
        """
        super().__init__("clr_transform")
        self.pseudocount = pseudocount
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply CLR transformation"""
        X_shifted = X + self.pseudocount
        
        # Compute geometric mean along OTU axis
        # Log-transform then take mean, then exp
        log_X = np.log(X_shifted)
        geom_mean = np.exp(log_X.mean(axis=1, keepdims=True))
        
        # CLR = log(x / geom_mean)
        clr = np.log(X_shifted / geom_mean)
        
        return clr


class FilterLowAbundance(DataTransform):
    """
    Filter out OTUs with low abundance or prevalence.
    
    Removes rare OTUs that may be noise or sequencing artifacts.
    """
    
    def __init__(self, min_abundance: float = 0.0001, 
                 min_prevalence: float = 0.1,
                 keep_indices: Optional[np.ndarray] = None):
        """
        Initialize filtering transform.
        
        Args:
            min_abundance: Minimum mean abundance to keep OTU
            min_prevalence: Minimum fraction of samples where OTU present
            keep_indices: Pre-computed indices to keep (overrides filtering)
        """
        super().__init__("filter_low_abundance")
        self.min_abundance = min_abundance
        self.min_prevalence = min_prevalence
        self.keep_indices = keep_indices
        self._computed_indices = None
        
    def compute_filter_indices(self, X: np.ndarray) -> np.ndarray:
        """
        Compute which OTUs to keep based on filtering criteria.
        
        Args:
            X: Input data (n_subjects, n_otus, n_timepoints)
            
        Returns:
            Boolean array of OTUs to keep
        """
        if self.keep_indices is not None:
            return self.keep_indices
        
        # Flatten across subjects and time for prevalence calculation
        X_flat = X.reshape(-1, X.shape[1])  # (n_subjects * n_timepoints, n_otus)
        
        # Abundance filter: mean abundance across all samples
        mean_abundance = X_flat.mean(axis=0)
        abundance_mask = mean_abundance >= self.min_abundance
        
        # Prevalence filter: fraction of non-zero samples
        prevalence = (X_flat > 0).mean(axis=0)
        prevalence_mask = prevalence >= self.min_prevalence
        
        # Keep OTUs passing both filters
        keep_mask = abundance_mask & prevalence_mask
        
        return keep_mask
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply filtering"""
        if self._computed_indices is None:
            self._computed_indices = self.compute_filter_indices(X)
        
        n_original = X.shape[1]
        n_kept = self._computed_indices.sum()
        
        if n_kept < n_original:
            warnings.warn(
                f"Filtered {n_original - n_kept} OTUs, keeping {n_kept} "
                f"(min_abundance={self.min_abundance}, min_prevalence={self.min_prevalence})"
            )
        
        # Apply filter
        return X[:, self._computed_indices, :]
    
    def get_kept_indices(self) -> Optional[np.ndarray]:
        """Get indices of OTUs that were kept"""
        return self._computed_indices


class TransformPipeline:
    """
    Pipeline for applying multiple transforms in sequence.
    
    Example:
        pipeline = TransformPipeline([
            NormalizeTransform(),
            FilterLowAbundance(min_abundance=0.001),
            CLRTransform()
        ])
        X_transformed = pipeline(X)
    """
    
    def __init__(self, transforms: list):
        """
        Initialize pipeline.
        
        Args:
            transforms: List of DataTransform instances
        """
        self.transforms = transforms
        
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply all transforms in sequence"""
        for transform in self.transforms:
            X = transform(X)
        return X
    
    def __repr__(self) -> str:
        transform_names = [t.name for t in self.transforms]
        return f"TransformPipeline({' -> '.join(transform_names)})"


class ZScoreTransform(DataTransform):
    """
    Z-score normalization (standardization).
    
    Transforms data to have mean 0 and std 1.
    """
    
    def __init__(self, axis: int = 1, epsilon: float = 1e-10):
        """
        Initialize z-score transform.
        
        Args:
            axis: Axis along which to compute statistics
            epsilon: Small constant for numerical stability
        """
        super().__init__("zscore")
        self.axis = axis
        self.epsilon = epsilon
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization"""
        mean = X.mean(axis=self.axis, keepdims=True)
        std = X.std(axis=self.axis, keepdims=True)
        return (X - mean) / (std + self.epsilon)


class RobustScaleTransform(DataTransform):
    """
    Robust scaling using median and IQR.
    
    More robust to outliers than z-score normalization.
    """
    
    def __init__(self, axis: int = 1):
        """
        Initialize robust scaling.
        
        Args:
            axis: Axis along which to compute statistics
        """
        super().__init__("robust_scale")
        self.axis = axis
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply robust scaling"""
        median = np.median(X, axis=self.axis, keepdims=True)
        q75 = np.percentile(X, 75, axis=self.axis, keepdims=True)
        q25 = np.percentile(X, 25, axis=self.axis, keepdims=True)
        iqr = q75 - q25
        # Avoid division by zero
        iqr = np.where(iqr == 0, 1.0, iqr)
        return (X - median) / iqr
