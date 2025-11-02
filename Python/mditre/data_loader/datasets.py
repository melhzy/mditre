"""
PyTorch Dataset Implementations

Provides PyTorch Dataset classes for MDITRE training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, Tuple, Union


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for microbiome time-series trajectories.
    
    Handles abundance data, labels, and temporal masks for MDITRE training.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 mask: Optional[np.ndarray] = None,
                 return_numpy: bool = False):
        """
        Initialize trajectory dataset.
        
        Args:
            X: Abundance data (n_subjects, n_otus, n_timepoints)
            y: Binary labels (n_subjects,)
            mask: Optional temporal mask (n_subjects, n_timepoints)
                  1 where samples exist, 0 otherwise
            return_numpy: If True, return numpy arrays instead of tensors
        """
        super(TrajectoryDataset, self).__init__()
        
        # Validate inputs
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
        
        if mask is not None and len(mask) != len(X):
            raise ValueError(f"mask must have same length as X: {len(mask)} vs {len(X)}")
        
        self.X = X
        self.y = y
        self.mask = mask
        self.return_numpy = return_numpy
        
    def __len__(self) -> int:
        """Return number of subjects"""
        return len(self.X)
    
    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Dict[str, Any]:
        """
        Get a single subject's data.
        
        Args:
            idx: Subject index (int or Tensor)
            
        Returns:
            Dictionary with 'data', 'label', and optionally 'mask'
        """
        # Convert tensor index to int
        if torch.is_tensor(idx):
            idx = int(idx.item()) if idx.numel() == 1 else int(idx.tolist()[0])
        
        traj = self.X[idx]
        label = self.y[idx]
        
        # Convert to tensors unless requested otherwise
        if not self.return_numpy:
            traj = torch.from_numpy(traj).float()
            label = torch.tensor(label, dtype=torch.long)
        
        sample = {'data': traj, 'label': label}
        
        if self.mask is not None:
            mask_val = self.mask[idx]
            if not self.return_numpy:
                mask_val = torch.from_numpy(mask_val).float()
            sample['mask'] = mask_val
        
        return sample
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset info
        """
        stats = {
            'n_subjects': len(self.X),
            'n_otus': self.X.shape[1],
            'n_timepoints': self.X.shape[2],
            'n_positive': (self.y == 1).sum(),
            'n_negative': (self.y == 0).sum(),
            'class_balance': (self.y == 1).sum() / len(self.y),
        }
        
        if self.mask is not None:
            stats['avg_samples_per_subject'] = self.mask.sum(axis=1).mean()
            stats['min_samples_per_subject'] = self.mask.sum(axis=1).min()
            stats['max_samples_per_subject'] = self.mask.sum(axis=1).max()
        
        return stats


class TrajectoryDatasetWithMetadata(TrajectoryDataset):
    """
    Extended trajectory dataset that includes additional metadata.
    
    Useful for storing subject IDs, time points, and other covariates.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 times: Optional[np.ndarray] = None,
                 subject_ids: Optional[np.ndarray] = None,
                 covariates: Optional[Dict[str, np.ndarray]] = None,
                 return_numpy: bool = False):
        """
        Initialize dataset with metadata.
        
        Args:
            X: Abundance data
            y: Labels
            mask: Temporal mask
            times: Time indices for each subject
            subject_ids: Subject identifiers
            covariates: Additional covariates dictionary
            return_numpy: Return numpy arrays instead of tensors
        """
        super().__init__(X, y, mask, return_numpy)
        
        self.times = times
        self.subject_ids = subject_ids
        self.covariates = covariates or {}
        
    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Dict[str, Any]:
        """Get subject data with metadata"""
        sample = super().__getitem__(idx)
        
        # Ensure idx is int for indexing
        if torch.is_tensor(idx):
            idx = int(idx.item()) if idx.numel() == 1 else int(idx.tolist()[0])
        
        if self.times is not None:
            sample['times'] = self.times[idx]
        
        if self.subject_ids is not None:
            sample['subject_id'] = self.subject_ids[idx]
        
        # Add covariates
        for key, values in self.covariates.items():
            sample[key] = values[idx]
        
        return sample


def create_data_loader(X: np.ndarray, y: np.ndarray,
                      mask: Optional[np.ndarray] = None,
                      batch_size: int = 32,
                      shuffle: bool = False,
                      num_workers: int = 0,
                      pin_memory: bool = False,
                      drop_last: bool = False) -> DataLoader:
    """
    Create a PyTorch DataLoader for MDITRE training.
    
    Args:
        X: Abundance data (n_subjects, n_otus, n_timepoints)
        y: Labels (n_subjects,)
        mask: Optional temporal mask
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        
    Returns:
        PyTorch DataLoader instance
    """
    dataset = TrajectoryDataset(X, y, mask)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    return loader


def create_stratified_loaders(X: np.ndarray, y: np.ndarray,
                              mask: Optional[np.ndarray] = None,
                              train_ratio: float = 0.8,
                              batch_size: int = 32,
                              num_workers: int = 0,
                              random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create stratified train/validation data loaders.
    
    Args:
        X: Abundance data
        y: Labels
        mask: Optional temporal mask
        train_ratio: Fraction of data for training
        batch_size: Batch size
        num_workers: Number of workers
        random_seed: Random seed for reproducibility
        
    Returns:
        (train_loader, val_loader) tuple
    """
    from sklearn.model_selection import train_test_split
    
    # Stratified split
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=y,
        random_state=random_seed
    )
    
    # Create datasets
    train_dataset = TrajectoryDataset(
        X[train_idx], y[train_idx],
        mask[train_idx] if mask is not None else None
    )
    
    val_dataset = TrajectoryDataset(
        X[val_idx], y[val_idx],
        mask[val_idx] if mask is not None else None
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_kfold_loaders(X: np.ndarray, y: np.ndarray,
                        mask: Optional[np.ndarray] = None,
                        n_splits: int = 5,
                        batch_size: int = 32,
                        num_workers: int = 0,
                        random_seed: int = 42):
    """
    Create k-fold cross-validation data loaders.
    
    Args:
        X: Abundance data
        y: Labels
        mask: Optional temporal mask
        n_splits: Number of folds
        batch_size: Batch size
        num_workers: Number of workers
        random_seed: Random seed
        
    Yields:
        (train_loader, val_loader) for each fold
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_dataset = TrajectoryDataset(
            X[train_idx], y[train_idx],
            mask[train_idx] if mask is not None else None
        )
        
        val_dataset = TrajectoryDataset(
            X[val_idx], y[val_idx],
            mask[val_idx] if mask is not None else None
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        yield fold, train_loader, val_loader
