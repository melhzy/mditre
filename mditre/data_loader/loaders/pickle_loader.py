"""
Pickle Data Loader

Loader for MDITRE's native pickle format (backward compatibility).
"""

import pickle
import numpy as np
from typing import Dict, Any, Optional
from ete3 import Tree

from ..base_loader import BaseDataLoader, DataLoaderRegistry


@DataLoaderRegistry.register('pickle')
class PickleDataLoader(BaseDataLoader):
    """
    Load data from MDITRE's native pickle format.
    
    Expected pickle structure:
        {
            'X': list of arrays or array,
            'y': array of labels,
            'T': array of time indices,
            'variable_tree': ete3 Tree,
            'variable_names': list of OTU names,
            ...
        }
    """
    
    def __init__(self, data_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pickle loader.
        
        Args:
            data_path: Path to pickle file
            config: Optional configuration
        """
        super().__init__(data_path, config)
        
    def load(self) -> Dict[str, Any]:
        """Load data from pickle file"""
        with open(self.data_path, 'rb') as f:
            try:
                data = pickle.load(f, encoding='latin1')
            except:
                data = pickle.load(f)
        
        return data
    
    def preprocess(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess pickle data into MDITRE format.
        
        Args:
            raw_data: Raw pickle data
            
        Returns:
            Preprocessed data dictionary
        """
        # Extract components
        X_raw = raw_data['X']
        y = np.array(raw_data['y'])
        times = raw_data.get('T', None)
        phylo_tree = raw_data.get('variable_tree', None)
        variable_names = raw_data.get('variable_names', [])
        
        # Process X - handle list of arrays (variable length trajectories)
        if isinstance(X_raw, list):
            X, mask = self._process_variable_length_trajectories(X_raw, times)
        else:
            X = X_raw
            mask = np.ones((X.shape[0], X.shape[2]), dtype=np.float32)
        
        # Ensure proper dtypes
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        
        # Create default tree if missing
        if phylo_tree is None:
            from ..base_loader import compute_phylo_distance_matrix
            phylo_tree = self._create_default_tree(variable_names or list(range(X.shape[1])))
        
        # Prepare metadata
        metadata = {
            'n_subjects': len(X),
            'n_otus': X.shape[1],
            'n_timepoints': X.shape[2],
            'variable_names': variable_names,
            'experiment_start': raw_data.get('experiment_start', 0),
            'experiment_end': raw_data.get('experiment_end', X.shape[2] - 1),
            'variable_annotations': raw_data.get('variable_annotations', {}),
            'subject_IDs': raw_data.get('subject_IDs', None),
            'subject_data': raw_data.get('subject_data', None),
        }
        
        return {
            'X': X,
            'y': y,
            'times': times,
            'mask': mask,
            'phylo_tree': phylo_tree,
            'metadata': metadata
        }
    
    def _process_variable_length_trajectories(self, X_list, times):
        """
        Convert list of variable-length trajectories to fixed-size array with mask.
        
        Args:
            X_list: List of arrays (n_otus, n_samples)
            times: Array of time indices per subject
            
        Returns:
            (X, mask) tuple
        """
        n_subjects = len(X_list)
        n_otus = X_list[0].shape[0]
        
        # Find maximum time point
        max_time = 0
        for t in times:
            max_time = max(max_time, t.max())
        n_timepoints = int(max_time) + 1
        
        # Initialize arrays
        X = np.zeros((n_subjects, n_otus, n_timepoints), dtype=np.float32)
        mask = np.zeros((n_subjects, n_timepoints), dtype=np.float32)
        
        # Fill in data
        for i, (x, t) in enumerate(zip(X_list, times)):
            t_int = t.astype(int)
            for j, time_idx in enumerate(t_int):
                if time_idx < n_timepoints:
                    X[i, :, time_idx] = x[:, j]
                    mask[i, time_idx] = 1.0
        
        return X, mask
    
    def _create_default_tree(self, variable_names):
        """Create a simple star tree if no phylogeny provided"""
        if not variable_names:
            raise ValueError("No variable names provided and no tree available")
        
        # Create root
        root = Tree()
        root.name = "root"
        
        # Add all variables as children of root
        for name in variable_names:
            child = root.add_child(name=str(name))
            child.dist = 1.0
        
        return root


@DataLoaderRegistry.register('pickle_trajectory')
class PickleTrajectoryLoader(PickleDataLoader):
    """
    Specialized loader for trajectory-format pickle files.
    
    Handles cases where subjects have variable numbers of samples.
    """
    
    def preprocess(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess with special handling for irregular sampling.
        
        Args:
            raw_data: Raw pickle data
            
        Returns:
            Preprocessed data dictionary
        """
        result = super().preprocess(raw_data)
        
        # Additional filtering: remove subjects with too few samples
        min_samples = self.config.get('min_samples_per_subject', 2)
        
        if 'mask' in result:
            samples_per_subject = result['mask'].sum(axis=1)
            valid_subjects = samples_per_subject >= min_samples
            
            if not valid_subjects.all():
                n_removed = (~valid_subjects).sum()
                print(f"Filtering {n_removed} subjects with < {min_samples} samples")
                
                result['X'] = result['X'][valid_subjects]
                result['y'] = result['y'][valid_subjects]
                result['mask'] = result['mask'][valid_subjects]
                if result['times'] is not None:
                    result['times'] = result['times'][valid_subjects]
                
                # Update metadata
                result['metadata']['n_subjects'] = valid_subjects.sum()
                result['metadata']['n_filtered'] = n_removed
        
        return result
