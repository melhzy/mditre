"""
16S rRNA Sequencing Data Loader

Loaders for 16S amplicon sequencing data (DADA2, QIIME2, Mothur output).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from ete3 import Tree

from ..base_loader import BaseDataLoader, DataLoaderRegistry


@DataLoaderRegistry.register('16s_dada2')
class DADA2Loader(BaseDataLoader):
    """
    Load DADA2 output format.
    
    Expected files:
        - abundance.csv: OTU/ASV abundance table
        - tax_table.csv: Taxonomy assignments
        - sample_metadata.csv: Sample metadata with time points
        - phylo_tree.nwk or placement.jplace: Phylogenetic tree
    """
    
    def __init__(self, data_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DADA2 loader.
        
        Args:
            data_path: Path to directory containing DADA2 output
            config: Configuration dict with keys:
                - abundance_file: filename for abundance table
                - tax_file: filename for taxonomy
                - metadata_file: filename for metadata
                - tree_file: filename for phylogenetic tree
                - subject_col: column name for subject IDs
                - time_col: column name for time points
                - label_col: column name for outcome labels
        """
        super().__init__(data_path, config)
        
        # Set default file names
        self.abundance_file = config.get('abundance_file', 'abundance.csv')
        self.tax_file = config.get('tax_file', 'tax_table.csv')
        self.metadata_file = config.get('metadata_file', 'sample_metadata.csv')
        self.tree_file = config.get('tree_file', 'phylo_tree.nwk')
        
        # Column names
        self.subject_col = config.get('subject_col', 'subject_id')
        self.time_col = config.get('time_col', 'time')
        self.label_col = config.get('label_col', 'outcome')
        
    def load(self) -> Dict[str, Any]:
        """Load DADA2 output files"""
        data_dir = Path(self.data_path)
        
        # Load abundance table
        abundance_path = data_dir / self.abundance_file
        abundance_df = pd.read_csv(abundance_path, index_col=0)
        
        # Load metadata
        metadata_path = data_dir / self.metadata_file
        metadata_df = pd.read_csv(metadata_path, index_col=0)
        
        # Load taxonomy (optional)
        tax_df = None
        tax_path = data_dir / self.tax_file
        if tax_path.exists():
            tax_df = pd.read_csv(tax_path, index_col=0)
        
        # Load tree
        tree = None
        tree_path = data_dir / self.tree_file
        if tree_path.exists():
            if tree_path.suffix == '.nwk':
                tree = Tree(str(tree_path))
            elif tree_path.suffix == '.jplace':
                # Parse jplace format
                tree = self._parse_jplace(tree_path)
        
        return {
            'abundance': abundance_df,
            'metadata': metadata_df,
            'taxonomy': tax_df,
            'tree': tree
        }
    
    def preprocess(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess DADA2 data into MDITRE format.
        
        Args:
            raw_data: Raw data dictionary from load()
            
        Returns:
            Preprocessed data dictionary
        """
        abundance_df = raw_data['abundance']
        metadata_df = raw_data['metadata']
        
        # Extract subject IDs and organize by subject
        subjects = metadata_df[self.subject_col].unique()
        n_subjects = len(subjects)
        n_otus = abundance_df.shape[1]
        
        # Determine time range
        all_times = metadata_df[self.time_col].values
        min_time = int(all_times.min())
        max_time = int(all_times.max())
        n_timepoints = max_time - min_time + 1
        
        # Initialize arrays
        X = np.zeros((n_subjects, n_otus, n_timepoints), dtype=np.float32)
        mask = np.zeros((n_subjects, n_timepoints), dtype=np.float32)
        y = np.zeros(n_subjects, dtype=np.int64)
        times_list = []
        
        # Organize data by subject
        for i, subject_id in enumerate(subjects):
            # Get samples for this subject
            subject_mask = metadata_df[self.subject_col] == subject_id
            subject_samples = metadata_df[subject_mask]
            subject_abundance = abundance_df.loc[subject_samples.index]
            
            # Get label (assume consistent within subject)
            label_val = subject_samples[self.label_col].iloc[0]
            if isinstance(label_val, str):
                y[i] = 1 if label_val.lower() in ['true', 'yes', '1', 'positive'] else 0
            else:
                y[i] = int(label_val)
            
            # Fill in time points
            subject_times = subject_samples[self.time_col].values
            times_list.append(subject_times.astype(int))
            
            for j, (_, row) in enumerate(subject_samples.iterrows()):
                time_idx = int(row[self.time_col]) - min_time
                if 0 <= time_idx < n_timepoints:
                    X[i, :, time_idx] = subject_abundance.iloc[j].values
                    mask[i, time_idx] = 1.0
        
        # Get variable names
        variable_names = abundance_df.columns.tolist()
        
        # Get or create phylogenetic tree
        phylo_tree = raw_data.get('tree')
        if phylo_tree is None:
            phylo_tree = self._create_star_tree(variable_names)
        
        # Prepare metadata
        metadata = {
            'n_subjects': n_subjects,
            'n_otus': n_otus,
            'n_timepoints': n_timepoints,
            'variable_names': variable_names,
            'experiment_start': min_time,
            'experiment_end': max_time,
            'subject_IDs': subjects.tolist(),
            'taxonomy': raw_data.get('taxonomy'),
        }
        
        return {
            'X': X,
            'y': y,
            'times': np.array(times_list, dtype=object),
            'mask': mask,
            'phylo_tree': phylo_tree,
            'metadata': metadata
        }
    
    def _parse_jplace(self, jplace_path):
        """Parse jplace format phylogenetic placement file"""
        import json
        
        with open(jplace_path) as f:
            jplace_data = json.load(f)
        
        # Extract tree from jplace
        tree_str = jplace_data['tree']
        tree = Tree(tree_str, format=1)
        
        return tree
    
    def _create_star_tree(self, variable_names):
        """Create simple star tree if no phylogeny available"""
        root = Tree()
        root.name = "root"
        
        for name in variable_names:
            child = root.add_child(name=str(name))
            child.dist = 1.0
        
        return root


@DataLoaderRegistry.register('16s_qiime2')
class QIIME2Loader(BaseDataLoader):
    """
    Load QIIME2 output format.
    
    Expected files:
        - feature-table.csv: OTU/ASV abundance table
        - taxonomy.csv: Taxonomy assignments
        - metadata.tsv: Sample metadata
        - tree.nwk: Phylogenetic tree
    """
    
    def __init__(self, data_path: str, config: Optional[Dict[str, Any]] = None):
        """Initialize QIIME2 loader"""
        super().__init__(data_path, config)
        
        # QIIME2 default file names
        self.feature_table_file = config.get('feature_table_file', 'feature-table.csv')
        self.metadata_file = config.get('metadata_file', 'metadata.tsv')
        self.tree_file = config.get('tree_file', 'tree.nwk')
        
        # Column names
        self.subject_col = config.get('subject_col', 'subject-id')
        self.time_col = config.get('time_col', 'collection-time')
        self.label_col = config.get('label_col', 'outcome')
        
    def load(self) -> Dict[str, Any]:
        """Load QIIME2 output files"""
        data_dir = Path(self.data_path)
        
        # Load feature table
        feature_path = data_dir / self.feature_table_file
        feature_df = pd.read_csv(feature_path, index_col=0, sep=',')
        
        # Load metadata
        metadata_path = data_dir / self.metadata_file
        metadata_df = pd.read_csv(metadata_path, index_col=0, sep='\t')
        
        # Load tree
        tree = None
        tree_path = data_dir / self.tree_file
        if tree_path.exists():
            tree = Tree(str(tree_path))
        
        return {
            'feature_table': feature_df,
            'metadata': metadata_df,
            'tree': tree
        }
    
    def preprocess(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess QIIME2 data.
        
        Similar to DADA2 preprocessing but handles QIIME2 format specifics.
        """
        # Use DADA2 loader logic with QIIME2-specific adjustments
        dada2_data = {
            'abundance': raw_data['feature_table'],
            'metadata': raw_data['metadata'],
            'tree': raw_data['tree'],
            'taxonomy': None
        }
        
        # Create temporary DADA2 loader to reuse preprocessing
        temp_loader = DADA2Loader(self.data_path, self.config)
        return temp_loader.preprocess(dada2_data)
