# MDITRE Data Loader System

## Overview

The MDITRE Data Loader module provides a modular, extensible system for loading and preprocessing microbiome time-series data from various sources and formats. The design enables easy integration of new data modalities while maintaining a consistent interface for the MDITRE model.

## Architecture

```
mditre/data_loader/
├── __init__.py                 # Main module exports
├── base_loader.py              # Abstract base classes and registry
├── datasets.py                 # PyTorch Dataset implementations
├── transforms.py               # Data transformation utilities
└── loaders/                    # Format-specific loaders
    ├── __init__.py
    ├── pickle_loader.py        # MDITRE native format
    └── amplicon_loader.py      # 16S/amplicon sequencing
```

## Core Components

### 1. Base Loader (`base_loader.py`)

**`BaseDataLoader`**: Abstract base class for all loaders
- `load()`: Load raw data from source
- `preprocess()`: Convert to MDITRE format
- `validate()`: Check data integrity
- `load_and_preprocess()`: Convenience method

**`DataLoaderRegistry`**: Registry for dynamic loader management
- `@DataLoaderRegistry.register(name)`: Decorator to register loaders
- `get_loader(name)`: Retrieve loader class
- `list_loaders()`: List all registered loaders
- `create_loader(name, path, config)`: Factory method

**Utility Functions**:
- `compute_phylo_distance_matrix(tree)`: Compute pairwise phylogenetic distances
- `get_otu_embeddings(tree, method, emb_dim)`: Generate OTU embeddings from tree

### 2. Data Transforms (`transforms.py`)

Composable transformations for preprocessing:

**Available Transforms**:
- `NormalizeTransform`: Convert to relative abundances (sum to 1)
- `LogTransform`: Log transformation with pseudocount
- `CLRTransform`: Centered log-ratio for compositional data
- `FilterLowAbundance`: Remove rare OTUs based on abundance/prevalence
- `ZScoreTransform`: Standardization (mean=0, std=1)
- `RobustScaleTransform`: Robust scaling using median and IQR

**`TransformPipeline`**: Chain multiple transforms
```python
pipeline = TransformPipeline([
    NormalizeTransform(),
    FilterLowAbundance(min_abundance=0.001),
    CLRTransform()
])
X_transformed = pipeline(X)
```

### 3. PyTorch Datasets (`datasets.py`)

**`TrajectoryDataset`**: Basic dataset for MDITRE
- Handles (X, y, mask) tuples
- Returns dictionaries with 'data', 'label', 'mask'
- Provides dataset statistics

**`TrajectoryDatasetWithMetadata`**: Extended dataset
- Includes time points, subject IDs, covariates
- Useful for analysis and visualization

**Data Loader Creation**:
- `create_data_loader()`: Single data loader
- `create_stratified_loaders()`: Stratified train/val split
- `create_kfold_loaders()`: K-fold cross-validation loaders

### 4. Format-Specific Loaders

#### Pickle Loader (`loaders/pickle_loader.py`)
- `PickleDataLoader`: Load MDITRE native pickle format
- `PickleTrajectoryLoader`: Handle variable-length trajectories

#### Amplicon Sequencing Loaders (`loaders/amplicon_loader.py`)
- `DADA2Loader`: Load DADA2 output (abundance + metadata + tree)
- `QIIME2Loader`: Load QIIME2 artifacts

## Usage Examples

### Example 1: List Available Loaders

```python
from mditre.data_loader import DataLoaderRegistry

# List all registered loaders
loaders = DataLoaderRegistry.list_loaders()
print(f"Available loaders: {loaders}")
# Output: ['pickle', 'pickle_trajectory', '16s_dada2', '16s_qiime2']
```

### Example 2: Load Pickle Data

```python
from mditre.data_loader import PickleDataLoader

# Method 1: Direct instantiation
loader = PickleDataLoader('path/to/data.pkl')
data = loader.load_and_preprocess()

# Method 2: Using registry
loader = DataLoaderRegistry.create_loader(
    'pickle',
    'path/to/data.pkl',
    config={'min_samples_per_subject': 2}
)
data = loader.load_and_preprocess()

# Access preprocessed data
X = data['X']           # (n_subjects, n_otus, n_timepoints)
y = data['y']           # (n_subjects,)
mask = data['mask']      # (n_subjects, n_timepoints)
tree = data['phylo_tree']
metadata = data['metadata']
```

### Example 3: Load 16S Data (DADA2)

```python
from mditre.data_loader import DADA2Loader

config = {
    'abundance_file': 'abundance.csv',
    'metadata_file': 'sample_metadata.csv',
    'tree_file': 'phylo_tree.nwk',
    'subject_col': 'subject_id',
    'time_col': 'collection_time',
    'label_col': 'outcome'
}

loader = DADA2Loader('path/to/dada2_output/', config=config)
data = loader.load_and_preprocess()
```

### Example 4: Apply Transformations

```python
from mditre.data_loader import (
    TransformPipeline,
    NormalizeTransform,
    FilterLowAbundance,
    CLRTransform
)

# Single transformation
normalize = NormalizeTransform()
X_norm = normalize(X)

# Pipeline of transformations
pipeline = TransformPipeline([
    NormalizeTransform(),                    # Convert to relative abundance
    FilterLowAbundance(                      # Filter rare OTUs
        min_abundance=0.001,
        min_prevalence=0.1
    ),
    CLRTransform()                           # Apply CLR transformation
])

X_transformed = pipeline(X)
```

### Example 5: Create PyTorch DataLoaders

```python
from mditre.data_loader import create_data_loader, create_stratified_loaders

# Single data loader
train_loader = create_data_loader(
    X, y, mask,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Stratified train/val split
train_loader, val_loader = create_stratified_loaders(
    X, y, mask,
    train_ratio=0.8,
    batch_size=32,
    random_seed=42
)

# Use in training loop
for batch in train_loader:
    data = batch['data']    # (batch_size, n_otus, n_timepoints)
    labels = batch['label']  # (batch_size,)
    masks = batch['mask']    # (batch_size, n_timepoints)
    # ... train model ...
```

### Example 6: K-Fold Cross-Validation

```python
from mditre.data_loader import create_kfold_loaders

for fold, train_loader, val_loader in create_kfold_loaders(
    X, y, mask,
    n_splits=5,
    batch_size=32
):
    print(f"Fold {fold}:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    # ... train and evaluate ...
```

### Example 7: Complete Workflow

```python
from mditre.data_loader import (
    DataLoaderRegistry,
    TransformPipeline,
    NormalizeTransform,
    FilterLowAbundance,
    create_stratified_loaders,
    compute_phylo_distance_matrix,
    get_otu_embeddings
)

# Step 1: Load data
loader = DataLoaderRegistry.create_loader('pickle', 'data.pkl')
data = loader.load_and_preprocess()

# Step 2: Preprocess
pipeline = TransformPipeline([
    NormalizeTransform(),
    FilterLowAbundance(min_abundance=0.001, min_prevalence=0.1)
])
X_processed = pipeline(data['X'])

# Step 3: Compute phylogenetic features
dist_matrix = compute_phylo_distance_matrix(data['phylo_tree'])
otu_embeddings = get_otu_embeddings(data['phylo_tree'], method='mds', emb_dim=5)

# Step 4: Create data loaders
train_loader, val_loader = create_stratified_loaders(
    X_processed, data['y'], data['mask'],
    train_ratio=0.8,
    batch_size=32
)

# Step 5: Train MDITRE model
# ... (see MDITRE training examples) ...
```

## Adding New Data Loaders

To add support for a new data format:

### Step 1: Create Loader Class

```python
# mditre/data_loader/loaders/my_format_loader.py
from ..base_loader import BaseDataLoader, DataLoaderRegistry

@DataLoaderRegistry.register('my_format')
class MyFormatLoader(BaseDataLoader):
    """Load data from MyFormat"""
    
    def __init__(self, data_path, config=None):
        super().__init__(data_path, config)
        # Initialize format-specific parameters
        
    def load(self):
        """Load raw data"""
        # Implement format-specific loading
        return raw_data
    
    def preprocess(self, raw_data):
        """Convert to MDITRE format"""
        # Convert to standard format:
        # X: (n_subjects, n_otus, n_timepoints)
        # y: (n_subjects,)
        # times: time indices
        # phylo_tree: ete3.Tree
        # metadata: dict
        
        return {
            'X': X,
            'y': y,
            'times': times,
            'mask': mask,
            'phylo_tree': tree,
            'metadata': metadata
        }
```

### Step 2: Register and Use

```python
# Import to register
from mditre.data_loader.loaders.my_format_loader import MyFormatLoader

# Use via registry
loader = DataLoaderRegistry.create_loader(
    'my_format',
    'path/to/data',
    config={'param': 'value'}
)
data = loader.load_and_preprocess()
```

## Data Format Specification

All loaders must produce data in this standard format:

### Required Fields

**`X`**: Abundance matrix
- Shape: `(n_subjects, n_otus, n_timepoints)`
- Type: `np.ndarray` (float32)
- Values: Non-negative abundances (can be absolute or relative)

**`y`**: Binary outcome labels
- Shape: `(n_subjects,)`
- Type: `np.ndarray` (int64)
- Values: 0 or 1

**`times`**: Time indices per subject
- Shape: `(n_subjects,)` or `(n_subjects, n_samples)`
- Type: `np.ndarray` (int) or list of arrays
- Values: Integer time indices

**`phylo_tree`**: Phylogenetic tree
- Type: `ete3.Tree`
- Leaves: Must correspond to OTUs in X

**`metadata`**: Additional information
- Type: `dict`
- Required keys:
  - `n_subjects`: Number of subjects
  - `n_otus`: Number of OTUs
  - `n_timepoints`: Number of time points
  - `variable_names`: List of OTU names

### Optional Fields

**`mask`**: Temporal mask indicating sample availability
- Shape: `(n_subjects, n_timepoints)`
- Type: `np.ndarray` (float32)
- Values: 1.0 where samples exist, 0.0 otherwise

## Best Practices

1. **Always normalize data**: Use `NormalizeTransform` to convert to relative abundances
2. **Filter rare OTUs**: Use `FilterLowAbundance` to remove noise
3. **Use stratified splits**: Maintain class balance with `create_stratified_loaders`
4. **Set random seeds**: Ensure reproducibility
5. **Validate data**: Check shapes and ranges after preprocessing
6. **Document metadata**: Store preprocessing steps in metadata dict

## Integration with MDITRE

The data loader integrates seamlessly with MDITRE training:

```python
from mditre.data_loader import DataLoaderRegistry, create_stratified_loaders
from mditre.models import MDITRE
from mditre.trainer import Trainer

# Load and preprocess data
loader = DataLoaderRegistry.create_loader('pickle', 'data.pkl')
data = loader.load_and_preprocess()

# Create data loaders
train_loader, val_loader = create_stratified_loaders(
    data['X'], data['y'], data['mask'],
    train_ratio=0.8,
    batch_size=32
)

# Initialize MDITRE model
model = MDITRE(
    num_rules=5,
    num_otus=data['metadata']['n_otus'],
    num_time=data['metadata']['n_timepoints'],
    ...
)

# Train using MDITRE trainer
# (See trainer documentation for details)
```

## Supported Data Formats

| Format | Loader Name | Description |
|--------|-------------|-------------|
| Pickle | `pickle` | MDITRE native format |
| Pickle Trajectory | `pickle_trajectory` | Variable-length trajectories |
| DADA2 | `16s_dada2` | DADA2 amplicon pipeline output |
| QIIME2 | `16s_qiime2` | QIIME2 artifacts |

## Future Extensions

Planned support for additional formats:
- Metaphlan taxonomic profiles
- WGS (Whole Genome Sequencing) data
- Metagenomic functional profiles (HUMAnN)
- Multi-omics data integration

## Troubleshooting

### Problem: Shape mismatch errors
**Solution**: Check that X, y, times, and mask have consistent dimensions

### Problem: Missing phylogenetic tree
**Solution**: Provide tree file or set `create_default_tree=True` in config

### Problem: Memory issues with large datasets
**Solution**: Use batch loading or increase `num_workers` parameter

### Problem: Inconsistent time indices
**Solution**: Ensure time values are integers and start from 0

## References

- MDITRE Paper: Maringanti et al. (2022), mSystems 7:e00132-22
- Repository: https://github.com/melhzy/mditre
- Documentation: See MODULAR_ARCHITECTURE.md for layer details

## Support

For questions or issues:
- Check examples in `mditre/examples/data_loader_example.py`
- Review loader implementations in `mditre/data_loader/loaders/`
- Open an issue on GitHub
