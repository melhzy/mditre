# MDITRE Data Loader Module

A modular, extensible data loading system for microbiome time-series data supporting multiple sequencing formats and preprocessing pipelines.

## Quick Start

```python
from mditre.data_loader import (
    DataLoaderRegistry,
    create_data_loader,
    NormalizeTransform,
    FilterLowAbundance,
    TransformPipeline
)

# List available loaders
loaders = DataLoaderRegistry.list_loaders()
print(f"Available loaders: {list(loaders.keys())}")
# Output: ['pickle', 'pickle_trajectory', '16s_dada2', '16s_qiime2']

# Load data using a registered loader
loader = DataLoaderRegistry.create_loader('16s_dada2')
data = loader.load('path/to/data', metadata_path='path/to/metadata.csv')

# Apply transformations
pipeline = TransformPipeline([
    NormalizeTransform(),
    FilterLowAbundance(min_abundance=0.001, min_prevalence=0.1)
])
data['X'] = pipeline(data['X'])

# Create PyTorch DataLoader
train_loader = create_data_loader(
    data['X'], 
    data['y'], 
    data['mask'],
    batch_size=16,
    shuffle=True
)
```

## Architecture

### Design Pattern
**Registry Pattern** with decorator-based registration enables dynamic loader selection and easy extensibility.

```
BaseDataLoader (Abstract)
    ├── load() - Load raw data
    ├── preprocess() - Apply transformations  
    └── validate() - Check data integrity

DataLoaderRegistry
    ├── @register() - Decorator for registration
    ├── list_loaders() - Show available loaders
    └── create_loader() - Instantiate loader
```

### Components

#### 1. Base Infrastructure (`base_loader.py`)
- `BaseDataLoader`: Abstract base class enforcing consistent interface
- `DataLoaderRegistry`: Manages loader registration and instantiation
- Phylogenetic utilities: distance matrices and embeddings

#### 2. Data Transformations (`transforms.py`)
Composable preprocessing operations:
- `NormalizeTransform`: Sum-to-one normalization
- `LogTransform`: Log transformation with pseudocount
- `CLRTransform`: Centered log-ratio (compositional data)
- `FilterLowAbundance`: Remove rare OTUs
- `ZScoreTransform`: Standardize to zero mean, unit variance
- `RobustScaleTransform`: Median-based robust scaling
- `TransformPipeline`: Chain multiple transforms

#### 3. PyTorch Integration (`datasets.py`)
- `TrajectoryDataset`: Basic dataset with X/y/mask
- `TrajectoryDatasetWithMetadata`: Extended with temporal/subject info
- `create_data_loader()`: Create PyTorch DataLoader
- `create_stratified_loaders()`: Stratified train/val/test splits
- `create_kfold_loaders()`: K-fold cross-validation

#### 4. Format-Specific Loaders

**Pickle Loaders** (`loaders/pickle_loader.py`)
- `PickleDataLoader`: MDITRE native pickle format
- `PickleTrajectoryLoader`: Variable-length trajectories

**Amplicon Loaders** (`loaders/amplicon_loader.py`)
- `DADA2Loader`: DADA2 pipeline output
- `QIIME2Loader`: QIIME2 artifacts

## Supported Data Formats

### 1. Pickle Format (Native MDITRE)
```python
loader = DataLoaderRegistry.create_loader('pickle')
data = loader.load('data.pkl')
```

**Expected structure:**
```python
{
    'X': np.ndarray,      # (n_subjects, n_otus, n_timepoints)
    'y': np.ndarray,      # (n_subjects,)
    'mask': np.ndarray,   # (n_subjects, n_timepoints)
    'phylo_tree': Tree,   # ete3 Tree object
    'metadata': dict      # Optional metadata
}
```

### 2. DADA2 Output
```python
loader = DataLoaderRegistry.create_loader('16s_dada2')
data = loader.load(
    data_path='abundance.csv',
    metadata_path='sample_metadata.csv',
    tree_path='placement.jplace',
    subject_col='SubjectID',
    time_col='CollectionDay',
    label_col='Disease'
)
```

**Required files:**
- `abundance.csv`: OTU abundance matrix (samples × OTUs)
- `sample_metadata.csv`: Sample information with subject, time, label
- `placement.jplace`: Phylogenetic placement (optional)

### 3. QIIME2 Artifacts
```python
loader = DataLoaderRegistry.create_loader('16s_qiime2')
data = loader.load(
    feature_table_path='table.qza',
    metadata_path='metadata.tsv',
    tree_path='tree.qza',
    subject_col='subject_id',
    time_col='days_from_baseline',
    label_col='diagnosis'
)
```

**Required files:**
- `table.qza`: Feature table artifact
- `metadata.tsv`: Metadata with subject IDs, timepoints, labels
- `tree.qza`: Phylogenetic tree artifact (optional)

## Data Transformations

### Individual Transforms

```python
from mditre.data_loader import (
    NormalizeTransform,
    LogTransform,
    CLRTransform,
    FilterLowAbundance,
    ZScoreTransform
)

# Normalize to sum=1 per sample
normalize = NormalizeTransform()
X_norm = normalize(X)

# Log transform with pseudocount
log_transform = LogTransform(pseudocount=0.5)
X_log = log_transform(X)

# Centered log-ratio (for compositional data)
clr = CLRTransform(pseudocount=0.5)
X_clr = clr(X)

# Filter rare OTUs
filter_transform = FilterLowAbundance(
    min_abundance=0.001,  # 0.1% minimum abundance
    min_prevalence=0.1    # Present in 10% of samples
)
X_filtered = filter_transform(X)

# Z-score normalization
zscore = ZScoreTransform()
X_zscore = zscore(X)
```

### Transform Pipeline

```python
from mditre.data_loader import TransformPipeline

# Chain multiple transforms
pipeline = TransformPipeline([
    NormalizeTransform(),
    FilterLowAbundance(min_abundance=0.001, min_prevalence=0.1),
    CLRTransform(pseudocount=0.5)
])

X_transformed = pipeline(X)
```

## PyTorch Integration

### Basic DataLoader

```python
from mditre.data_loader import create_data_loader

# Create PyTorch DataLoader
train_loader = create_data_loader(
    X=X_train,
    y=y_train,
    mask=mask_train,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# Iterate through batches
for batch in train_loader:
    data = batch['data']      # torch.Tensor: (batch_size, n_otus, n_timepoints)
    labels = batch['label']   # torch.Tensor: (batch_size,)
    masks = batch['mask']     # torch.Tensor: (batch_size, n_timepoints)
```

### Stratified Splits

```python
from mditre.data_loader import create_stratified_loaders

# Create stratified train/val/test splits
loaders = create_stratified_loaders(
    X=X,
    y=y,
    mask=mask,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    batch_size=16,
    random_state=42
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

### K-Fold Cross-Validation

```python
from mditre.data_loader import create_kfold_loaders

# Create k-fold loaders
kfold_loaders = create_kfold_loaders(
    X=X,
    y=y,
    mask=mask,
    n_splits=5,
    batch_size=16,
    random_state=42
)

# Train with cross-validation
for fold_idx, train_loader, val_loader in kfold_loaders:
    print(f"Training fold {fold_idx}")
    # Train model
    # Validate model
```

## Phylogenetic Processing

### Distance Matrix

```python
from mditre.data_loader import compute_phylo_distance_matrix
from ete3 import Tree

tree = Tree("((A:1,B:1):0.5,(C:0.8,D:0.8):0.7);")
dist_matrix = compute_phylo_distance_matrix(tree)

# Output: (n_otus, n_otus) distance matrix
print(dist_matrix.shape)  # (4, 4)
```

### OTU Embeddings

```python
from mditre.data_loader import get_otu_embeddings

# Method 1: Distance-based (MDS on phylogenetic distances)
embeddings = get_otu_embeddings(tree, method='distance', emb_dim=10)

# Method 2: MDS on distance matrix
dist_matrix = compute_phylo_distance_matrix(tree)
embeddings = get_otu_embeddings(tree, method='mds', emb_dim=10, 
                                dist_matrix=dist_matrix)

# Method 3: Random embeddings (baseline)
embeddings = get_otu_embeddings(tree, method='random', emb_dim=10, seed=42)

# Output: (n_otus, emb_dim) embeddings
print(embeddings.shape)  # (4, 10)
```

## Adding New Data Loaders

### Step 1: Create Loader Class

```python
from mditre.data_loader import BaseDataLoader, DataLoaderRegistry
import numpy as np

@DataLoaderRegistry.register('my_format')
class MyFormatLoader(BaseDataLoader):
    """Load data from my custom format"""
    
    def load(self, data_path, **kwargs):
        """
        Load data from file
        
        Returns:
            dict with keys: 'X', 'y', 'mask', 'phylo_tree', 'metadata'
        """
        # Load your data
        X = np.load(data_path)  # (n_subjects, n_otus, n_timepoints)
        
        # Extract labels and metadata
        y = kwargs.get('labels')
        metadata = kwargs.get('metadata')
        
        # Create temporal mask
        mask = np.ones((X.shape[0], X.shape[2]), dtype=np.float32)
        
        # Load or create phylogenetic tree
        phylo_tree = self._create_default_tree(X.shape[1])
        
        return {
            'X': X,
            'y': y,
            'mask': mask,
            'phylo_tree': phylo_tree,
            'metadata': metadata
        }
    
    def preprocess(self, data, **kwargs):
        """Apply preprocessing transformations"""
        # Apply transforms if provided
        if 'transforms' in kwargs:
            data['X'] = kwargs['transforms'](data['X'])
        return data
    
    def validate(self, data):
        """Validate data structure and content"""
        required_keys = ['X', 'y', 'mask', 'phylo_tree']
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"
        
        assert data['X'].ndim == 3, "X must be 3D array"
        assert data['y'].ndim == 1, "y must be 1D array"
        assert data['mask'].ndim == 2, "mask must be 2D array"
```

### Step 2: Use New Loader

```python
# Loader is automatically registered via decorator
loader = DataLoaderRegistry.create_loader('my_format')
data = loader.load('path/to/data.npy', labels=y, metadata=metadata)
```

## Complete Example Workflow

```python
from mditre.data_loader import (
    DataLoaderRegistry,
    TransformPipeline,
    NormalizeTransform,
    FilterLowAbundance,
    CLRTransform,
    create_stratified_loaders,
    get_otu_embeddings
)

# 1. Load data
loader = DataLoaderRegistry.create_loader('16s_dada2')
data = loader.load(
    data_path='abundance.csv',
    metadata_path='sample_metadata.csv',
    tree_path='placement.jplace'
)

# 2. Create preprocessing pipeline
pipeline = TransformPipeline([
    NormalizeTransform(),
    FilterLowAbundance(min_abundance=0.001, min_prevalence=0.1),
    CLRTransform(pseudocount=0.5)
])

# 3. Preprocess data
data['X'] = pipeline(data['X'])

# 4. Create stratified train/val/test splits
loaders = create_stratified_loaders(
    X=data['X'],
    y=data['y'],
    mask=data['mask'],
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    batch_size=16,
    random_state=42
)

# 5. Get OTU embeddings for model
otu_embeddings = get_otu_embeddings(
    data['phylo_tree'], 
    method='distance', 
    emb_dim=10
)

# 6. Train model (pseudo-code)
from mditre.models import MDITRE

model = MDITRE(
    num_rules=10,
    num_otus=data['X'].shape[1],
    num_otu_centers=20,
    num_time=data['X'].shape[2],
    num_time_centers=5,
    dist=otu_embeddings,
    emb_dim=10
)

for epoch in range(num_epochs):
    for batch in loaders['train']:
        # Training step
        outputs = model(batch['data'], mask=batch['mask'])
        loss = criterion(outputs, batch['label'])
        # Backprop
```

## Testing

Run validation script to test all components:

```bash
python validate_package.py
```

Expected output:
```
================================================================================
MDITRE Package Integrity Validation
================================================================================

Testing Core Module                                                    [OK]
Testing Layers Module                                                  [OK]
Testing Data Loader Module                                             [OK]
Testing Models Module                                                  [OK]
Testing Complete Integration                                           [OK]
Testing Backward Compatibility                                         [OK]

================================================================================
ALL TESTS PASSED [OK]
================================================================================
```

## Documentation

- **Full Guide**: See `DATA_LOADER_GUIDE.md` for comprehensive documentation
- **Examples**: See `examples/data_loader_example.py` for working examples
- **Integrity Report**: See `PACKAGE_INTEGRITY_REPORT.md` for validation results

## API Reference

### Core Classes

**BaseDataLoader**
```python
class BaseDataLoader:
    def load(self, data_path, **kwargs) -> Dict
    def preprocess(self, data, **kwargs) -> Dict
    def validate(self, data) -> None
```

**DataLoaderRegistry**
```python
class DataLoaderRegistry:
    @staticmethod
    def register(name: str) -> Callable
    @staticmethod
    def list_loaders() -> Dict[str, List[str]]
    @staticmethod
    def get_loader(name: str) -> Type[BaseDataLoader]
    @staticmethod
    def create_loader(name: str, **kwargs) -> BaseDataLoader
```

**DataTransform**
```python
class DataTransform:
    def __call__(self, X: np.ndarray) -> np.ndarray
    def fit(self, X: np.ndarray) -> 'DataTransform'
    def transform(self, X: np.ndarray) -> np.ndarray
```

### Utility Functions

**Phylogenetic Processing**
```python
def compute_phylo_distance_matrix(tree: Tree) -> np.ndarray
def get_otu_embeddings(tree: Tree, method: str, emb_dim: int, **kwargs) -> np.ndarray
```

**PyTorch Integration**
```python
def create_data_loader(X, y, mask=None, batch_size=32, shuffle=False, **kwargs) -> DataLoader
def create_stratified_loaders(X, y, mask=None, train_size=0.7, val_size=0.15, 
                             test_size=0.15, batch_size=32, **kwargs) -> Dict[str, DataLoader]
def create_kfold_loaders(X, y, mask=None, n_splits=5, batch_size=32, 
                        **kwargs) -> Generator[Tuple[int, DataLoader, DataLoader]]
```

## License

See main MDITRE package license.

## Citation

If you use MDITRE data loader module, please cite the MDITRE paper:

```
[Citation to be added]
```

## Support

For questions or issues:
- GitHub Issues: https://github.com/gerberlab/mditre/issues
- Documentation: See `DATA_LOADER_GUIDE.md`
- Examples: See `examples/data_loader_example.py`
