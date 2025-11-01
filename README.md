# MDITRE: Microbiome Dynamics using Interpretable Temporal Rules

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](CHANGELOG.md)

**MDITRE** (Microbiome DynamIc Time-series Rule Extraction) is a scalable and interpretable machine learning framework for predicting host status from temporal microbiome dynamics. The model learns human-readable rules that combine phylogenetic relationships and temporal patterns in longitudinal microbiome data.

## ✨ Key Features

- 🔬 **Interpretable Rules**: Learn human-readable IF-THEN rules from microbiome time-series
- 🌳 **Phylogenetic Integration**: Leverage evolutionary relationships between microbes
- ⏱️ **Temporal Dynamics**: Discover critical time windows and rate-of-change patterns
- 🔧 **Modular Architecture**: Extensible 5-layer design for easy customization
- 📊 **Multiple Data Formats**: Support for 16S rRNA, shotgun metagenomics (DADA2, QIIME2, Metaphlan)
- 🎨 **Visualization GUI**: Interactive exploration of learned rules and patterns
- � **Reproducibility**: Deterministic seeding system for consistent results
- �🚀 **Production Ready**: v1.0.0 with comprehensive test coverage and modern infrastructure

## 📚 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Data Loading](#data-loading)
- [Training Models](#training-models)
- [Interpreting Results](#interpreting-results)
- [Development](#development)
- [Tutorials](#tutorials)
- [Advanced Usage](#advanced-usage)
- [Citation](#citation)
- [Contributing](#contributing)

---

## 🔧 Installation

### Requirements

- Python 3.8+ (tested with 3.8-3.12)
- PyTorch 2.0+ (tested with 2.5.1)
- CUDA 11.0+ for GPU support (optional but recommended)

### Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/melhzy/mditre.git
cd mditre

# Install with pip (includes all dependencies)
pip install -e .

# For development (includes testing, formatting, type checking tools)
pip install -r requirements-dev.txt

# Or use Makefile:
make install-dev
```

### Platform-Specific Instructions

#### Ubuntu 24.04 / Linux

**With GPU Support (CUDA 12.x)**
```bash
# Create virtual environment
python3 -m venv mditre_env
source mditre_env/bin/activate

# Install MDITRE with pinned dependencies
pip install -r requirements.txt
pip install -e .

# Or install PyTorch separately then MDITRE
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

**CPU Only**
```bash
python3 -m venv mditre_env
source mditre_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

#### Windows 11

**With GPU Support**
```powershell
# Using conda (recommended for Windows)
conda create -n mditre python=3.12 -y
conda activate mditre

# Install MDITRE
pip install -r requirements.txt
pip install -e .

# Or with CUDA-specific PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

#### macOS (Apple Silicon / Intel)

```bash
python3 -m venv mditre_env
source mditre_env/bin/activate

# PyTorch optimized for Apple Silicon
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```python
python -c "import mditre; import torch; print(f'MDITRE installed. PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### Basic Workflow

```python
from mditre.data_loader import DataLoaderRegistry, TransformPipeline
from mditre.data_loader import NormalizeTransform, FilterLowAbundance, get_otu_embeddings
from mditre.models import MDITRE
import torch

# 1. Load data (DADA2 format)
loader = DataLoaderRegistry.create_loader('16s_dada2')
data = loader.load(
    data_path='abundance.csv',
    metadata_path='metadata.csv',
    tree_path='phylogenetic_tree.jplace',
    subject_col='SubjectID',
    time_col='CollectionDay',
    label_col='Disease'
)

# 2. Preprocess data
pipeline = TransformPipeline([
    NormalizeTransform(),
    FilterLowAbundance(min_abundance=0.001, min_prevalence=0.1)
])
data['X'] = pipeline(data['X'])

# 3. Get phylogenetic embeddings
otu_embeddings = get_otu_embeddings(data['phylo_tree'], method='distance', emb_dim=10)

# 4. Create MDITRE model
model = MDITRE(
    num_rules=5,
    num_otus=data['X'].shape[1],
    num_otu_centers=10,
    num_time=data['X'].shape[2],
    num_time_centers=5,
    dist=otu_embeddings,
    emb_dim=10
)

# 5. Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ... training loop ...

# 6. Visualize rules
from mditre.visualize import visualize_rules
visualize_rules(model, data, output_dir='./results/')
```

### What MDITRE Learns

```
Rule 1: Predicts "Healthy" (weight: 2.3)
  ✓ Clostridiales group abundance > 7% during days 120-180
  AND
  ✓ Bacteroides acidifaciens increasing during days 90-150

Rule 2: Predicts "Disease" (weight: -1.8)
  ✓ Bacteroidetes abundance < 3% during days 30-90
  AND
  ✓ Proteobacteria decreasing during days 60-120
```

---

## 🏗️ Architecture

MDITRE uses a modular 5-layer architecture that mirrors biological interpretation:

```
┌────────────────────────────────────────────────────────┐
│  Layer 1: Phylogenetic Focus                           │
│  Aggregate microbes by evolutionary relationships      │
│  → SpatialAgg, SpatialAggDynamic                       │
└──────────────┬─────────────────────────────────────────┘
               ↓
┌────────────────────────────────────────────────────────┐
│  Layer 2: Temporal Focus                               │
│  Select important time windows                         │
│  → TimeAgg, TimeAggAbun                                │
└──────────────┬─────────────────────────────────────────┘
               ↓
┌────────────────────────────────────────────────────────┐
│  Layer 3: Detector                                     │
│  Apply thresholds to detect patterns                   │
│  → Threshold, Slope                                    │
└──────────────┬─────────────────────────────────────────┘
               ↓
┌────────────────────────────────────────────────────────┐
│  Layer 4: Rule                                         │
│  Combine detectors via logical AND                     │
│  → Rules                                               │
└──────────────┬─────────────────────────────────────────┘
               ↓
┌────────────────────────────────────────────────────────┐
│  Layer 5: Classification                               │
│  Weighted rule combination for prediction              │
│  → DenseLayer, DenseLayerAbun                          │
└────────────────────────────────────────────────────────┘
```

### Package Structure

```
mditre/
├── core/                      # Foundation
│   ├── base_layer.py         # Abstract base class
│   ├── registry.py           # Dynamic layer registration
│   └── math_utils.py         # Mathematical utilities
│
├── layers/                    # Five-layer architecture
│   ├── phylogenetic_focus.py # Layer 1
│   ├── temporal_focus.py     # Layer 2
│   ├── detector.py           # Layer 3
│   ├── rule.py               # Layer 4
│   └── classification.py     # Layer 5
│
├── data_loader/              # Modular data loading
│   ├── base_loader.py        # Abstract loader + registry
│   ├── transforms.py         # Preprocessing (7 transforms)
│   ├── datasets.py           # PyTorch integration
│   └── loaders/
│       ├── pickle_loader.py      # Native format
│       └── amplicon_loader.py    # 16S (DADA2, QIIME2)
│
├── models.py                 # MDITRE models
├── trainer.py                # Training infrastructure
├── visualize.py              # Visualization tools
└── examples/                 # Working examples
```

**Design Benefits:**
- ✅ **Extensible**: Add new layers via registry pattern
- ✅ **Interpretable**: Each layer has biological meaning
- ✅ **Flexible**: Mix and match implementations
- ✅ **Maintainable**: Clean separation of concerns

---

## 📊 Data Loading

### Supported Formats

| Format | Loader | Description |
|--------|--------|-------------|
| **DADA2** | `16s_dada2` | 16S amplicon sequencing (abundance + tree) |
| **QIIME2** | `16s_qiime2` | QIIME2 artifacts (table.qza + tree.qza) |
| **Pickle** | `pickle` | Native MDITRE format |
| **Trajectory** | `pickle_trajectory` | Variable-length time series |

### Loading Examples

#### DADA2 Output

```python
from mditre.data_loader import DataLoaderRegistry

loader = DataLoaderRegistry.create_loader('16s_dada2')
data = loader.load(
    data_path='abundance.csv',
    metadata_path='sample_metadata.csv',
    tree_path='placement.jplace',
    subject_col='SubjectID',
    time_col='CollectionDay',
    label_col='Disease'
)

print(f"Shape: {data['X'].shape}")  # (subjects, OTUs, timepoints)
```

#### QIIME2 Artifacts

```python
loader = DataLoaderRegistry.create_loader('16s_qiime2')
data = loader.load(
    feature_table_path='table.qza',
    metadata_path='metadata.tsv',
    tree_path='tree.qza',
    subject_col='subject_id',
    time_col='days',
    label_col='diagnosis'
)
```

#### Native Pickle

```python
loader = DataLoaderRegistry.create_loader('pickle')
data = loader.load('preprocessed_data.pkl')
```

### Data Preprocessing

Composable transformation pipeline:

```python
from mditre.data_loader import (
    TransformPipeline,
    NormalizeTransform,      # Sum to 1 per sample
    FilterLowAbundance,      # Remove rare OTUs
    CLRTransform,            # Centered log-ratio
    LogTransform,            # Log with pseudocount
    ZScoreTransform          # Z-score standardization
)

pipeline = TransformPipeline([
    NormalizeTransform(),
    FilterLowAbundance(min_abundance=0.001, min_prevalence=0.1),
    CLRTransform(pseudocount=0.5)
])

X_transformed = pipeline(data['X'])
```

### PyTorch DataLoaders

Create train/val/test splits:

```python
from mditre.data_loader import create_stratified_loaders

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

for batch in loaders['train']:
    data_batch = batch['data']      # (batch, OTUs, time)
    labels_batch = batch['label']   # (batch,)
    mask_batch = batch['mask']      # (batch, time)
```

### Adding Custom Loaders

```python
from mditre.data_loader import BaseDataLoader, DataLoaderRegistry

@DataLoaderRegistry.register('my_format')
class MyLoader(BaseDataLoader):
    def load(self, data_path, **kwargs):
        X = load_my_data(data_path)  # Your loading logic
        return {
            'X': X,    # (subjects, OTUs, timepoints)
            'y': y,    # (subjects,)
            'mask': mask,
            'phylo_tree': tree,
            'metadata': metadata
        }
```

---

## 🎯 Training Models

### Configuration File

Create `config.cfg`:

```ini
[Paths]
data_dir = /path/to/data
output_dir = /path/to/output

[Data]
abundance_file = abundance.csv
metadata_file = metadata.csv
tree_file = tree.nwk

[Model]
num_rules = 5
num_otu_centers = 10
num_time_centers = 5
embedding_dim = 10

[Training]
epochs = 2000
batch_size = 32
learning_rate = 0.001
num_folds = 5
```

Run training:
```bash
python mditre/tutorials/model_run_tutorial.py --data config.cfg
```

### Python API

```python
from mditre.models import MDITRE
import torch

model = MDITRE(
    num_rules=5,
    num_otus=100,
    num_otu_centers=10,
    num_time=15,
    num_time_centers=5,
    dist=otu_embeddings,
    emb_dim=10
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['data'].to(device), mask=batch['mask'].to(device))
        loss = criterion(outputs, batch['label'].float().to(device))
        loss.backward()
        optimizer.step()
```

### Hyperparameter Guidelines

| Dataset Size | `num_rules` | `num_otu_centers` | `num_time_centers` | `epochs` | `batch_size` |
|--------------|-------------|-------------------|-------------------|----------|--------------|
| **Small** (<50) | 3 | 5 | 3 | 2000 | 16 |
| **Medium** (50-200) | 5 | 10 | 5 | 3000 | 32 |
| **Large** (>200) | 7-10 | 15-20 | 7-10 | 5000 | 64 |

---

## 🔍 Interpreting Results

### Rule Structure

Each MDITRE rule contains:

1. **Phylogenetic Group**: Which microbes (e.g., Clostridiales)
2. **Time Window**: When to look (e.g., days 120-180)
3. **Detector Type**: Abundance or slope threshold
4. **Threshold**: Significance cutoff (e.g., > 7%)
5. **Weight**: Contribution to prediction

### Visualization GUI

```python
from mditre.visualize import visualize_rules

visualize_rules(
    model=trained_model,
    data=data,
    output_dir='./results/'
)
```

**Features:**
- View all learned rules
- Explore which subjects activate each rule
- Examine phylogenetic groupings
- Analyze temporal patterns
- Export rule summaries

### Example Rule Interpretation

```
Rule 1: Predicts "Preterm Birth Risk" (weight: 2.1)

Detector 1: Phylogenetic Focus
  - Microbes: Lactobacillus spp. (OTUs 12, 15, 23, 31)
  - Time: Weeks 20-28 of pregnancy
  - Type: Abundance threshold
  - Condition: < 15%
  - Meaning: Low Lactobacillus in mid-pregnancy

Detector 2: Phylogenetic Focus  
  - Microbes: Gardnerella vaginalis cluster
  - Time: Weeks 24-32
  - Type: Slope (rate of change)
  - Condition: Increasing (positive slope)
  - Meaning: Rising Gardnerella levels

Combined Rule:
  IF (Lactobacillus < 15% at weeks 20-28) AND 
     (Gardnerella increasing at weeks 24-32)
  THEN: Predict High Preterm Birth Risk
```

---

## 📖 Tutorials

### Tutorial 1: 16S rRNA Analysis
**[Tutorial_Bokulich_16S_data.ipynb](jupyter/tutorials/Tutorial_Bokulich_16S_data.ipynb)**

- Load 16S amplicon data (DADA2)
- Build phylogenetic trees
- Train model to predict infant diet
- Interpret rules with GUI

**Dataset:** Bokulich et al. infant gut microbiome (2016)

### Tutorial 2: Shotgun Metagenomics
**[Tutorial_2_metaphlan_data.ipynb](jupyter/tutorials/Tutorial_2_metaphlan_data.ipynb)**

- Work with MetaPhlAn profiles
- Handle shotgun metagenomic data
- Process taxonomic hierarchies

### Tutorial 3: Complete Workflow
**[Tutorial_1_16s_data.ipynb](jupyter/tutorials/Tutorial_1_16s_data.ipynb)**

- DADA2 sequence processing
- Phylogenetic placement (pplacer)
- Data QC and preprocessing
- Hyperparameter tuning
- Rule visualization

---

## 🔬 Advanced Usage

### K-Fold Cross-Validation

```python
from mditre.data_loader import create_kfold_loaders

kfold_loaders = create_kfold_loaders(
    X=data['X'], y=data['y'], mask=data['mask'],
    n_splits=5, batch_size=16, random_state=42
)

results = []
for fold_idx, train_loader, val_loader in kfold_loaders:
    model = MDITRE(...)  # Fresh model per fold
    # Train and evaluate
    fold_result = train_and_evaluate(model, train_loader, val_loader)
    results.append(fold_result)

print(f"Average AUC: {np.mean([r['auc'] for r in results]):.3f}")
```

### Phylogenetic Embeddings

```python
from mditre.data_loader import get_otu_embeddings, compute_phylo_distance_matrix
from ete3 import Tree

tree = Tree("tree.nwk")

# Method 1: Distance-based (recommended)
embeddings = get_otu_embeddings(tree, method='distance', emb_dim=10)

# Method 2: MDS on distance matrix
dist_matrix = compute_phylo_distance_matrix(tree)
embeddings = get_otu_embeddings(tree, method='mds', emb_dim=10, 
                                dist_matrix=dist_matrix)

# Method 3: Random baseline
embeddings = get_otu_embeddings(tree, method='random', emb_dim=10, seed=42)
```

### Custom Layer Implementation

```python
from mditre.core import BaseLayer, LayerRegistry

@LayerRegistry.register('phylogenetic_focus', 'custom')
class CustomSpatialAgg(BaseLayer):
    def __init__(self, num_rules, emb_dim, num_otus):
        super().__init__()
        # Your initialization
    
    def forward(self, x):
        # Your forward logic
        return x_aggregated
    
    def init_params(self, init_args):
        # Parameter initialization
        pass
```

---

## 🎯 Use Cases

### 1. Disease Onset Prediction
```python
# Example: Type 1 diabetes risk from infant gut microbiome
# - Monthly samples over 3 years
# - Identifies: Pre-disease microbial changes, critical time windows
```

### 2. Dietary Intervention Classification
```python
# Example: Distinguish diet types from microbiome
# - Samples before/during/after diet change
# - Reveals: Responsive microbes, adaptation speed
```

### 3. Clinical Outcome Prediction
```python
# Example: Preterm birth from vaginal microbiome
# - Weekly pregnancy samples
# - Discovers: Early warning signals, dysbiotic patterns
```

---

## 🐛 Troubleshooting

### Common Issues

**CUDA out of memory:**
```bash
python script.py --batch-size 16 --device cpu
```

**Slow training:**
- Use GPU acceleration
- Reduce complexity (`num_rules`, `num_otu_centers`)
- Increase batch size if memory allows

**Poor performance:**
- Increase `epochs`
- Adjust learning rate (try 0.01, 0.001, 0.0001)
- Check data quality and class balance

### Package Validation

```bash
python validate_package.py
```

Expected:
```
================================================================================
ALL TESTS PASSED [OK]
================================================================================

[PASS] Core module
[PASS] Layers module
[PASS] Data loader module
[PASS] Models module
[PASS] Complete integration
[PASS] Backward compatibility
```

---

## 📈 Performance

| Dataset | Subjects | Timepoints | OTUs | Training Time* | Memory |
|---------|----------|------------|------|----------------|--------|
| Small | <50 | 5-15 | 50-200 | 5-10 min | 2 GB |
| Medium | 50-200 | 10-20 | 100-500 | 20-40 min | 4 GB |
| Large | >200 | 15-30 | 200-1000 | 1-3 hours | 8 GB |

*NVIDIA RTX 3090 GPU, ~3000 epochs

**Optimization:**
- Use GPU: 10-50x speedup
- Larger batch sizes (if memory allows)
- Early stopping on validation loss
- Mixed precision training (`torch.cuda.amp`)

---

## 📝 Data Format

### Required Structure

1. **Abundance Table**: (samples × OTUs) with counts/relative abundances
2. **Metadata**: Subject IDs, timepoints, binary labels
3. **Phylogenetic Tree**: Newick format or distance matrix

### Example

```python
import pandas as pd

# Abundance
abundance_df = pd.DataFrame({
    'OTU1': [0.10, 0.15, 0.08, ...],
    'OTU2': [0.05, 0.03, 0.12, ...],
})

# Metadata
metadata_df = pd.DataFrame({
    'subject_id': ['S1', 'S1', 'S2', ...],
    'timepoint': [0, 30, 0, ...],
    'label': [0, 0, 1, ...]
})

# Tree (Newick)
tree_file = "tree.nwk"
```

---

## 🛠️ Development

### For Contributors

MDITRE v1.0.0 includes modern development infrastructure for easy contribution.

#### Quick Development Setup

```bash
# Clone and setup
git clone https://github.com/melhzy/mditre.git
cd mditre

# Install development dependencies
make install-dev
# Or: pip install -r requirements-dev.txt

# Run tests
make test

# Format code
make format

# Check code quality
make quality
```

#### Development Commands

| Command | Description |
|---------|-------------|
| `make install` | Install package in editable mode |
| `make install-dev` | Install with development tools |
| `make test` | Run all tests (pytest) |
| `make test-cov` | Run tests with coverage report |
| `make test-fast` | Skip slow tests |
| `make format` | Format code (black + isort) |
| `make lint` | Check code style (flake8) |
| `make typecheck` | Run type checker (mypy) |
| `make quality` | Run all quality checks |
| `make dev` | Quick dev cycle (format + fast test) |
| `make ci` | Full CI simulation |
| `make clean` | Remove build artifacts |

#### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_mditre_comprehensive.py -v

# With coverage
pytest tests/ --cov=mditre --cov-report=html

# By marker
pytest tests/ -m architecture
pytest tests/ -m "not slow"

# Latest: 28/28 passing in 2.17s ✅
```

#### Project Structure

```
mditre/
├── .gitignore              # Version control config
├── pyproject.toml          # Modern packaging (PEP 518)
├── requirements.txt        # Pinned dependencies
├── requirements-dev.txt    # Development tools
├── Makefile                # Task automation
├── CHANGELOG.md            # Version history
├── CONTRIBUTING.md         # Contribution guidelines
├── tests/                  # Test suite
│   ├── conftest.py         # Shared fixtures
│   ├── test_*.py           # Test files
│   └── README.md           # Test documentation
├── mditre/                 # Package source
└── docs/                   # Documentation
```

#### Code Style

- **Formatter**: Black (100 char line length)
- **Import Sort**: isort (black profile)
- **Linter**: Flake8
- **Type Checker**: mypy
- **All configured in**: `pyproject.toml`

#### Quality Standards

- ✅ All tests must pass
- ✅ Code coverage >80%
- ✅ Black formatted
- ✅ No linting errors
- ✅ Type hints for public APIs
- ✅ Docstrings (Google style)

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- 🚀 Getting started guide
- 📋 Contribution workflow
- ✅ Code quality standards
- 📝 Documentation guidelines
- 🐛 Bug report templates
- ✨ Feature request process

### Quick Contribution Guide

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test**: `make test`
5. **Format**: `make format`
6. **Quality Check**: `make quality`
7. **Commit** with clear message
8. **Push** and create Pull Request

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New features (layers, loaders, transforms)
- 📝 Documentation improvements
- 🧪 Additional tests
- 🎨 Visualization enhancements
- 📊 Example notebooks

---

## 📖 Documentation

### Core Documentation
- **[README.md](README.md)** - This file: Installation, usage, quick start
- **[QA.md](QA.md)** - Quality assurance and test status
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

### Technical Documentation (`docs/`)
- **[docs/README.md](docs/README.md)** - Documentation index and navigation
- **[docs/MODULAR_ARCHITECTURE.md](docs/MODULAR_ARCHITECTURE.md)** - 5-layer architecture reference
- **[docs/DATA_LOADER_GUIDE.md](docs/DATA_LOADER_GUIDE.md)** - Data loading system API
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Development guide & performance
- **[docs/SEEDING_GUIDE.md](docs/SEEDING_GUIDE.md)** - Reproducibility & seeding
- **[docs/TRAINER_FIXES.md](docs/TRAINER_FIXES.md)** - Bug fixes documentation

### Examples & Tutorials
- **Examples**: `mditre/examples/` - Working code examples
- **Tutorials**: `jupyter/` - Jupyter notebooks with step-by-step guides
- **Quick Test**: `jupyter/run_mditre_test.ipynb` - Fast training demo

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/melhzy/mditre/issues)
- **Discussions**: [GitHub Discussions](https://github.com/melhzy/mditre/discussions)
- **Tests**: `tests/` - Comprehensive test suite documentation

---

## 📄 Citation

If you use MDITRE, please cite:

```bibtex
@article{maringanti2022mditre,
  title={MDITRE: Scalable and Interpretable Machine Learning for Predicting Host Status from Temporal Microbiome Dynamics},
  author={Maringanti, Veda Sheersh and Bucci, Vanni and Gerber, Georg K},
  journal={mSystems},
  volume={7},
  number={3},
  pages={e00132--22},
  year={2022},
  publisher={American Society for Microbiology},
  doi={10.1128/msystems.00132-22}
}
```

**Paper:** Maringanti VS, Bucci V, Gerber GK. 2022. MDITRE: Scalable and Interpretable Machine Learning for Predicting Host Status from Temporal Microbiome Dynamics. mSystems 7:e00132-22. https://doi.org/10.1128/msystems.00132-22

---

## 📜 License

GNU General Public License v3.0 (GPL-3.0) - see [LICENSE](LICENSE) file

---

## 👥 Contributors

- **Venkata Suhas Maringanti** - Original implementation
- **Georg K. Gerber** - Principal Investigator  
- **Vanni Bucci** - Co-Principal Investigator
- **Ziyuan Huang** - Maintainer, v1.0.0 infrastructure improvements

---

## 🙏 Acknowledgments

We thank the microbiome research community for public data and the developers of PyTorch, scikit-learn, and other open-source tools.

---

## 📌 Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

- **v1.0.0** (November 2025): Production release with modern infrastructure
  - Modular 5-layer architecture
  - Extensible data loading system
  - Deterministic seeding module
  - Comprehensive test suite (28 tests, 100% passing)
  - Modern development tools (pyproject.toml, Makefile, requirements.txt)
  - Documentation reorganization (9 docs → 6 focused guides)
  - Deprecation warnings for legacy code
  - Task automation with 20+ Makefile commands
- **v0.1.6** (2022): Initial beta release

---

**🎉 MDITRE v1.0.0 is production-ready with modern development infrastructure!**
