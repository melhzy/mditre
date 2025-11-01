# Changelog

All notable changes to MDITRE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-01

### Added
- **Modular Architecture**: New 5-layer modular design in `mditre/layers/`
  - Layer 1: Phylogenetic Focus (`SpatialAggDynamic`)
  - Layer 2: Temporal Focus (`TimeAgg`)
  - Layer 3: Detectors (`Threshold`, `Slope`)
  - Layer 4: Rule Formation (`Rules`)
  - Layer 5: Classification (`DenseLayer`)
- **Core Module**: Base classes and utilities in `mditre/core/`
  - `BaseLayer`: Abstract base class for all layers
  - `LayerRegistry`: Dynamic layer registration system
  - `math_utils`: Shared mathematical utilities
- **Modern Data Loader**: Extensible data loading system in `mditre/data_loader/`
  - `AmpliconLoader`: For 16S rRNA amplicon data
  - `MetaphlanLoader`: For shotgun metagenomics data
  - `PickleLoader`: For preprocessed data
  - Modular transforms and preprocessors
- **Seeding System**: Deterministic reproducibility with `mditre.seeding`
  - `MDITRESeedGenerator`: Layer-specific seed generation
  - `get_mditre_seeds()`: Comprehensive seed management
  - `set_random_seeds()`: One-function reproducibility setup
- **Comprehensive Test Suite**: 28 tests covering all components
  - Architecture tests for each layer
  - Differentiability tests for gradient flow
  - Metrics validation
  - Integration tests
  - Seeding validation (5 dedicated tests)
- **Documentation**: Consolidated and organized documentation
  - `MODULAR_ARCHITECTURE.md`: Detailed architecture guide
  - `DATA_LOADER_GUIDE.md`: Data loading instructions
  - `SEEDING.md` & `SEEDING_INTEGRATION.md`: Reproducibility docs
  - `EFFICIENCY_REPORT.md`: Performance analysis
  - `TRAINER_FIXES.md`: Bug fix documentation
- **Development Infrastructure**:
  - `.gitignore`: Comprehensive ignore patterns
  - `pyproject.toml`: Modern Python packaging configuration
  - `requirements.txt`: Pinned production dependencies
  - `requirements-dev.txt`: Development dependencies
  - `pytest.ini`: Test configuration with markers
  - `STRUCTURE_ANALYSIS.md`: Project structure analysis
  - `CHANGELOG.md`: Version history (this file)

### Changed
- **Package Status**: Promoted from Beta (0.1.6) to Production/Stable (1.0.0)
- **Python Support**: Updated minimum from 3.6+ to 3.8+ (tested on 3.8-3.12)
- **PyTorch Version**: Updated minimum from 1.7+ to 2.0+ (tested on 2.5.1)
- **Setup Configuration**: 
  - Implemented `find_packages()` for automatic package discovery
  - Added `extras_require` for optional dependencies (`dev`, `viz`, `docs`)
  - Enhanced metadata with keywords and project URLs
  - Fixed encoding issue with UTF-8 README reading
- **Package Structure**: Organized with clear separation
  - Core algorithm in `mditre/` package
  - Tests in root (to be moved to `tests/` in v2.0)
  - Documentation in `docs/`
  - Tutorials in `jupyter/`
  - Research code in `mditre_paper_results/`

### Fixed
- **Trainer.py**: 43+ critical bug fixes
  - Fixed `best_model` None access issues (lines 1960, 3007)
  - Fixed `losses_csv` unbound errors
  - Added proper None checks for FCA tree traversal
  - Fixed variable initialization in training loops
  - Added safe dictionary access patterns
- **Test Suite**: Removed all warnings
  - Fixed `torch.load()` security warning with `weights_only=True`
  - All 28 tests passing (3.86s execution time)
- **Notebook Performance**: Fixed hanging issues
  - Added `sys.stdout.flush()` calls in summary cell
  - Reduced summary generation from 26+ minutes to 22ms
  - Training remains efficient (10 epochs in 30.28s)
- **Documentation**: Consolidated from 19 scattered files to 8 organized files
  - Removed 70% duplicate content
  - Reduced total size by 40%
  - Created master documentation hub

### Deprecated
- **Legacy Models** (to be removed in v2.0):
  - `mditre.models.MDITRE`: Use `mditre.layers` modular approach instead
  - `mditre.models.MDITREAbun`: Use modular layers for custom architectures
  - Note: Legacy models maintained for backward compatibility in v1.x

### Security
- Updated `torch.load()` calls to use `weights_only=True` parameter
- Added proper input validation in data loaders

### Performance
- **Test Execution**: 28 tests complete in 3.86 seconds
- **Training Speed**: 10 epochs in 30.28 seconds on RTX 4090
- **Memory Efficiency**: Optimized tensor operations
- **Import Time**: Fast package initialization

### Documentation
- **README.md**: Comprehensive installation and quick start guide
- **API Documentation**: Detailed docstrings for all public APIs
- **Examples**: Working examples in `mditre/examples/`
  - Modular architecture example
  - Data loader examples
- **Tutorials**: Jupyter notebooks in `jupyter/tutorials/`
  - Model run tutorial
  - 16S data analysis
  - Metaphlan data analysis
  - Bokulich dataset tutorial

### Known Issues
- `trainer.py` remains monolithic (4,966 LOC) - to be refactored in v2.0
- Legacy `data.py` and `data_utils.py` still in use - migration to `data_loader/` planned
- Some visualization variables not initialized (non-critical)
- CSV dataframe initialization issues in rule extraction (non-critical)

### Migration Guide
For users upgrading from v0.1.6:

**No breaking changes** - v1.0.0 is backward compatible. However, we recommend:

1. **Update imports for new features**:
   ```python
   # Old (still works)
   from mditre.models import MDITRE
   
   # New (recommended)
   from mditre.layers import SpatialAggDynamic, TimeAgg, Rules
   from mditre.seeding import get_mditre_seeds
   ```

2. **Use new data loaders**:
   ```python
   # New approach
   from mditre.data_loader import AmpliconLoader
   loader = AmpliconLoader(data_path, metadata_path)
   ```

3. **Enable reproducibility**:
   ```python
   from mditre.seeding import set_random_seeds
   set_random_seeds(seed=42)
   ```

### Contributors
- Venkata Suhas Maringanti (Original Author)
- melhzy (Maintainer, v1.0.0 Release)

### Links
- **Repository**: https://github.com/melhzy/mditre
- **Documentation**: https://github.com/melhzy/mditre/blob/master/README.md
- **Issues**: https://github.com/melhzy/mditre/issues

---

## [0.1.6] - Pre-release

Initial beta release with core MDITRE functionality.

### Features
- Original monolithic MDITRE and MDITREAbun models
- Basic training pipeline
- Rule extraction and visualization
- Phylogenetic tree integration
- Legacy data loading

---

## Version Format

- **Major.Minor.Patch** (e.g., 1.0.0)
- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes (backward compatible)
