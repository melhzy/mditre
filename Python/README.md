# MDITRE Python Implementation

**Version**: 1.0.0  
**Status**: ✅ Production Ready

## Overview

This directory contains the complete Python implementation of MDITRE (Microbiome Differentiable Interpretable Temporal Rule Engine).

## Directory Structure

```
Python/
├── mditre/              # Main package code
│   ├── core/           # Core functionality (base layers, math utils)
│   ├── layers/         # 5-layer architecture implementation
│   ├── data_loader/    # Modular data loading system
│   ├── models.py       # MDITRE and MDITREAbun models
│   ├── seeding.py      # Reproducibility utilities
│   └── ...
├── tests/              # Comprehensive test suite (39 tests)
│   ├── test_all.py     # Consolidated test suite
│   ├── conftest.py     # Shared fixtures
│   └── README.md       # Test documentation
├── docs/               # Technical documentation
│   ├── MODULAR_ARCHITECTURE.md  # 5-layer architecture
│   ├── DATA_LOADER_GUIDE.md     # Data loading API
│   ├── DEVELOPMENT.md           # Development workflow
│   ├── SEEDING_GUIDE.md         # Reproducibility guide
│   └── TRAINER_FIXES.md         # Bug fixes
├── jupyter/            # Tutorials & example notebooks
│   ├── run_mditre_test.ipynb   # Quick test
│   └── tutorials/              # Comprehensive tutorials
├── mditre_outputs/     # Model outputs & trained weights
├── mditre_paper_results/  # Paper reproduction code
├── setup.py            # Package installation
├── pyproject.toml      # Modern Python packaging
├── requirements.txt    # Production dependencies
├── requirements-dev.txt # Development dependencies
├── pytest.ini          # Test configuration
└── Makefile            # Development automation

```

## Quick Start

### Installation

```bash
cd Python/
pip install -e .
```

### Running Tests

```bash
cd Python/
pytest tests/test_all.py -v
```

### Using Makefile

```bash
cd Python/
make install      # Install package
make test         # Run tests
make test-cov     # Run tests with coverage
make format       # Format code
make lint         # Lint code
```

## Features

- ✅ 5-layer interpretable neural network architecture
- ✅ Phylogenetic and temporal focus mechanisms
- ✅ Modular data loading (DADA2, QIIME2, etc.)
- ✅ GPU acceleration support
- ✅ Comprehensive test suite (39 tests, 100% passing)
- ✅ Reproducibility via seeding module
- ✅ Extensive documentation

## Documentation

See the documentation in `docs/`:
- Architecture: `docs/MODULAR_ARCHITECTURE.md`
- Data Loading: `docs/DATA_LOADER_GUIDE.md`
- Development: `docs/DEVELOPMENT.md`
- Seeding: `docs/SEEDING_GUIDE.md`

Also see tutorials in `jupyter/tutorials/` for practical examples.

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- NumPy >= 1.20
- scikit-learn >= 0.24
- ete3 >= 3.1.2

See `requirements.txt` for complete dependency list.

## Citation

If you use MDITRE in your research, please cite:

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

## License

GPL-3.0 License - See `../LICENSE` for details.

## Related Implementations

- **R Implementation**: See `../R/` (under development)

---

For more information, see the main [README.md](../README.md) in the repository root.
