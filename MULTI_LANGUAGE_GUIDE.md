# MDITRE Multi-Language Implementation Guide

**Date**: November 1, 2025  
**Status**: Python ✅ Production (v1.0.0) | R 🚧 Planned (v2.0)

---

## Repository Structure

The MDITRE repository is organized to support multiple programming language implementations:

```
mditre/
├── Python/          # Python implementation (v1.0.0) ✅ PRODUCTION
│   ├── mditre/             # Core package
│   ├── tests/              # Test suite
│   ├── docs/               # Documentation
│   ├── jupyter/            # Example notebooks
│   ├── mditre_outputs/     # Output files
│   ├── mditre_paper_results/  # Paper reproduction code
│   └── ...                 # Config files, requirements, etc.
│
├── R/              # R implementation (v2.0) 🚧 PLANNED
│   └── README.md   # Planned features
│
├── README.md       # Main documentation
├── CHANGELOG.md    # Version history
└── ...
```

---

## Python Implementation (v1.0.0)

### Location
```
Python/
```

### Status
✅ **Production Ready**
- 39/39 tests passing (100%)
- Comprehensive documentation
- Full feature set implemented

### Installation

```bash
cd Python/
pip install -e .
```

### Testing

```bash
cd Python/
pytest tests/test_all.py -v
```

### Key Files

```
Python/
├── mditre/              # Main package
│   ├── core/           # Base classes
│   ├── layers/         # 5-layer architecture
│   ├── data_loader/    # Data loading
│   ├── models.py       # MDITRE models
│   └── seeding.py      # Reproducibility
│
├── tests/              # Test suite (39 tests)
│   ├── test_all.py     # All tests
│   ├── conftest.py     # Fixtures
│   └── README.md       # Test docs
│
├── docs/               # Documentation
├── jupyter/            # Jupyter notebooks & tutorials
├── mditre_outputs/     # Model outputs & results
├── mditre_paper_results/  # Paper reproduction scripts
│
├── setup.py            # Installation
├── pyproject.toml      # Modern packaging
├── requirements.txt    # Dependencies
├── Makefile            # Automation
└── README.md           # Python docs
```

---

## R Implementation (v2.0)

### Location
```
R/
```

### Status
🚧 **Planned for v2.0**

### Planned Features

- Core MDITRE models in R
- Integration with phyloseq/microbiome packages
- R-native visualization with ggplot2
- Statistical analysis integration
- Interoperability with Python implementation

### Timeline

**Target**: v2.0 release (TBD)

### How to Contribute

Contributions to the R implementation are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Shared Resources

### Python-Specific Resources

#### Documentation (`Python/docs/`)

Technical documentation for the Python implementation:

- `MODULAR_ARCHITECTURE.md` - 5-layer architecture details
- `DATA_LOADER_GUIDE.md` - Data loading patterns
- `DEVELOPMENT.md` - Development workflow
- `SEEDING_GUIDE.md` - Reproducibility guide

#### Examples (`Python/jupyter/`)

Jupyter notebooks demonstrating MDITRE usage:
- Tutorial notebooks for different data types
- Example analyses and visualizations
- Sample datasets and preprocessing scripts

#### Paper Results (`Python/mditre_paper_results/`)

Code to reproduce results from:
> Maringanti et al. (2022). Explainable Deep Relational Networks for Longitudinal Genomic Sequence Data. mSystems, 7(5).

#### Outputs (`Python/mditre_outputs/`)

Directory for model outputs, trained weights, and analysis results.

---

## Development Workflow

### Working on Python Implementation

```bash
# Navigate to Python directory
cd Python/

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/test_all.py -v

# Use Makefile commands
make test          # Run tests
make test-cov      # With coverage
make format        # Format code
make lint          # Check style
```

### Working on R Implementation (Future)

```bash
# Navigate to R directory
cd R/

# (Commands TBD when R implementation begins)
```

---

## Version History

### v1.0.0 (November 1, 2025)
- ✅ Python implementation production ready
- ✅ 39 comprehensive tests
- ✅ Multi-language repository structure
- 🚧 R implementation directory created

### v2.0 (Planned)
- R implementation
- Cross-language compatibility
- Extended documentation

---

## Quick Links

- **Python README**: [Python/README.md](Python/README.md)
- **R README**: [R/README.md](R/README.md)
- **Main README**: [README.md](README.md)
- **Documentation**: [docs/](docs/)
- **Tests**: [Python/tests/](Python/tests/)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

## Language-Specific Resources

### Python
- **Package**: `Python/mditre/`
- **Tests**: `Python/tests/` (39 tests)
- **Docs**: `Python/README.md`
- **Install**: `cd Python/ && pip install -e .`
- **Test**: `cd Python/ && pytest tests/test_all.py -v`

### R (Coming Soon)
- **Package**: `R/` (TBD)
- **Docs**: `R/README.md` (planning doc)
- **Status**: 🚧 Under Development

---

## FAQs

### Why separate directories?

To support multiple language implementations while:
- Keeping each implementation clean and self-contained
- Allowing language-specific dependencies
- Facilitating independent development and testing
- Supporting language-specific tooling (pytest for Python, testthat for R)

### Which implementation should I use?

- **For production use**: Python implementation (v1.0.0) ✅
- **For R users**: Wait for v2.0 or contribute to R implementation 🚧

### Can I use both implementations?

Yes! The v2.0 R implementation will be designed for interoperability with Python.

### How do I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines for both implementations.

---

**Last Updated**: November 1, 2025  
**Maintainer**: melhzy  
**Repository**: https://github.com/melhzy/mditre
