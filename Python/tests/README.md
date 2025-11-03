# MDITRE Test Suite Documentation

**Version**: 1.0.0  
**Last Updated**: November 1, 2025  
**Status**: 39/39 tests passing (3.29s runtime)

---

## Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Current Test Status](#current-test-status)
4. [Running Tests](#running-tests)
5. [Test Categories](#test-categories)
6. [Shared Fixtures](#shared-fixtures)
7. [Writing New Tests](#writing-new-tests)
8. [Future Test Plan](#future-test-plan)
9. [Common Issues](#common-issues)

---

## Overview

The MDITRE test suite is consolidated into a single comprehensive test file `test_all.py` that validates all aspects of the package based on the paper "MDITRE: Scalable and Interpretable Machine Learning for Predicting Host Status from Temporal Microbiome Dynamics" (Maringanti, Bucci, & Gerber, 2022, mSystems).

**Key Features**:
- ✅ Single integrated test file (no duplicates)
- ✅ 39 comprehensive tests covering all components
- ✅ Fast execution (<5s for full suite)
- ✅ Clear test organization by functionality
- ✅ Automatic cleanup and isolation

---

## Test Structure

```
tests/
├── conftest.py       # Shared fixtures and pytest configuration
├── test_all.py       # Consolidated comprehensive test suite (39 tests)
└── README.md         # This file
```

### Consolidated Test File: `test_all.py`

All tests are organized into 8 major test classes:

1. **TestSection1_1_FiveLayerArchitecture** (8 tests) - 5-layer model validation
2. **TestSection1_2_Differentiability** (3 tests) - Gradient flow
3. **TestSection1_3_ModelVariants** (2 tests) - MDITRE & MDITREAbun
4. **TestSection2_PhylogeneticFocus** (4 tests) - Phylogenetic mechanisms
5. **TestSection3_TemporalFocus** (4 tests) - Temporal mechanisms
6. **TestSection10_1_PerformanceMetrics** (3 tests) - F1, AUC-ROC
7. **TestSection12_1_PyTorchIntegration** (3 tests) - PyTorch APIs
8. **TestEndToEndWorkflow** (1 test) - Complete pipeline
9. **TestSeeding** (5 tests) - Reproducibility
10. **TestPackageIntegrity** (6 tests) - Module validation

---

## Current Test Status

### Latest Results (November 1, 2025)

```
Platform: Windows 11
Python: 3.12.4
PyTorch: 2.5.1+cu121
CUDA: 12.1 (RTX 4090, 16GB VRAM)
pytest: 7.4.4

========================================
RESULT: 39/39 PASSED in 3.29s
========================================
```

### Test Breakdown by Category

| Category | Tests | Status | Description |
|----------|-------|--------|-------------|
| **Architecture** | 13 | ✅ 13/13 | 5-layer model + variants |
| **Phylogenetic** | 4 | ✅ 4/4 | Embeddings & selection |
| **Temporal** | 4 | ✅ 4/4 | Time windows & slopes |
| **Metrics** | 3 | ✅ 3/3 | F1, AUC-ROC, etc. |
| **Integration** | 4 | ✅ 4/4 | PyTorch & E2E |
| **Seeding** | 5 | ✅ 5/5 | Reproducibility |
| **Integrity** | 6 | ✅ 6/6 | Package validation |
| **TOTAL** | **39** | **✅ 100%** | **All passing** |

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/test_all.py

# Run with verbose output
pytest tests/test_all.py -v

# Run with short traceback
pytest tests/test_all.py -v --tb=short

# Run specific test class
pytest tests/test_all.py::TestSection1_1_FiveLayerArchitecture -v

# Run specific test
pytest tests/test_all.py::TestSeeding::test_seeding_reproducibility -v
```

### With Coverage

```bash
# Coverage with terminal report
pytest tests/test_all.py --cov=mditre --cov-report=term-missing

# Coverage with HTML report
pytest tests/test_all.py --cov=mditre --cov-report=html
```

### Using Markers

```bash
# Run only architecture tests
pytest tests/test_all.py -m architecture

# Run only critical tests
pytest tests/test_all.py -m critical

# Skip slow tests
pytest tests/test_all.py -m "not slow"

# Run seeding tests
pytest tests/test_all.py -m seeding

# Run package integrity tests
pytest tests/test_all.py -m integrity
```

### Parallel Execution

```bash
# Use all CPU cores
pytest tests/test_all.py -n auto
```

---

## Test Categories

Tests are organized with pytest markers:

- `architecture`: General architecture tests
- `layer1` - `layer5`: Layer-specific tests
- `differentiability`: Gradient flow
- `model`: Model instantiation
- `phylogenetic`: Phylogenetic operations
- `temporal`: Time series handling
- `metrics`: Performance metrics
- `integration`: End-to-end workflows
- `seeding`: Reproducibility
- `integrity`: Package validation
- `critical`: Must-pass tests
- `slow`: Long-running tests
- `gpu`: GPU-specific tests

---

## Shared Fixtures

Defined in `conftest.py`:

- `device`: PyTorch device (CPU/CUDA)
- `test_config`: Standard test parameters
- `synthetic_data`: Synthetic microbiome data
- `phylo_dist_matrix`: Phylogenetic distances
- `otu_embeddings`: OTU embeddings
- `init_args_full`: Model initialization args

---

## Writing New Tests

### Test Template

```python
@pytest.mark.architecture
@pytest.mark.layer1
def test_new_feature(device, synthetic_data):
    """
    Test description: What this test validates.
    
    Paper Reference: Quote from paper if applicable.
    """
    # Arrange
    model = MDITRE(...).to(device)
    X = synthetic_data['X']
    
    # Act
    output = model(X)
    
    # Assert
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
```

### Best Practices

1. Use descriptive test names
2. Follow AAA pattern (Arrange, Act, Assert)
3. Test one behavior per test
4. Use fixtures for common setup
5. Add appropriate markers
6. Include docstrings with paper references
7. Test edge cases
8. Clean up resources

### Correct Parameter Names

```python
model = MDITRE(
    num_rules=5,           # NOT num_rule
    num_otus=50,           # NOT num_otu
    num_otu_centers=10,
    num_time=10,
    num_time_centers=5,
    dist=dist_array,       # numpy array, shape (num_otus, emb_dim)
    emb_dim=8
)
```

---

## Future Test Plan

### Phase 1: Core Functionality (COMPLETED ✅)
**Status**: 39/39 tests passing
- ✅ Five-layer architecture
- ✅ Model variants
- ✅ Differentiability
- ✅ Phylogenetic and temporal focus
- ✅ Performance metrics
- ✅ PyTorch integration
- ✅ End-to-end pipeline
- ✅ Seeding and reproducibility
- ✅ Package integrity

### Phase 2: Data Loading (Planned - 15 tests)
- Data loader tests (8 tests)
- Transform tests (7 tests)

### Phase 3: Training & Optimization (Planned - 20 tests)
- Training loop (8 tests)
- Optimization (6 tests)
- Hyperparameter tuning (6 tests)

### Total Progress

| Phase | Status | Tests | Priority |
|-------|--------|-------|----------|
| Phase 1: Core | ✅ Complete | 39/39 | Critical |
| Phase 2: Data | 📋 Planned | 0/15 | High |
| Phase 3: Training | 📋 Planned | 0/20 | High |
| **TOTAL** | **53%** | **39/74** | - |

---

## Common Issues

### CUDA out of memory
```bash
pytest tests/test_all.py -m "not gpu"
```

### Import errors
```bash
pip install -e .
```

### Slow execution
```bash
pip install pytest-xdist
pytest tests/test_all.py -n auto
```

### Model dimension mismatch
Ensure `dist` has shape `(num_otus, emb_dim)`:
```python
dist = np.random.randn(num_otus, emb_dim).astype(np.float32)
```

---

## Resources

- **Main README**: `../README.md`
- **QA Documentation**: `../QA.md`
- **Architecture Docs**: `../docs/MODULAR_ARCHITECTURE.md`
- **Paper**: Maringanti et al. (2022) - mSystems Volume 7 Issue 5
- **Pytest Documentation**: https://docs.pytest.org/

---

**Test Suite Version**: 1.0.0  
**Last Updated**: November 1, 2025  
**Maintainer**: melhzy  
**Status**: ✅ Production Ready (39/39 tests passing, 3.29s runtime)
