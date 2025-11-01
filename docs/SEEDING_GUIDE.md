# MDITRE Seeding Guide

**Version**: 1.0.0  
**Last Updated**: November 1, 2025  
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [API Reference](#api-reference)
4. [Repository Integration](#repository-integration)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Technical Details](#technical-details)

---

## Overview

The MDITRE seeding module provides deterministic seed generation for reproducible experiments using the [seedhash](https://github.com/melhzy/seedhash) library. It creates reproducible random seeds from the MDITRE master seed string using MD5 hashing.

**Master Seed String**:
```
"MDITRE: Scalable and Interpretable Machine Learning for Predicting Host Status from Temporal Microbiome Dynamics"
```

**MD5 Hash**: `144d0c1d3178b54e1171fcb281dad562`

### Features

- ✅ **Deterministic**: Same experiment name always produces same seeds
- ✅ **Traceable**: Each seed has an MD5 hash for verification
- ✅ **Reproducible**: Sets seeds for Python, NumPy, and PyTorch
- ✅ **CUDA-aware**: Configures cuDNN for deterministic operations
- ✅ **Configurable**: Supports experiment-specific seed generation
- ✅ **Documented**: Complete seed information saved with results

---

## Quick Start

### Installation

The seeding module is automatically installed with MDITRE:

```bash
pip install git+https://github.com/melhzy/seedhash.git#subdirectory=Python
```

### Basic Usage

```python
from mditre.seeding import MDITRESeedGenerator, set_random_seeds

# Create generator with default MDITRE master seed
seed_gen = MDITRESeedGenerator()

# Generate 5 random seeds
seeds = seed_gen.generate_seeds(5)
print(f"Generated seeds: {seeds}")

# Get the hash for reproducibility tracking
print(f"Seed hash: {seed_gen.get_hash()}")

# Set the first seed globally for all libraries
set_random_seeds(seeds[0])
```

### With Experiment Name

```python
from mditre.seeding import MDITRESeedGenerator

# Create generator with experiment identifier
seed_gen = MDITRESeedGenerator(experiment_name="experiment_v1_baseline")

# Generate seeds for cross-validation folds
cv_seeds = seed_gen.generate_seeds(5)

# Get complete seed information
info = seed_gen.get_seed_info()
print(f"Experiment: {info['seed_string']}")
print(f"Hash: {info['hash']}")
```

### Convenience Function

```python
from mditre.seeding import get_mditre_seeds

# Quick seed generation without creating a generator object
seeds = get_mditre_seeds(10, experiment_name="monte_carlo_sim")
print(seeds)
```

---

## API Reference

### `MDITRESeedGenerator`

Main class for generating MDITRE seeds.

#### Constructor

```python
MDITRESeedGenerator(
    experiment_name: Optional[str] = None,
    min_value: int = 0,
    max_value: int = 2147483647
)
```

**Parameters**:
- `experiment_name`: Optional experiment identifier appended to master seed
- `min_value`: Minimum seed value (default: 0)
- `max_value`: Maximum seed value (default: 2^31-1)

**Example**:
```python
# Default master seed only
gen = MDITRESeedGenerator()

# With experiment name
gen = MDITRESeedGenerator(experiment_name="baseline_v1")

# Custom range
gen = MDITRESeedGenerator(experiment_name="test", min_value=1000, max_value=9999)
```

#### Methods

##### `generate_seeds(count: int) -> List[int]`

Generate deterministic random seeds.

```python
seeds = seed_gen.generate_seeds(10)
# Returns: [951483900, 1751077464, ...]
```

##### `get_hash() -> str`

Get the MD5 hash of the seed string.

```python
hash_value = seed_gen.get_hash()
# Returns: '144d0c1d3178b54e1171fcb281dad562'
```

##### `get_seed_info() -> dict`

Get comprehensive seed information.

```python
info = seed_gen.get_seed_info()
# Returns:
# {
#     'seed_string': 'MDITRE: Scalable...',
#     'hash': '144d0c1d...',
#     'min_value': 0,
#     'max_value': 2147483647,
#     'master_seed': 'MDITRE: Scalable...',
#     'seed_number': 26984612475648148687220031640980608354
# }
```

---

### `get_mditre_seeds()`

Convenience function for quick seed generation.

```python
get_mditre_seeds(
    count: int,
    experiment_name: Optional[str] = None,
    min_value: int = 0,
    max_value: int = 2147483647
) -> List[int]
```

**Example**:
```python
seeds = get_mditre_seeds(5, experiment_name="baseline_model")
```

---

### `set_random_seeds()`

Set random seeds for all common libraries.

```python
set_random_seeds(seed: int)
```

**Sets seeds for**:
- Python's `random` module
- NumPy
- PyTorch (CPU and CUDA)
- PyTorch deterministic mode (cuDNN)

**Example**:
```python
from mditre.seeding import get_mditre_seeds, set_random_seeds

# Generate and set seed
seeds = get_mditre_seeds(1)
set_random_seeds(seeds[0])

# Now all random operations are reproducible
import random
import numpy as np
import torch

print(random.randint(0, 100))  # Reproducible
print(np.random.rand())         # Reproducible
print(torch.rand(3))            # Reproducible
```

---

## Repository Integration

The seeding module has been integrated throughout the entire MDITRE repository for consistency and reproducibility.

### Files Using Seeding Module

#### 1. `mditre/trainer.py`

**Location**: Line ~1395-1410

**Implementation**:
```python
# Use MDITRE seeding module for consistent seed management
from mditre.seeding import set_random_seeds
set_random_seeds(seed)

# Additional environment configuration
os.environ['PYTHONHASHSEED'] = str(seed)
```

**Benefits**:
- Consistent seeding logic across repository
- Automatic CUDA configuration (deterministic + benchmark settings)
- Cleaner, more maintainable code

---

#### 2. `mditre/examples/modular_architecture_example.py`

**Location**: Line ~170-171

**Before**:
```python
torch.manual_seed(42)
np.random.seed(42)
```

**After**:
```python
from mditre.seeding import set_random_seeds
set_random_seeds(42)
```

**Benefits**: CUDA operations now properly seeded

---

#### 3. `tests/test_mditre_comprehensive.py`

**Multiple fixtures updated**:
- `synthetic_data` (Line ~74-75)
- `phylo_dist_matrix` (Line ~104)
- `otu_embeddings` (Line ~118)
- `init_args_full` (Line ~129)

**Implementation**:
```python
from mditre.seeding import set_random_seeds
set_random_seeds(test_config['random_seed'])
```

**Benefits**: Improved test reproducibility, CUDA operations now deterministic

---

#### 4. `jupyter/run_mditre_test.ipynb`

**Usage in Notebook**:
```python
from mditre.seeding import MDITRESeedGenerator, set_random_seeds

# Initialize seeding
seed_gen = MDITRESeedGenerator()
seeds = seed_gen.generate_seeds(1)
master_seed = seeds[0]

# Get seed info for tracking
seed_info = seed_gen.get_seed_info()
print(f"Master Seed: {seed_info['master_seed'][:50]}...")
print(f"Seed Hash: {seed_info['hash']}")
print(f"Generated Seed: {master_seed}")

# Set all random seeds
set_random_seeds(master_seed)

# Seed info is saved with results
results = {
    'seed': int(master_seed),
    'seed_hash': seed_info['hash'],
    'seed_string': seed_info['seed_string'],
    # ... other results
}
```

---

## Usage Examples

### Example 1: Cross-Validation

```python
from mditre.seeding import MDITRESeedGenerator, set_random_seeds

# Generate seeds for 5-fold CV
cv_gen = MDITRESeedGenerator(experiment_name="5fold_cv_baseline")
fold_seeds = cv_gen.generate_seeds(5)

for fold, seed in enumerate(fold_seeds, 1):
    print(f"Fold {fold}: seed={seed}")
    set_random_seeds(seed)
    # Use seed for train_test_split or data sampling
```

### Example 2: Hyperparameter Tuning

```python
from mditre.seeding import MDITRESeedGenerator, set_random_seeds

# Generate seeds for hyperparameter search
hp_gen = MDITRESeedGenerator(experiment_name="hp_tuning_lr_wd")
trial_seeds = hp_gen.generate_seeds(100)  # 100 random trials

for trial, seed in enumerate(trial_seeds):
    set_random_seeds(seed)
    # Each trial gets a unique but reproducible seed
    # Run experiment with hyperparameters
```

### Example 3: Ensemble Models

```python
from mditre.seeding import MDITRESeedGenerator, set_random_seeds

# Generate seeds for ensemble members
ensemble_gen = MDITRESeedGenerator(experiment_name="ensemble_10_models")
model_seeds = ensemble_gen.generate_seeds(10)

for i, seed in enumerate(model_seeds):
    print(f"Training model {i+1} with seed {seed}")
    set_random_seeds(seed)
    # Train model with unique initialization
```

### Example 4: Training Script

```python
from mditre.seeding import MDITRESeedGenerator, set_random_seeds
from mditre.models import MDITRE
import torch

# Generate master seed
seed_gen = MDITRESeedGenerator(experiment_name="experiment_001")
master_seed = seed_gen.generate_seeds(1)[0]

# Set all RNGs for reproducibility
set_random_seeds(master_seed)

# Now everything is deterministic
model = MDITRE(...)
# Training will be reproducible
```

---

## Best Practices

### 1. Always Use Experiment Names

```python
# Good - traceable
gen = MDITRESeedGenerator(experiment_name="baseline_v1")

# Less traceable
gen = MDITRESeedGenerator()  # Uses only master seed
```

### 2. Save Seed Information with Results

```python
info = seed_gen.get_seed_info()
results['seed_hash'] = info['hash']
results['seed'] = master_seed

# Save to file
seed_info_path = OUTPUT_DIR / "seed_info.json"
with open(seed_info_path, 'w') as f:
    json.dump({
        'seed': int(master_seed),
        'seed_hash': info['hash'],
        'seed_string': info['seed_string'],
        'master_seed': info['master_seed'],
        'experiment_name': EXPERIMENT_NAME
    }, f, indent=4)
```

### 3. Verify Reproducibility by Checking Hashes

```python
assert seed_gen.get_hash() == expected_hash
```

### 4. Use set_random_seeds() After Generation

```python
seeds = get_mditre_seeds(1, experiment_name="test")
set_random_seeds(seeds[0])  # Sets all library seeds
```

---

## Reproducibility

### Reproducing an Experiment

**1. Save seed information** (automatically done in notebook):
```python
seed_info_path = OUTPUT_DIR / "seed_info.json"
with open(seed_info_path, 'w') as f:
    json.dump({
        'seed': int(master_seed),
        'seed_hash': seed_info['hash'],
        'seed_string': seed_info['seed_string'],
        'master_seed': seed_info['master_seed'],
        'experiment_name': EXPERIMENT_NAME
    }, f, indent=4)
```

**2. Load and use saved seed**:
```python
import json
from mditre.seeding import set_random_seeds

# Load saved seed
with open('mditre_outputs/seed_info.json', 'r') as f:
    seed_data = json.load(f)

# Set the seed
set_random_seeds(seed_data['seed'])

# Now reproduce the exact experiment
```

**3. Verify hash**:
```python
from mditre.seeding import MDITRESeedGenerator

gen = MDITRESeedGenerator()
assert gen.get_hash() == seed_data['seed_hash']  # Verify correctness
```

### Verification Test

Run this to verify seeding works:

```python
from mditre.seeding import set_random_seeds
import numpy as np
import torch

# First run
set_random_seeds(42)
a1 = np.random.rand(5)
t1 = torch.rand(5)

# Second run (should be identical)
set_random_seeds(42)
a2 = np.random.rand(5)
t2 = torch.rand(5)

assert np.allclose(a1, a2), "NumPy not reproducible!"
assert torch.allclose(t1, t2), "PyTorch not reproducible!"
print("✅ Seeding is reproducible!")
```

---

## Technical Details

### Seed Generation Workflow

```
1. seed_string (text) → MD5 hash (32 hex chars)
                    ↓
2. MD5 hash → seed_number (large integer ~128-bit)
                    ↓
3. seed_number → master_seed (first generated seed)
                    ↓
4. master_seed → set_random_seeds(master_seed)
                    ↓
5. All RNGs initialized (Python, NumPy, PyTorch CPU+CUDA)
```

### Seed String Construction

When an experiment name is provided, the seed string becomes:
```python
f"{MDITRE_MASTER_SEED}::{experiment_name}"
```

This ensures:
- Different experiments get different seeds
- Same experiment always gets same seed
- All seeds are traceable to MDITRE master seed

### Deterministic Behavior

The `set_random_seeds()` function ensures:
- Python's `random.seed(seed)`
- NumPy's `np.random.seed(seed)`
- PyTorch's `torch.manual_seed(seed)`
- CUDA's `torch.cuda.manual_seed_all(seed)`
- Deterministic CUDA operations: `torch.backends.cudnn.deterministic = True`
- Benchmark mode disabled: `torch.backends.cudnn.benchmark = False`

### Integration Benefits

1. **Consistency**: Single seeding implementation across entire codebase
2. **Reproducibility**: Guaranteed deterministic behavior across Python, NumPy, PyTorch (CPU + CUDA)
3. **Maintainability**: Changes to seeding logic only need to happen in one place
4. **Testing**: All tests use same seeding mechanism
5. **Documentation**: Clear workflow from seed_string → hash → seed_number → master_seed

---

## Testing

Run the seeding tests:

```bash
python tests/test_seeding.py
```

**Expected Results**:
- ✅ Basic seed generation
- ✅ Seed information retrieval
- ✅ Experiment-specific seeding
- ✅ Convenience function
- ✅ Reproducibility verification (Python/NumPy/PyTorch)

---

## See Also

- [seedhash GitHub repository](https://github.com/melhzy/seedhash)
- [MDITRE Documentation](../README.md)
- [Development Guide](DEVELOPMENT.md)

---

## References

1. Rivest, R. (1992). The MD5 Message-Digest Algorithm (RFC 1321). MIT Laboratory for Computer Science and RSA Data Security, Inc.
2. Python Software Foundation. (2024). random — Generate pseudo-random numbers.

---

**Note**: This module uses MD5 hashing for seed generation. MD5 is suitable for non-cryptographic purposes like reproducible seed generation. Do not use this module for cryptographic or security-sensitive applications.

---

**Version**: 1.0.0  
**Status**: Production Ready  
**Last Updated**: November 1, 2025
