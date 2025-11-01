# MDITRE Seeding Module

Deterministic seed generation for reproducible MDITRE experiments using the [seedhash](https://github.com/melhzy/seedhash) library.

## Overview

The MDITRE seeding module provides a standardized way to generate reproducible random seeds across all MDITRE experiments. It uses MD5 hashing to create deterministic seeds from the MDITRE master seed string:

```
"MDITRE: Scalable and Interpretable Machine Learning for Predicting Host Status from Temporal Microbiome Dynamics"
```

## Features

- ✅ **Deterministic**: Same experiment name always produces the same seeds
- ✅ **Traceable**: Each seed has an MD5 hash for verification
- ✅ **Reproducible**: Sets seeds for Python, NumPy, and PyTorch
- ✅ **Configurable**: Supports experiment-specific seed generation
- ✅ **Documented**: Complete seed information saved with results

## Installation

The seeding module is automatically installed with MDITRE. The `seedhash` dependency is included in `setup.py`:

```bash
pip install git+https://github.com/melhzy/seedhash.git#subdirectory=Python
```

## Quick Start

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
print(f"Master seed: {info['master_seed']}")
```

### Convenience Function

```python
from mditre.seeding import get_mditre_seeds

# Quick seed generation without creating a generator object
seeds = get_mditre_seeds(10, experiment_name="monte_carlo_sim")
print(seeds)
```

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

**Parameters:**
- `experiment_name`: Optional experiment identifier appended to master seed
- `min_value`: Minimum seed value (default: 0)
- `max_value`: Maximum seed value (default: 2^31-1)

**Example:**
```python
# Default master seed only
gen = MDITRESeedGenerator()

# With experiment name
gen = MDITRESeedGenerator(experiment_name="baseline_v1")

# Custom range
gen = MDITRESeedGenerator(
    experiment_name="test",
    min_value=1000,
    max_value=9999
)
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

**Example:**
```python
seeds = get_mditre_seeds(5, experiment_name="baseline_model")
```

### `set_random_seeds()`

Set random seeds for all common libraries.

```python
set_random_seeds(seed: int)
```

Sets seeds for:
- Python's `random` module
- NumPy
- PyTorch (CPU and CUDA)
- PyTorch deterministic mode

**Example:**
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

## Usage in MDITRE Notebook

The seeding module is integrated into `jupyter/run_mditre_test.ipynb`:

```python
# Initialize seeding
from mditre.seeding import MDITRESeedGenerator, set_random_seeds

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

## Reproducibility

To reproduce an experiment exactly:

1. **Save seed information** (automatically done in notebook):
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

2. **Load and use saved seed**:
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

3. **Verify hash**:
```python
from mditre.seeding import MDITRESeedGenerator

gen = MDITRESeedGenerator()
assert gen.get_hash() == seed_data['seed_hash']  # Verify correctness
```

## Examples

### Example 1: Cross-Validation Folds

```python
from mditre.seeding import MDITRESeedGenerator

# Generate seeds for 5-fold CV
cv_gen = MDITRESeedGenerator(experiment_name="5fold_cv_baseline")
fold_seeds = cv_gen.generate_seeds(5)

for fold, seed in enumerate(fold_seeds, 1):
    print(f"Fold {fold}: seed={seed}")
    # Use seed for train_test_split or data sampling
```

### Example 2: Hyperparameter Tuning

```python
from mditre.seeding import MDITRESeedGenerator

# Generate seeds for hyperparameter search
hp_gen = MDITRESeedGenerator(experiment_name="hp_tuning_lr_wd")
trial_seeds = hp_gen.generate_seeds(100)  # 100 random trials

for trial, seed in enumerate(trial_seeds):
    # Each trial gets a unique but reproducible seed
    set_random_seeds(seed)
    # Run experiment with hyperparameters
```

### Example 3: Ensemble Models

```python
from mditre.seeding import MDITRESeedGenerator

# Generate seeds for ensemble members
ensemble_gen = MDITRESeedGenerator(experiment_name="ensemble_10_models")
model_seeds = ensemble_gen.generate_seeds(10)

for i, seed in enumerate(model_seeds):
    print(f"Training model {i+1} with seed {seed}")
    set_random_seeds(seed)
    # Train model with unique initialization
```

## Best Practices

1. **Always use experiment names** for different runs:
```python
# Good
gen = MDITRESeedGenerator(experiment_name="baseline_v1")

# Less traceable
gen = MDITRESeedGenerator()  # Uses only master seed
```

2. **Save seed information** with results:
```python
info = seed_gen.get_seed_info()
results['seed_hash'] = info['hash']
results['seed'] = master_seed
```

3. **Verify reproducibility** by checking hashes:
```python
assert seed_gen.get_hash() == expected_hash
```

4. **Use set_random_seeds()** after generation:
```python
seeds = get_mditre_seeds(1, experiment_name="test")
set_random_seeds(seeds[0])  # Sets all library seeds
```

## Technical Details

### Master Seed String

The MDITRE master seed is:
```
"MDITRE: Scalable and Interpretable Machine Learning for Predicting Host Status from Temporal Microbiome Dynamics"
```

MD5 Hash: `144d0c1d3178b54e1171fcb281dad562`

### Seed Construction

When an experiment name is provided, the seed string becomes:
```
f"{MDITRE_MASTER_SEED}::{experiment_name}"
```

This ensures:
- Different experiments get different seeds
- Same experiment always gets same seed
- All seeds are traceable to MDITRE master seed

### Deterministic Behavior

The `set_random_seeds()` function ensures:
- Python's `random.seed()`
- NumPy's `np.random.seed()`
- PyTorch's `torch.manual_seed()`
- CUDA's `torch.cuda.manual_seed_all()`
- Deterministic CUDA operations: `torch.backends.cudnn.deterministic = True`

## Testing

Run the seeding tests:

```bash
python test_seeding.py
```

Expected output shows:
- Seed generation
- Hash verification
- Reproducibility across libraries
- Experiment-specific seeding

## See Also

- [seedhash GitHub repository](https://github.com/melhzy/seedhash)
- [MDITRE Documentation](../README.md)
- [Reproducibility Best Practices](../docs/reproducibility.md)

## References

1. Rivest, R. (1992). The MD5 Message-Digest Algorithm (RFC 1321). MIT Laboratory for Computer Science and RSA Data Security, Inc.
2. Python Software Foundation. (2024). random — Generate pseudo-random numbers.

---

**Note**: This module uses MD5 hashing for seed generation. MD5 is suitable for non-cryptographic purposes like reproducible seed generation. Do not use this module for cryptographic or security-sensitive applications.
