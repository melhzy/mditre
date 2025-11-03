# MDITRE Installation Guide

**Version**: 1.0.1  
**Last Updated**: November 3, 2025  
**Test Status**: ✅ 39/39 Python tests passing (100%)  
**Supported Platforms**: Windows, macOS, Linux/Ubuntu, **Docker** 🐳

---

## Overview

MDITRE v1.0.1 is designed for **zero-configuration cross-platform installation**. After installing via `pip install mditre` (Python) or `install.packages("mditre")` (R), the package works immediately on Windows, macOS, and Ubuntu without any manual path configuration or edits.

**NEW**: Docker support for guaranteed reproducibility and zero version conflicts! 🎉

**Verified Performance**: Python test suite completes in 3.31 seconds with 100% pass rate.

## Installation Methods

### 🐳 Docker Installation (Recommended)

**Best for**: Reproducible environments, avoiding version conflicts, quick setup

```bash
# Clone repository
git clone https://github.com/melhzy/mditre.git
cd mditre

# Option A: Python-only environment (lightweight)
docker-compose up -d mditre-python
docker exec -it mditre-python bash
cd /workspace/Python && pytest tests/test_all.py -v

# Option B: Full environment with R support
docker-compose up -d mditre-full
docker exec -it mditre-full bash

# Option C: Jupyter Lab for interactive development
docker-compose up -d mditre-jupyter
# Access at http://localhost:8888
```

**Advantages**:
- ✅ No version conflicts (Python 3.12.3, R 4.5.2, PyTorch 2.5.1)
- ✅ Consistent across all platforms
- ✅ GPU support included (CUDA 12.4)
- ✅ All dependencies pre-installed
- ✅ Isolated from system packages

See [DOCKER.md](DOCKER.md) for complete documentation.

---

### Python Installation

#### Option 1: Install from PyPI (Recommended)

```bash
# Install latest release
pip install mditre

# Verify installation
python -c "import mditre; print(f'MDITRE {mditre.__version__} installed successfully')"
```

#### Option 2: Install from GitHub

```bash
# Install from source
pip install git+https://github.com/melhzy/mditre.git#subdirectory=Python

# Or clone and install in development mode
git clone https://github.com/melhzy/mditre.git
cd mditre/Python
pip install -e .
```

#### Option 3: Install with Optional Dependencies

```bash
# Install with visualization tools
pip install mditre[viz]

# Install with development tools
pip install mditre[dev]

# Install everything
pip install mditre[all]
```

#### Verify Installation

```bash
# Quick verification (recommended after installation)
python scripts/verify_cross_platform.py
# Expected output: 3/3 tests passed

# Or verify programmatically
python -c "import mditre; print(f'MDITRE v{mditre.__version__} - Ready!')"
```

### R Installation

#### Option 1: Install from CRAN (When Available)

```r
install.packages("mditre")
```

#### Option 2: Install from GitHub

```r
# Install remotes if needed
install.packages("remotes")

# Install MDITRE R package
remotes::install_github("melhzy/mditre", subdir = "R")

# Install seedhash dependency
remotes::install_github("melhzy/seedhash", subdir = "R")
```

#### Option 3: Development Installation

```r
# Clone repository
# git clone https://github.com/melhzy/mditre.git

# Install in development mode
devtools::install("path/to/mditre/R")
```

---

## Cross-Platform Verification

### Verify Python Installation

```python
from mditre.utils.path_utils import (
    get_package_root,
    get_project_root,
    get_platform_info
)

# Check installation location
print(f"Package installed at: {get_package_root()}")

# Check platform
info = get_platform_info()
print(f"Platform: {info['os']} on {info['os_name']}")

# Development mode check
dev_root = get_project_root()
if dev_root:
    print(f"Development mode: {dev_root}")
else:
    print("Production install (via pip)")
```

### Verify R Installation

```r
library(mditre)

# Check installation location
cat("Package location:", get_package_root(), "\n")

# Check platform
info <- get_platform_info()
cat("Platform:", info$os, info$os_version, "\n")

# Development mode check
dev_root <- get_project_root()
if (!is.null(dev_root)) {
  cat("Development mode:", dev_root, "\n")
} else {
  cat("Production install (via install.packages)\n")
}
```

---

## Platform-Specific Notes

### Windows

**Requirements**:
- Python 3.8+ or R 4.0+
- PyTorch 2.0+ (automatically installed with mditre)
- Visual C++ Redistributable (usually pre-installed)

**GPU Support**:
```bash
# For CUDA 12.x
pip install mditre
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Path Handling**:
- MDITRE automatically handles Windows paths (`C:\Users\...`)
- Internally uses forward slashes for cross-platform compatibility
- Automatically detects user home directory via `Path.home()`

### macOS

**Requirements**:
- Python 3.8+ or R 4.0+
- PyTorch 2.0+ (automatically installed with mditre)
- Xcode Command Line Tools (for some dependencies)

**Apple Silicon (M1/M2/M3)**:
```bash
# PyTorch optimized for Apple Silicon
pip install mditre
# PyTorch will automatically use MPS (Metal Performance Shaders)
```

**Intel Mac**:
```bash
pip install mditre
```

**Path Handling**:
- MDITRE automatically handles macOS paths (`/Users/...`)
- Uses native forward slashes
- Automatically detects user home directory

### Linux/Ubuntu

**Requirements**:
- Python 3.8+ or R 4.0+
- PyTorch 2.0+ (automatically installed with mditre)

**Ubuntu 24.04 Example**:
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv

# Create virtual environment
python3 -m venv mditre_env
source mditre_env/bin/activate

# Install MDITRE
pip install mditre

# Verify
python -c "import mditre; print('Success!')"
```

**GPU Support (CUDA)**:
```bash
# For CUDA 12.x
pip install mditre
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Path Handling**:
- MDITRE automatically handles Linux paths (`/home/...`)
- Uses native forward slashes
- Automatically detects user home directory

---

## Usage After Installation

### Python Usage

When MDITRE is installed via pip, your data and outputs are in your working directory:

```python
from mditre.models import MDITRE
from mditre.data_loader import DataLoaderRegistry
from mditre.utils.path_utils import get_data_dir, get_output_dir

# Option 1: Use current directory (default)
data_dir = get_data_dir()  # Returns cwd/data/
output_dir = get_output_dir()  # Returns cwd/outputs/

# Option 2: Specify your own paths
data_dir = get_data_dir(base_path='/my/project/data')
output_dir = get_output_dir(base_path='/my/project/results')

# Load your data
loader = DataLoaderRegistry.create_loader('16s_dada2')
data = loader.load(
    data_path=str(data_dir / 'abundance.csv'),
    metadata_path=str(data_dir / 'metadata.csv'),
    tree_path=str(data_dir / 'tree.jplace')
)

# Train model and save results
model = MDITRE(...)
# ... training ...

# Save results
results_path = output_dir / 'my_model_results.pkl'
# ... save ...
```

### R Usage

When MDITRE is installed via install.packages(), your data and outputs are in your working directory:

```r
library(mditre)

# Option 1: Use current directory (default)
data_dir <- get_data_dir()  # Returns getwd()/data/
output_dir <- get_output_dir()  # Returns getwd()/outputs/

# Option 2: Specify your own paths
data_dir <- get_data_dir(base_path = '/my/project/data')
output_dir <- get_output_dir(base_path = '/my/project/results')

# Load your data
data <- load_mditre_data(
  abundance_file = file.path(data_dir, 'abundance.csv'),
  metadata_file = file.path(data_dir, 'metadata.csv'),
  tree_file = file.path(data_dir, 'tree.jplace')
)

# Train model and save results
model <- mditre_model(...)
# ... training ...

# Save results
saveRDS(results, file.path(output_dir, 'my_model_results.rds'))
```

---

## Troubleshooting

### Issue: "Module 'mditre' not found"

**Solution**:
```bash
# Check pip installation
pip show mditre

# Reinstall if needed
pip install --upgrade --force-reinstall mditre
```

### Issue: "Cannot find package 'mditre'" (R)

**Solution**:
```r
# Check installed packages
installed.packages()["mditre",]

# Reinstall if needed
remove.packages("mditre")
install.packages("mditre")
```

### Issue: torch Not Available

**Python**:
```bash
# Install PyTorch separately
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mditre
```

**R**:
```r
# Install torch R package
install.packages("torch")
torch::install_torch()
```

### Issue: Permission Denied During Installation

**Solution**:
```bash
# Python - use user install
pip install --user mditre

# Or use virtual environment (recommended)
python -m venv mditre_env
source mditre_env/bin/activate  # On Windows: mditre_env\Scripts\activate
pip install mditre
```

---

## Verification Checklist

After installation on **any platform**, verify:

### Python Checklist

- [ ] Package imports: `import mditre`
- [ ] Version check: `mditre.__version__`
- [ ] Utils available: `from mditre.utils.path_utils import get_package_root`
- [ ] Models available: `from mditre.models import MDITRE`
- [ ] Data loaders work: `from mditre.data_loader import DataLoaderRegistry`
- [ ] PyTorch detected: `import torch; torch.__version__`
- [ ] Path utilities work: Run verification script

### R Checklist

- [ ] Package loads: `library(mditre)`
- [ ] Functions available: `get_package_root()`
- [ ] Platform info: `get_platform_info()`
- [ ] torch available: `library(torch)`
- [ ] Path utilities work: `print_path_info()`

---

## Development vs Production Modes

### Development Mode (Editable Install)

**Python**:
```bash
git clone https://github.com/melhzy/mditre.git
cd mditre/Python
pip install -e .
```

**Characteristics**:
- `get_project_root()` returns repository root
- Full access to Python/, R/, data/, examples/
- Changes to source code take effect immediately
- Suitable for contributing to MDITRE

**R**:
```r
devtools::load_all("path/to/mditre/R")
```

**Characteristics**:
- `get_project_root()` returns repository root
- Full repository structure available
- Changes take effect after reload
- Suitable for development and testing

### Production Mode (Standard Install)

**Python**:
```bash
pip install mditre
```

**Characteristics**:
- `get_project_root()` returns `None`
- Package installed in site-packages
- Only installed package files available
- Suitable for end users

**R**:
```r
install.packages("mditre")  # or remotes::install_github(...)
```

**Characteristics**:
- `get_project_root()` returns `NULL`
- Package installed in R library
- Only installed package files available
- Suitable for end users

---

## Continuous Integration Testing

MDITRE is tested on all three platforms:

```yaml
# .github/workflows/test.yml example
matrix:
  os: [ubuntu-latest, macos-latest, windows-latest]
  python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
  r-version: ['4.0', '4.1', '4.2', '4.3']
```

See `CROSS_PLATFORM_PATHS.md` for CI/CD integration details.

---

## Summary

✅ **Zero Configuration**: Works immediately after `pip install` or `install.packages()`  
✅ **Cross-Platform**: Identical behavior on Windows, macOS, and Linux  
✅ **No Hardcoded Paths**: Automatically detects installation location  
✅ **User Flexibility**: Specify custom data/output paths when needed  
✅ **Development Friendly**: Supports both production and development modes  

For detailed path handling documentation, see `CROSS_PLATFORM_PATHS.md`.

For API documentation, see `README.md` and inline documentation.
