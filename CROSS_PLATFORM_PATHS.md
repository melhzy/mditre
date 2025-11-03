# MDITRE Cross-Platform Path Handling Guide

**Version**: 1.0.1  
**Date**: November 2, 2025  
**Status**: ✅ Implemented

---

## Overview

MDITRE now implements **comprehensive cross-platform path handling** that automatically adapts to Windows, macOS, and Linux/Ubuntu. All file paths are handled in a platform-independent manner, ensuring the project works seamlessly across different operating systems.

## Key Principles

### 1. **Automatic OS Detection**
MDITRE automatically detects the operating system and uses appropriate path conventions:
- **Windows**: Supports both `C:\Users\...` and `C:/Users/...` formats
- **macOS**: Uses `/Users/...` Unix-style paths
- **Linux/Ubuntu**: Uses `/home/...` Unix-style paths

### 2. **No Hardcoded Paths**
All absolute paths are computed dynamically based on:
- Project root detection
- User home directory (`~`)
- Platform-specific conventions

### 3. **Forward Slash Normalization**
Internally, MDITRE uses forward slashes (`/`) for consistency, converting to platform-specific separators only when needed.

---

## Implementation Details

### Python Implementation

#### Path Utilities Module
Location: `Python/mditre/utils/path_utils.py`

**Key Functions**:
```python
from mditre.utils.path_utils import (
    get_project_root,     # Auto-detect project root
    get_python_dir,       # Get Python/ directory
    get_r_dir,           # Get R/ directory
    get_data_dir,        # Get data directory
    get_output_dir,      # Get outputs directory
    normalize_path,      # Platform-independent normalization
    ensure_dir_exists,   # Create directories safely
    join_paths,          # Cross-platform path joining
    to_unix_path,        # Convert to Unix-style
    to_platform_path,    # Convert to platform-specific
    get_platform_info    # Get OS information
)
```

**Example Usage**:
```python
from pathlib import Path
from mditre.utils.path_utils import get_project_root, get_data_dir

# Automatically works on all platforms
project_root = get_project_root()
data_file = get_data_dir('raw') / 'abundance.csv'

# Path objects work seamlessly
with open(data_file, 'r') as f:
    data = f.read()
```

#### Updated Files
1. **`Python/mditre/examples/data_loader_example.py`**
   - Changed from: `sys.path.insert(0, 'd:/Github/mditre')`
   - Changed to: Dynamic project root detection using `Path(__file__)`

2. **All Python examples** now use:
   ```python
   from pathlib import Path
   current_file = Path(__file__).resolve()
   project_root = current_file.parent.parent  # Adjust levels as needed
   ```

---

### R Implementation

#### Path Utilities Module
Location: `R/R/path_utils.R`

**Key Functions**:
```r
library(rmditre)

get_project_root()        # Auto-detect project root
get_python_dir()          # Get Python/ directory
get_r_dir()              # Get R/ directory
get_data_dir()           # Get data directory
get_output_dir()         # Get outputs directory
normalize_path()         # Platform-independent normalization
ensure_dir_exists()      # Create directories safely
join_paths()             # Cross-platform path joining
to_unix_path()           # Convert to Unix-style
to_platform_path()       # Convert to platform-specific
get_platform_info()      # Get OS information
print_path_info()        # Print diagnostic info
```

**Example Usage**:
```r
# Automatically works on Windows, macOS, and Linux
project_root <- get_project_root()
data_file <- file.path(get_data_dir('raw'), 'abundance.csv')

# Read data - platform-independent
data <- read.csv(data_file)

# Diagnostic information
print_path_info()
```

#### Updated Files
1. **`R/.Renviron`**
   - Removed hardcoded Windows-specific paths
   - Added documentation about platform-specific configuration
   - Torch paths now handled automatically by torch package

2. **`R/R/zzz.R`** (already cross-platform)
   - Uses `normalizePath(path, winslash = "/")` 
   - Dynamic Python directory detection
   - Platform-independent conda/pip commands

---

## Platform-Specific Behaviors

### Windows
```r
# Example outputs on Windows
get_platform_info()
# $os: "Windows"
# $sep: "\\"
# $home: "C:/Users/username"

get_project_root()
# "C:/Github/mditre"  # Forward slashes internally

to_platform_path("data/raw/file.txt")
# "data\\raw\\file.txt"  # Windows separators for system calls
```

### macOS
```r
# Example outputs on macOS
get_platform_info()
# $os: "Darwin"
# $sep: "/"
# $home: "/Users/username"

get_project_root()
# "/Users/username/mditre"

to_platform_path("data/raw/file.txt")
# "data/raw/file.txt"  # Unix separators
```

### Linux/Ubuntu
```r
# Example outputs on Linux
get_platform_info()
# $os: "Linux"
# $sep: "/"
# $home: "/home/username"

get_project_root()
# "/home/username/mditre"

to_platform_path("data/raw/file.txt")
# "data/raw/file.txt"  # Unix separators
```

---

## Best Practices for Contributors

### 1. **Never Hardcode Absolute Paths**
❌ **Bad**:
```python
data_path = "C:/Users/huang/data/file.csv"
data_path = "/home/john/projects/mditre/data/file.csv"
```

✅ **Good**:
```python
from mditre.utils.path_utils import get_data_dir
from pathlib import Path
data_path = get_data_dir() / "file.csv"
```

### 2. **Use Path Objects in Python**
❌ **Bad**:
```python
import os
path = os.path.join("data", "raw", "file.csv")
path = path.replace("\\", "/")  # Manual conversion
```

✅ **Good**:
```python
from pathlib import Path
path = Path("data") / "raw" / "file.csv"
# Automatically handles platform differences
```

### 3. **Use file.path() in R**
❌ **Bad**:
```r
path <- paste("data", "raw", "file.csv", sep = "/")
path <- "C:/Users/data/file.csv"  # Hardcoded
```

✅ **Good**:
```r
path <- file.path("data", "raw", "file.csv")
path <- file.path(get_data_dir("raw"), "file.csv")
```

### 4. **Normalize Paths for Display**
```python
# Python
from mditre.utils.path_utils import to_unix_path
display_path = to_unix_path(path)  # Always use / for display
print(f"Loading data from: {display_path}")
```

```r
# R
display_path <- to_unix_path(path)
cat("Loading data from:", display_path, "\n")
```

### 5. **Test on Multiple Platforms**
Before committing, verify your code works on:
- ✅ Windows (if available)
- ✅ macOS (if available)
- ✅ Linux/Ubuntu (GitHub Actions, Docker, or VM)

---

## Environment Variables

### Python
No special environment variables needed. Uses:
- `__file__` for current script location
- `Path.home()` for user home directory
- `Path.cwd()` for current working directory

### R
MDITRE automatically handles torch paths. If custom configuration needed:

```r
# Optional: Set custom torch home (cross-platform)
torch_home <- normalizePath(
  path.expand("~/custom/torch"), 
  winslash = "/", 
  mustWork = FALSE
)
Sys.setenv(TORCH_HOME = torch_home)
```

---

## Migrating Existing Code

### Step 1: Identify Hardcoded Paths
```bash
# Search for hardcoded paths
grep -r "C:\\Users" .
grep -r "/home/" .
grep -r "D:\\Github" .
```

### Step 2: Replace with Dynamic Paths

**Python**:
```python
# Before
sys.path.insert(0, "D:/Github/mditre")

# After
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**R**:
```r
# Before
setwd("d:/Github/mditre/R")

# After
setwd(get_r_dir())
```

### Step 3: Use Path Utilities

**Python**:
```python
# Before
data_path = "d:/Github/mditre/data/raw/file.csv"

# After
from mditre.utils.path_utils import get_data_dir
data_path = get_data_dir("raw") / "file.csv"
```

**R**:
```r
# Before
data_path <- "d:/Github/mditre/data/raw/file.csv"

# After
data_path <- file.path(get_data_dir("raw"), "file.csv")
```

---

## Testing Cross-Platform Paths

### Python
```python
# Run path diagnostics
python -m mditre.utils.path_utils

# Expected output shows:
# - Platform information
# - Project paths
# - Path conversion examples
```

### R
```r
# Run path diagnostics
library(rmditre)
print_path_info()

# Expected output shows:
# - Platform: Windows/Darwin/Linux
# - Project root directory
# - Python and R directories
# - Path conversion examples
```

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Cross-Platform Tests

on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: [3.8, 3.9, 3.10, 3.11, 3.12]
        
    runs-on: ${{ matrix.os }}
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        pip install -e Python/
        
    - name: Test cross-platform paths
      run: |
        python -m mditre.utils.path_utils
        pytest Python/tests/ -v
```

---

## Troubleshooting

### Issue: "Path not found"
**Solution**: Use dynamic path detection instead of hardcoded paths
```python
from mditre.utils.path_utils import get_project_root
root = get_project_root()  # Always correct for current system
```

### Issue: "Permission denied on path"
**Solution**: Check path exists and is writable
```python
from mditre.utils.path_utils import ensure_dir_exists
output_dir = ensure_dir_exists("outputs/models")  # Creates if needed
```

### Issue: "Torch libraries not found" (R)
**Solution**: Let torch package handle paths automatically
```r
# Don't set TORCH_HOME manually
# Remove hardcoded paths from .Renviron
# Torch package will configure automatically
```

---

## Summary

✅ **Python**: All paths use `pathlib.Path` and `path_utils` module  
✅ **R**: All paths use `file.path()` and `path_utils.R` functions  
✅ **No hardcoded paths** in production code  
✅ **Automatic OS detection** and path adaptation  
✅ **Comprehensive utilities** for path manipulation  
✅ **Backward compatible** with existing code using relative paths  

MDITRE is now **fully cross-platform** and works identically on Windows, macOS, and Linux! 🎉

---

**For Questions**: See `Python/mditre/utils/path_utils.py` and `R/R/path_utils.R` for implementation details.
