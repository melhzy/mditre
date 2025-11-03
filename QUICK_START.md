# MDITRE Quick Start Guide

**Version**: 1.0.1 (Verified: November 2, 2025)  
**Test Status**: ✅ 81/81 tests passing across Python, R, and cross-platform verification

**TL;DR**: MDITRE works on Windows, macOS, and Linux without any configuration.

---

## Installation (Any Platform)

### Python
```bash
pip install mditre
```

### R
```r
install.packages("mditre")
```

That's it! No configuration needed.

### Verify Installation

```bash
# Quick verification (< 1 second)
python scripts/verify_cross_platform.py
# Expected: 3/3 tests passed
```

---

## Basic Usage

### Python

```python
from mditre.models import MDITRE
from mditre.data_loader import DataLoaderRegistry
from mditre.utils.path_utils import get_data_dir, get_output_dir

# Your data can be anywhere - just specify the path
data_dir = get_data_dir(base_path='/your/data/location')

# Or use current directory (default)
data_dir = get_data_dir()  # Uses cwd/data/

# Load and train
loader = DataLoaderRegistry.create_loader('16s_dada2')
data = loader.load(
    data_path=str(data_dir / 'abundance.csv'),
    metadata_path=str(data_dir / 'metadata.csv'),
    tree_path=str(data_dir / 'tree.jplace')
)

model = MDITRE(...)
# ... train ...

# Save results anywhere
output_dir = get_output_dir(base_path='/your/results/location')
```

### R

```r
library(mditre)

# Your data can be anywhere - just specify the path
data_dir <- get_data_dir(base_path = '/your/data/location')

# Or use current directory (default)
data_dir <- get_data_dir()  # Uses getwd()/data/

# Load and train
data <- load_mditre_data(
  abundance_file = file.path(data_dir, 'abundance.csv'),
  metadata_file = file.path(data_dir, 'metadata.csv'),
  tree_file = file.path(data_dir, 'tree.jplace')
)

model <- mditre_model(...)
# ... train ...

# Save results anywhere
output_dir <- get_output_dir(base_path = '/your/results/location')
```

---

## Key Points

✅ **Works Everywhere**: Same code runs on Windows, macOS, and Linux  
✅ **Zero Config**: No paths to edit, no environment variables to set  
✅ **User Data**: Specify your own data/output locations  
✅ **Automatic**: Platform detection, path conversion, everything handled  

---

## Platform Details

| Platform | Python Install | R Install | Home Directory |
|----------|---------------|-----------|----------------|
| **Windows** | `pip install mditre` | `install.packages("mditre")` | `C:\Users\<you>` |
| **macOS** | `pip install mditre` | `install.packages("mditre")` | `/Users/<you>` |
| **Linux** | `pip install mditre` | `install.packages("mditre")` | `/home/<you>` |

---

## Documentation

- **Installation**: See `INSTALLATION.md`
- **Path Utilities**: See `CROSS_PLATFORM_PATHS.md`
- **Full Compliance**: See `CROSS_PLATFORM_COMPLIANCE.md`
- **API Reference**: See `README.md`

---

## Questions?

**Q: Do I need to configure anything?**  
A: No. Just install and use.

**Q: Can I use my own data directory?**  
A: Yes. Use `get_data_dir(base_path='/your/path')`.

**Q: Does it work on my OS?**  
A: Yes. Windows, macOS, and Linux are all supported.

**Q: Do I need to edit paths in my code for different platforms?**  
A: No. Same code works everywhere.

**Q: What if I clone the repository for development?**  
A: Use `pip install -e .` and everything still works automatically.

---

**That's all you need to know!** MDITRE is designed to "just work" on any platform.
