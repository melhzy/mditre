# Google Colab Setup Guide

**Last Updated**: November 2025  
**Repository**: https://github.com/melhzy/mditre

---

## Quick Start

All MDITRE Jupyter notebooks are fully compatible with Google Colab. Each tutorial notebook includes setup cells that automatically:
- Installs the MDITRE package from the Python subdirectory
- Downloads required dataset files from GitHub
- Configures the appropriate matplotlib backend

**For tutorials: Run cells #2 and #3 to get started!**
- Cell #2: Package installation and environment setup
- Cell #3: Automatic dataset download (tutorials only)

---

## Compatibility Status

| Notebook | Local Jupyter | Google Colab | Notes |
|----------|---------------|--------------|-------|
| run_mditre_test.ipynb | ✅ | ✅ | Works on CPU/GPU |
| Tutorial_1_16s_data.ipynb | ✅ | ✅ | Full compatibility |
| Tutorial_2_metaphlan_data.ipynb | ✅ | ✅ | Full compatibility |
| Tutorial_Bokulich_16S_data.ipynb | ✅ | ✅ | Full compatibility |

---

## What Was Fixed

### Issue 1: `%matplotlib qt` Incompatibility ❌→✅

**Problem:** The Qt backend opens matplotlib in a separate window, which Google Colab does not support.

**Solution:** Conditional backend selection based on environment detection:

```python
# Check if running in Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("📊 Running in Google Colab - using inline matplotlib")
except ImportError:
    IN_COLAB = False
    print("💻 Running locally - using Qt matplotlib for interactive GUI")

# Set appropriate matplotlib backend
if IN_COLAB:
    %matplotlib inline  # Static images in Colab
else:
    %matplotlib qt      # Interactive GUI locally
```

**Result:** Visualizations work in both environments.

### Issue 2: Package Installation

**Problem:** MDITRE package needs to be installed before use. The Python package is located in the `Python/` subdirectory of the repository.

**Solution:** Automatic installation in Colab setup cell with subdirectory specification:

```python
# Install MDITRE package from Python subdirectory
print("📦 Installing MDITRE package...")
!pip install -q git+https://github.com/melhzy/mditre.git#subdirectory=Python
print("✅ MDITRE installed\n")
```

### Issue 3: File Access

**Problem:** Data files located in local directory structure won't be accessible in Colab.

**Solution:** Automatic dataset download from GitHub:

All tutorial notebooks now include an automatic dataset download cell (right after the Colab setup cell) that downloads the required data files directly from GitHub. This eliminates the need to manually upload files or configure Google Drive paths.

```python
# Download dataset files (for Colab users)
if IN_COLAB:
    import os
    import urllib.request
    
    # Create directory structure and download files
    os.makedirs('./datasets/raw/david', exist_ok=True)
    base_url = "https://raw.githubusercontent.com/melhzy/mditre/master/Python/jupyter/tutorials/datasets/raw/david/"
    
    # Downloads all required files automatically
    for filename in files:
        urllib.request.urlretrieve(base_url + filename, f'./datasets/raw/david/{filename}')
```

**Alternative:** Mount Google Drive if you prefer to use your own data files:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Colab Setup Cell (Template)

This cell is automatically included as **Cell #2** in all tutorial notebooks:

```python
# ============================================
# 🌐 GOOGLE COLAB SETUP (Run this cell first!)
# ============================================

try:
    import google.colab
    IN_COLAB = True
    print("✅ Running in Google Colab\n")
    
    # Install MDITRE package from Python subdirectory
    print("📦 Installing MDITRE package...")
    !pip install -q git+https://github.com/melhzy/mditre.git#subdirectory=Python
    print("✅ MDITRE installed\n")
    
    # Mount Google Drive
    print("💾 Mounting Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Drive mounted\n")
    
    print("📁 Please ensure your data files are in Google Drive at:")
    print("   /content/drive/MyDrive/mditre_data/")
    
except ImportError:
    IN_COLAB = False
    print("✅ Running locally - skipping Colab setup")
```

---

## GPU Setup (for run_mditre_test.ipynb)

### Enabling GPU in Colab

1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU (T4)**
3. Click **Save**
4. Runtime will reconnect with GPU enabled

### GPU Detection Code

The notebook automatically detects and reports GPU status:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🔧 Device: {device}")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    if IN_COLAB:
        print("\n⚠️  GPU NOT DETECTED in Colab!")
        print("To enable GPU:")
        print("  1. Runtime → Change runtime type")
        print("  2. Hardware accelerator → GPU (T4)")
        print("  3. Save and reconnect")
        print("\nContinuing with CPU (training will be slower)...")
```

---

## Feature Comparison: Local vs Colab

| Feature | Local Jupyter | Google Colab |
|---------|---------------|--------------|
| **Package Installation** | Manual (`pip install`) | Automatic (in setup cell) |
| **Matplotlib Backend** | Qt (interactive GUI) | Inline (static images) |
| **RuleVisualizer** | Full interactivity | Static visualization |
| **Data Files** | Local filesystem | Google Drive mount |
| **GPU Access** | If available locally | Free T4 GPU |
| **Setup Time** | None (if installed) | ~30 seconds (first run) |

---

## Known Limitations in Colab

### 1. Interactive GUI Features ⚠️

The `RuleVisualizer` utility provides an interactive GUI locally but displays static images in Colab.

**Local Features:**
- Click to explore rules
- Dynamic filtering
- Interactive threshold adjustment

**Colab Features:**
- Static visualization
- All information visible
- Download images for local viewing

**Workaround:** Run visualization sections locally for full interactivity, use Colab for training and preprocessing.

### 2. File Path Management

Data files must be uploaded to Google Drive or mounted from a shared location.

**Recommended structure:**
```
/content/drive/MyDrive/mditre_data/
├── david/
│   ├── abundance.csv
│   ├── sample_metadata.csv
│   └── ...
├── kostic/
└── bokulich/
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'mditre'"

**Solution:** Run the Colab setup cell (Cell #2) to install the package.

### "Drive mount failed"

**Solution:** 
1. Check your Google account permissions
2. Click the link provided by Colab to authorize Drive access
3. Re-run the setup cell

### "FileNotFoundError: No such file or directory"

**Solution:** 
1. Verify data files are uploaded to Google Drive
2. Update file paths to match your Drive structure:
   ```python
   filename = '/content/drive/MyDrive/mditre_data/david/david_benchmark.cfg'
   ```

### Visualizations not displaying

**Solution:** 
1. Ensure you're using `%matplotlib inline` in Colab
2. Check that the conditional backend cell was executed
3. Re-run visualization cells

---

## Backward Compatibility

✅ **100% backward compatible with local Jupyter**

- Local users: Setup cells auto-detect local environment and skip Colab-specific steps
- Qt backend: Still used for interactive GUI in local Jupyter
- No breaking changes to existing workflows

---

## Testing Checklist

Before running notebooks:

**In Google Colab:**
- [ ] Run Colab setup cell (Cell #2)
- [ ] Verify package installation completes
- [ ] Check Drive mount succeeded
- [ ] Enable GPU runtime (for run_mditre_test.ipynb)
- [ ] Confirm visualizations display inline

**In Local Jupyter:**
- [ ] Verify setup cells are skipped automatically
- [ ] Confirm Qt backend works for interactive GUI
- [ ] Check all original functionality preserved

---

## Quick Links

- **Main Repository:** https://github.com/melhzy/mditre
- **Open in Colab:** See badges in [README.md](README.md) or [COLAB_BADGES.md](COLAB_BADGES.md)
- **Issues:** https://github.com/melhzy/mditre/issues

---

## Support

For questions or issues:
1. Check this guide first
2. Review [COLAB_BADGES.md](COLAB_BADGES.md) for direct Colab links
3. Open an issue on GitHub with:
   - Notebook name
   - Error message
   - Environment (Colab or local)
   - Steps to reproduce
