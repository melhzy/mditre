# MDITRE Utility Scripts

This directory contains utility scripts for testing, validation, and development of the MDITRE package.

## Available Scripts

### 1. `verify_cross_platform.py` ⭐ **Recommended**

**Purpose**: Verifies that MDITRE works correctly across Windows, macOS, and Linux.

**Use Cases**:
- ✅ **After Installation**: Quick smoke test to verify everything works
- ✅ **CI/CD**: Automated cross-platform validation in GitHub Actions
- ✅ **Development**: Check that path utilities work correctly
- ✅ **Debugging**: First-line diagnostic tool for installation issues

**Usage**:
```bash
# Run from repository root
python scripts/verify_cross_platform.py
```

**Expected Output**:
```
======================================================================
MDITRE Cross-Platform Verification Suite
======================================================================
...
Results: 3/3 tests passed
🎉 All verification tests passed! Cross-platform support is working.
```

**Tests Performed**:
1. Package imports and version detection
2. Cross-platform path utilities (Windows/macOS/Linux)
3. Validation that examples don't contain hardcoded paths

**Run Time**: < 1 second

---

### 2. `test_mditre_python.py`

**Purpose**: Comprehensive test of the Python MDITRE package functionality.

**Use Cases**:
- ✅ **Development Testing**: Manual verification during development
- ✅ **GPU Verification**: Tests CUDA availability and GPU support
- ✅ **Package Validation**: Checks all modules and functionality
- ✅ **Integration Testing**: End-to-end package verification

**Usage**:
```bash
# Run from repository root
python scripts/test_mditre_python.py
```

**Tests Performed**:
1. Environment setup (Python, PyTorch, CUDA)
2. Package import and version
3. Available modules
4. Data loading and processing
5. Model training and prediction
6. GPU acceleration (if available)

**Run Time**: Variable (depends on data and GPU availability)

---

## Quick Reference

| Script | Purpose | When to Use | Run Time |
|--------|---------|-------------|----------|
| `verify_cross_platform.py` | Cross-platform validation | After install, CI/CD, debugging | < 1s |
| `test_mditre_python.py` | Comprehensive package test | Development, GPU testing | Variable |

---

## For Developers

### Running Scripts in Development Mode

Both scripts automatically handle development mode (running from repository root):

```bash
# From repository root
cd /path/to/mditre
python scripts/verify_cross_platform.py
python scripts/test_mditre_python.py
```

### Running Scripts After Installation

After installing via pip, only `verify_cross_platform.py` is designed to work:

```bash
pip install mditre
python -c "from mditre.utils.path_utils import get_platform_info; print(get_platform_info())"
```

---

## CI/CD Integration

To use `verify_cross_platform.py` in GitHub Actions:

```yaml
- name: Verify cross-platform support
  run: python scripts/verify_cross_platform.py
```

---

## Troubleshooting

### Import Errors

If you see import errors when running scripts:

1. **Check you're in the repository root**:
   ```bash
   pwd  # Should show .../mditre
   ls   # Should show Python/, R/, scripts/, etc.
   ```

2. **Verify Python path**:
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

3. **Check MDITRE installation**:
   ```bash
   python -c "import mditre; print(mditre.__version__)"
   ```

### Version Mismatch

If version reported doesn't match expected (should be 1.0.1):

1. Reinstall the package:
   ```bash
   cd Python
   pip install -e .
   ```

2. Clear Python cache:
   ```bash
   find . -type d -name "__pycache__" -exec rm -rf {} +
   rm -rf .pytest_cache
   ```

---

## Adding New Scripts

When adding new utility scripts to this directory:

1. **Follow naming convention**: Use descriptive names with underscores
   - ✅ Good: `validate_data_format.py`, `benchmark_performance.py`
   - ❌ Bad: `test.py`, `script1.py`

2. **Include docstring**: Add clear documentation at the top
   ```python
   """
   Script Name and Purpose
   
   Detailed description of what the script does.
   
   Usage:
       python scripts/your_script.py
   """
   ```

3. **Update this README**: Add entry in the "Available Scripts" section

4. **Make executable** (optional):
   ```bash
   chmod +x scripts/your_script.py
   ```

---

## Related Documentation

- **Unit Tests**: See `Python/tests/` for pytest-based unit tests
- **R Tests**: See `R/tests/` for testthat-based R tests
- **CI/CD**: See `.github/workflows/` for automated testing
- **Contributing**: See `CONTRIBUTING.md` for development guidelines

---

**Last Updated**: November 2025  
**MDITRE Version**: 1.0.1
