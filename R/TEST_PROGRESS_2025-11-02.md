# R MDITRE Test Suite Progress Report
**Date**: November 2, 2025  
**Status**: 37/39 tests passing (94.9%)  
**Session**: Test synchronization and fixing phase

---

## Current Status

### Test Results Summary
```
Total Tests:  39
Passed:       37 (94.9%)
Failed:       2

Failing Tests:
- Test 21: Slope computation correctness
- Test 31: Reproducible forward pass
```

### Test Coverage by Section
- ✅ Section 1.1: Five-layer architecture (8/8 tests)
- ✅ Section 1.2: Differentiability & gradients (3/3 tests)
- ✅ Section 1.3: Model variants (2/2 tests)
- ✅ Section 2: Phylogenetic focus (4/4 tests)
- ⚠️  Section 3: Temporal focus (3/4 tests) - **Test 21 failing**
- ✅ Section 10: Performance metrics (3/3 tests)
- ✅ Section 11: Training pipeline (1/1 test)
- ✅ Section 12: PyTorch integration (3/3 tests)
- ⚠️  Section 13: Seeding & reproducibility (4/5 tests) - **Test 31 failing**
- ✅ Section 14: Package integrity (6/6 tests)

---

## Recent Accomplishments

### 1. Test Synchronization ✅
- Added Test 28: Complete Training Pipeline (was missing in R)
- Renumbered tests 29-39 to match Python structure
- R MDITRE now has all 39 tests matching Python MDITRE

### 2. Major Fixes Applied ✅
- **Seeding Range Fix**: Changed `max_value` from 2147483647 to 2147483646 in `R/R/seeding.R`
  - Reason: seedhash validates (max - min) < 2147483647
  
- **Accuracy Test Fix**: Changed expected value from 0.75 to 0.875 in Test 24
  - Manually verified: 7/8 correct predictions = 0.875
  
- **Tensor Dimension Fixes**: Multiple tests (17, 19, 20)
  - TimeAgg expects 4D input: (batch, num_rules, num_otus, num_time)
  - Fixed shape comparison method: Use `dim(py_to_r(tensor$cpu()$numpy()))` instead of `output$shape`
  
- **Model Initialization**: Tests 28, 31
  - Added all 12 required parameters for complete model initialization
  - Fixed DenseLayer weight dimensions: `w_init` = (1, num_rules), `bias_init` = (1,)

### 3. Progress Trajectory
- Started session: 29/38 tests (76.3%)
- After Test 28 added: 29/39 tests (74.4%)
- After seeding fix: 33/39 tests (84.6%)
- After dimension fixes: 36/39 tests (92.3%)
- **Current**: 37/39 tests (94.9%)

---

## Remaining Issues

### Test 21: Slope Computation Correctness
**Error**: `IndexError: index 10 is out of bounds for dimension 3 with size 10`

**Root Cause**: Mixed R/Python indexing in tensor assignment loop

**Current Code** (lines ~740-770):
```r
# Create linearly increasing signal: X[:, :, :, t] = t
# Create the tensor directly with numpy then convert
X_np <- array(0, dim = c(5, test_config$num_rules, test_config$num_otu_centers, test_config$num_time))
for (t in 0:(test_config$num_time - 1)) {
  X_np[, , , t + 1] <- t  # R is 1-based for array indexing
}
X <- torch_py$from_numpy(X_np)$float()$to(device)
```

**Status**: Fix applied but not yet tested

**Python Reference** (test_all.py lines 730-735):
```python
X = torch.zeros(5, num_rules, num_otus, num_time, device=device)
for t in range(num_time):
    X[:, :, :, t] = float(t)

mask = torch.ones(5, num_time, device=device)
```

**Next Action**: Run tests to verify the numpy array approach works

---

### Test 31: Reproducible Forward Pass
**Error**: `diff < 1e-04 is not TRUE`

**Root Cause**: `rnorm()` used in `eta_init` generation wasn't seeded for R's RNG

**Fix Applied** (lines ~1070-1120):
```r
# Seed both R and PyTorch RNGs for full reproducibility
set.seed(test_seed)
torch_py$manual_seed(as.integer(test_seed))
model1 <- mditre_models$MDITRE(...)$to(device)

# Re-seed for second model to get same initialization
set.seed(test_seed)
torch_py$manual_seed(as.integer(test_seed))
model2 <- mditre_models$MDITRE(...)$to(device)

# Initialize both models with same parameters (using seeded rnorm)
set.seed(test_seed)
init_args <- list(
  ...
  eta_init = np$array(array(rnorm(...), dim = c(...))),
  ...
)
```

**Status**: Fix applied but not yet tested

**Key Insight**: Need to seed BOTH R's RNG (`set.seed()`) and PyTorch's RNG (`torch_py$manual_seed()`) for full reproducibility when using R functions like `rnorm()` in initialization

**Next Action**: Run tests to verify reproducibility

---

## Technical Details

### Model Initialization Parameters (All 12 Required)
```r
init_args <- list(
  # Spatial Aggregation
  kappa_init = ...,      # Phylogenetic distance weights (num_rules, num_otu_centers)
  eta_init = ...,        # Embeddings (num_rules, num_otu_centers, emb_dim)
  
  # Temporal Aggregation
  abun_a_init = ...,     # Abundance window start (num_rules, num_otu_centers)
  abun_b_init = ...,     # Abundance window end (num_rules, num_otu_centers)
  slope_a_init = ...,    # Slope window start (num_rules, num_otu_centers)
  slope_b_init = ...,    # Slope window end (num_rules, num_otu_centers)
  
  # Detectors
  thresh_init = ...,     # Threshold values (num_rules, num_otu_centers)
  slope_init = ...,      # Slope detector values (num_rules, num_otu_centers)
  
  # Rules Layer
  alpha_init = ...,      # Soft AND parameter (num_rules, num_otu_centers)
  
  # Dense Layer
  w_init = ...,          # Weight matrix (1, num_rules) - IMPORTANT: (out_feat, in_feat)
  bias_init = ...,       # Bias vector (1,) - IMPORTANT: Single value
  beta_init = ...        # Rule selection parameter (num_rules,)
)
```

### Critical Dimension Requirements
- **TimeAgg Input**: (batch, num_rules, num_otus, num_time) - 4D tensor
- **DenseLayer Weight**: (out_feat, in_feat) = (1, num_rules)
- **DenseLayer Bias**: (out_feat,) = (1,) - single value, not per-rule

### Seeding Architecture
- **Python MDITRE**: Uses Python seedhash from github.com/melhzy/seedhash/tree/main/Python
- **R MDITRE**: Uses R seedhash from github.com/melhzy/seedhash/tree/main/R
- **Both versions**: Provide `SeedHashGenerator` class for deterministic seeding
- **R Implementation**: `R/R/seeding.R` with `max_value = 2147483646` (not 2147483647)

---

## Next Steps (Priority Order)

### Immediate Actions
1. **Run Tests**: Execute `Rscript run_mditre_tests.R` to verify Test 21 and Test 31 fixes
   ```powershell
   cd D:\Github\mditre\R
   & "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" run_mditre_tests.R
   ```

2. **If Tests Pass (39/39)**: ✅ SUCCESS!
   - Document final results
   - Update README if needed
   - Consider running tests multiple times to ensure stability

3. **If Test 21 Still Fails**:
   - Verify the numpy array approach creates correct dimensions
   - Check if mask parameter is being passed correctly
   - Compare with Python test implementation more carefully
   - Consider printing intermediate values to debug

4. **If Test 31 Still Fails**:
   - Check if `set.seed()` is actually affecting `rnorm()` calls
   - Verify both models receive identical initialization
   - Try using fixed values instead of `rnorm()` for `eta_init`
   - Compare actual tensor differences to understand magnitude

### Alternative Approaches (If Current Fixes Don't Work)

**For Test 21**:
- Option A: Use `torch_py$zeros()` and fill via Python-side loop
- Option B: Create full 4D numpy array at once using broadcasting
- Option C: Use simpler test with fixed values instead of loop

**For Test 31**:
- Option A: Replace `rnorm()` with fixed values for reproducibility test
- Option B: Pre-generate random values once, reuse for both models
- Option C: Use `np$random$seed()` for numpy-based random generation

---

## Files Modified in This Session

1. **R/R/seeding.R**
   - Changed `max_value` from 2147483647 to 2147483646

2. **R/run_mditre_tests.R** (Major updates throughout)
   - Added Test 28: Complete Training Pipeline (~lines 910-1000)
   - Renumbered tests 29-39 accordingly
   - Fixed Test 17: Tensor creation and shape comparison
   - Fixed Tests 19-20: TimeAgg input dimensions
   - Fixed Test 21: Linearly increasing signal creation (in progress)
   - Fixed Test 24: Accuracy expectation (0.75 → 0.875)
   - Fixed Test 28: Complete model initialization with 12 parameters
   - Fixed Test 31: Added R RNG seeding for reproducibility (in progress)
   - Updated shape comparison throughout: `dim(py_to_r(tensor$cpu()$numpy()))`

---

## Test Execution Command

```powershell
# Navigate to R directory
cd D:\Github\mditre\R

# Run full test suite
& "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" run_mditre_tests.R
```

**Expected Output** (if fixes work):
```
Total Tests:  39
Passed:       39 (100%)
Failed:       0

✓ 39/39 tests passed
```

---

## Environment Details

- **OS**: Windows
- **Shell**: PowerShell
- **R Version**: 4.5.2
- **Python**: 3.12 (MDITRE conda environment)
- **PyTorch**: 2.6.0+cu124
- **CUDA**: Available
- **GPU**: NVIDIA GeForce RTX 4090 Laptop GPU
- **Bridge**: reticulate

---

## Key Learnings

1. **R-Python Indexing**: R is 1-based, Python/PyTorch is 0-based. When creating numpy arrays in R for PyTorch, use R indexing; when slicing PyTorch tensors, remember Python indexing.

2. **Dual RNG Seeding**: When using R functions (like `rnorm()`) alongside PyTorch operations, must seed both:
   - `set.seed()` for R's RNG
   - `torch_py$manual_seed()` for PyTorch's RNG

3. **Shape Inspection**: To get tensor dimensions in R, use `dim(py_to_r(tensor$cpu()$numpy()))` rather than trying to access `tensor$shape` directly.

4. **Model Initialization**: MDITRE requires all 12 parameters for proper initialization. Missing any parameter causes KeyError or NaN propagation.

5. **Dimension Constraints**: 
   - DenseLayer expects weight shape (out_feat, in_feat), not (in_feat, out_feat)
   - TimeAgg processes 4D tensors after spatial aggregation adds num_rules dimension

---

## Contact Points for Issues

- **Seeding Issues**: Check `R/R/seeding.R` and seedhash package installation
- **Model Initialization**: See "Model Initialization Parameters" section above
- **Tensor Dimensions**: Refer to "Critical Dimension Requirements" section
- **Test Failures**: Full test output saved in terminal history

---

**Resume Point**: Run test suite and verify if Tests 21 and 31 now pass with the applied fixes.
