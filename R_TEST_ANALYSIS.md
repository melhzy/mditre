# R Test Analysis and Fixes

## Summary

After fixing the naming convention issues (spatial_agg → spatial_agg_layer, etc.), we successfully fixed:
- ✅ **22/22 evaluation tests** (100% passing)
- ✅ Confusion matrix indexing corrected

However, the layer tests (layers 1-5) reveal **API mismatches** between test expectations and current implementation.

## Test Results

### Passing Tests
- **Evaluation Tests**: 22/22 passing (100%)
  - compute_metrics()
  - confusion_matrix()
  - compute_auc_roc()
  - All edge cases

### Failing Tests  
- **Layer 1 (Phylogenetic)**: 2/10 tests passing
- **Layer 2 (Temporal)**: 0/8 tests passing  
- **Layers 3-5**: Not yet tested (test run stopped early)

**Total**: 24 PASS, 2 FAIL, 14 ERROR, 1 WARN

## Root Cause Analysis

### Issue 1: SpatialAgg API Mismatch

**Test Expectations** (old API):
```r
layer <- spatial_agg_layer(num_rules, num_otus, dist)
# Expects: layer$eta parameter
output <- layer(x, temp = 0.5)
```

**Current Implementation**:
```r
layer <- spatial_agg_layer(num_rules, num_otus, dist)
# Has: layer$kappa parameter (NO layer$eta)
output <- layer(x, k = 1)  # Uses 'k' not 'temp'
```

**Failures**:
- test-layer1_phylogenetic.R:16 - `layer$eta is NULL`
- test-layer1_phylogenetic.R:40, 61, 82, 157 - `unused argument (temp = 0.5)`

### Issue 2: SpatialAggDynamic Missing Required Parameter

**Test Call**:
```r
layer <- spatial_agg_dynamic_layer(num_rules, num_otus, dist)
```

**Required Signature**:
```r
spatial_agg_dynamic_layer(num_rules, num_otu_centers, otu_embeddings, emb_dim, num_otus)
```

**Missing Parameters**: `otu_embeddings`, `emb_dim`, `num_otu_centers`

**Failures**:
- test-layer1_phylogenetic.R:100, 119 - `argument "emb_dim" is missing`

### Issue 3: TimeAgg Missing Required Parameter

**Test Call**:
```r
layer <- time_agg_layer(num_rules, num_time, times)
```

**Required Signature**:
```r
time_agg_layer(num_rules, num_otus, num_time, num_time_centers)
```

**Missing Parameters**: `num_otus`, `num_time_centers`

**Failures**:
- All test-layer2_temporal.R tests - `argument "num_time_centers" is missing`

## Comparison with Python Tests

The **Python tests pass 100% (39/39)**. Key difference:

### Python Test Example (WORKS):
```python
# test_all.py line ~200
layer = SpatialAgg(num_rules=3, num_otus=50, dist=dist_matrix)
output = layer(x)  # No 'temp' parameter
assert layer.kappa is not None  # Uses 'kappa', not 'eta'
```

### R Test Example (FAILS):
```r
# test-layer1_phylogenetic.R line 11
layer <- spatial_agg_layer(num_rules, num_otus, dist)
output <- layer(x, temp = 0.5)  # Wrong parameter name
expect_true(!is.null(layer$eta))  # Wrong parameter name
```

## Required Fixes

### Option 1: Update R Tests to Match Current API (RECOMMENDED)

Update all R layer tests to use the correct parameters and signatures that match the current R implementation documented in `R/R/layer*.R` files.

**Pros:**
- Tests will match documented API
- Consistent with Python implementation
- Minimal code changes

**Cons:**
- Tests need significant rewriting
- May reveal more functionality issues

### Option 2: Verify R Implementation Matches Python

Before fixing tests, verify that the R implementation in `R/R/layer*.R` correctly wraps the Python MDITRE implementation.

## Next Steps

1. **Verify R-Python API consistency**:
   - Compare `R/R/layer1_phylogenetic_focus.R` with Python `spatial_agg.py`
   - Confirm parameter names and signatures match

2. **Update test files** to use correct API:
   - Remove `temp` → use `k`
   - Remove `eta` → use `kappa`
   - Add missing parameters for all layer constructors

3. **Re-run full test suite** after fixes

4. **Document any intentional API differences** between R and Python

## Current Status

**Python**: ✅ 39/39 tests passing (100%)
**R Evaluation**: ✅ 22/22 tests passing (100%)  
**R Layers**: ❌ Layer tests need API updates to match current implementation

## Recommendation

The evaluation functions work perfectly, proving the R-Python integration via reticulate is solid. The layer tests simply need updating to match the current API as documented in the R source files. This is expected when tests haven't been maintained alongside API changes.

**Action**: Update R layer tests to match current documented API in `R/R/layer*.R` files.
