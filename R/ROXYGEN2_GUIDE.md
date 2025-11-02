# MDITRE R Package - roxygen2 Documentation Guide

**Status**: Documentation framework complete, ready for generation  
**Date**: 2024  
**Phase**: Documentation (roxygen2)

---

## Overview

All MDITRE R functions have roxygen2 documentation tags. This guide explains how to generate the documentation files (NAMESPACE and man/*.Rd).

---

## Prerequisites

```r
# Install required packages
install.packages(c("roxygen2", "devtools"))
```

---

## Quick Start

### Generate Documentation

```r
# From R/
source("generate_docs.R")

# Or from command line
Rscript generate_docs.R
```

This will:
1. Process all `#' @...` roxygen2 comments
2. Generate `NAMESPACE` file
3. Generate `man/*.Rd` files (one per exported function)
4. Validate documentation completeness

---

## Documentation Structure

### Existing roxygen2 Tags

All R files in `R/R/` have documentation with:

- **@title**: Function title
- **@description**: Detailed description
- **@param**: Parameter documentation
- **@return**: Return value documentation
- **@export**: Mark function for export
- **@examples**: Usage examples
- **@details**: Additional details
- **@references**: Citations (where applicable)
- **@seealso**: Related functions
- **@keywords**: Keywords for indexing

### Documented Files

1. **base_layer.R** (3 exports)
   - `base_layer`: Abstract base class
   - `layer_registry`: Layer registration system
   - `get_registered_layers`: Retrieve registered layers

2. **math_utils.R** (6 exports)
   - `binary_concrete`: Gumbel-Softmax relaxation
   - `unitboxcar`: Soft boxcar function
   - `soft_and`: Differentiable AND
   - `soft_or`: Differentiable OR
   - `transf_log`: Bounded transformation
   - `inv_transf_log`: Inverse transformation

3. **layer1_phylogenetic_focus.R** (2 exports)
   - `spatial_agg_layer`: Static phylogenetic focus
   - `spatial_agg_dynamic_layer`: Dynamic phylogenetic focus

4. **layer2_temporal_focus.R** (2 exports)
   - `time_agg_layer`: Temporal focus with slopes
   - `time_agg_abun_layer`: Temporal focus (abundance only)

5. **layer3_detector.R** (2 exports)
   - `threshold_detector_layer`: Threshold detection
   - `slope_detector_layer`: Slope detection

6. **layer4_rule.R** (1 export)
   - `rules_layer`: Rule formation (soft AND)

7. **layer5_classification.R** (2 exports)
   - `dense_layer`: Classification (with slopes)
   - `dense_layer_abun`: Classification (abundance only)

8. **models.R** (2 exports)
   - `mditre_model`: Complete MDITRE model
   - `mditre_abun_model`: MDITRE abundance-only model

9. **seeding.R** (5 exports)
   - `set_mditre_seeds`: Set all random seeds
   - `generate_seed`: Generate deterministic seed
   - `seed_generator`: Seed generator class
   - `seedhash_available`: Check seedhash availability
   - `get_seed_info`: Get current seed information

10. **phyloseq_loader.R** (4 exports)
    - `phyloseq_to_mditre`: Convert phyloseq to MDITRE format
    - `split_train_test`: Train/test splitting
    - `create_dataloader`: Create PyTorch-style dataloaders
    - `print_mditre_data_summary`: Print data summary

11. **trainer.R** (3 exports)
    - `train_mditre`: Complete training pipeline
    - `print_training_metrics`: Print training metrics
    - `print_metrics`: Print evaluation metrics

12. **evaluation.R** (6 exports)
    - `compute_metrics`: Compute all metrics
    - `compute_roc_curve`: ROC curve computation
    - `evaluate_model_on_data`: Evaluate model
    - `cross_validate_mditre`: K-fold cross-validation
    - `compare_models`: Model comparison
    - `print_metrics`: Print metrics (duplicate, will merge)

13. **visualize.R** (8 exports)
    - `plot_training_history`: Training curves
    - `plot_roc_curve`: ROC curves
    - `plot_confusion_matrix`: Confusion matrix heatmap
    - `plot_cv_results`: Cross-validation results
    - `plot_model_comparison`: Model comparison plots
    - `plot_phylogenetic_tree`: Phylogenetic tree with weights
    - `plot_parameter_distributions`: Parameter histograms
    - `create_evaluation_report`: Comprehensive report

**Total**: 46+ exported functions/classes

---

## Generated Files

### NAMESPACE

Automatically generated file listing:
- Exported functions (`export(...)`)
- Imported packages (`import(...)` or `importFrom(...)`)
- S3/S4 methods

**Do NOT edit NAMESPACE manually** - regenerate with roxygen2

### man/*.Rd Files

One `.Rd` file per exported function/class containing:
- Title and description
- Usage syntax
- Parameter descriptions
- Return value description
- Examples
- See also links
- Keywords

These files are used by R's help system:
```r
?mditre_model
help(train_mditre)
```

---

## Documentation Standards

### Function Documentation Template

```r
#' Function Title (Brief, One Line)
#'
#' @description
#' Detailed description of what the function does.
#' Can span multiple lines.
#'
#' @param param1 Description of first parameter
#' @param param2 Description of second parameter
#' @param ... Additional arguments passed to other functions
#'
#' @return Description of return value(s)
#'
#' @details
#' Additional implementation details, algorithms used,
#' special considerations, etc.
#'
#' @references
#' Author (Year). Title. Journal. DOI/URL.
#'
#' @seealso
#' \code{\link{related_function}}
#'
#' @export
#' @examples
#' \dontrun{
#' # Example usage
#' result <- my_function(param1 = 10, param2 = "value")
#' }
my_function <- function(param1, param2) {
  # Implementation
}
```

### Common Tags

| Tag | Purpose | Required? |
|-----|---------|-----------|
| `@title` | Short title | Auto-generated from first sentence |
| `@description` | Detailed description | Recommended |
| `@param` | Document parameters | Yes (for each param) |
| `@return` | Document return value | Yes |
| `@export` | Export function | Yes (for public functions) |
| `@examples` | Usage examples | Recommended |
| `@details` | Additional details | Optional |
| `@references` | Citations | Optional |
| `@seealso` | Related functions | Optional |
| `@keywords` | Index keywords | Optional |
| `@rdname` | Merge docs | Optional |

---

## Validation

### Check Documentation Completeness

```r
# Load devtools
library(devtools)

# Check package (includes documentation check)
check()

# Just check documentation
document()

# Specific checks
check_man()  # Check .Rd files
```

### Common Issues

1. **Missing @param tags**
   ```r
   # BAD: No @param documentation
   #' @export
   my_func <- function(x, y) { }
   
   # GOOD: All params documented
   #' @param x First parameter
   #' @param y Second parameter
   #' @export
   my_func <- function(x, y) { }
   ```

2. **Missing @return tag**
   ```r
   # BAD: No return documentation
   #' @export
   my_func <- function(x) { x * 2 }
   
   # GOOD: Return documented
   #' @return Numeric vector, doubled input
   #' @export
   my_func <- function(x) { x * 2 }
   ```

3. **Broken @examples**
   ```r
   # BAD: Example will fail
   #' @examples
   #' result <- undefined_function()
   
   # GOOD: Working or wrapped in \dontrun
   #' @examples
   #' \dontrun{
   #' result <- my_function(10)
   #' }
   ```

---

## Workflow

### Complete Documentation Workflow

```r
# 1. Add/update roxygen2 comments in R/*.R files
# Edit: R/R/my_function.R

# 2. Generate documentation
source("generate_docs.R")

# 3. Check for issues
devtools::check()

# 4. Preview documentation
?my_function

# 5. Build package
devtools::build()

# 6. Install
devtools::install()
```

### Iterative Development

```r
# Quick cycle during development
devtools::document()  # Regenerate docs
devtools::load_all()  # Reload package
?my_function          # Check documentation
```

---

## Integration with Package

### R CMD check

roxygen2 documentation is validated during:
```bash
R CMD check mditre_*.tar.gz
```

Requirements:
- All exported functions have @title, @param, @return
- Examples run without errors (or wrapped in \dontrun)
- No broken cross-references (@seealso links)
- No undocumented parameters

### CRAN Submission

For CRAN submission, documentation must:
- Pass R CMD check with no ERRORs or WARNINGs
- Have complete @param and @return tags
- Include working @examples (demonstrating key functionality)
- Use \dontrun{} for examples requiring data/dependencies
- Have no spelling errors (use `devtools::spell_check()`)

---

## Advanced Features

### Inheriting Documentation

```r
#' Base function
#' @param x Parameter description
#' @export
base_function <- function(x) { }

#' @rdname base_function
#' @export
variant_function <- function(x) { }
```

### S3 Method Documentation

```r
#' @export
print.my_class <- function(x, ...) {
  # Implementation
}

# Automatically creates print.my_class.Rd
```

### Internal Functions

```r
#' Internal helper function
#' @keywords internal
helper_function <- function() { }

# Not exported, documented but hidden
```

---

## Next Steps

1. **Generate Documentation**
   ```r
   source("generate_docs.R")
   ```

2. **Review Generated Files**
   - Check `NAMESPACE` for correct exports
   - Verify `man/*.Rd` files are complete

3. **Validate**
   ```r
   devtools::check()
   ```

4. **Build pkgdown Site** (next phase)
   - Configure `_pkgdown.yml`
   - Generate static website
   - Deploy to GitHub Pages

---

## Documentation Statistics

- **Total Functions**: 46+ exported functions
- **Documentation Coverage**: 100% (all have roxygen2 tags)
- **man/ Files**: Will generate 46+ .Rd files
- **NAMESPACE**: Will list all exports and imports

---

## Troubleshooting

### Issue: NAMESPACE not generated

**Solution**: Ensure roxygen2 version >= 7.0
```r
packageVersion("roxygen2")
update.packages("roxygen2")
```

### Issue: Missing man/ files

**Solution**: Check for roxygen2 syntax errors
```r
roxygen2::roxygenise(package.dir = ".", clean = TRUE)
```

### Issue: Examples fail

**Solution**: Wrap in \dontrun{} or \donttest{}
```r
#' @examples
#' \dontrun{
#' # Example requiring external data
#' result <- my_function(data)
#' }
```

### Issue: Cross-references broken

**Solution**: Use correct syntax for links
```r
# BAD
#' @seealso related_function

# GOOD
#' @seealso \code{\link{related_function}}
```

---

## References

- [roxygen2 documentation](https://roxygen2.r-lib.org/)
- [Writing R Extensions](https://cran.r-project.org/doc/manuals/R-exts.html)
- [R Packages book](https://r-pkgs.org/man.html)

---

**Document Status**: Complete  
**Last Updated**: 2024  
**Phase**: Documentation (roxygen2)  
**Next Phase**: pkgdown website
