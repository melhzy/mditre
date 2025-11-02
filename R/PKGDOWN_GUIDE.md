# MDITRE R Package - pkgdown Website Guide

**Status**: Configuration complete, ready for build  
**Date**: 2024  
**Phase**: Documentation (pkgdown)

---

## Overview

pkgdown generates a static website for the MDITRE R package, providing:
- Package homepage
- Function reference (from roxygen2)
- Vignettes (as articles)
- News/changelog
- Search functionality

---

## Prerequisites

```r
# Install pkgdown
install.packages("pkgdown")

# Install other dependencies
install.packages(c("roxygen2", "devtools", "rmarkdown", "knitr"))
```

---

## Quick Start

### Build Website Locally

```r
# From R/ directory
library(pkgdown)

# Build complete site
pkgdown::build_site()

# Or individual components
pkgdown::build_home()       # Homepage
pkgdown::build_reference()  # Function reference
pkgdown::build_articles()   # Vignettes
pkgdown::build_news()       # Changelog
```

Website will be in `docs/` directory.

### Preview Website

```r
# Build and preview
pkgdown::build_site(preview = TRUE)

# Or manually open
browseURL("docs/index.html")
```

---

## Configuration

### _pkgdown.yml Structure

The `_pkgdown.yml` file configures:

1. **Site Metadata**
   ```yaml
   url: https://melhzy.github.io/mditre/
   title: "MDITRE"
   description: "Microbiome Interpretable Temporal Rule Engine"
   ```

2. **Theme**
   ```yaml
   template:
     bootstrap: 5
     bootswatch: cosmo
     theme: arrow-light
   ```

3. **Navigation Bar**
   ```yaml
   navbar:
     structure:
       left:  [intro, reference, articles, tutorials, news]
       right: [search, github, twitter]
   ```

4. **Function Reference Organization**
   ```yaml
   reference:
   - title: "Model Construction"
     contents:
     - mditre_model
     - mditre_abun_model
   ```

5. **Articles (Vignettes)**
   ```yaml
   articles:
   - title: "Getting Started"
     contents:
     - quickstart
   ```

---

## Website Structure

```
docs/                          # Generated website
├── index.html                 # Homepage (from README.md)
├── reference/                 # Function reference
│   ├── index.html            # Reference index
│   ├── mditre_model.html     # Individual function pages
│   └── ...
├── articles/                  # Vignettes as articles
│   ├── index.html
│   ├── quickstart.html
│   ├── training.html
│   ├── evaluation.html
│   └── interpretation.html
├── news/                      # Changelog
│   └── index.html
├── search.json               # Search index
└── pkgdown.yml               # Build metadata
```

---

## Function Reference Organization

Functions are organized into logical groups:

### 1. Model Construction (2 functions)
- `mditre_model`: Complete MDITRE model
- `mditre_abun_model`: Abundance-only variant

### 2. Neural Network Layers (9 functions)
- Layer 1: `spatial_agg_layer`, `spatial_agg_dynamic_layer`
- Layer 2: `time_agg_layer`, `time_agg_abun_layer`
- Layer 3: `threshold_detector_layer`, `slope_detector_layer`
- Layer 4: `rules_layer`
- Layer 5: `dense_layer`, `dense_layer_abun`

### 3. Data Loading (4 functions)
- `phyloseq_to_mditre`: phyloseq conversion
- `split_train_test`: Train/test splitting
- `create_dataloader`: DataLoader creation
- `print_mditre_data_summary`: Data summary

### 4. Model Training (2 functions)
- `train_mditre`: Complete training pipeline
- `print_training_metrics`: Training metrics display

### 5. Model Evaluation (6 functions)
- `compute_metrics`: Calculate all metrics
- `compute_roc_curve`: ROC curve
- `evaluate_model_on_data`: Model evaluation
- `cross_validate_mditre`: K-fold CV
- `compare_models`: Model comparison
- `print_metrics`: Metrics display

### 6. Visualization (8 functions)
- `plot_training_history`: Training curves
- `plot_roc_curve`: ROC curves
- `plot_confusion_matrix`: Confusion matrices
- `plot_cv_results`: CV results
- `plot_model_comparison`: Model comparison
- `plot_phylogenetic_tree`: Phylogenetic trees
- `plot_parameter_distributions`: Parameter histograms
- `create_evaluation_report`: Comprehensive reports

### 7. Mathematical Utilities (6 functions)
- `binary_concrete`: Gumbel-Softmax
- `unitboxcar`: Soft boxcar
- `soft_and`: Differentiable AND
- `soft_or`: Differentiable OR
- `transf_log`: Bounded transformation
- `inv_transf_log`: Inverse transformation

### 8. Reproducibility (5 functions)
- `set_mditre_seeds`: Set all seeds
- `generate_seed`: Generate deterministic seed
- `seed_generator`: Seed generator class
- `seedhash_available`: Check seedhash
- `get_seed_info`: Seed information

### 9. Architecture (3 functions)
- `base_layer`: Abstract base class
- `layer_registry`: Layer registration
- `get_registered_layers`: List layers

**Total**: 46+ documented functions

---

## Customization

### Homepage

Edit `README.md` to customize homepage content:
```markdown
# MDITRE

<!-- badges -->

## Overview

MDITRE (Microbiome Interpretable Temporal Rule Engine) is...

## Installation

## Quick Start

## Features

## Citation
```

### Logo

Add package logo:
```r
# Create logo (hexagonal sticker)
# Save as: man/figures/logo.png (240x278 px recommended)

# pkgdown will automatically use it in navbar
```

### Theme Customization

Modify `_pkgdown.yml`:
```yaml
template:
  bootstrap: 5
  bootswatch: cosmo  # Choose: cosmo, flatly, lumen, etc.
  theme: arrow-light # Or: arrow-dark
  
  # Custom CSS
  includes:
    in_header: |
      <link rel="stylesheet" href="extra.css">
```

### Navbar Customization

```yaml
navbar:
  structure:
    left:  [intro, reference, articles, tutorials, news]
    right: [search, github]
  components:
    github:
      icon: fab fa-github fa-lg
      href: https://github.com/melhzy/mditre
```

---

## Articles (Vignettes)

pkgdown automatically converts vignettes to articles:

### Vignette → Article Mapping

| Vignette File | Article URL | Title |
|---------------|-------------|-------|
| `quickstart.Rmd` | `articles/quickstart.html` | Quickstart Guide |
| `training.Rmd` | `articles/training.html` | Training Guide |
| `evaluation.Rmd` | `articles/evaluation.html` | Evaluation Guide |
| `interpretation.Rmd` | `articles/interpretation.html` | Interpretation Guide |

### Article Organization

Configure in `_pkgdown.yml`:
```yaml
articles:
- title: "Getting Started"
  navbar: ~
  contents:
  - quickstart

- title: "User Guides"
  navbar: ~
  contents:
  - training
  - evaluation
  - interpretation
```

---

## Deployment

### GitHub Pages

#### Method 1: Automatic (GitHub Actions)

Create `.github/workflows/pkgdown.yaml`:
```yaml
name: pkgdown

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  pkgdown:
    runs-on: ubuntu-latest
    env:
      GITHUB_PAT: ${{ secrets.GITHUB_TOKEN }}
    steps:
      - uses: actions/checkout@v3
      - uses: r-lib/actions/setup-r@v2
      - uses: r-lib/actions/setup-pandoc@v2
      - uses: r-lib/actions/setup-r-dependencies@v2
        with:
          extra-packages: any::pkgdown, local::.
          needs: website
      - name: Build site
        run: pkgdown::build_site_github_pages(new_process = FALSE, install = FALSE)
        shell: Rscript {0}
      - name: Deploy to GitHub pages
        uses: JamesIves/github-pages-deploy-action@v4.4.1
        with:
          branch: gh-pages
          folder: docs
```

#### Method 2: Manual

```r
# 1. Build site
pkgdown::build_site()

# 2. Commit docs/
git add docs/
git commit -m "Update pkgdown site"
git push

# 3. Enable GitHub Pages
# Repository Settings → Pages → Source: main branch /docs folder
```

### Custom Domain

1. Add `CNAME` file to `docs/`:
   ```
   mditre.example.com
   ```

2. Configure DNS:
   ```
   CNAME   mditre   melhzy.github.io
   ```

3. Enable in GitHub Settings → Pages

---

## Build Workflow

### Complete Build Process

```r
# 1. Ensure roxygen2 documentation is current
devtools::document()

# 2. Build vignettes
devtools::build_vignettes()

# 3. Build pkgdown site
pkgdown::build_site()

# 4. Preview
pkgdown::preview_site()
```

### Incremental Updates

```r
# Update specific components
pkgdown::build_reference_index()  # Reference index
pkgdown::build_article("quickstart")  # Single vignette
pkgdown::build_news()  # Changelog
```

### Development Mode

```r
# Fast builds during development
pkgdown::build_site(
  preview = TRUE,
  devel = TRUE,  # Development mode
  lazy = TRUE    # Only rebuild changed files
)
```

---

## Quality Checks

### Pre-Build Checklist

- [ ] All roxygen2 documentation complete
- [ ] All vignettes render without errors
- [ ] README.md is up to date
- [ ] NEWS.md has latest changes
- [ ] DESCRIPTION file is current
- [ ] All examples work
- [ ] No broken cross-references

### Validation

```r
# Check package
devtools::check()

# Spell check
devtools::spell_check()

# URL check
pkgdown::check_pkgdown()

# Build without errors
pkgdown::build_site(devel = FALSE, preview = FALSE)
```

---

## Advanced Features

### Search

pkgdown automatically creates search index:
- Searches functions, articles, and help topics
- Powered by Fuse.js
- Accessible via search box in navbar

### Changelog

Create `NEWS.md`:
```markdown
# mditre 2.0.0

## New Features

* Complete R implementation with all 5 layers
* phyloseq integration
* Comprehensive training pipeline
* Evaluation utilities
* Visualization toolkit

## Bug Fixes

* Fixed handling of missing timepoints
* Improved numerical stability

# mditre 1.0.0

* Initial Python release
```

pkgdown converts to `news/index.html`

### Code Snippets

Syntax highlighting for R, Python, bash, etc.:
```r
# Automatically detected from code blocks
```

### LaTeX Math

Use LaTeX math in documentation:
```r
#' The model computes \eqn{y = \beta_0 + \beta_1 x}
#' 
#' Where:
#' \deqn{\beta_0 = \frac{\sum y_i}{n}}
```

---

## Maintenance

### Regular Updates

```r
# After changing code/docs
devtools::document()
pkgdown::build_site()

# After adding vignettes
pkgdown::build_articles()

# After updating README
pkgdown::build_home()
```

### Version Updates

1. Update `DESCRIPTION` version
2. Update `NEWS.md`
3. Rebuild site
4. Tag release in git

---

## Troubleshooting

### Issue: Site not building

**Solution**: Check package builds first
```r
devtools::check()
devtools::build()
```

### Issue: Vignettes not appearing

**Solution**: Ensure proper YAML headers
```yaml
---
title: "My Vignette"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{My Vignette}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---
```

### Issue: Examples failing

**Solution**: Use \dontrun{} or \donttest{}
```r
#' @examples
#' \dontrun{
#' # Example requiring data
#' }
```

### Issue: Broken links

**Solution**: Check cross-references
```r
pkgdown::check_pkgdown()
```

---

## Examples

### Complete Workflow

```r
# Starting from scratch

# 1. Add roxygen2 documentation to all functions
# (Already complete for MDITRE)

# 2. Generate man/ files
devtools::document()

# 3. Create _pkgdown.yml configuration
# (Already created)

# 4. Build site
pkgdown::build_site()

# 5. Preview
pkgdown::preview_site()

# 6. Deploy to GitHub Pages
# Commit docs/ and push
```

### Updating After Changes

```r
# Made changes to functions?
devtools::document()
pkgdown::build_reference()

# Updated vignettes?
pkgdown::build_articles()

# Changed README?
pkgdown::build_home()

# Full rebuild
pkgdown::build_site()
```

---

## Resources

- [pkgdown documentation](https://pkgdown.r-lib.org/)
- [Bootstrap themes](https://bootswatch.com/)
- [GitHub Pages setup](https://pages.github.com/)
- [R Packages book](https://r-pkgs.org/website.html)

---

## Next Steps

1. **Generate roxygen2 docs** (if not done)
   ```r
   source("generate_docs.R")
   ```

2. **Build pkgdown site**
   ```r
   pkgdown::build_site()
   ```

3. **Preview locally**
   ```r
   pkgdown::preview_site()
   ```

4. **Deploy to GitHub Pages**
   - Commit `docs/` directory
   - Enable Pages in repository settings

---

## Website Features

### Current Configuration

- **Theme**: Cosmo (Bootstrap 5)
- **Navbar**: Home, Reference, Articles, News, GitHub
- **Search**: Enabled
- **Function Reference**: 9 categories, 46+ functions
- **Articles**: 4 vignettes (quickstart, training, evaluation, interpretation)
- **Responsive**: Mobile-friendly design

### Planned Enhancements

- [ ] Add package logo
- [ ] Custom CSS styling
- [ ] Tutorial videos
- [ ] Gallery of examples
- [ ] Interactive plots (plotly)
- [ ] Cheatsheet PDF

---

**Document Status**: Complete  
**Last Updated**: 2024  
**Phase**: Documentation (pkgdown)  
**Website URL**: https://melhzy.github.io/mditre/ (after deployment)
