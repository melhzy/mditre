# MDITRE: Python to R Conversion Guide

**Version**: 1.0.0  
**Date**: November 1, 2025  
**Purpose**: Comprehensive documentation for converting MDITRE from Python to R

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Library Mapping](#library-mapping)
3. [Architecture Overview](#architecture-overview)
4. [File-by-File Conversion Plan](#file-by-file-conversion-plan)
5. [Core Components](#core-components)
6. [Data Structures](#data-structures)
7. [Neural Network Implementation](#neural-network-implementation)
8. [Testing Strategy](#testing-strategy)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Key Challenges](#key-challenges)

---

## Executive Summary

MDITRE is a deep learning framework for analyzing longitudinal microbiome time-series data. The conversion from Python to R requires:

- **Deep Learning**: PyTorch → TensorFlow/Keras R or torch R
- **Microbiome Analysis**: Custom Python → phyloseq + microbiome R packages
- **Phylogenetic**: ete3 (Python) → ape + phangorn (R)
- **Data Processing**: NumPy/Pandas → tidyverse + data.table
- **Visualization**: matplotlib/seaborn → ggplot2 + plotly

### Key Paper Citations
From Maringanti et al. (2022) mSystems:
- MDITRE learns human-interpretable rules from longitudinal microbiome data
- 5-layer fully differentiable architecture
- Phylogenetic and temporal focusing mechanisms
- Orders of magnitude faster than MITRE (original Bayesian version)
- Handles 16S rRNA and metagenomic sequencing data

---

## Library Mapping

### 1. Deep Learning Framework

| Python (PyTorch) | R Equivalent | Notes |
|------------------|--------------|-------|
| `torch` | `torch` (torch R package) | Preferred: Native R bindings to LibTorch |
| `torch.nn` | `torch::nn_*` | Neural network modules |
| `torch.optim` | `torch::optim_*` | Optimizers (Adam, SGD, etc.) |
| Alternative: | `tensorflow` + `keras` | Mature but requires different paradigm |

**Recommendation**: Use `torch` R package (https://torch.mlverse.org/)
- Direct port of PyTorch to R
- Similar API structure
- GPU support via CUDA
- Active development by RStudio/Posit

### 2. Scientific Computing

| Python | R Equivalent | Package |
|--------|--------------|---------|
| `numpy` | `array()` operations | Base R + `Matrix` |
| `numpy.random` | `stats::rnorm()` etc | Base R |
| `scipy.special` | `gtools::logit()`, `boot::inv.logit()` | Various |
| `scipy.stats` | `stats::*` | Base R |
| `pandas` | `dplyr`, `tidyr`, `data.table` | tidyverse |

### 3. Microbiome-Specific Libraries

| Python | R Equivalent | Package | URL |
|--------|--------------|---------|-----|
| Custom data loaders | `phyloseq` | phyloseq | https://bioconductor.org/packages/release/bioc/html/phyloseq.html |
| OTU/ASV processing | `microbiome` | microbiome | https://microbiome.github.io/ |
| Diversity analysis | `vegan` | vegan | CRAN |
| Compositional data | `compositions` | compositions | CRAN |

**phyloseq Integration**:
```r
# phyloseq object structure
phyloseq_obj <- phyloseq(
  otu_table(otu_mat, taxa_are_rows = TRUE),
  sample_data(metadata),
  tax_table(taxonomy),
  phy_tree(tree)
)
```

### 4. Phylogenetic Analysis

| Python | R Equivalent | Package |
|--------|--------------|---------|
| `ete3.Tree` | `ape::read.tree()` | ape |
| `ete3` distances | `ape::cophenetic.phylo()` | ape |
| `dendropy` | `phangorn` | phangorn |
| Tree manipulation | `tidytree` | tidytree |

**Key Functions**:
```r
library(ape)
library(phangorn)

# Read phylogenetic tree
tree <- read.tree("tree.newick")

# Calculate distances
dist_matrix <- cophenetic.phylo(tree)

# Tree operations
pruned_tree <- drop.tip(tree, tips_to_remove)
```

### 5. Machine Learning

| Python | R Equivalent | Package |
|--------|--------------|---------|
| `sklearn.linear_model` | `glmnet` | glmnet |
| `sklearn.metrics` | `caret`, `yardstick` | caret/tidymodels |
| `sklearn.preprocessing` | `recipes` | recipes |
| `sklearn.model_selection` | `rsample` | rsample |

### 6. Visualization

| Python | R Equivalent | Package |
|--------|--------------|---------|
| `matplotlib` | `ggplot2` | ggplot2 |
| `seaborn` | `ggplot2` + themes | ggplot2 |
| Interactive plots | `plotly` | plotly |
| Tree visualization | `ggtree` | ggtree |

---

## Architecture Overview

### MDITRE 5-Layer Architecture

From the paper (Maringanti et al., 2022):

```
Layer 1: Phylogenetic Focus
├── Aggregates microbes by evolutionary relationships
├── Classes: SpatialAgg, SpatialAggDynamic
└── Output: Phylogenetically aggregated abundances

Layer 2: Temporal Focus
├── Selects important time windows
├── Classes: TimeAgg, TimeAggAbun
└── Output: Temporally focused features

Layer 3: Detector
├── Applies thresholds to detect patterns
├── Classes: Threshold, Slope
└── Output: Binary detector activations

Layer 4: Rule
├── Combines detectors via logical AND
├── Class: Rules
└── Output: Rule activations

Layer 5: Classification
├── Weighted rule combination for prediction
├── Classes: DenseLayer, DenseLayerAbun
└── Output: Host status predictions (binary)
```

### R Implementation Strategy

**Option 1: torch R (Recommended)**
```r
library(torch)

# Layer 1: Phylogenetic Focus
phylo_focus_module <- nn_module(
  "PhylogeneticFocus",
  initialize = function(n_taxa, emb_dim, dist_matrix) {
    self$n_taxa <- n_taxa
    self$emb_dim <- emb_dim
    self$embeddings <- nn_parameter(torch_randn(n_taxa, emb_dim))
    self$register_buffer("dist_matrix", torch_tensor(dist_matrix))
  },
  forward = function(x) {
    # Implement soft selection over taxa
    # ...
  }
)
```

**Option 2: TensorFlow/Keras R**
```r
library(tensorflow)
library(keras)

phylo_focus_layer <- layer_lambda(
  object = NULL,
  f = function(x) {
    # Custom phylogenetic focus operation
  }
)
```

---

## File-by-File Conversion Plan

### Core Package Structure

```
R/
├── mditre/
│   ├── DESCRIPTION
│   ├── NAMESPACE
│   ├── R/
│   │   ├── core/
│   │   │   ├── base_layer.R
│   │   │   ├── math_utils.R
│   │   │   └── registry.R
│   │   ├── layers/
│   │   │   ├── layer1_phylogenetic_focus.R
│   │   │   ├── layer2_temporal_focus.R
│   │   │   ├── layer3_detector.R
│   │   │   ├── layer4_rule.R
│   │   │   └── layer5_classification.R
│   │   ├── data_loader/
│   │   │   ├── base_loader.R
│   │   │   ├── phyloseq_loader.R
│   │   │   ├── dada2_loader.R
│   │   │   └── transforms.R
│   │   ├── models.R
│   │   ├── seeding.R
│   │   ├── trainer.R
│   │   └── visualize.R
│   ├── tests/
│   │   └── testthat/
│   ├── vignettes/
│   │   ├── quickstart.Rmd
│   │   └── tutorial_16s.Rmd
│   └── man/
└── README.md
```

### Priority Files for Conversion

#### 1. Core Infrastructure (Priority: HIGH)

**Python**: `mditre/core/base_layer.py`  
**R**: `R/core/base_layer.R`

```python
# Python (PyTorch)
class BaseLayer(nn.Module):
    def __init__(self):
        super().__init__()
```

```r
# R (torch)
base_layer <- nn_module(
  "BaseLayer",
  initialize = function() {
    # Initialization
  },
  forward = function(x) {
    # Forward pass
  }
)
```

#### 2. Mathematical Utilities (Priority: HIGH)

**Python**: `mditre/core/math_utils.py`  
**R**: `R/core/math_utils.R`

Key functions to convert:
- `binary_concrete()`: Gumbel-Softmax relaxation
- `soft_and()`: Differentiable AND operation
- `logit()`, `sigmoid()`: Available in R

```r
# R implementation
binary_concrete <- function(logits, temperature = 0.5, hard = FALSE) {
  # Gumbel-Softmax trick
  u <- torch_rand_like(logits)
  gumbel <- -torch_log(-torch_log(u + 1e-20) + 1e-20)
  y <- torch_sigmoid((logits + gumbel) / temperature)
  
  if (hard) {
    y_hard <- torch_round(y)
    y <- (y_hard - y)$detach() + y  # Straight-through estimator
  }
  
  return(y)
}
```

#### 3. Models (Priority: HIGH)

**Python**: `mditre/models.py`  
**R**: `R/models.R`

Core models:
- `MDITRE`: Full model with phylogenetic + temporal + slope detectors
- `MDITREAbun`: Abundance-only variant

```r
# R implementation skeleton
mditre_model <- nn_module(
  "MDITRE",
  initialize = function(n_subjects, n_taxa, n_timepoints, 
                       n_rules, n_detectors_per_rule,
                       phylo_dist, emb_dim = 10) {
    
    # Layer 1: Phylogenetic Focus
    self$phylo_focus <- phylo_focus_module(n_taxa, emb_dim, phylo_dist)
    
    # Layer 2: Temporal Focus  
    self$temporal_focus <- temporal_focus_module(n_timepoints)
    
    # Layer 3: Detectors
    self$threshold_detectors <- threshold_module(n_detectors_per_rule)
    self$slope_detectors <- slope_module(n_detectors_per_rule)
    
    # Layer 4: Rules
    self$rules <- rule_module(n_rules, n_detectors_per_rule)
    
    # Layer 5: Classification
    self$classifier <- classification_module(n_rules)
  },
  
  forward = function(x, mask = NULL) {
    # Implement 5-layer forward pass
    # ...
  }
)
```

#### 4. Data Loaders (Priority: HIGH)

**Python**: `mditre/data_loader/`  
**R**: `R/data_loader/`

Convert to use phyloseq objects:

```r
#' Load data from phyloseq object
#'
#' @param phyloseq_obj A phyloseq object
#' @param outcome_var Column name in sample_data for outcome
#' @return List with X (abundance), y (labels), mask, tree, metadata
#' @export
load_from_phyloseq <- function(phyloseq_obj, outcome_var) {
  # Extract components
  otu_table <- as.matrix(phyloseq::otu_table(phyloseq_obj))
  metadata <- phyloseq::sample_data(phyloseq_obj)
  tree <- phyloseq::phy_tree(phyloseq_obj)
  
  # Process for MDITRE
  # - Reshape to (n_subjects, n_taxa, n_timepoints)
  # - Extract outcomes
  # - Calculate phylogenetic distances
  # ...
  
  return(list(
    X = abundance_array,
    y = outcomes,
    mask = mask_matrix,
    tree = tree,
    metadata = metadata,
    phylo_dist = dist_matrix
  ))
}
```

#### 5. Seeding (Priority: MEDIUM)

**Python**: `mditre/seeding.py`  
**R**: `R/seeding.R`

```r
#' Set random seeds for reproducibility
#'
#' @param seed Integer seed value
#' @export
set_mditre_seeds <- function(seed) {
  set.seed(seed)  # R base
  torch::torch_manual_seed(seed)  # torch
  
  if (torch::cuda_is_available()) {
    torch::cuda_manual_seed_all(seed)
  }
}
```

#### 6. Visualization (Priority: MEDIUM)

**Python**: `mditre/visualize.py`  
**R**: `R/visualize.R`

Convert matplotlib → ggplot2:

```r
#' Visualize MDITRE rules
#'
#' @param model Trained MDITRE model
#' @param rule_idx Rule index to visualize
#' @return ggplot object
#' @export
visualize_rule <- function(model, rule_idx) {
  library(ggplot2)
  library(ggtree)
  
  # Extract rule parameters
  # - Selected taxa
  # - Time windows
  # - Thresholds
  
  # Create visualization
  p <- ggplot(...) +
    geom_point() +
    theme_minimal() +
    labs(title = paste("Rule", rule_idx))
  
  return(p)
}
```

#### 7. Trainer (Priority: LOW - Can be simplified)

**Python**: `mditre/trainer.py` (1000+ lines)  
**R**: `R/trainer.R` (Simplified)

The trainer can be simplified in R:

```r
#' Train MDITRE model
#'
#' @param model MDITRE model
#' @param train_data Training data list
#' @param val_data Validation data list  
#' @param epochs Number of epochs
#' @param lr Learning rate
#' @export
train_mditre <- function(model, train_data, val_data, 
                        epochs = 100, lr = 0.001) {
  
  optimizer <- optim_adam(model$parameters, lr = lr)
  criterion <- nn_bce_with_logits_loss()
  
  for (epoch in seq_len(epochs)) {
    # Training loop
    model$train()
    optimizer$zero_grad()
    
    outputs <- model(train_data$X, mask = train_data$mask)
    loss <- criterion(outputs, train_data$y)
    
    loss$backward()
    optimizer$step()
    
    # Validation
    if (!is.null(val_data)) {
      model$eval()
      # Compute validation metrics
    }
  }
  
  return(model)
}
```

---

## Core Components

### 1. Phylogenetic Focus Layer (Layer 1)

**Key Concepts** (from paper):
- Embeds taxa in phylogenetic space
- Performs "soft" selection over phylogenetic subtrees
- Uses phylogenetic distance matrix as prior information

**Python Implementation Pattern**:
```python
# Compute weights based on phylogenetic distance
weights = F.softmax(-self.dist * temperature, dim=-1)
aggregated = torch.matmul(weights, abundances)
```

**R Conversion**:
```r
phylo_focus_forward <- function(self, x) {
  # x: (batch, n_taxa, n_timepoints)
  
  # Compute phylogenetic weights
  weights <- torch_softmax(-self$dist_matrix * self$temperature, dim = -1L)
  
  # Aggregate abundances
  aggregated <- torch_matmul(weights, x)
  
  return(aggregated)
}
```

### 2. Temporal Focus Layer (Layer 2)

**Key Concepts**:
- Selects time windows of interest
- Computes average or rate of change within windows
- Soft selection using Gaussian kernels

**R Implementation**:
```r
temporal_focus_forward <- function(self, x) {
  # x: (batch, n_features, n_timepoints)
  
  # Generate soft time windows
  window_centers <- torch_sigmoid(self$window_params)
  window_widths <- torch_softplus(self$width_params)
  
  # Apply Gaussian windowing
  time_grid <- torch_linspace(0, 1, steps = self$n_timepoints)
  windows <- torch_exp(
    -((time_grid - window_centers)^2) / (2 * window_widths^2)
  )
  
  # Aggregate over windows
  focused <- torch_matmul(windows, x)
  
  return(focused)
}
```

### 3. Detector Layer (Layer 3)

**Types**:
- Threshold detectors: abundance > threshold
- Slope detectors: rate of change > threshold

**R Implementation**:
```r
threshold_detector <- function(x, threshold, temperature = 0.1) {
  # Soft thresholding using sigmoid
  activation <- torch_sigmoid((x - threshold) / temperature)
  return(activation)
}

slope_detector <- function(x, threshold, temperature = 0.1) {
  # Compute slope (finite differences)
  slopes <- x[, , 2:self$n_timepoints] - x[, , 1:(self$n_timepoints-1)]
  avg_slope <- torch_mean(slopes, dim = 3)
  
  # Soft thresholding
  activation <- torch_sigmoid((avg_slope - threshold) / temperature)
  return(activation)
}
```

### 4. Rule Layer (Layer 4)

**Key Concept**: Soft AND operation (from Neural Arithmetic Units)

**R Implementation**:
```r
soft_and <- function(x, dim = -1) {
  # Product approximation to AND
  # AND(x1, x2, ...) ≈ x1 * x2 * ...
  result <- torch_prod(x, dim = dim)
  return(result)
}
```

### 5. Classification Layer (Layer 5)

**Key Concept**: Weighted sum of rule activations

**R Implementation**:
```r
classification_forward <- function(self, rule_activations) {
  # rule_activations: (batch, n_rules)
  
  # Weighted sum
  logits <- torch_matmul(rule_activations, self$rule_weights)
  
  return(logits)
}
```

---

## Data Structures

### Input Data Format

**Python** (NumPy arrays):
```python
X: np.ndarray  # Shape: (n_subjects, n_taxa, n_timepoints)
y: np.ndarray  # Shape: (n_subjects,)
mask: np.ndarray  # Shape: (n_subjects, n_timepoints)
phylo_dist: np.ndarray  # Shape: (n_taxa, n_taxa)
```

**R** (torch tensors):
```r
X <- torch_tensor(data$X)  # Shape: (n_subjects, n_taxa, n_timepoints)
y <- torch_tensor(data$y)  # Shape: (n_subjects)
mask <- torch_tensor(data$mask)  # Shape: (n_subjects, n_timepoints)
phylo_dist <- torch_tensor(data$phylo_dist)  # Shape: (n_taxa, n_taxa)
```

### phyloseq Integration

```r
#' Convert phyloseq to MDITRE format
#'
#' @param ps phyloseq object with sample_data containing:
#'   - subject_id: Subject identifier
#'   - timepoint: Time point  
#'   - outcome: Binary outcome variable
phyloseq_to_mditre <- function(ps, outcome_var = "outcome") {
  
  # Extract data
  otu <- as.matrix(otu_table(ps))
  meta <- as.data.frame(sample_data(ps))
  tree <- phy_tree(ps)
  
  # Reshape to (subject, taxa, time)
  subjects <- unique(meta$subject_id)
  n_subjects <- length(subjects)
  n_taxa <- ntaxa(ps)
  
  # Create 3D array
  X_list <- lapply(subjects, function(subj) {
    subj_samples <- meta$subject_id == subj
    subj_otu <- otu[, subj_samples]
    # Organize by timepoint
    # ...
  })
  
  X <- array_from_list(X_list)
  
  # Extract outcomes
  y <- meta %>%
    group_by(subject_id) %>%
    slice(1) %>%
    pull(!!sym(outcome_var))
  
  # Calculate phylogenetic distances
  phylo_dist <- cophenetic.phylo(tree)
  
  return(list(
    X = X,
    y = y,
    phylo_dist = phylo_dist,
    tree = tree,
    taxa_names = taxa_names(ps)
  ))
}
```

---

## Neural Network Implementation

### torch R vs TensorFlow/Keras

**Recommendation: torch R**

Advantages:
- Direct PyTorch API translation
- Similar syntax and concepts
- GPU support
- Active development
- Better for research code

**Example Comparison**:

```r
# torch R (Recommended)
library(torch)

model <- nn_module(
  "MDITRE",
  initialize = function(n_taxa) {
    self$fc <- nn_linear(n_taxa, 64)
  },
  forward = function(x) {
    self$fc(x)
  }
)

# TensorFlow/Keras (Alternative)
library(keras)

model <- keras_model_sequential() %>%
  layer_dense(units = 64, input_shape = c(n_taxa))
```

### Custom Layers in torch R

```r
# Custom phylogenetic focus layer
phylo_focus_layer <- nn_module(
  "PhylogeneticFocus",
  
  initialize = function(n_taxa, emb_dim, dist_matrix) {
    self$n_taxa <- n_taxa
    self$emb_dim <- emb_dim
    
    # Learnable embeddings
    self$embeddings <- nn_parameter(
      torch_randn(n_taxa, emb_dim) * 0.01
    )
    
    # Fixed distance matrix
    self$register_buffer(
      "dist_matrix",
      torch_tensor(dist_matrix, dtype = torch_float())
    )
    
    # Temperature parameter
    self$temperature <- nn_parameter(torch_tensor(1.0))
  },
  
  forward = function(x, mask = NULL) {
    # x: (batch, n_taxa, n_timepoints)
    batch_size <- x$size(1)
    n_timepoints <- x$size(3)
    
    # Compute attention weights based on distance
    # weights: (n_taxa, n_taxa)
    weights <- torch_softmax(
      -self$dist_matrix * self$temperature,
      dim = -1L
    )
    
    # Aggregate: (batch, n_taxa, n_timepoints)
    aggregated <- torch_einsum(
      "ij,bjt->bit",
      list(weights, x)
    )
    
    # Apply mask if provided
    if (!is.null(mask)) {
      mask_expanded <- mask$unsqueeze(2)$expand_as(aggregated)
      aggregated <- aggregated * mask_expanded
    }
    
    return(aggregated)
  }
)
```

---

## Testing Strategy

### Unit Tests with testthat

**Structure**:
```r
# tests/testthat/test-phylo-focus.R
library(testthat)
library(mditre)

test_that("Phylogenetic focus layer processes input correctly", {
  n_taxa <- 10
  n_timepoints <- 5
  batch_size <- 2
  
  # Create test data
  x <- torch_randn(batch_size, n_taxa, n_timepoints)
  dist_matrix <- matrix(runif(n_taxa * n_taxa), n_taxa, n_taxa)
  
  # Create layer
  layer <- phylo_focus_layer(n_taxa, emb_dim = 8, dist_matrix)
  
  # Forward pass
  output <- layer(x)
  
  # Tests
  expect_equal(output$size(1), batch_size)
  expect_equal(output$size(2), n_taxa)
  expect_equal(output$size(3), n_timepoints)
  expect_true(all(torch_isfinite(output)))
})
```

### Integration Tests

```r
# tests/testthat/test-end-to-end.R
test_that("Full MDITRE model can be trained", {
  # Load test data
  data <- generate_synthetic_data(n_subjects = 20, n_taxa = 50)
  
  # Create model
  model <- mditre_model(
    n_taxa = data$n_taxa,
    n_timepoints = data$n_timepoints,
    n_rules = 2,
    phylo_dist = data$phylo_dist
  )
  
  # Train
  trained_model <- train_mditre(
    model,
    train_data = data,
    epochs = 5,
    lr = 0.01
  )
  
  # Test predictions
  model$eval()
  predictions <- model(data$X)
  
  expect_equal(predictions$size(1), data$n_subjects)
  expect_true(all(torch_isfinite(predictions)))
})
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-3)

**Week 1: Setup & Base Classes**
- [ ] Create R package structure (DESCRIPTION, NAMESPACE)
- [ ] Implement `base_layer.R` with torch nn_module
- [ ] Implement `math_utils.R` (binary_concrete, soft_and)
- [ ] Set up testthat framework
- [ ] Create basic documentation

**Week 2: Data Loading**
- [ ] Implement phyloseq integration (`phyloseq_loader.R`)
- [ ] Create data transformation functions
- [ ] Support DADA2 output format
- [ ] Support QIIME2 output format
- [ ] Add data validation functions

**Week 3: Seeding & Utilities**
- [ ] Implement reproducible seeding (`seeding.R`)
- [ ] Add logging utilities
- [ ] Create helper functions for data preprocessing

### Phase 2: Model Layers (Weeks 4-7)

**Week 4: Layer 1 - Phylogenetic Focus**
- [ ] Implement `SpatialAgg` (static selection)
- [ ] Implement `SpatialAggDynamic` (learnable selection)
- [ ] Add phylogenetic distance calculations
- [ ] Unit tests for Layer 1

**Week 5: Layer 2 - Temporal Focus**
- [ ] Implement `TimeAgg` (abundance averaging)
- [ ] Implement `TimeAggAbun` (rate of change)
- [ ] Add soft time window selection
- [ ] Unit tests for Layer 2

**Week 6: Layer 3 - Detectors**
- [ ] Implement `Threshold` detector
- [ ] Implement `Slope` detector
- [ ] Add temperature annealing
- [ ] Unit tests for Layer 3

**Week 7: Layers 4-5 - Rules & Classification**
- [ ] Implement `Rules` (soft AND logic)
- [ ] Implement `DenseLayer` (classification)
- [ ] Implement `DenseLayerAbun` (abundance variant)
- [ ] Unit tests for Layers 4-5

### Phase 3: Complete Models (Weeks 8-9)

**Week 8: MDITRE Model**
- [ ] Assemble 5-layer MDITRE model
- [ ] Implement `init_params()` initialization
- [ ] Add forward pass
- [ ] Integration tests

**Week 9: MDITREAbun Variant**
- [ ] Implement abundance-only variant
- [ ] Add model serialization (save/load)
- [ ] Integration tests

### Phase 4: Training & Optimization (Weeks 10-11)

**Week 10: Training Loop**
- [ ] Implement `train_mditre()` function
- [ ] Add loss functions (BCE with logits)
- [ ] Implement validation loop
- [ ] Add early stopping
- [ ] Add model checkpointing

**Week 11: Optimization**
- [ ] Implement Adam optimizer configuration
- [ ] Add learning rate scheduling
- [ ] Add gradient clipping
- [ ] Performance optimization

### Phase 5: Visualization (Weeks 12-13)

**Week 12: Rule Visualization**
- [ ] Plot selected taxa (ggplot2 + ggtree)
- [ ] Plot time windows
- [ ] Plot detector activations
- [ ] Create rule summary plots

**Week 13: Model Interpretation**
- [ ] Rule extraction functions
- [ ] Create interactive plots (plotly)
- [ ] Add phylogenetic tree annotations
- [ ] Generate interpretation reports

### Phase 6: Documentation & Vignettes (Weeks 14-15)

**Week 14: Documentation**
- [ ] Complete roxygen2 documentation for all functions
- [ ] Add examples to all exported functions
- [ ] Create package website (pkgdown)
- [ ] Write troubleshooting guide

**Week 15: Vignettes**
- [ ] Quickstart vignette (Rmd)
- [ ] 16S rRNA tutorial (Rmd)
- [ ] Metagenomic data tutorial (Rmd)
- [ ] Advanced customization guide

### Phase 7: Testing & Benchmarking (Weeks 16-17)

**Week 16: Comprehensive Testing**
- [ ] Achieve >90% code coverage
- [ ] Test on semi-synthetic data
- [ ] Test on real datasets (Bokulich, David, etc.)
- [ ] Cross-platform testing (Windows, Mac, Linux)

**Week 17: Benchmarking**
- [ ] Performance benchmarks vs Python
- [ ] Memory profiling
- [ ] GPU vs CPU benchmarks
- [ ] Scalability tests

### Phase 8: Release Preparation (Week 18)

- [ ] Code review and refactoring
- [ ] Final documentation review
- [ ] Create CRAN submission materials
- [ ] Submit to Bioconductor (if applicable)
- [ ] Create release announcement

---

## Key Challenges

### 1. PyTorch → torch R Translation

**Challenge**: Some PyTorch features may not be directly available in torch R

**Solutions**:
- Use torch R's custom layer API
- Implement missing operations manually
- Contribute to torch R if needed

**Example Workaround**:
```r
# Python: torch.einsum()
# R: torch_einsum() - available in torch R
aggregated <- torch_einsum("ij,bjt->bit", list(weights, x))
```

### 2. Phylogenetic Data Structures

**Challenge**: Python uses ete3 Tree objects, R uses ape phylo objects

**Solutions**:
- Use ape and phangorn for phylogenetic operations
- Convert between formats when needed
- Leverage phyloseq for unified interface

```r
# Convert between formats
ape_to_ete3 <- function(ape_tree) {
  newick_str <- write.tree(ape_tree)
  # Could call Python via reticulate if needed
}

# Better: Work entirely in R ecosystem
tree <- read.tree("tree.newick")
dist_matrix <- cophenetic.phylo(tree)
```

### 3. GPU Compatibility

**Challenge**: Ensuring GPU support works across platforms

**Solutions**:
- Use torch R's CUDA detection
- Provide CPU fallbacks
- Test on multiple GPU configurations

```r
get_device <- function() {
  if (torch::cuda_is_available()) {
    return(torch::torch_device("cuda"))
  } else {
    message("CUDA not available, using CPU")
    return(torch::torch_device("cpu"))
  }
}
```

### 4. Reproducibility

**Challenge**: Ensuring exact reproducibility between Python and R versions

**Solutions**:
- Implement deterministic seeding
- Document random number generation
- Provide seed management utilities

```r
set_mditre_seeds <- function(seed = 42) {
  set.seed(seed)  # R base random
  torch::torch_manual_seed(seed)
  
  if (torch::cuda_is_available()) {
    torch::cuda_manual_seed_all(seed)
  }
  
  # Make torch operations deterministic
  torch::torch_set_deterministic(TRUE)
}
```

### 5. Data Format Compatibility

**Challenge**: Python users may have NumPy arrays, R users expect tibbles/data.frames

**Solutions**:
- Support multiple input formats
- Provide conversion utilities
- Integrate with phyloseq standard

```r
#' Flexible data input
#' 
#' Accepts: phyloseq, arrays, data.frames, paths to files
prepare_mditre_data <- function(data, format = c("phyloseq", "array", "csv")) {
  format <- match.arg(format)
  
  if (format == "phyloseq") {
    return(phyloseq_to_mditre(data))
  } else if (format == "array") {
    # Process 3D array
  } else if (format == "csv") {
    # Read from files
  }
}
```

### 6. Performance Optimization

**Challenge**: Maintaining performance parity with Python implementation

**Solutions**:
- Profile code to find bottlenecks
- Use vectorized operations
- Leverage torch R's JIT compilation
- Consider Rcpp for critical paths

```r
# Use torch operations (compiled)
fast_computation <- function(x) {
  torch_matmul(
    torch_softmax(x, dim = -1L),
    self$weights
  )
}

# Avoid loops
# BAD:
for (i in seq_len(n)) {
  result[i] <- compute(x[i])
}

# GOOD:
result <- torch_vmap(compute)(x)
```

---

## Notebook Conversion: .ipynb → .Rmd

### Jupyter Notebook Structure

**Python (.ipynb)**:
```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "source": ["# Title"]
    },
    {
      "cell_type": "code",
      "source": ["import torch"],
      "outputs": []
    }
  ]
}
```

**R Markdown (.Rmd)**:
````markdown
# Title

```{r}
library(torch)
```
````

### Conversion Mapping

| Python | R | Notes |
|--------|---|-------|
| `import numpy as np` | `library(torch)` | Use torch tensors |
| `import torch` | `library(torch)` | torch R package |
| `import matplotlib.pyplot as plt` | `library(ggplot2)` | ggplot2 for viz |
| `from mditre import MDITRE` | `library(mditre)` | Load R package |

### Example Notebook Conversion

**Python (`run_mditre_test.ipynb`)**:
```python
# Cell 1: Import libraries
import numpy as np
import torch
from mditre.models import MDITRE
from mditre.seeding import set_random_seeds

# Cell 2: Load data
data = load_from_pickle("data.pkl")
X = torch.from_numpy(data['X'])

# Cell 3: Train model
model = MDITRE(n_taxa=50, n_rules=5)
trained = train(model, X, y)
```

**R (`run_mditre_test.Rmd`)**:
````markdown
---
title: "MDITRE Quick Test"
output: html_document
---

## Setup

```{r setup}
library(torch)
library(mditre)
library(phyloseq)
```

## Load Data

```{r load-data}
# Load from phyloseq
ps <- readRDS("data.rds")
data <- phyloseq_to_mditre(ps, outcome_var = "disease")

# Convert to torch tensors
X <- torch_tensor(data$X)
y <- torch_tensor(data$y)
```

## Train Model

```{r train}
set_mditre_seeds(42)

model <- mditre_model(
  n_taxa = ncol(data$X),
  n_timepoints = dim(data$X)[3],
  n_rules = 5,
  phylo_dist = data$phylo_dist
)

trained_model <- train_mditre(
  model,
  train_data = list(X = X, y = y),
  epochs = 100,
  lr = 0.001
)
```

## Visualize Results

```{r visualize}
library(ggplot2)

# Plot training curves
plot_training_history(trained_model)

# Visualize rules
visualize_rule(trained_model, rule_idx = 1)
```
````

---

## Additional Resources

### R Package Development

- **R Packages Book**: https://r-pkgs.org/
- **torch R Documentation**: https://torch.mlverse.org/
- **Bioconductor Guidelines**: https://bioconductor.org/developers/
- **pkgdown**: https://pkgdown.r-lib.org/

### Microbiome R Packages

- **phyloseq**: https://joey711.github.io/phyloseq/
- **microbiome**: https://microbiome.github.io/
- **vegan**: https://cran.r-project.org/package=vegan
- **DADA2**: https://benjjneb.github.io/dada2/
- **ggtree**: https://yulab-smu.top/treedata-book/

### Deep Learning in R

- **torch**: https://torch.mlverse.org/
- **keras**: https://keras.rstudio.com/
- **reticulate**: https://rstudio.github.io/reticulate/ (Python interop)

---

## Conclusion

This guide provides a comprehensive roadmap for converting MDITRE from Python to R. The key strategies are:

1. **Use torch R** for neural network implementation (maintains API similarity)
2. **Integrate phyloseq** for microbiome data handling (R standard)
3. **Leverage Bioconductor** ecosystem for phylogenetic and microbiome tools
4. **Prioritize core functionality** before advanced features
5. **Maintain interpretability** as the primary design goal
6. **Test extensively** against Python implementation
7. **Document thoroughly** with vignettes and examples

The estimated timeline is **18 weeks** for a complete, well-tested, documented R implementation.

**Next Steps**:
1. Review this guide with R developers
2. Set up package structure
3. Begin Phase 1 implementation
4. Establish continuous testing against Python version
5. Create minimal working example as proof-of-concept

---

**Document Version**: 1.0.0  
**Last Updated**: November 1, 2025  
**Maintainer**: MDITRE Development Team
