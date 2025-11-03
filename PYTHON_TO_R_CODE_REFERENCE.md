# MDITRE: Python to R Code Translation Reference

**Companion to**: PYTHON_TO_R_CONVERSION_GUIDE.md  
**Version**: 1.0.1  
**Date**: November 1, 2025

This document provides side-by-side code translations for key MDITRE components.

---

## Quick Reference Table

| Component | Python File | R File | Priority | Complexity |
|-----------|-------------|--------|----------|------------|
| Base Layer | `core/base_layer.py` | `R/core/base_layer.R` | HIGH | Medium |
| Math Utils | `core/math_utils.py` | `R/core/math_utils.R` | HIGH | Medium |
| Phylo Focus | `layers/layer1_phylogenetic_focus/` | `R/layers/layer1_phylogenetic_focus.R` | HIGH | High |
| Temporal Focus | `layers/layer2_temporal_focus/` | `R/layers/layer2_temporal_focus.R` | HIGH | High |
| Detectors | `layers/layer3_detector/` | `R/layers/layer3_detector.R` | HIGH | Medium |
| Rules | `layers/layer4_rule/` | `R/layers/layer4_rule.R` | HIGH | Low |
| Classification | `layers/layer5_classification/` | `R/layers/layer5_classification.R` | HIGH | Low |
| MDITRE Model | `models.py` | `R/models.R` | HIGH | High |
| Data Loader | `data_loader/` | `R/data_loader/` | HIGH | Medium |
| Seeding | `seeding.py` | `R/seeding.R` | MEDIUM | Low |
| Visualization | `visualize.py` | `R/visualize.R` | MEDIUM | Medium |
| Trainer | `trainer.py` | `R/trainer.R` | LOW | High |

---

## 1. Base Layer (`core/base_layer.py` → `R/core/base_layer.R`)

### Python (PyTorch)

```python
# Python: core/base_layer.py
import torch
import torch.nn as nn

class BaseLayer(nn.Module):
    """Abstract base class for MDITRE layers."""
    
    def __init__(self):
        super(BaseLayer, self).__init__()
        self._registry = {}
    
    def forward(self, x, mask=None):
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def get_params(self):
        """Return learnable parameters."""
        return {name: param.data for name, param in self.named_parameters()}
    
    def set_params(self, params_dict):
        """Set parameters from dictionary."""
        for name, param in self.named_parameters():
            if name in params_dict:
                param.data = params_dict[name]
```

### R (torch R)

```r
# R: R/core/base_layer.R
library(torch)

#' Base Layer for MDITRE
#'
#' @description Abstract base class for all MDITRE layers
#' @export
base_layer <- nn_module(
  "BaseLayer",
  
  initialize = function() {
    self$registry <- list()
  },
  
  forward = function(x, mask = NULL) {
    stop("Subclasses must implement forward()")
  },
  
  get_params = function() {
    # Return named list of parameter values
    params <- list()
    for (name in names(self$parameters)) {
      params[[name]] <- self$parameters[[name]]$data
    }
    return(params)
  },
  
  set_params = function(params_dict) {
    # Set parameters from named list
    for (name in names(params_dict)) {
      if (name %in% names(self$parameters)) {
        self$parameters[[name]]$data <- params_dict[[name]]
      }
    }
  }
)
```

---

## 2. Math Utilities (`core/math_utils.py` → `R/core/math_utils.R`)

### Python

```python
# Python: core/math_utils.py
import torch
import torch.nn.functional as F

def binary_concrete(logits, temperature=0.5, hard=False):
    """
    Binary Concrete (Gumbel-Softmax) relaxation.
    
    Args:
        logits: Input logits
        temperature: Temperature parameter (lower = sharper)
        hard: If True, use straight-through estimator
        
    Returns:
        Soft binary values in [0, 1]
    """
    # Sample Gumbel noise
    u = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    
    # Apply Gumbel-Softmax
    y = torch.sigmoid((logits + gumbel) / temperature)
    
    if hard:
        # Straight-through estimator
        y_hard = torch.round(y)
        y = (y_hard - y).detach() + y
    
    return y

def soft_and(x, dim=-1, epsilon=1e-10):
    """
    Differentiable AND operation using product.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce
        epsilon: Small value for numerical stability
        
    Returns:
        Product along dimension (approximates AND)
    """
    return torch.prod(x + epsilon, dim=dim)

def soft_or(x, dim=-1, epsilon=1e-10):
    """
    Differentiable OR operation.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce
        epsilon: Small value for numerical stability
        
    Returns:
        Probabilistic OR
    """
    return 1 - torch.prod(1 - x + epsilon, dim=dim)
```

### R

```r
# R: R/core/math_utils.R
library(torch)

#' Binary Concrete (Gumbel-Softmax) Relaxation
#'
#' @param logits Input logits tensor
#' @param temperature Temperature parameter (default: 0.5)
#' @param hard Whether to use straight-through estimator (default: FALSE)
#' @return Soft binary values in [0, 1]
#' @export
binary_concrete <- function(logits, temperature = 0.5, hard = FALSE) {
  # Sample Gumbel noise
  u <- torch_rand_like(logits)
  gumbel <- -torch_log(-torch_log(u + 1e-20) + 1e-20)
  
  # Apply Gumbel-Softmax
  y <- torch_sigmoid((logits + gumbel) / temperature)
  
  if (hard) {
    # Straight-through estimator
    y_hard <- torch_round(y)
    y <- (y_hard - y)$detach() + y
  }
  
  return(y)
}

#' Soft AND Operation
#'
#' @param x Input tensor
#' @param dim Dimension to reduce (default: -1)
#' @param epsilon Small value for numerical stability (default: 1e-10)
#' @return Product along dimension
#' @export
soft_and <- function(x, dim = -1L, epsilon = 1e-10) {
  torch_prod(x + epsilon, dim = dim)
}

#' Soft OR Operation
#'
#' @param x Input tensor
#' @param dim Dimension to reduce (default: -1)
#' @param epsilon Small value for numerical stability (default: 1e-10)
#' @return Probabilistic OR
#' @export
soft_or <- function(x, dim = -1L, epsilon = 1e-10) {
  1 - torch_prod(1 - x + epsilon, dim = dim)
}
```

---

## 3. Phylogenetic Focus Layer

### Python

```python
# Python: layers/layer1_phylogenetic_focus/spatial_agg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAgg(nn.Module):
    """Phylogenetic focus layer with learnable embeddings."""
    
    def __init__(self, n_taxa, emb_dim, dist_matrix):
        super().__init__()
        self.n_taxa = n_taxa
        self.emb_dim = emb_dim
        
        # Learnable embeddings
        self.embeddings = nn.Parameter(
            torch.randn(n_taxa, emb_dim) * 0.01
        )
        
        # Fixed phylogenetic distance matrix
        self.register_buffer(
            'dist_matrix',
            torch.tensor(dist_matrix, dtype=torch.float32)
        )
        
        # Temperature parameter
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, n_taxa, n_timepoints)
            mask: (batch, n_timepoints) or None
            
        Returns:
            (batch, n_taxa, n_timepoints) aggregated abundances
        """
        # Compute attention weights from phylogenetic distances
        # weights: (n_taxa, n_taxa)
        weights = F.softmax(
            -self.dist_matrix * torch.abs(self.temperature),
            dim=-1
        )
        
        # Aggregate: (batch, n_taxa, n_timepoints)
        aggregated = torch.einsum('ij,bjt->bit', weights, x)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(aggregated)
            aggregated = aggregated * mask_expanded
        
        return aggregated
```

### R

```r
# R: R/layers/layer1_phylogenetic_focus.R
library(torch)

#' Phylogenetic Focus Layer
#'
#' @description Aggregates taxa based on phylogenetic relationships
#' @param n_taxa Number of taxa
#' @param emb_dim Embedding dimension
#' @param dist_matrix Phylogenetic distance matrix (n_taxa x n_taxa)
#' @export
spatial_agg_layer <- nn_module(
  "SpatialAgg",
  
  initialize = function(n_taxa, emb_dim, dist_matrix) {
    self$n_taxa <- n_taxa
    self$emb_dim <- emb_dim
    
    # Learnable embeddings
    self$embeddings <- nn_parameter(
      torch_randn(n_taxa, emb_dim) * 0.01
    )
    
    # Fixed phylogenetic distance matrix
    self$register_buffer(
      "dist_matrix",
      torch_tensor(dist_matrix, dtype = torch_float())
    )
    
    # Temperature parameter
    self$temperature <- nn_parameter(torch_tensor(1.0))
  },
  
  forward = function(x, mask = NULL) {
    # x: (batch, n_taxa, n_timepoints)
    
    # Compute attention weights from phylogenetic distances
    # weights: (n_taxa, n_taxa)
    weights <- torch_softmax(
      -self$dist_matrix * torch_abs(self$temperature),
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

## 4. Temporal Focus Layer

### Python

```python
# Python: layers/layer2_temporal_focus/time_agg.py
class TimeAgg(nn.Module):
    """Temporal focus layer with soft time windows."""
    
    def __init__(self, n_timepoints, n_windows=1):
        super().__init__()
        self.n_timepoints = n_timepoints
        self.n_windows = n_windows
        
        # Window center positions (0 to 1)
        self.window_centers = nn.Parameter(
            torch.rand(n_windows)
        )
        
        # Window widths
        self.window_widths = nn.Parameter(
            torch.ones(n_windows) * 0.1
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, n_features, n_timepoints)
            mask: (batch, n_timepoints) or None
            
        Returns:
            (batch, n_features, n_windows) temporally focused features
        """
        batch_size, n_features, _ = x.shape
        
        # Normalize window parameters
        centers = torch.sigmoid(self.window_centers)  # [0, 1]
        widths = torch.softplus(self.window_widths)   # > 0
        
        # Create time grid
        time_grid = torch.linspace(
            0, 1, self.n_timepoints,
            device=x.device
        ).view(1, 1, -1)
        
        # Compute Gaussian windows
        # Shape: (n_windows, 1, n_timepoints)
        centers_exp = centers.view(-1, 1, 1)
        widths_exp = widths.view(-1, 1, 1)
        
        windows = torch.exp(
            -((time_grid - centers_exp) ** 2) / (2 * widths_exp ** 2)
        )
        
        # Normalize windows
        windows = windows / (windows.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Apply windows: (batch, n_features, n_windows)
        focused = torch.einsum('wt,bft->bfw', windows.squeeze(1), x)
        
        return focused
```

### R

```r
# R: R/layers/layer2_temporal_focus.R
#' Temporal Focus Layer
#'
#' @description Selects important time windows using soft attention
#' @param n_timepoints Number of time points
#' @param n_windows Number of temporal windows to learn
#' @export
time_agg_layer <- nn_module(
  "TimeAgg",
  
  initialize = function(n_timepoints, n_windows = 1) {
    self$n_timepoints <- n_timepoints
    self$n_windows <- n_windows
    
    # Window center positions (0 to 1)
    self$window_centers <- nn_parameter(
      torch_rand(n_windows)
    )
    
    # Window widths
    self$window_widths <- nn_parameter(
      torch_ones(n_windows) * 0.1
    )
  },
  
  forward = function(x, mask = NULL) {
    # x: (batch, n_features, n_timepoints)
    batch_size <- x$size(1)
    n_features <- x$size(2)
    
    # Normalize window parameters
    centers <- torch_sigmoid(self$window_centers)  # [0, 1]
    widths <- torch_softplus(self$window_widths)   # > 0
    
    # Create time grid
    time_grid <- torch_linspace(
      0, 1, self$n_timepoints,
      device = x$device
    )$view(c(1, 1, -1))
    
    # Compute Gaussian windows
    # Shape: (n_windows, 1, n_timepoints)
    centers_exp <- centers$view(c(-1, 1, 1))
    widths_exp <- widths$view(c(-1, 1, 1))
    
    windows <- torch_exp(
      -((time_grid - centers_exp)^2) / (2 * widths_exp^2)
    )
    
    # Normalize windows
    windows <- windows / (windows$sum(dim = -1L, keepdim = TRUE) + 1e-10)
    
    # Apply windows: (batch, n_features, n_windows)
    focused <- torch_einsum(
      "wt,bft->bfw",
      list(windows$squeeze(2), x)
    )
    
    return(focused)
  }
)
```

---

## 5. Detector Layer

### Python

```python
# Python: layers/layer3_detector/threshold.py
class Threshold(nn.Module):
    """Threshold detector with soft activation."""
    
    def __init__(self, n_detectors):
        super().__init__()
        self.n_detectors = n_detectors
        
        # Learnable thresholds
        self.thresholds = nn.Parameter(
            torch.randn(n_detectors) * 0.1
        )
        
        # Temperature for soft thresholding
        self.temperature = nn.Parameter(
            torch.tensor(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_features) input features
            
        Returns:
            (batch, n_detectors) soft binary activations
        """
        # Expand dimensions for broadcasting
        # x: (batch, n_features, 1)
        # thresholds: (1, 1, n_detectors)
        x_exp = x.unsqueeze(-1)
        thresh_exp = self.thresholds.view(1, 1, -1)
        
        # Soft thresholding with sigmoid
        # activation: (batch, n_features, n_detectors)
        activation = torch.sigmoid(
            (x_exp - thresh_exp) / torch.abs(self.temperature)
        )
        
        # Aggregate over features (max pooling)
        activation = activation.max(dim=1)[0]
        
        return activation
```

### R

```r
# R: R/layers/layer3_detector.R
#' Threshold Detector Layer
#'
#' @description Detects patterns by soft thresholding
#' @param n_detectors Number of detectors
#' @export
threshold_layer <- nn_module(
  "Threshold",
  
  initialize = function(n_detectors) {
    self$n_detectors <- n_detectors
    
    # Learnable thresholds
    self$thresholds <- nn_parameter(
      torch_randn(n_detectors) * 0.1
    )
    
    # Temperature for soft thresholding
    self$temperature <- nn_parameter(
      torch_tensor(0.1)
    )
  },
  
  forward = function(x) {
    # x: (batch, n_features)
    
    # Expand dimensions for broadcasting
    # x: (batch, n_features, 1)
    # thresholds: (1, 1, n_detectors)
    x_exp <- x$unsqueeze(-1L)
    thresh_exp <- self$thresholds$view(c(1, 1, -1))
    
    # Soft thresholding with sigmoid
    # activation: (batch, n_features, n_detectors)
    activation <- torch_sigmoid(
      (x_exp - thresh_exp) / torch_abs(self$temperature)
    )
    
    # Aggregate over features (max pooling)
    activation <- activation$max(dim = 2L)[[1]]
    
    return(activation)
  }
)
```

---

## 6. Complete MDITRE Model

### Python

```python
# Python: models.py (simplified)
class MDITRE(nn.Module):
    """Complete MDITRE model."""
    
    def __init__(self, n_taxa, n_timepoints, n_rules=5,
                 n_detectors_per_rule=3, phylo_dist=None, emb_dim=10):
        super().__init__()
        
        # Layer 1: Phylogenetic Focus
        self.phylo_focus = SpatialAgg(n_taxa, emb_dim, phylo_dist)
        
        # Layer 2: Temporal Focus
        self.temporal_focus = TimeAgg(n_timepoints, n_windows=n_detectors_per_rule)
        
        # Layer 3: Detectors
        self.threshold_detectors = Threshold(n_detectors_per_rule)
        self.slope_detectors = Slope(n_detectors_per_rule)
        
        # Layer 4: Rules
        self.rules = Rules(n_rules, n_detectors_per_rule * 2)
        
        # Layer 5: Classification
        self.classifier = DenseLayer(n_rules)
    
    def forward(self, x, mask=None):
        # x: (batch, n_taxa, n_timepoints)
        
        # Layer 1: Phylogenetic aggregation
        x1 = self.phylo_focus(x, mask)
        
        # Layer 2: Temporal focusing
        x2 = self.temporal_focus(x1, mask)
        
        # Layer 3: Detectors
        x3_thresh = self.threshold_detectors(x2)
        x3_slope = self.slope_detectors(x2)
        x3 = torch.cat([x3_thresh, x3_slope], dim=-1)
        
        # Layer 4: Rules
        x4 = self.rules(x3)
        
        # Layer 5: Classification
        logits = self.classifier(x4)
        
        return logits.squeeze(-1)
```

### R

```r
# R: R/models.R
#' MDITRE Model
#'
#' @description Complete 5-layer MDITRE model
#' @param n_taxa Number of taxa
#' @param n_timepoints Number of time points
#' @param n_rules Number of rules (default: 5)
#' @param n_detectors_per_rule Detectors per rule (default: 3)
#' @param phylo_dist Phylogenetic distance matrix
#' @param emb_dim Embedding dimension (default: 10)
#' @export
mditre_model <- nn_module(
  "MDITRE",
  
  initialize = function(n_taxa, n_timepoints, n_rules = 5,
                       n_detectors_per_rule = 3, 
                       phylo_dist, emb_dim = 10) {
    
    # Layer 1: Phylogenetic Focus
    self$phylo_focus <- spatial_agg_layer(n_taxa, emb_dim, phylo_dist)
    
    # Layer 2: Temporal Focus
    self$temporal_focus <- time_agg_layer(n_timepoints, n_detectors_per_rule)
    
    # Layer 3: Detectors
    self$threshold_detectors <- threshold_layer(n_detectors_per_rule)
    self$slope_detectors <- slope_layer(n_detectors_per_rule)
    
    # Layer 4: Rules
    self$rules <- rule_layer(n_rules, n_detectors_per_rule * 2)
    
    # Layer 5: Classification
    self$classifier <- classification_layer(n_rules)
  },
  
  forward = function(x, mask = NULL) {
    # x: (batch, n_taxa, n_timepoints)
    
    # Layer 1: Phylogenetic aggregation
    x1 <- self$phylo_focus(x, mask)
    
    # Layer 2: Temporal focusing
    x2 <- self$temporal_focus(x1, mask)
    
    # Layer 3: Detectors
    x3_thresh <- self$threshold_detectors(x2)
    x3_slope <- self$slope_detectors(x2)
    x3 <- torch_cat(list(x3_thresh, x3_slope), dim = -1L)
    
    # Layer 4: Rules
    x4 <- self$rules(x3)
    
    # Layer 5: Classification
    logits <- self$classifier(x4)
    
    return(logits$squeeze(-1L))
  }
)
```

---

## 7. Data Loading with phyloseq

### Python

```python
# Python: data_loader/loaders/pickle_loader.py
def load_from_pickle(filepath):
    """Load preprocessed data from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return {
        'X': data['X'],  # (n_subjects, n_taxa, n_timepoints)
        'y': data['y'],  # (n_subjects,)
        'mask': data['mask'],  # (n_subjects, n_timepoints)
        'phylo_dist': data['phylo_dist'],  # (n_taxa, n_taxa)
        'taxa_names': data['taxa_names'],
        'subject_ids': data['subject_ids']
    }
```

### R

```r
# R: R/data_loader/phyloseq_loader.R
#' Load Data from phyloseq Object
#'
#' @param ps phyloseq object
#' @param outcome_var Name of outcome variable in sample_data
#' @param subject_var Name of subject ID variable (default: "subject_id")
#' @param time_var Name of time variable (default: "timepoint")
#' @return Named list with MDITRE data components
#' @export
load_from_phyloseq <- function(ps, outcome_var, 
                               subject_var = "subject_id",
                               time_var = "timepoint") {
  
  library(phyloseq)
  library(ape)
  
  # Extract components
  otu <- as.matrix(otu_table(ps))
  meta <- as.data.frame(sample_data(ps))
  tree <- phy_tree(ps)
  
  # Get unique subjects and timepoints
  subjects <- unique(meta[[subject_var]])
  n_subjects <- length(subjects)
  n_taxa <- ntaxa(ps)
  
  # Determine max timepoints
  max_timepoints <- meta %>%
    group_by(!!sym(subject_var)) %>%
    summarize(n_time = n()) %>%
    pull(n_time) %>%
    max()
  
  # Initialize arrays
  X <- array(0, dim = c(n_subjects, n_taxa, max_timepoints))
  mask <- array(0, dim = c(n_subjects, max_timepoints))
  y <- numeric(n_subjects)
  
  # Fill arrays
  for (i in seq_along(subjects)) {
    subj <- subjects[i]
    subj_rows <- meta[[subject_var]] == subj
    subj_meta <- meta[subj_rows, ]
    subj_otu <- otu[, subj_rows]
    
    # Order by time
    time_order <- order(subj_meta[[time_var]])
    subj_otu <- subj_otu[, time_order]
    subj_meta <- subj_meta[time_order, ]
    
    n_time <- ncol(subj_otu)
    
    # Fill data
    X[i, , 1:n_time] <- subj_otu
    mask[i, 1:n_time] <- 1
    y[i] <- subj_meta[[outcome_var]][1]  # Assume same for all timepoints
  }
  
  # Calculate phylogenetic distances
  phylo_dist <- cophenetic.phylo(tree)
  
  return(list(
    X = X,
    y = y,
    mask = mask,
    phylo_dist = phylo_dist,
    taxa_names = taxa_names(ps),
    subject_ids = subjects,
    tree = tree
  ))
}
```

---

## 8. Training Function

### Python

```python
# Python: Simplified training loop
def train_mditre(model, train_data, val_data, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        outputs = model(train_data['X'], mask=train_data['mask'])
        loss = criterion(outputs, train_data['y'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_data['X'], mask=val_data['mask'])
                val_loss = criterion(val_outputs, val_data['y'])
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    return model
```

### R

```r
# R: R/trainer.R
#' Train MDITRE Model
#'
#' @param model MDITRE model
#' @param train_data List with X, y, mask
#' @param val_data Validation data (optional)
#' @param epochs Number of epochs (default: 100)
#' @param lr Learning rate (default: 0.001)
#' @return Trained model
#' @export
train_mditre <- function(model, train_data, val_data = NULL,
                         epochs = 100, lr = 0.001) {
  
  # Convert to torch tensors if needed
  if (!inherits(train_data$X, "torch_tensor")) {
    train_data$X <- torch_tensor(train_data$X, dtype = torch_float())
    train_data$y <- torch_tensor(train_data$y, dtype = torch_float())
    train_data$mask <- torch_tensor(train_data$mask, dtype = torch_float())
  }
  
  # Setup optimizer and loss
  optimizer <- optim_adam(model$parameters, lr = lr)
  criterion <- nn_bce_with_logits_loss()
  
  # Training loop
  for (epoch in seq_len(epochs)) {
    model$train()
    
    # Forward pass
    outputs <- model(train_data$X, mask = train_data$mask)
    loss <- criterion(outputs, train_data$y)
    
    # Backward pass
    optimizer$zero_grad()
    loss$backward()
    optimizer$step()
    
    # Validation
    if (!is.null(val_data)) {
      model$eval()
      with_no_grad({
        val_outputs <- model(val_data$X, mask = val_data$mask)
        val_loss <- criterion(val_outputs, val_data$y)
      })
    }
    
    # Progress
    if (epoch %% 10 == 0) {
      cat(sprintf("Epoch %d: Loss = %.4f\n", epoch, loss$item()))
    }
  }
  
  return(model)
}
```

---

## 9. Visualization

### Python

```python
# Python: visualize.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rule(model, rule_idx, data):
    """Visualize a specific rule."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract rule parameters
    rule_params = model.rules.get_rule_params(rule_idx)
    
    # Plot 1: Selected taxa
    ax = axes[0, 0]
    taxa_weights = rule_params['taxa_weights']
    top_taxa_idx = taxa_weights.argsort()[-10:]
    ax.barh(range(10), taxa_weights[top_taxa_idx])
    ax.set_title(f'Rule {rule_idx}: Selected Taxa')
    
    # Plot 2: Time windows
    ax = axes[0, 1]
    time_windows = rule_params['time_windows']
    ax.plot(time_windows)
    ax.set_title('Temporal Window')
    
    # Plot 3: Detector activations
    ax = axes[1, 0]
    activations = rule_params['detector_activations']
    sns.heatmap(activations, ax=ax, cmap='RdYlBu_r')
    ax.set_title('Detector Activations')
    
    # Plot 4: Rule predictions
    ax = axes[1, 1]
    predictions = rule_params['predictions']
    ax.scatter(range(len(predictions)), predictions)
    ax.set_title('Rule Output')
    
    plt.tight_layout()
    return fig
```

### R

```r
# R: R/visualize.R
library(ggplot2)
library(patchwork)
library(ggtree)

#' Plot MDITRE Rule
#'
#' @param model Trained MDITRE model
#' @param rule_idx Rule index
#' @param data Data list with taxa_names, tree, etc.
#' @return ggplot object (patchwork composite)
#' @export
plot_rule <- function(model, rule_idx, data) {
  
  # Extract rule parameters
  rule_params <- get_rule_params(model, rule_idx)
  
  # Plot 1: Selected taxa
  p1 <- {
    taxa_weights <- rule_params$taxa_weights
    top_idx <- order(taxa_weights, decreasing = TRUE)[1:10]
    
    df <- data.frame(
      taxa = data$taxa_names[top_idx],
      weight = taxa_weights[top_idx]
    )
    
    ggplot(df, aes(x = reorder(taxa, weight), y = weight)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(title = paste("Rule", rule_idx, ": Selected Taxa"),
           x = "Taxa", y = "Weight") +
      theme_minimal()
  }
  
  # Plot 2: Time windows
  p2 <- {
    time_windows <- rule_params$time_windows
    
    df <- data.frame(
      time = seq_along(time_windows),
      weight = time_windows
    )
    
    ggplot(df, aes(x = time, y = weight)) +
      geom_line(color = "darkred", size = 1) +
      geom_area(alpha = 0.3, fill = "darkred") +
      labs(title = "Temporal Window",
           x = "Time Point", y = "Weight") +
      theme_minimal()
  }
  
  # Plot 3: Detector activations heatmap
  p3 <- {
    activations <- rule_params$detector_activations
    
    df <- reshape2::melt(activations)
    names(df) <- c("Sample", "Detector", "Activation")
    
    ggplot(df, aes(x = Detector, y = Sample, fill = Activation)) +
      geom_tile() +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                          midpoint = 0.5) +
      labs(title = "Detector Activations") +
      theme_minimal()
  }
  
  # Plot 4: Rule predictions
  p4 <- {
    predictions <- rule_params$predictions
    
    df <- data.frame(
      sample = seq_along(predictions),
      prediction = predictions,
      outcome = data$y
    )
    
    ggplot(df, aes(x = sample, y = prediction, color = factor(outcome))) +
      geom_point(size = 3) +
      scale_color_manual(values = c("0" = "blue", "1" = "red")) +
      labs(title = "Rule Output",
           x = "Sample", y = "Prediction", color = "True Outcome") +
      theme_minimal()
  }
  
  # Combine plots
  combined <- (p1 + p2) / (p3 + p4)
  combined + plot_annotation(
    title = paste("MDITRE Rule", rule_idx, "Visualization"),
    theme = theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))
  )
}
```

---

## 10. Seeding for Reproducibility

### Python

```python
# Python: seeding.py
import random
import numpy as np
import torch

def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### R

```r
# R: R/seeding.R
#' Set Random Seeds for Reproducibility
#'
#' @param seed Integer seed value (default: 42)
#' @export
set_mditre_seeds <- function(seed = 42) {
  # R base random
  set.seed(seed)
  
  # torch random
  torch::torch_manual_seed(seed)
  
  # CUDA if available
  if (torch::cuda_is_available()) {
    torch::cuda_manual_seed_all(seed)
  }
  
  # Make operations deterministic
  torch::torch_set_deterministic(TRUE)
  
  message(paste("All random seeds set to:", seed))
}

#' Get MDITRE Seed Generator
#'
#' @param base_seed Base seed for generation
#' @return Function that generates deterministic seeds
#' @export
get_mditre_seed_generator <- function(base_seed = 42) {
  counter <- 0
  
  function() {
    counter <<- counter + 1
    digest::digest2int(paste(base_seed, counter))
  }
}
```

---

## Summary

This reference provides direct Python-to-R translations for MDITRE's core components. Key patterns:

1. **Module Definition**: `class X(nn.Module)` → `x_module <- nn_module("X", ...)`
2. **Parameters**: `nn.Parameter()` → `nn_parameter()`
3. **Buffers**: `register_buffer()` → `register_buffer()`
4. **Forward Pass**: `def forward(self, x)` → `forward = function(x)`
5. **Tensor Operations**: Most translate directly (e.g., `torch.softmax` → `torch_softmax`)
6. **Data Loading**: NumPy/pickle → phyloseq/tidyverse
7. **Visualization**: matplotlib → ggplot2

**Next Steps**:
1. Use these examples as templates
2. Test each component independently
3. Build up to full model
4. Validate against Python outputs

---

**Document Version**: 1.0.1  
**Last Updated**: November 1, 2025
