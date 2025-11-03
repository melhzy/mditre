# MDITRE Modular Architecture

## Overview

This document describes the new modular architecture for MDITRE (Microbiome Differentiable Interpretable Temporal Rule Engine). The codebase has been refactored from a monolithic structure into a clean, 5-layer modular design that mirrors the biological and mathematical structure described in the publication (Maringanti et al. 2022).

## Architecture Layers

MDITRE consists of five distinct computational layers, each now implemented as an independent, composable module:

### Layer 1: Phylogenetic Focus (`mditre/layers/layer1_phylogenetic_focus/`)

**Purpose**: Aggregate microbial time-series based on phylogenetic relationships between OTUs.

**Classes**:
- `SpatialAgg`: Uses fixed phylogenetic distance matrix for aggregation
- `SpatialAggDynamic`: Learns OTU embeddings dynamically in phylogenetic space

**Key Innovation**: Allows the model to focus on taxonomically related groups rather than individual OTUs, improving both interpretability and generalization.

**Input**: `(batch, num_otus, time_points)`  
**Output**: `(batch, num_rules, num_otus, time_points)` or `(batch, num_rules, num_otu_centers, time_points)`

### Layer 2: Temporal Focus (`mditre/layers/layer2_temporal_focus/`)

**Purpose**: Select contiguous time windows that are important for the prediction task.

**Classes**:
- `TimeAgg`: Computes both average abundance and average slope within learned time windows
- `TimeAggAbun`: Simplified version computing only average abundance

**Key Innovation**: Automatically discovers critical time periods (e.g., gestational weeks, disease onset phases) without manual feature engineering.

**Input**: `(batch, num_rules, num_otus, time_points)`  
**Output**: 
- `TimeAgg`: `(abundance, slope)` both `(batch, num_rules, num_otus)`
- `TimeAggAbun`: `(batch, num_rules, num_otus)`

### Layer 3: Detector (`mditre/layers/layer3_detector/`)

**Purpose**: Apply learned thresholds to identify significant abundance and slope patterns.

**Classes**:
- `Threshold`: Detects significant abundance levels
- `Slope`: Detects significant slope (trend) patterns

**Key Innovation**: Differentiable threshold detection using sigmoid approximation, enabling end-to-end gradient-based learning.

**Input**: `(batch, num_rules, num_otus)`  
**Output**: `(batch, num_rules, num_otus)` - binary-like responses

### Layer 4: Rule (`mditre/layers/layer4_rule/`)

**Purpose**: Combine detector responses using approximate logical AND operations.

**Classes**:
- `Rules`: Implements differentiable AND via product: `AND(x1, x2, ...) ≈ ∏(1 - αi(1 - xi))`

**Key Innovation**: Uses binary concrete relaxation for detector selection, enabling automated discovery of which OTU patterns matter for each rule.

**Input**: `(batch, num_rules, num_otus)`  
**Output**: `(batch, num_rules)` - rule activations

### Layer 5: Classification (`mditre/layers/layer5_classification/`)

**Purpose**: Linear classifier with rule selection for outcome prediction.

**Classes**:
- `DenseLayer`: Combines abundance and slope rules for prediction
- `DenseLayerAbun`: Abundance-only version for simpler models

**Key Innovation**: Binary concrete rule selection learns which rules contribute to final prediction, improving interpretability.

**Input**: `(batch, num_rules)` [and optionally `(batch, num_rules)` for slope]  
**Output**: `(batch,)` - log odds for binary classification

## Core Utilities (`mditre/core/`)

### Base Layer (`base_layer.py`)

**`BaseLayer`**: Abstract base class providing common interface for all layers
- `forward(*args, **kwargs)`: Forward pass (must be implemented)
- `init_params(init_args)`: Parameter initialization (must be implemented)
- `get_config()`: Returns layer configuration
- `get_layer_info()`: Returns detailed layer metadata

**`LayerRegistry`**: Registry for dynamic layer management
- `@LayerRegistry.register(layer_type, layer_name)`: Decorator to register implementations
- `LayerRegistry.get_layer(layer_type, layer_name)`: Retrieve registered layer class
- `LayerRegistry.list_layers()`: List all available implementations

### Mathematical Utilities (`math_utils.py`)

**Functions**:
- `binary_concrete(x, k, hard, use_noise)`: Differentiable binary selection with Gumbel noise
- `unitboxcar(x, mu, l, k)`: Smooth boxcar function using sigmoid approximation
- `transf_log(x, u, l)`: Bounded logarithmic transformation
- `inv_transf_log(x, u, l)`: Inverse bounded transformation

## Directory Structure

```
mditre/
├── models.py                    # Original monolithic implementation (backward compatibility)
├── core/
│   ├── __init__.py
│   ├── base_layer.py            # Base classes and registry
│   └── math_utils.py            # Shared mathematical operations
└── layers/
    ├── __init__.py              # Main layers exports
    ├── layer1_phylogenetic_focus/
    │   ├── __init__.py
    │   └── spatial_agg.py
    ├── layer2_temporal_focus/
    │   ├── __init__.py
    │   └── time_agg.py
    ├── layer3_detector/
    │   ├── __init__.py
    │   └── threshold.py
    ├── layer4_rule/
    │   ├── __init__.py
    │   └── rules.py
    └── layer5_classification/
        ├── __init__.py
        └── dense_layer.py
```

## Migration Guide

### From Original Models (models.py) to Modular Architecture

#### Old Way (Monolithic):
```python
from mditre.models import SpatialAggDynamic, TimeAgg, Threshold, Slope, Rules, DenseLayer

# All classes imported from single monolithic file
model = MDITRE(num_rules, num_otus, num_otu_centers, num_time, num_time_centers, dist, emb_dim)
```

#### New Way (Modular):
```python
# Option 1: Import from main layers module
from mditre.layers import SpatialAggDynamic, TimeAgg, Threshold, Slope, Rules, DenseLayer

# Option 2: Import from specific layer modules
from mditre.layers.layer1_phylogenetic_focus import SpatialAggDynamic
from mditre.layers.layer2_temporal_focus import TimeAgg
from mditre.layers.layer3_detector import Threshold, Slope
from mditre.layers.layer4_rule import Rules
from mditre.layers.layer5_classification import DenseLayer

# Build custom model with modular components
class CustomMDITRE(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.phylo = SpatialAggDynamic(num_rules, num_otu_centers, otu_embeddings, emb_dim, num_otus)
        self.temporal = TimeAgg(num_rules, num_otu_centers, num_time, num_time_centers)
        self.detector_abun = Threshold(num_rules, num_otu_centers, num_time_centers)
        self.detector_slope = Slope(num_rules, num_otu_centers, num_time_centers)
        self.rule_abun = Rules(num_rules, num_otu_centers, num_time_centers)
        self.rule_slope = Rules(num_rules, num_otu_centers, num_time_centers)
        self.classifier = DenseLayer(num_rules, 1)
```

### Using LayerRegistry for Dynamic Models

```python
from mditre.core import LayerRegistry

# Dynamically select layer implementations
phylo_class = LayerRegistry.get_layer('layer1', 'spatial_agg_dynamic')
temporal_class = LayerRegistry.get_layer('layer2', 'time_agg')

# List available implementations
available_layers = LayerRegistry.list_layers()
print(available_layers)
# {'layer1': ['spatial_agg', 'spatial_agg_dynamic'],
#  'layer2': ['time_agg', 'time_agg_abun'],
#  ...}
```

## Benefits of Modular Architecture

### 1. **Enhanced Maintainability**
- Each layer is self-contained with clear responsibilities
- Changes to one layer don't affect others
- Easier to debug and test individual components

### 2. **Improved Extensibility**
- Easy to add new layer implementations (e.g., different phylogenetic aggregation strategies)
- Can mix and match layer implementations to create custom models
- LayerRegistry enables runtime layer selection

### 3. **Better Code Organization**
- Clear separation between layers mirrors the mathematical architecture
- Shared utilities extracted to core module
- Consistent interface via BaseLayer abstract class

### 4. **Facilitated Research**
- Swap individual layers to test hypotheses
- Compare different implementations of same layer type
- Add new data modalities by creating new layer implementations

### 5. **Backward Compatibility**
- Original `models.py` still works for existing code
- Saved model checkpoints can still be loaded
- Gradual migration path for existing projects

## Extending the Architecture

### Adding a New Layer Implementation

1. **Create the layer class**:
```python
# mditre/layers/layer1_phylogenetic_focus/my_new_agg.py
from ...core.base_layer import BaseLayer, LayerRegistry

@LayerRegistry.register('layer1', 'my_new_agg')
class MyNewAggregation(BaseLayer):
    def __init__(self, num_rules, num_otus, layer_name="my_new_agg", **kwargs):
        config = {'num_rules': num_rules, 'num_otus': num_otus}
        super().__init__(layer_name, config)
        # Initialize parameters
        
    def forward(self, x, **kwargs):
        # Implement forward pass
        return x
    
    def init_params(self, init_args):
        # Implement parameter initialization
        pass
```

2. **Register in `__init__.py`**:
```python
# mditre/layers/layer1_phylogenetic_focus/__init__.py
from .spatial_agg import SpatialAgg, SpatialAggDynamic
from .my_new_agg import MyNewAggregation

__all__ = ['SpatialAgg', 'SpatialAggDynamic', 'MyNewAggregation']
```

3. **Use in models**:
```python
from mditre.layers.layer1_phylogenetic_focus import MyNewAggregation

model = CustomMDITRE(
    phylo_layer=MyNewAggregation(num_rules, num_otus),
    ...
)
```

## Best Practices

1. **Always inherit from `BaseLayer`** when creating new layers
2. **Implement both `forward()` and `init_params()`** methods
3. **Register layers with `@LayerRegistry.register`** for discoverability
4. **Document layer configurations** in the config dict passed to super().__init__()
5. **Use type hints** for better IDE support and documentation
6. **Store intermediate results** as instance attributes for debugging/visualization

## Future Enhancements

Possible future developments enabled by this architecture:

1. **Alternative Phylogenetic Representations**
   - Tree-based aggregation
   - Hierarchical clustering approaches
   - Graph neural network-based aggregation

2. **New Temporal Patterns**
   - Multi-scale time windows
   - Periodic pattern detection
   - Attention-based temporal mechanisms

3. **Advanced Detectors**
   - Non-linear threshold functions
   - Multi-threshold detection
   - Context-dependent thresholds

4. **Rule Combinations**
   - OR operations alongside AND
   - Hierarchical rule structures
   - Probabilistic rule logic

5. **Classification Extensions**
   - Multi-class classification
   - Regression tasks
   - Structured output prediction

## Testing

The original test suite in `test_mditre_comprehensive.py` validates the complete MDITRE pipeline. When developing new layers:

1. Test individual layer forward passes
2. Test parameter initialization
3. Test gradient flow through layers
4. Test layer composition in full models
5. Validate against known good outputs

## References

- Maringanti, Veda Sheersh, Bucci, Vanni, and Gerber, Georg K. (2022). "MDITRE: Scalable and Interpretable Machine Learning for Predicting Host Status from Temporal Microbiome Dynamics." *mSystems* 7(3), e00132-22. https://doi.org/10.1128/msystems.00132-22
- GitHub Repository: https://github.com/melhzy/mditre

## Support

For questions or issues:
- Open an issue on GitHub
- Refer to the publication for mathematical details
- Check the tutorial notebooks in `mditre/tutorials/`
