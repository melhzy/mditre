# MDITRE Code Refactoring Plan
## Modularized and Dynamic Design for Enhanced Maintainability and Extensibility

### Current Issues
1. **Monolithic files**: `trainer.py` (4875 lines), `data.py` (2988 lines), `models.py` (541 lines)
2. **Mixed concerns**: Training, visualization, data processing in single files
3. **Hard-coded dependencies**: Difficult to extend with new models or data modalities
4. **Limited plugin architecture**: No dynamic model/data loader registration
5. **Tight coupling**: Models, trainers, and data loaders tightly coupled

### Proposed Modular Structure

```
mditre/
├── __init__.py                      # Package initialization, version info
├── config/                          # Configuration management
│   ├── __init__.py
│   ├── base_config.py              # Base configuration classes
│   ├── model_config.py             # Model-specific configs
│   ├── training_config.py          # Training hyperparameters
│   └── data_config.py              # Data loading configs
│
├── core/                           # Core abstractions and interfaces
│   ├── __init__.py
│   ├── base_model.py               # Abstract base model class
│   ├── base_trainer.py             # Abstract base trainer class
│   ├── base_data_loader.py         # Abstract data loader class
│   ├── registry.py                 # Dynamic registration system
│   └── exceptions.py               # Custom exceptions
│
├── models/                         # Model implementations
│   ├── __init__.py
│   ├── components/                 # Reusable model components
│   │   ├── __init__.py
│   │   ├── spatial_aggregation.py  # Phylogenetic focus layers
│   │   ├── temporal_aggregation.py # Temporal focus layers
│   │   ├── threshold.py            # Threshold layers
│   │   ├── rules.py                # Rule layers
│   │   └── utils.py                # Model utility functions
│   │
│   ├── mditre.py                   # MDITRE model
│   ├── mditre_abun.py              # Abundance-only variant
│   └── factory.py                  # Model factory for dynamic creation
│
├── data/                           # Data handling
│   ├── __init__.py
│   ├── loaders/                    # Data loaders by type
│   │   ├── __init__.py
│   │   ├── base_loader.py
│   │   ├── amplicon_16s_loader.py  # 16S rRNA data
│   │   ├── metaphlan_loader.py     # Shotgun metagenomics
│   │   ├── phyloseq_loader.py      # R phyloseq objects
│   │   └── custom_loader.py        # User custom format
│   │
│   ├── preprocessors/              # Data preprocessing
│   │   ├── __init__.py
│   │   ├── normalizer.py
│   │   ├── filter.py
│   │   └── transformer.py
│   │
│   ├── phylogenetic/               # Phylogenetic tree handling
│   │   ├── __init__.py
│   │   ├── tree_parser.py
│   │   ├── distance_calculator.py
│   │   └── embedder.py
│   │
│   └── datasets.py                 # PyTorch Dataset classes
│
├── training/                       # Training infrastructure
│   ├── __init__.py
│   ├── trainer.py                  # Main trainer class (refactored)
│   ├── callbacks/                  # Training callbacks
│   │   ├── __init__.py
│   │   ├── early_stopping.py
│   │   ├── checkpoint.py
│   │   ├── logging.py
│   │   └── visualization.py
│   │
│   ├── optimizers/                 # Optimizer configurations
│   │   ├── __init__.py
│   │   └── param_groups.py
│   │
│   ├── schedulers/                 # Learning rate schedulers
│   │   ├── __init__.py
│   │   └── custom_schedulers.py
│   │
│   └── losses/                     # Loss functions
│       ├── __init__.py
│       ├── bce_loss.py
│       └── custom_losses.py
│
├── evaluation/                     # Model evaluation
│   ├── __init__.py
│   ├── metrics.py                  # Evaluation metrics
│   ├── cross_validation.py         # CV strategies
│   └── statistical_tests.py        # Statistical comparisons
│
├── visualization/                  # Visualization components
│   ├── __init__.py
│   ├── rule_viz.py                 # Rule visualization (refactored)
│   ├── plots/                      # Different plot types
│   │   ├── __init__.py
│   │   ├── phylogenetic_plots.py
│   │   ├── temporal_plots.py
│   │   ├── abundance_plots.py
│   │   └── rule_plots.py
│   │
│   └── gui/                        # GUI components
│       ├── __init__.py
│       ├── main_window.py
│       └── widgets.py
│
├── utils/                          # Utilities
│   ├── __init__.py
│   ├── logging.py                  # Logging utilities
│   ├── device.py                   # GPU/CPU management
│   ├── io.py                       # File I/O
│   ├── math_utils.py               # Mathematical utilities
│   └── validation.py               # Input validation
│
├── cli/                            # Command-line interface
│   ├── __init__.py
│   ├── train.py                    # Training CLI
│   ├── evaluate.py                 # Evaluation CLI
│   ├── visualize.py                # Visualization CLI
│   └── convert.py                  # Data conversion CLI
│
├── extensions/                     # Extension system
│   ├── __init__.py
│   ├── README.md                   # How to create extensions
│   └── examples/                   # Example extensions
│       ├── custom_model.py
│       └── custom_data_loader.py
│
└── tests/                          # Unit tests
    ├── __init__.py
    ├── test_models.py
    ├── test_data_loaders.py
    ├── test_training.py
    └── test_visualization.py
```

### Key Design Patterns

#### 1. Registry Pattern
Allows dynamic registration of models, data loaders, and other components:

```python
from mditre.core.registry import Registry

# Register models
MODEL_REGISTRY = Registry('model')

@MODEL_REGISTRY.register('mditre')
class MDITRE(BaseModel):
    pass

# Use models dynamically
model = MODEL_REGISTRY.get('mditre')(config)
```

#### 2. Factory Pattern
Create objects based on configuration:

```python
from mditre.models.factory import ModelFactory

model = ModelFactory.create(
    model_type='mditre',
    config=model_config
)
```

#### 3. Strategy Pattern
Different strategies for data loading, training, etc.:

```python
from mditre.data.loaders import DataLoaderStrategy

loader = DataLoaderStrategy.get_loader(
    data_type='16s',
    config=data_config
)
```

#### 4. Observer Pattern
Callbacks for training events:

```python
from mditre.training.callbacks import CallbackList

callbacks = CallbackList([
    EarlyStopping(patience=10),
    ModelCheckpoint(save_best=True),
    TensorBoardLogger()
])
```

### Implementation Phases

#### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Create base classes and interfaces
- [ ] Implement registry system
- [ ] Set up configuration management
- [ ] Create factory classes

#### Phase 2: Model Refactoring (Week 3-4)
- [ ] Extract model components
- [ ] Create component library
- [ ] Implement model factory
- [ ] Add model registry

#### Phase 3: Data Layer (Week 5-6)
- [ ] Separate data loaders by type
- [ ] Create preprocessing pipeline
- [ ] Implement data registry
- [ ] Add validation

#### Phase 4: Training Infrastructure (Week 7-8)
- [ ] Refactor trainer class
- [ ] Implement callback system
- [ ] Create optimizer/scheduler management
- [ ] Add logging infrastructure

#### Phase 5: Visualization (Week 9-10)
- [ ] Modularize visualization code
- [ ] Create plot library
- [ ] Separate GUI components
- [ ] Add export functionality

#### Phase 6: Testing & Documentation (Week 11-12)
- [ ] Write unit tests
- [ ] Integration tests
- [ ] Update documentation
- [ ] Create migration guide

### Benefits

1. **Maintainability**
   - Single Responsibility Principle: Each module has one purpose
   - Clear code organization and navigation
   - Easier debugging and testing

2. **Extensibility**
   - Easy to add new models via registry
   - Plugin architecture for custom components
   - No need to modify core code for extensions

3. **Testability**
   - Small, focused modules are easier to test
   - Mock dependencies easily
   - Better test coverage

4. **Reusability**
   - Model components can be reused
   - Data loaders work across projects
   - Training utilities are generic

5. **Collaboration**
   - Multiple developers can work on different modules
   - Clear interfaces reduce conflicts
   - Better code review process

6. **Future-Proofing**
   - Easy to add new data modalities (metabolomics, transcriptomics)
   - Support for multi-modal learning
   - Integration with other frameworks

### Migration Strategy

1. **Backward Compatibility**
   - Keep old API with deprecation warnings
   - Provide adapter classes
   - Gradual migration path

2. **Documentation**
   - Migration guide for existing users
   - New tutorials for refactored code
   - API reference documentation

3. **Testing**
   - Ensure all existing tests pass
   - Add tests for new functionality
   - Performance benchmarking

### Example Extension: Adding a New Model

```python
# extensions/custom_model.py
from mditre.core.base_model import BaseModel
from mditre.core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register('custom_lstm_mditre')
class LSTMMDITREModel(BaseModel):
    """Custom MDITRE variant with LSTM layers"""
    
    def __init__(self, config):
        super().__init__(config)
        # Initialize LSTM components
        
    def forward(self, x, mask=None):
        # Custom forward pass
        pass
```

### Example Extension: Adding a New Data Loader

```python
# extensions/metabolomics_loader.py
from mditre.data.loaders.base_loader import BaseDataLoader
from mditre.core.registry import DATA_LOADER_REGISTRY

@DATA_LOADER_REGISTRY.register('metabolomics')
class MetabolomicsLoader(BaseDataLoader):
    """Load metabolomics time-series data"""
    
    def load(self, path):
        # Custom loading logic
        pass
    
    def preprocess(self, data):
        # Custom preprocessing
        pass
```

### Configuration Example

```python
# config/experiment_config.yaml
model:
  type: 'mditre'
  num_rules: 5
  num_otu_centers: 10
  num_time_centers: 5
  embedding_dim: 10

data:
  type: '16s'
  path: '/path/to/data'
  preprocessing:
    - normalize
    - filter_low_abundance

training:
  epochs: 2000
  batch_size: 32
  optimizer:
    type: 'adam'
    lr: 0.001
  scheduler:
    type: 'step_lr'
    step_size: 500
  callbacks:
    - early_stopping:
        patience: 100
    - checkpoint:
        save_best: true

evaluation:
  metrics:
    - f1_score
    - auc_roc
  cross_validation:
    n_folds: 5
```

### Next Steps

1. Review and approve refactoring plan
2. Set up development branch
3. Begin Phase 1 implementation
4. Iterative development with code reviews
5. Continuous integration testing
6. Documentation updates
7. Community feedback
