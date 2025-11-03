# MDITRE Neural Network Architecture Demo

This folder contains interactive Jupyter notebooks demonstrating the MDITRE 5-layer architecture.

## Notebooks

### `five_layer_architecture_demo.ipynb`
**Comprehensive demonstration of MDITRE's 5-layer neural network architecture**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/mditre/blob/master/neural_network/five_layer_architecture_demo.ipynb)

**What it demonstrates:**
- 🧬 **Layer 1**: Phylogenetic Focus (groups taxa by evolutionary relationships)
- ⏰ **Layer 2**: Temporal Focus (captures time-series patterns)
- 🎯 **Layer 3**: Detector (identifies presence/absence patterns)
- 📊 **Layer 4**: Rule Layer (learns interpretable weights)
- 🔮 **Layer 5**: Classification (makes final predictions)

**Features:**
- ✅ **Reproducible seeding** with [seedhash](https://github.com/melhzy/seedhash) - guaranteed identical results
- ✅ Synthetic data generation aligned with MDITRE paper methodology
- ✅ Biologically-inspired temporal patterns (abundance + slope dynamics)
- ✅ Layer-by-layer data flow visualization
- ✅ Complete training example with gradient descent
- ✅ Interpretability analysis with visualizations
- ✅ Architecture diagrams
- ✅ **Google Colab compatible** - runs with one click!

## Quick Start

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" badge above to run in your browser with GPU support. No installation required!

### Option 2: Local Jupyter
```bash
# From the mditre root directory
cd neural_network
jupyter notebook five_layer_architecture_demo.ipynb
```

### Option 3: VS Code
Open the `.ipynb` file directly in VS Code with the Jupyter extension.

## Requirements

The notebook automatically installs dependencies when run in Colab. For local use:
```bash
pip install torch numpy matplotlib seaborn
# Install seedhash for reproducible seeding
pip install "git+https://github.com/melhzy/seedhash.git#subdirectory=Python[torch]"
# Install MDITRE
cd ../Python
pip install -e .
```

**Dependencies:**
- PyTorch (deep learning framework)
- NumPy (numerical operations)
- matplotlib & seaborn (visualizations)
- **seedhash** (reproducible seed generation from experiment names)

## What You'll Learn

1. **Reproducible experiments** with hash-based seed generation (seedhash)
2. **How to structure microbiome time-series data** for MDITRE with biological realism
3. **Paper-aligned synthetic data** with compositional constraints and phylogenetic structure
4. **How each layer transforms the data** with visual examples
5. **How to train the model** with gradient descent
6. **How to interpret learned rules** and visualize model parameters
7. **The complete architecture** from input to prediction

## Reproducibility 🔐

This notebook uses **[seedhash](https://github.com/melhzy/seedhash)** for deterministic seed generation:

```python
import random
from seedhash import SeedHashGenerator

# Generate seed from experiment name
seed_gen = SeedHashGenerator("mditre_demo_v1")

# Manual seeding for all frameworks (compatible with all seedhash versions)
# Note: NumPy and PyTorch require seed < 2^32, so we use modulo for compatibility
numpy_seed = seed_gen.seed_number % (2**32)
torch_seed = seed_gen.seed_number % (2**32)

random.seed(seed_gen.seed_number)
np.random.seed(numpy_seed)
torch.manual_seed(torch_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Benefits:**
- ✅ Same experiment name → Same results everywhere
- ✅ Seeds Python random, NumPy, and PyTorch
- ✅ Deterministic mode for CUDA/cuDNN operations
- ✅ MD5 hash verification: `seed_gen.get_hash()`
- ✅ Version-compatible: Works with any seedhash version
- ✅ Framework-safe: Handles large seeds automatically
- ✅ Run notebook anytime, anywhere - get identical outputs!


## Output Examples

The notebook generates:
- � **Reproducibility reports** (seed numbers, MD5 hashes, framework status)
- 📊 **Pattern analysis** (quantitative validation of synthetic patterns)
- �📈 **Training curves** (loss and accuracy over epochs)
- 🎨 **Data flow visualizations** (heatmaps, bar charts, time series)
- 🧬 **Biological pattern plots** (clade abundances, temporal slopes)
- 🧠 **Learned parameter visualizations** (embeddings, thresholds, weights)
- 📐 **Architecture diagrams** (complete model structure)

## Use Cases

- **Learning**: Understand how MDITRE works internally
- **Development**: Test modifications to the architecture
- **Teaching**: Demonstrate interpretable deep learning for microbiome analysis
- **Debugging**: Visualize intermediate outputs for troubleshooting

## Next Steps

After completing this demo:
1. Try the real-data tutorials in `../tutorials/`
2. Experiment with different hyperparameters
3. Apply MDITRE to your own microbiome datasets
4. Read the full documentation in `../docs/`

## Support

- 📖 [Full Documentation](../README.md)
- 🐛 [Report Issues](https://github.com/melhzy/mditre/issues)
- 💬 [GitHub Discussions](https://github.com/melhzy/mditre/discussions)

---

**MDITRE v1.0.1** - Microbiome Differentiable Interpretable Temporal Rule Engine
