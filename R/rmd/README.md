# R MDITRE Tutorials

Interactive RMarkdown tutorials for learning and using R MDITRE.

## Overview

This directory contains RMarkdown (`.Rmd`) files that demonstrate R MDITRE functionality. These tutorials are similar to Jupyter notebooks but use R's RMarkdown format.

## Architecture

**R MDITRE** is an R interface that bridges to **Python MDITRE**:

```
┌─────────────────────────────────────┐
│    R Environment (R 4.5.2+)        │
│  ┌───────────────────────────────┐ │
│  │     R MDITRE Interface        │ │
│  │  • R workflows & syntax       │ │
│  │  • Visualization (ggplot2)    │ │
│  │  • phyloseq integration       │ │
│  └──────────┬────────────────────┘ │
│             │ reticulate             │
└─────────────┼──────────────────────┘
              │
┌─────────────▼──────────────────────┐
│  Python MDITRE Backend             │
│  (MDITRE conda environment)        │
│                                     │
│  • Python 3.12+                    │
│  • PyTorch 2.6.0+cu124             │
│  • mditre package v1.0.0           │
│  • GPU acceleration (CUDA)         │
└─────────────────────────────────────┘
```

**Benefits:**
- ✅ R users work in familiar R environment
- ✅ Computation leverages native PyTorch performance
- ✅ Same models as Python MDITRE (consistency)
- ✅ GPU acceleration via CUDA

## Prerequisites

### Two-Package System

The MDITRE ecosystem requires both packages:

1. **Python MDITRE** (Backend) - Must be installed first
2. **R MDITRE** (Frontend) - Connects to Python via reticulate

### Python MDITRE Setup (Backend)

All tutorials require the **MDITRE conda environment** with Python MDITRE:

```bash
# Create and activate conda environment
conda create -n MDITRE python=3.12
conda activate MDITRE

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Python MDITRE in development mode
cd path/to/mditre/Python
pip install -e .

# Verify installation
python -c "import mditre; print('mditre version:', mditre.__version__)"
```

### R MDITRE Setup (Frontend)

```r
# Install required R packages
install.packages(c("reticulate", "ggplot2", "dplyr", "tidyr", "patchwork", "knitr", "rmarkdown"))

# Configure reticulate to use MDITRE conda environment
library(reticulate)
use_condaenv("MDITRE", required = TRUE)

# Verify connection to Python MDITRE
torch_py <- import("torch")
mditre <- import("mditre.models")
cat("PyTorch:", torch_py$`__version__`, "\n")
cat("mditre:", mditre$`__version__`, "\n")
```

## Available Tutorials

### 1. `example_quick_start.Rmd` - Quick Start Demo (5 minutes)

**Purpose**: Rapid demonstration of complete R MDITRE workflow

**Content**:
- Generate synthetic microbiome data (50 subjects, 30 OTUs, 8 timepoints)
- Create and initialize MDITRE model (3 rules)
- Train for 20 epochs
- Evaluate performance with metrics
- Visualize results (training curve, predictions, confusion matrix)

**Run Time**: ~5 minutes

**To Run**:
```r
# In RStudio
# Open example_quick_start.Rmd
# Click "Knit" button

# Or in R console
rmarkdown::render("example_quick_start.Rmd")
```

### 2. `tutorial_1_getting_started.Rmd` - Getting Started with R MDITRE

**Purpose**: Comprehensive introduction to R MDITRE architecture

**Content**:
- Setup and environment configuration
- Five-layer architecture deep dive:
  - Layer 1: Phylogenetic Focus (SpatialAgg)
  - Layer 2: Temporal Focus (TimeAgg)
  - Layer 3: Detectors (Threshold, Slope)
  - Layer 4: Rule Logic (soft AND)
  - Layer 5: Classification (DenseLayer)
- Complete MDITRE model creation
- Inference on synthetic data
- Visualization with ggplot2

**Run Time**: ~10 minutes

**To Run**:
```r
rmarkdown::render("tutorial_1_getting_started.Rmd")
```

### 3. `tutorial_2_training.Rmd` - Training on Microbiome Data

**Purpose**: Full training pipeline demonstration

**Content**:
- **Data Preparation**:
  - Synthetic microbiome generation (Dirichlet distributions)
  - Train-test split (70/30)
  - PyTorch tensor conversion
- **Model Setup**:
  - MDITRE initialization with proper parameters
  - Optimizer and loss configuration
- **Training Loop**:
  - Mini-batch gradient descent (16 samples, 50 epochs)
  - Progress monitoring
- **Evaluation**:
  - Accuracy, F1, AUC-ROC metrics
  - Confusion matrix visualization
  - ROC curve analysis
- **Interpretation**:
  - Rule importance extraction
  - Temporal focus analysis
  - Weight visualizations

**Run Time**: ~15 minutes

**To Run**:
```r
rmarkdown::render("tutorial_2_training.Rmd")
```

## Tutorial Features

All tutorials include:

✅ **Automatic Python setup** - Installs mditre in development mode  
✅ **GPU support** - Automatically uses CUDA if available  
✅ **Clean visualizations** - ggplot2 + patchwork plots  
✅ **Inline code execution** - See results as you work  
✅ **HTML output** - Floating TOC, code folding, syntax highlighting  
✅ **Copy-paste ready** - All code chunks are self-contained

## Running Tutorials in RStudio

1. **Open RStudio**
2. **Open a `.Rmd` file**
3. **Option A - Interactive**: Run code chunks individually with `Ctrl+Enter` (Windows) or `Cmd+Enter` (Mac)
4. **Option B - Full render**: Click the "Knit" button to generate HTML output

## Running Tutorials from Command Line

```bash
# Render all tutorials
Rscript -e "rmarkdown::render('example_quick_start.Rmd')"
Rscript -e "rmarkdown::render('tutorial_1_getting_started.Rmd')"
Rscript -e "rmarkdown::render('tutorial_2_training.Rmd')"
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'mditre'"

**Solution**: The tutorials automatically install mditre, but you can manually install:

```bash
conda activate MDITRE
cd path/to/mditre/Python
pip install -e .
```

### Issue: "CUDA not available"

**Solution**: This is normal if you don't have an NVIDIA GPU. Models will run on CPU (slower but functional).

### Issue: "reticulate can't find conda environment"

**Solution**: Specify the full conda path:

```r
library(reticulate)
use_condaenv("MDITRE", conda = "path/to/conda", required = TRUE)

# Or use full Python path
use_python("path/to/conda/envs/MDITRE/bin/python", required = TRUE)
```

### Issue: Slow rendering

**Solution**: RMarkdown tutorials can take 5-15 minutes to render completely. For faster iteration, run code chunks interactively in RStudio instead of knitting.

## Directory Structure

```
R/rmd/
├── README.md                          # This file
├── example_quick_start.Rmd            # 5-minute demo
├── tutorial_1_getting_started.Rmd     # Architecture introduction
└── tutorial_2_training.Rmd            # Training pipeline
```

## Output Files

After rendering, you'll get HTML files:

```
R/rmd/
├── example_quick_start.html
├── tutorial_1_getting_started.html
└── tutorial_2_training.html
```

Open these in your web browser for interactive documentation!

## Next Steps

After completing these tutorials:

1. **Try real data**: Load your own microbiome datasets (phyloseq format)
2. **Experiment with parameters**: Adjust num_rules, num_otu_centers, learning rates
3. **Explore visualizations**: Use the R visualization functions to interpret your models
4. **Run comprehensive tests**: Execute `R/run_mditre_tests.R` to verify your setup

## Additional Resources

- **R MDITRE Source**: `R/R/*.R` - All layer implementations
- **Test Suite**: `R/run_mditre_tests.R` - 15 comprehensive tests
- **Python Reference**: `Python/tests/test_all.py` - 39 Python tests
- **Examples**: `R/examples/*.R` - Additional code examples

---

**R MDITRE** | RMarkdown Tutorials | November 2, 2025
