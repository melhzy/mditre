# MDITRE: scalable and interpretable machine learning for predicting host status from temporal microbiome dynamics
We present a new differentiable model that learns human interpretable rules from microbiome time-series data for classifying the status of the human host.

# Installation

## Requirements
- Python 3.8+ (tested with Python 3.8-3.12)
- PyTorch 1.7+ (tested with PyTorch 2.6+)
- CUDA 11.0+ for GPU support (optional but recommended)

## Quick Install

### Ubuntu 24.04 / Linux

#### With GPU Support (CUDA 12.x)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv

# Create virtual environment (recommended)
python3 -m venv mditre_env
source mditre_env/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install MDITRE and dependencies
pip install scikit-learn matplotlib seaborn pandas scipy dendropy ete3 ipykernel PyQt5

# Install MDITRE from source
git clone https://github.com/melhzy/mditre.git
cd mditre
pip install -e .
```

#### CPU Only
```bash
# Create virtual environment
python3 -m venv mditre_env
source mditre_env/bin/activate

# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install MDITRE and dependencies
pip install scikit-learn matplotlib seaborn pandas scipy dendropy ete3 ipykernel PyQt5

# Install MDITRE from source
git clone https://github.com/melhzy/mditre.git
cd mditre
pip install -e .
```

### Windows 11

#### With GPU Support (CUDA 12.x)
```powershell
# Create conda environment (recommended for Windows)
conda create -n mditre python=3.12 -y
conda activate mditre

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install MDITRE and dependencies
pip install scikit-learn matplotlib seaborn pandas scipy dendropy ete3 ipykernel PyQt5

# Install MDITRE from source
git clone https://github.com/melhzy/mditre.git
cd mditre
pip install -e .
```

#### CPU Only
```powershell
# Create conda environment
conda create -n mditre python=3.12 -y
conda activate mditre

# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install MDITRE and dependencies
pip install scikit-learn matplotlib seaborn pandas scipy dendropy ete3 ipykernel PyQt5

# Install MDITRE from source
git clone https://github.com/melhzy/mditre.git
cd mditre
pip install -e .
```

### macOS (Apple Silicon M1/M2/M3 or Intel)

```bash
# Create virtual environment
python3 -m venv mditre_env
source mditre_env/bin/activate

# Install PyTorch (optimized for Apple Silicon)
pip install torch torchvision torchaudio

# Install MDITRE and dependencies
pip install scikit-learn matplotlib seaborn pandas scipy dendropy ete3 ipykernel PyQt5

# Install MDITRE from source
git clone https://github.com/melhzy/mditre.git
cd mditre
pip install -e .
```

**Note for macOS:** CUDA is not supported on macOS. PyTorch will use Metal Performance Shaders (MPS) on Apple Silicon for GPU acceleration, or CPU for Intel Macs.

## Verify Installation

```python
# Test the installation
python -c "import mditre; import torch; print(f'MDITRE installed. PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

## Using PyPI (Alternative)

```bash
pip install mditre
pip install torch torchvision torchaudio
```

**Note:** PyPI package may not have the latest features. Installing from source is recommended.

# Usage and Tutorials

MDITRE is designed to predict host status (e.g., healthy vs. diseased) from longitudinal microbiome data using interpretable, human-readable rules. The method works with both 16S rRNA amplicon sequencing and shotgun metagenomic data.

## Quick Start Example

```python
from mditre.trainer import parse, Trainer

# Parse command line arguments or create args object
args = parse()

# Initialize trainer
trainer = Trainer(args)

# Load your microbiome time-series data
trainer.load_data()

# Run cross-validation training loop
trainer.train_loop()

# Visualize learned rules using the GUI
# The trained model will output interpretable rules that can be explored
```

## Tutorials

We provide comprehensive Jupyter notebook tutorials demonstrating MDITRE workflows:

### Tutorial 1: 16S rRNA Data Analysis
**[Tutorial_Bokulich_16S_data.ipynb](https://github.com/melhzy/mditre/blob/master/mditre/tutorials/Tutorial_Bokulich_16S_data.ipynb)**

This tutorial demonstrates:
- Loading and preprocessing 16S rRNA amplicon sequencing data
- Creating phylogenetic trees from sequence data
- Setting up MDITRE configuration
- Training the model to predict infant diet (breastfed vs. formula)
- Interpreting learned rules with the graphical user interface
- Analyzing which microbial taxa and time windows are important

**Dataset:** Bokulich et al. infant gut microbiome study (2016)

### Tutorial 2: Shotgun Metagenomics Data
**[Tutorial_2_metaphlan_data.ipynb](https://github.com/melhzy/mditre/blob/master/mditre/tutorials/Tutorial_2_metaphlan_data.ipynb)**

This tutorial demonstrates:
- Loading MetaPhlAn taxonomic abundance profiles
- Working with shotgun metagenomic data
- Handling MetaPhlAn's taxonomic hierarchy
- Training MDITRE on larger-scale data
- Post-hoc analysis of learned patterns

**Dataset:** Example shotgun metagenomics data processed with MetaPhlAn

### Tutorial 3: Complete 16S Workflow
**[Tutorial_1_16s_data.ipynb](https://github.com/melhzy/mditre/blob/master/mditre/tutorials/Tutorial_1_16s_data.ipynb)**

A comprehensive tutorial covering:
- DADA2 sequence variant processing
- Phylogenetic placement with pplacer
- Data preprocessing and quality control
- Model training and hyperparameter tuning
- Rule interpretation and visualization

## Data Format Requirements

### Input Data Structure

MDITRE expects data in the following format:

1. **Microbiome Abundance Table**: 
   - Rows: samples
   - Columns: microbial features (OTUs, ASVs, or taxa)
   - Values: relative abundances or counts

2. **Sample Metadata**:
   - Subject IDs
   - Time points (numeric, e.g., days from start)
   - Binary labels (e.g., 0 = healthy, 1 = diseased)

3. **Phylogenetic Information**:
   - Newick format phylogenetic tree
   - OR pairwise distance matrix
   - Relates microbial features based on evolutionary relationships

### Example Data Structure

```python
import pandas as pd
import numpy as np

# Abundance table (samples x OTUs)
abundance_df = pd.DataFrame({
    'OTU1': [0.10, 0.15, 0.08, ...],
    'OTU2': [0.05, 0.03, 0.12, ...],
    # ... more OTUs
})

# Metadata
metadata_df = pd.DataFrame({
    'subject_id': ['S1', 'S1', 'S2', 'S2', ...],
    'timepoint': [0, 30, 0, 30, ...],  # days
    'label': [0, 0, 1, 1, ...]  # binary outcome
})

# Phylogenetic tree (Newick format)
tree_file = "phylogenetic_tree.nwk"
```

## Running MDITRE

### Method 1: Command Line Interface

Create a configuration file (`.cfg`) with your experiment parameters:

```ini
[Paths]
data_dir = /path/to/your/data
output_dir = /path/to/output

[Data]
abundance_file = abundance_table.csv
metadata_file = sample_metadata.csv
tree_file = phylogenetic_tree.nwk

[Model]
num_rules = 5
num_otu_centers = 10
num_time_centers = 5
embedding_dim = 10

[Training]
epochs = 2000
batch_size = 32
learning_rate = 0.001
num_folds = 5

[Task]
label_column = disease_status
subject_column = subject_id
time_column = timepoint
```

Run MDITRE:

```bash
python mditre/tutorials/model_run_tutorial.py --data /path/to/config.cfg
```

### Method 2: Python API

```python
from mditre.data import load_from_pickle, get_dist_matrix
from mditre.models import MDITRE
from mditre.trainer import Trainer
import torch

# Load your preprocessed data
data = load_from_pickle('preprocessed_data.pkl')

# Extract components
X = data['abundances']  # (n_subjects, n_otus, n_timepoints)
y = data['labels']      # (n_subjects,)
dist_matrix = data['phylo_distances']  # (n_otus, n_otus)

# Initialize model
model = MDITRE(
    num_rules=5,
    num_otus=X.shape[1],
    num_otu_centers=10,
    num_time=X.shape[2],
    num_time_centers=5,
    dist=dist_matrix,
    emb_dim=10
)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ... training loop ...
```

## Interpreting Results

### Understanding MDITRE Rules

MDITRE learns interpretable rules of the form:

**Rule:** IF (detector1 AND detector2 AND ... detectorN) THEN predict class

Where each **detector** specifies:
- **Phylogenetic group**: Which related microbes to aggregate
- **Time window**: When to look in the time series
- **Threshold**: Abundance or slope threshold
- **Type**: Abundance-based or rate-of-change (slope)

### Example Learned Rule

```
Rule 1: Predicts "Healthy" status
  Detector 1: IF abundance of Clostridiales group 
              between days 120-180 is > 7%
  AND
  Detector 2: IF Bacteroides acidifaciens is increasing
              between days 90-150
  THEN: Predict Healthy (weight = 2.3)
```

### Visualization GUI

After training, use the graphical interface to:
1. View all learned rules
2. See which subjects activate each rule
3. Explore phylogenetic groupings
4. Examine temporal patterns
5. Export rule summaries

```python
# Launch visualization GUI (after training)
from mditre.visualize import visualize_rules

visualize_rules(
    model=trained_model,
    data=data,
    output_dir='./results/'
)
```

## Configuration Options

Detailed configuration options are available in the [configuration documentation](https://github.com/melhzy/mditre/blob/master/mditre/docs/config_doc.pdf).

### Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `num_rules` | Number of rules to learn | 3-10 |
| `num_otu_centers` | Phylogenetic focus points per rule | 5-20 |
| `num_time_centers` | Temporal focus points per rule | 3-10 |
| `embedding_dim` | Phylogenetic embedding dimension | 5-15 |
| `learning_rate` | Gradient descent step size | 0.0001-0.01 |
| `epochs` | Training iterations | 1000-5000 |
| `batch_size` | Samples per gradient update | 16-128 |

### Recommended Settings by Dataset Size

**Small datasets (< 50 subjects):**
```
num_rules = 3
num_otu_centers = 5
num_time_centers = 3
epochs = 2000
```

**Medium datasets (50-200 subjects):**
```
num_rules = 5
num_otu_centers = 10
num_time_centers = 5
epochs = 3000
```

**Large datasets (> 200 subjects):**
```
num_rules = 7-10
num_otu_centers = 15-20
num_time_centers = 7-10
epochs = 5000
```

## Example Use Cases

### 1. Predicting Disease Onset from Infant Gut Microbiome

```python
# Goal: Predict type 1 diabetes risk from gut microbiome time series
# Data: Monthly samples over first 3 years of life

# MDITRE can identify:
# - Which bacterial groups change before disease onset
# - Critical time windows (e.g., months 12-18)
# - Protective vs. risk-associated patterns
```

### 2. Classifying Diet from Microbiome Dynamics

```python
# Goal: Distinguish dietary interventions from microbiome changes
# Data: Samples before/during/after diet change

# MDITRE can reveal:
# - Microbes that respond to diet shifts
# - How quickly microbiome adapts (slope detectors)
# - Temporal succession patterns
```

### 3. Predicting Clinical Outcomes

```python
# Goal: Predict preterm birth from vaginal microbiome
# Data: Weekly samples throughout pregnancy

# MDITRE can discover:
# - Early warning signals (specific time windows)
# - Dysbiotic patterns associated with risk
# - Protective microbial communities
```

## Troubleshooting

### Common Issues

**1. CUDA out of memory:**
```bash
# Reduce batch size or use CPU
python script.py --batch-size 16
```

**2. Slow training:**
```bash
# Use GPU acceleration
# Reduce num_rules, num_otu_centers, or num_time_centers
```

**3. Poor performance:**
- Increase epochs (more training time)
- Adjust learning rates
- Check data quality and class balance
- Try different random seeds

**4. No rules activated:**
- Decrease threshold sensitivity
- Increase num_rules
- Check if labels are correct

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/melhzy/mditre/issues)
- **Documentation**: See `mditre/docs/` folder
- **Examples**: See `mditre/tutorials/` folder

## MDITRE workflow on 16S rRNA and shotgun metagenomics data 
We provide 2 tutorials, one for 16s-based data [here](https://github.com/melhzy/mditre/blob/master/mditre/tutorials/Tutorial_Bokulich_16S_data.ipynb) and another for shotgun metagenomics (Metaphlan) based data [here](https://github.com/melhzy/mditre/blob/master/mditre/tutorials/Tutorial_2_metaphlan_data.ipynb), which show how to use MDITRE for data loading and preprocessing, running the model code and using the GUI to interpret the learned rules for post-hoc analysis.

## Configuration options
MDITRE operation requires a list of configuration options to be passed as arguments as explained [here](https://github.com/melhzy/mditre/blob/master/mditre/docs/config_doc.pdf).

# Citation
If you use MDITRE in your research, please cite:

Maringanti VS, Bucci V, Gerber GK. 2022. MDITRE: Scalable and Interpretable Machine Learning for Predicting Host Status from Temporal Microbiome Dynamics. mSystems 7:e00132-22. https://doi.org/10.1128/msystems.00132-22
