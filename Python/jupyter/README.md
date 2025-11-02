# MDITRE Jupyter Notebooks

This folder contains all Jupyter notebook-related content for the MDITRE project.

## Contents

### Main Notebook
- **`run_mditre_test.ipynb`** - End-to-end MDITRE training demonstration
  - Complete workflow from data loading to model training
  - Uses 10 epochs for faster testing (configurable)
  - Demonstrates seeding module integration
  - Includes visualization of training curves and rule activations
  - See `NOTEBOOK_FIXES.md` for implementation details

### Tutorials
Located in `tutorials/` subdirectory:

1. **`Tutorial_Bokulich_16S_data.ipynb`** - 16S rRNA Analysis
   - Load 16S amplicon data (DADA2)
   - Build phylogenetic trees
   - Train model to predict infant diet
   - Interpret rules with GUI
   - Dataset: Bokulich et al. infant gut microbiome (2016)

2. **`Tutorial_2_metaphlan_data.ipynb`** - Shotgun Metagenomics
   - Work with MetaPhlAn profiles
   - Handle shotgun metagenomic data
   - Process taxonomic hierarchies

3. **`Tutorial_1_16s_data.ipynb`** - Complete Workflow
   - DADA2 sequence processing
   - Phylogenetic placement (pplacer)
   - Data QC and preprocessing
   - Hyperparameter tuning
   - Rule visualization

### Tutorial Data
Located in `tutorials/` subdirectory:

- **`datasets/`** - Tutorial datasets
  - `processed/` - Processed data for tutorials
  - `raw/` - Raw data files (16S, metagenomics)
    - `bokulich/` - Bokulich infant microbiome data
    - `david/` - David longitudinal data
    - `digiulio/` - DiGiulio pregnancy data
    - `karelia/` - Karelia allergy study data
    - `t1d/` - Type 1 diabetes data

- **`dada2/`** - DADA2 processing results
  - `dada2_results/` - Output files (abundance, sequences, taxonomy)
  - `dada2_scripts/` - R scripts for DADA2 analysis

- **`pplacer_files/`** - Phylogenetic placement files
  - Reference package for phylogenetic placement
  - Scripts for running pplacer

- **`logs/`** - Tutorial execution logs

## Documentation

- **`NOTEBOOK_FIXES.md`** - Detailed documentation of notebook debugging and fixes
  - Summary of 8 major issues fixed
  - Before/after code examples
  - Seeding updates explanation
  - Test results and usage examples

## Quick Start

### Run Main Notebook
```bash
# Navigate to jupyter folder
cd jupyter

# Launch Jupyter
jupyter notebook run_mditre_test.ipynb
```

### Run Tutorials
```bash
# Navigate to tutorials
cd jupyter/tutorials

# Launch any tutorial
jupyter notebook Tutorial_Bokulich_16S_data.ipynb
```

## Configuration

### Training Speed
The main notebook is configured for fast testing with 10 epochs:

```python
N_EPOCHS = 10  # Reduced for faster testing (from 50)
```

- **10 epochs**: ~30 seconds training time (testing/development)
- **50 epochs**: ~153 seconds training time (full training)

Modify `N_EPOCHS` in the notebook to adjust training duration.

### Reproducibility
All notebooks use the MDITRE seeding module for reproducibility:

```python
from mditre.seeding import MDITRESeedGenerator, set_random_seeds

# Generate master seed
seed_gen = MDITRESeedGenerator()
master_seed = seed_gen.generate_seeds(1)[0]

# Set all random number generators
set_random_seeds(master_seed)
```

See `../docs/SEEDING.md` for details on the seeding system.

## Requirements

### Python Packages
```bash
pip install jupyter
pip install torch numpy pandas matplotlib seaborn
pip install scikit-learn scipy
```

Or install MDITRE with all dependencies:
```bash
cd ..
pip install -e .
```

### Additional Tools (for tutorials)
- **R** with DADA2 package (for 16S processing)
- **pplacer** (for phylogenetic placement)

## Folder Structure

```
jupyter/
├── README.md                          # This file
├── NOTEBOOK_FIXES.md                  # Notebook debugging documentation
├── run_mditre_test.ipynb              # Main training notebook
└── tutorials/                         # Tutorial notebooks
    ├── Tutorial_Bokulich_16S_data.ipynb
    ├── Tutorial_2_metaphlan_data.ipynb
    ├── Tutorial_1_16s_data.ipynb
    ├── datasets/                      # Tutorial data
    │   ├── processed/                 # Processed datasets
    │   └── raw/                       # Raw data files
    ├── dada2/                         # DADA2 analysis
    │   ├── dada2_results/
    │   └── dada2_scripts/
    ├── pplacer_files/                 # Phylogenetic placement
    └── logs/                          # Execution logs
```

## Notes

- Notebooks are configured to use the MDITRE package from the parent directory
- All paths are relative to the notebook location
- Seeding ensures reproducibility across all notebook runs
- Training curves and visualizations are automatically saved to `../mditre_outputs/`

## Support

For issues or questions:
1. Check `NOTEBOOK_FIXES.md` for common problems and solutions
2. Refer to main project `README.md` for general MDITRE documentation
3. See `../docs/SEEDING.md` for seeding-related questions
