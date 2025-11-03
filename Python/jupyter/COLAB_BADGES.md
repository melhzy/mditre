# Open in Google Colab - Quick Links

Click the badges below to open notebooks directly in Google Colab:

---

## 🧪 Quick Start & Testing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/mditre/blob/master/Python/jupyter/run_mditre_test.ipynb)
**`run_mditre_test.ipynb`** - Quick training & validation with synthetic data

---

## 📚 Tutorials

### Tutorial 1: 16S Data (David et al., 2014)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/mditre/blob/master/Python/jupyter/tutorials/Tutorial_1_16s_data.ipynb)
**`Tutorial_1_16s_data.ipynb`** - Analyze diet-based microbiome changes

### Tutorial 2: Metaphlan Data (Kostic et al., 2015)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/mditre/blob/master/Python/jupyter/tutorials/Tutorial_2_metaphlan_data.ipynb)
**`Tutorial_2_metaphlan_data.ipynb`** - Type 1 diabetes prediction from shotgun metagenomics

### Tutorial 3: Bokulich 16S Data (Bokulich et al., 2016)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/mditre/blob/master/Python/jupyter/tutorials/Tutorial_Bokulich_16S_data.ipynb)
**`Tutorial_Bokulich_16S_data.ipynb`** - Diet classification (breastfed vs formula-fed)

---

## 🧠 Architecture Deep Dive

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/melhzy/mditre/blob/master/neural_network/five_layer_architecture_demo.ipynb)
**`five_layer_architecture_demo.ipynb`** - Interactive demonstration of MDITRE's 5-layer neural network architecture

---

## 💡 Usage Instructions

### First Time Setup (in Colab):

1. Click any badge above to open the notebook
2. **Run the Colab setup cell** (cell #2) to:
   - Install MDITRE package
   - Mount Google Drive (for data access)
3. **Enable GPU** (recommended for faster training):
   - Runtime → Change runtime type → GPU (T4)
4. Execute the remaining cells

### Data Files:

Tutorials require data files. You have two options:

**Option A: Upload to Google Drive**
- Upload data files to `/content/drive/MyDrive/mditre_data/`
- Modify paths in notebooks to match your Drive structure

**Option B: Download in Colab**
```python
# Example: Download data directly in Colab
!wget https://raw.githubusercontent.com/gerberlab/mditre/master/path/to/data.csv
```

---

## 🎯 Features

All notebooks are now fully compatible with Google Colab:

✅ Automatic package installation  
✅ Google Drive integration  
✅ GPU support (with instructions)  
✅ Static visualizations (inline matplotlib)  
✅ 100% backward compatible with local Jupyter  

---

## 📖 Documentation

- **Compatibility Report**: See `COLAB_COMPATIBILITY_REPORT.md` for technical details
- **Implementation Summary**: See `COLAB_FIXES_APPLIED.md` for changes made
- **Main README**: [github.com/gerberlab/mditre](https://github.com/gerberlab/mditre)

---

## ⚠️ Known Limitations

**Interactive GUI in Colab:**
- RuleVisualizer GUI displays as **static images** in Colab
- For full interactive features, run locally with `%matplotlib qt`
- This is a Colab limitation (no Qt backend support)

**Data Loading:**
- Tutorials assume local file paths by default
- Modify paths for Google Drive or use direct downloads

---

## 🔗 Alternative: Direct Links

Copy-paste these URLs into your browser:

**Run MDITRE Test:**
```
https://colab.research.google.com/github/melhzy/mditre/blob/master/Python/jupyter/run_mditre_test.ipynb
```

**Tutorial 1 (16S):**
```
https://colab.research.google.com/github/melhzy/mditre/blob/master/Python/jupyter/tutorials/Tutorial_1_16s_data.ipynb
```

**Tutorial 2 (Metaphlan):**
```
https://colab.research.google.com/github/melhzy/mditre/blob/master/Python/jupyter/tutorials/Tutorial_2_metaphlan_data.ipynb
```

**Tutorial 3 (Bokulich):**
```
https://colab.research.google.com/github/melhzy/mditre/blob/master/Python/jupyter/tutorials/Tutorial_Bokulich_16S_data.ipynb
```

---

## 🤝 Contributing

Found an issue with Colab compatibility? Please open an issue on GitHub:
https://github.com/melhzy/mditre/issues

---

**Last Updated**: November 3, 2025
