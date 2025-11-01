# MDITRE Comprehensive Testing Plan

**Based on:** Maringanti et al. (2022) - "MDITRE: Scalable and Interpretable Machine Learning for Predicting Host Status from Temporal Microbiome Dynamics" - mSystems Volume 7 Issue 5

**Prepared:** November 1, 2025

---

## Executive Summary

This testing plan covers all key features, innovations, and claims from the MDITRE publication. It is organized into 15 major test categories covering architecture, performance benchmarking, interpretability, scalability, and biological validation.

---

## 1. Core Architecture Tests

### 1.1 Five-Layer Neural Network Architecture
**Paper Reference:** "MDITRE can be represented as a five-layer neural network"

**Tests:**
- ✅ **T1.1.1:** Validate Layer 1 (Phylogenetic Focus) - SpatialAgg and SpatialAggDynamic
  - Test phylogenetic distance-based grouping
  - Validate embedding space anchoring
  - Test kappa (concentration) and eta (center) parameters
  
- ✅ **T1.1.2:** Validate Layer 2 (Temporal Focus) - TimeAgg and TimeAggAbun  
  - Test temporal window focusing (mu and sigma parameters)
  - Validate "soft" boxcar function approximation
  - Test both abundance aggregation and rate of change computation
  
- ✅ **T1.1.3:** Validate Layer 3 (Detector Layer) - Threshold and Slope
  - Test threshold detectors for abundance levels
  - Test slope detectors for rate of change
  - Validate continuous approximation to binary activation
  
- ✅ **T1.1.4:** Validate Layer 4 (Rule Layer)
  - Test "soft AND" operation over detectors
  - Validate conjunction logic approximation
  - Test rule activation computation
  
- ✅ **T1.1.5:** Validate Layer 5 (Classification Layer)
  - Test weighted aggregation of rule activations
  - Validate final prediction computation
  - Test binary classification output

### 1.2 Differentiability Tests
**Paper Reference:** "fully differentiable architecture that enables scalability while maintaining interpretability"

**Tests:**
- ✅ **T1.2.1:** Gradient flow through all layers
  - Validate gradients exist for all parameters
  - Test gradient computation for input data
  - Verify no gradient blocking
  
- ✅ **T1.2.2:** Relaxation techniques validation
  - Test binary concrete selector differentiability
  - Validate unitboxcar function smoothness
  - Test soft AND approximation continuity
  
- ⏳ **T1.2.3:** Straight-through estimator
  - Test hard vs soft binary concrete modes
  - Validate gradient estimation in discrete approximations

### 1.3 Model Variants
**Paper Reference:** "Both MDITRE and MDITREAbun variants"

**Tests:**
- ✅ **T1.3.1:** MDITRE (full model with slopes)
  - Test with both abundance and rate of change detectors
  - Validate temporal dynamics capture
  
- ✅ **T1.3.2:** MDITREAbun (abundance-only variant)
  - Test with only abundance detectors
  - Validate simplified architecture
  - Compare performance to full MDITRE

---

## 2. Phylogenetic Focus Mechanism Tests

### 2.1 Microbiome Group Focus Functions
**Paper Reference:** "phylogenetic focus functions that perform 'soft' selections over sets of microbes"

**Tests:**
- ⏳ **T2.1.1:** Phylogenetic embedding validation
  - Test embedding space initialization from distance matrix
  - Validate phylogenetic relationship preservation
  - Test anchor point learning
  
- ⏳ **T2.1.2:** Soft selection mechanism
  - Test concentration parameter (kappa) effect on selection sharpness
  - Validate smooth transition between selected/unselected taxa
  - Test gradient behavior during learning
  
- ⏳ **T2.1.3:** Phylogenetic subtree selection
  - Test ability to focus on coherent phylogenetic clades
  - Validate biological relevance of selected groups
  - Test multiple focus centers per rule

### 2.2 Distance-Based Aggregation
**Tests:**
- ⏳ **T2.2.1:** Distance metric validation
  - Test with phylogenetic distances
  - Test with alternative distance measures
  - Validate distance normalization
  
- ⏳ **T2.2.2:** Aggregation weight computation
  - Test exponential decay with distance
  - Validate weight normalization
  - Test edge cases (zero distance, infinite distance)

---

## 3. Temporal Focus Mechanism Tests

### 3.1 Time Window Selection
**Paper Reference:** "temporal focus functions that compute average or rate of change over temporally focused time windows"

**Tests:**
- ⏳ **T3.1.1:** Soft time window approximation
  - Test unitboxcar function implementation
  - Validate smooth Heaviside approximation
  - Test window center (mu) and width (sigma) learning
  
- ⏳ **T3.1.2:** Time window positioning
  - Test automatic focus on relevant time periods
  - Validate handling of study boundaries
  - Test multiple overlapping windows
  
- ⏳ **T3.1.3:** Rate of change computation
  - Test temporal derivative approximation
  - Validate slope calculation over focused windows
  - Test handling of sparse/irregular sampling

### 3.2 Temporal Mask Handling
**Tests:**
- ⏳ **T3.2.1:** Missing time point handling
  - Test mask application for irregular sampling
  - Validate computation with missing data
  - Test gradient behavior with masks
  
- ⏳ **T3.2.2:** Variable-length sequences
  - Test with different numbers of time points per subject
  - Validate proper averaging over available time points

---

## 4. Interpretability Tests

### 4.1 Human-Interpretable Rules
**Paper Reference:** "learns human-interpretable rules which explicitly incorporate microbiome- and temporal-specific features"

**Tests:**
- ⏳ **T4.1.1:** Rule readability
  - Generate English descriptions of learned rules
  - Validate logical structure (AND/OR combinations)
  - Test rule simplification for presentation
  
- ⏳ **T4.1.2:** Detector interpretation
  - Extract detector parameters (taxa, time windows, thresholds)
  - Validate biological meaningfulness
  - Test detector activation patterns
  
- ⏳ **T4.1.3:** Rule weight interpretation
  - Test odds ratio computation for rules
  - Validate contribution to final prediction
  - Test rule importance ranking

### 4.2 Visualization Capabilities
**Paper Reference:** "graphical user interface for visualizations of learned rules"

**Tests:**
- ⏳ **T4.2.1:** Rule visualization
  - Test phylogenetic tree highlighting for selected taxa
  - Validate time window overlay on longitudinal data
  - Test detector activation heatmaps
  
- ⏳ **T4.2.2:** Subject-level predictions
  - Visualize which rules activate for each subject
  - Test truth table generation (detectors × subjects)
  - Validate prediction explanation per subject
  
- ⏳ **T4.2.3:** Interactive exploration
  - Test clicking on rules/detectors to drill down
  - Validate parameter display and formatting
  - Test export of visualizations

---

## 5. Performance Benchmarking Tests

### 5.1 Semi-Synthetic Data Experiments
**Paper Reference:** "benchmarking on semi-synthetic time-series data sets"

**Tests:**
- ⏳ **T5.1.1:** One-perturbation scenario
  - Generate data with single clade perturbation
  - Test increasing numbers of subjects (20, 24, 32, 48, 64, 128, 256, 512, 1024)
  - Test increasing time points (6, 10, 15, 20, 25, 30)
  - Compare F1 score and AUC against baselines
  
- ⏳ **T5.1.2:** Two-perturbation scenario
  - Generate data with two clade perturbations
  - Test same subject/time point ranges
  - Validate performance on harder task
  
- ⏳ **T5.1.3:** Noise robustness
  - Generate test sets with increasing noise (1, 10, 100, 1000, 10000)
  - Test generalization across noise levels
  - Validate graceful degradation

### 5.2 Real Data Benchmarking
**Paper Reference:** "eight classification tasks from seven published human microbiome data sets"

**Tests:**
- ⏳ **T5.2.1:** Bokulich et al. - Diet classification (n=37)
  - Test breastfed vs formula classification
  - Use 16S amplicon data
  - 5-fold repeated cross-validation (5 reps, 10 seeds)
  
- ⏳ **T5.2.2:** Bokulich et al. - Birth mode classification (n=37)
  - Test vaginal vs C-section classification
  - Same data as T5.2.1, different labels
  
- ⏳ **T5.2.3:** David et al. - Dietary intervention (n=20)
  - Test animal vs plant-based diet
  - 16S amplicon data
  
- ⏳ **T5.2.4:** DiGiulio et al. - Preterm birth (n=37)
  - Test at-term vs preterm delivery
  - Vaginal microbiome data (highly imbalanced: 6 vs 31)
  
- ⏳ **T5.2.5:** Vatanen et al. - Nationality (n=117)
  - Test Russian vs Estonian/Finnish
  - Pediatric gut microbiome
  
- ⏳ **T5.2.6:** Kostic et al. - Type 1 diabetes (n=17)
  - Test normal vs T1D development
  - Shotgun metagenomics (smallest dataset)
  
- ⏳ **T5.2.7:** Brooks et al. - Birth mode (n=30)
  - Test vaginal vs C-section
  - Shotgun metagenomics
  
- ⏳ **T5.2.8:** Shao et al. - Birth mode (n=282)
  - Test vaginal vs C-section
  - Shotgun metagenomics (largest dataset)
  - True hold-out validation (75/25 split)

### 5.3 Comparator Methods
**Paper Reference:** "benchmarked against L1 regularized logistic regression and random forest"

**Tests:**
- ⏳ **T5.3.1:** L1 logistic regression baseline
  - Implement comparable feature engineering
  - Test on all benchmarking datasets
  - Compare F1 and AUC metrics
  
- ⏳ **T5.3.2:** Random forest baseline
  - Test black-box nonlinear method
  - Use same evaluation protocol
  - Statistical comparison (Mann-Whitney U test)
  
- ⏳ **T5.3.3:** MITRE comparison (when feasible)
  - Run original MITRE on small datasets
  - Compare predictive performance
  - Document runtime differences

---

## 6. Scalability and Runtime Tests

### 6.1 Computational Efficiency
**Paper Reference:** "orders-of-magnitude faster run-times... MDITRE completed analysis in 24 min vs MITRE unable to run"

**Tests:**
- ⏳ **T6.1.1:** Runtime scaling with subjects
  - Measure training time for 20, 32, 48, 64, 128, 256, 512, 1024 subjects
  - Plot runtime vs subject count
  - Compare to MITRE (up to 64 subjects)
  - Expected: 1000x+ speedup at 1024 subjects
  
- ⏳ **T6.1.2:** Runtime scaling with time points
  - Measure training time for 6, 10, 15, 20, 25, 30 time points
  - Validate linear/sub-linear scaling
  
- ⏳ **T6.1.3:** Runtime scaling with OTUs
  - Test with varying numbers of taxa (50, 100, 500, 1000, 5000)
  - Validate efficiency of phylogenetic focus
  
- ⏳ **T6.1.4:** GPU acceleration validation
  - Measure CPU vs GPU runtime
  - Test batch processing efficiency
  - Validate memory usage on GPU

### 6.2 Memory Efficiency
**Tests:**
- ⏳ **T6.2.1:** Memory scaling with dataset size
  - Monitor memory usage during training
  - Test on large datasets (>200 subjects)
  - Validate no memory leaks
  
- ⏳ **T6.2.2:** Batch processing
  - Test mini-batch gradient descent
  - Validate memory efficiency with batching
  - Test various batch sizes

### 6.3 Convergence Properties
**Tests:**
- ⏳ **T6.3.1:** Training convergence
  - Monitor loss curves over epochs
  - Test convergence speed
  - Validate stopping criteria
  
- ⏳ **T6.3.2:** Optimizer comparison
  - Test Adam, SGD, RMSprop
  - Validate learning rate schedules
  - Test warmup strategies

---

## 7. Model Learning and Optimization Tests

### 7.1 Maximum A Posteriori (MAP) Estimation
**Paper Reference:** "gradient-descent based approaches to perform maximum a posteriori estimation"

**Tests:**
- ⏳ **T7.1.1:** Prior distributions
  - Test sparsity-inducing priors on detectors
  - Validate regularization effects
  - Test hyperparameter sensitivity
  
- ⏳ **T7.1.2:** MAP optimization
  - Validate posterior computation
  - Test gradient-based learning
  - Compare to pure MLE
  
- ⏳ **T7.1.3:** Parameter initialization
  - Test initialization strategies
  - Validate stability across random seeds
  - Test warm-start from previous runs

### 7.2 Learning Rate Schedules
**Tests:**
- ⏳ **T7.2.1:** Layer-specific learning rates
  - Test different LR for each layer (kappa, eta, time, thresh, etc.)
  - Validate cited LR values (0.001, 0.01, 0.0001, etc.)
  - Test LR decay schedules
  
- ⏳ **T7.2.2:** Adaptive optimization
  - Test Adam optimizer with default/custom parameters
  - Validate momentum and weight decay
  - Test gradient clipping

### 7.3 Regularization
**Tests:**
- ⏳ **T7.3.1:** Model sparsity
  - Test alpha (detector selector) regularization
  - Test beta (rule selector) regularization
  - Validate number of active detectors/rules
  
- ⏳ **T7.3.2:** Overfitting prevention
  - Monitor train vs validation loss
  - Test early stopping
  - Validate generalization performance

---

## 8. Data Processing and Input Handling Tests

### 8.1 16S rRNA Amplicon Data
**Paper Reference:** "datasets consist of 16S rRNA amplicon sequencing data"

**Tests:**
- ⏳ **T8.1.1:** OTU/ASV table processing
  - Test relative abundance normalization
  - Validate compositionality handling
  - Test zero-inflation handling
  
- ⏳ **T8.1.2:** Phylogenetic tree processing
  - Parse Newick format trees
  - Compute pairwise phylogenetic distances
  - Validate distance matrix symmetry
  
- ⏳ **T8.1.3:** Taxonomy integration
  - Load taxonomic assignments
  - Map OTUs to phylogenetic tree
  - Test visualization with taxonomy labels

### 8.2 Shotgun Metagenomics Data
**Paper Reference:** "datasets consist of shotgun metagenomics data"

**Tests:**
- ⏳ **T8.2.1:** Species/strain-level data
  - Test with MetaPhlAn taxonomic profiles
  - Validate higher resolution than 16S
  - Test phylogenetic distance computation
  
- ⏳ **T8.2.2:** Functional profiling
  - Test with gene family abundances (HUMAnN)
  - Validate pathway-level analysis
  - Test functional distance metrics

### 8.3 Preprocessing Pipeline
**Tests:**
- ⏳ **T8.3.1:** Quality filtering
  - Test minimum abundance thresholds
  - Validate prevalence filtering
  - Test subject filtering criteria
  
- ⏳ **T8.3.2:** Transformation options
  - Test log-transformation
  - Test CLR (centered log-ratio) transformation
  - Validate numerical stability
  
- ⏳ **T8.3.3:** Time window truncation
  - Test alignment to common time range
  - Validate sufficient samples per window
  - Test handling of trailing time points

---

## 9. Cross-Validation and Model Selection Tests

### 9.1 Repeated Cross-Validation
**Paper Reference:** "repeated 5-fold cross-validation...5 repetitions and 10 random seeds"

**Tests:**
- ⏳ **T9.1.1:** K-fold splitting
  - Test stratified splits for balanced classes
  - Validate fold independence
  - Test k=5 configuration
  
- ⏳ **T9.1.2:** Repetition strategy
  - Test 5 repetitions with different fold splits
  - Validate 10 random seeds per repetition
  - Aggregate results properly
  
- ⏳ **T9.1.3:** Performance estimation
  - Compute mean and std across repetitions/seeds
  - Test statistical significance (Mann-Whitney U)
  - Validate confidence intervals

### 9.2 Hold-Out Validation
**Paper Reference:** "completely held-out 25% of the data as a test set"

**Tests:**
- ⏳ **T9.2.1:** Train/test split
  - Test 75/25 stratified split
  - Validate no data leakage
  - Test reproducibility with seeds
  
- ⏳ **T9.2.2:** Independent test set evaluation
  - Train on full training set
  - Evaluate once on held-out test set
  - Report unbiased performance estimates

### 9.3 Hyperparameter Tuning
**Tests:**
- ⏳ **T9.3.1:** Model selection within CV
  - Test nested cross-validation for hyperparameters
  - Validate no test set contamination
  - Test grid search or random search
  
- ⏳ **T9.3.2:** Key hyperparameters
  - Test num_rules (1, 2, 3, 5, 10)
  - Test num_otu_centers and num_time_centers
  - Test regularization strengths

---

## 10. Statistical Analysis Tests

### 10.1 Performance Metrics
**Paper Reference:** "F1-scores and area under the curve (AUC) of receiver operating characteristic curves"

**Tests:**
- ✅ **T10.1.1:** F1 score computation
  - Calculate precision and recall
  - Compute harmonic mean
  - Handle edge cases (all positive/negative)
  
- ✅ **T10.1.2:** AUC-ROC computation
  - Generate ROC curves
  - Calculate area under curve
  - Test with probability predictions
  
- ⏳ **T10.1.3:** Additional metrics
  - Test accuracy, sensitivity, specificity
  - Test confusion matrix generation
  - Validate metrics for imbalanced data

### 10.2 Statistical Testing
**Paper Reference:** "Mann-Whitney U test...Delong's method for statistical testing"

**Tests:**
- ⏳ **T10.2.1:** Mann-Whitney U test
  - Compare F1 scores between methods
  - Test non-parametric significance (p < 0.05)
  - Validate two-tailed tests
  
- ⏳ **T10.2.2:** DeLong's test for AUC
  - Compare AUC curves statistically
  - Test correlated ROC curves
  - Validate p-value computation
  
- ⏳ **T10.2.3:** Multiple comparison correction
  - Test Bonferroni or FDR correction
  - Validate family-wise error rate
  - Report adjusted p-values

---

## 11. Biological Case Study Validation Tests

### 11.1 Case Study 1: Diet and Infant Microbiome (Bokulich)
**Paper Reference:** "automatic focus on relevant time periods...4-6 months preceding solid food introduction"

**Tests:**
- ⏳ **T11.1.1:** Time window discovery
  - Validate learned windows align with 4-6 month period
  - Test biological interpretation (pre-solid-food)
  - Verify consistency across runs
  
- ⏳ **T11.1.2:** Taxa identification
  - Validate Clostridiales order selection (12 taxa)
  - Test genus-level findings (Clostridium, Blautia, Ruminococcus, etc.)
  - Verify phylogenetic coherence
  
- ⏳ **T11.1.3:** Bacteroides acidifaciens detection
  - Validate slope-type detector for increasing trend
  - Test biological explanation (fiber/solid foods)
  - Verify complementary information to other detectors
  
- ⏳ **T11.1.4:** Rule logic validation
  - Test two-rule OR combination
  - Validate AND logic within rules (two detectors)
  - Verify improved classification with combined rules

### 11.2 Case Study 2: Type 1 Diabetes Progression (Kostic)
**Paper Reference:** "temporal pattern of events detecting normal microbiome succession absent in T1D infants"

**Tests:**
- ⏳ **T11.2.1:** Progressive time window sequence
  - Validate three temporal windows: 5-15, 13-22, 17-26 months
  - Test progressive overlap pattern
  - Verify temporal ordering
  
- ⏳ **T11.2.2:** Succession pattern identification
  - Validate E. coli (earliest, facultative anaerobe)
  - Test Streptococcus/Coprobacillus (middle period)
  - Verify F. prausnitzii (latest, strict anaerobe)
  
- ⏳ **T11.2.3:** Single-rule classification
  - Test conjunction of three slope detectors
  - Validate detection of normal succession
  - Verify T1D cases lack this pattern
  
- ⏳ **T11.2.4:** Ecological interpretation
  - Validate progressive anaerobic specialization
  - Test consistency with infant gut colonization
  - Verify biological plausibility

---

## 12. Software Engineering and Deployment Tests

### 12.1 PyTorch Integration
**Paper Reference:** "implemented in Python using the PyTorch deep learning library"

**Tests:**
- ✅ **T12.1.1:** Standard PyTorch APIs
  - Test nn.Module subclassing
  - Validate forward/backward hooks
  - Test parameter registration
  
- ✅ **T12.1.2:** GPU support
  - Test .to(device) operations
  - Validate CUDA tensor operations
  - Test mixed CPU/GPU computation
  
- ✅ **T12.1.3:** Model serialization
  - Test state_dict() save/load
  - Validate checkpoint resuming
  - Test model export formats

### 12.2 Package Structure
**Tests:**
- ⏳ **T12.2.1:** Module organization
  - Test import structure (mditre.models, mditre.data, etc.)
  - Validate API consistency
  - Test documentation completeness
  
- ⏳ **T12.2.2:** Installation
  - Test pip installation
  - Validate dependency management (setup.py)
  - Test across Python versions (3.7+)
  
- ⏳ **T12.2.3:** Command-line interface
  - Test trainer.py script
  - Validate argument parsing
  - Test configuration file support

### 12.3 Cross-Platform Compatibility
**Tests:**
- ⏳ **T12.3.1:** Operating systems
  - Test on Linux, macOS, Windows
  - Validate file path handling
  - Test platform-specific optimizations
  
- ⏳ **T12.3.2:** Hardware configurations
  - Test CPU-only mode
  - Test single GPU
  - Test multi-GPU (if supported)

---

## 13. Graphical User Interface Tests

### 13.1 Rule Visualization Interface
**Paper Reference:** "graphical user interface for visualizations of learned rules"

**Tests:**
- ⏳ **T13.1.1:** Main rule dashboard
  - Display all learned rules
  - Show OR combination logic
  - Test subject classification overlay
  
- ⏳ **T13.1.2:** Rule drill-down
  - Click on rule to see English description
  - Display detector AND combination
  - Show truth table (detectors × subjects)
  
- ⏳ **T13.1.3:** Detector exploration
  - Click on detector to see details
  - Display time window on timeline
  - Show phylogenetic tree with highlighted taxa

### 13.2 Data Visualization
**Tests:**
- ⏳ **T13.2.1:** Longitudinal profiles
  - Plot abundance trajectories over time
  - Overlay temporal focus windows
  - Highlight selected taxa
  
- ⏳ **T13.2.2:** Phylogenetic visualization
  - Render phylogenetic tree
  - Color-code selected clades
  - Display aggregation weights
  
- ⏳ **T13.2.3:** Performance summaries
  - Show ROC curves
  - Display confusion matrices
  - Plot learning curves

### 13.3 Export and Reporting
**Tests:**
- ⏳ **T13.3.1:** Figure export
  - Save visualizations as PNG/PDF
  - Test high-resolution rendering
  - Validate publication-quality output
  
- ⏳ **T13.3.2:** Rule export
  - Export rules as text/JSON
  - Generate human-readable reports
  - Test batch export for all rules

---

## 14. Edge Cases and Robustness Tests

### 14.1 Data Quality Issues
**Tests:**
- ⏳ **T14.1.1:** Extreme class imbalance
  - Test with 90/10, 95/5, 99/1 splits
  - Validate with DiGiulio data (6/31)
  - Test weighted loss functions
  
- ⏳ **T14.1.2:** Small sample sizes
  - Test with n<20 subjects
  - Validate with Kostic data (n=17)
  - Test overfitting mitigation
  
- ⏳ **T14.1.3:** Sparse time series
  - Test with 2-3 time points per subject
  - Validate temporal focus with sparse data
  - Test missing data imputation

### 14.2 Numerical Stability
**Tests:**
- ⏳ **T14.2.1:** Zero abundance handling
  - Test with many zero entries
  - Validate log-transformation stability
  - Test epsilon addition
  
- ⏳ **T14.2.2:** Extreme parameter values
  - Test with very large/small kappa
  - Validate sigmoid saturation handling
  - Test gradient clipping
  
- ⏳ **T14.2.3:** NaN/Inf detection
  - Monitor for numerical issues during training
  - Test recovery mechanisms
  - Validate error reporting

### 14.3 Boundary Conditions
**Tests:**
- ⏳ **T14.3.1:** Minimum viable inputs
  - Test with single OTU
  - Test with single time point
  - Test with single subject (if applicable)
  
- ⏳ **T14.3.2:** Maximum scale inputs
  - Test with 10,000+ OTUs
  - Test with 100+ time points
  - Test with 1,000+ subjects
  
- ⏳ **T14.3.3:** Degenerate cases
  - Test with all same labels
  - Test with perfect separation
  - Test with completely random data

---

## 15. Comparison to MITRE (Original Method)

### 15.1 Approximation Quality
**Paper Reference:** "MDITRE is a highly scalable approximation to our previous MITRE method"

**Tests:**
- ⏳ **T15.1.1:** Predictive performance parity
  - Compare F1 scores on common datasets
  - Test cases where MDITRE ≈ MITRE (6/8 real datasets)
  - Analyze cases where MDITRE < MITRE (DiGiulio, Kostic)
  
- ⏳ **T15.1.2:** Rule similarity
  - Compare learned rules between methods
  - Test if similar taxa/time windows selected
  - Validate biological interpretation consistency
  
- ⏳ **T15.1.3:** Uncertainty quantification
  - Compare point estimates (MAP) to MITRE posteriors
  - Quantify approximation error
  - Test calibration of predictions

### 15.2 Computational Tradeoffs
**Tests:**
- ⏳ **T15.2.1:** Runtime comparison
  - Measure 86-1150x speedup on real data
  - Test 1000x+ speedup on n=1024 synthetic data
  - Validate MITRE timeout (>2 weeks) on large data
  
- ⏳ **T15.2.2:** Memory comparison
  - Compare peak memory usage
  - Test MITRE feature pre-computation overhead
  - Validate MDITRE gradient-based efficiency
  
- ⏳ **T15.2.3:** Scalability limits
  - Identify where MITRE becomes infeasible (n>64)
  - Test MDITRE on n=282 (Shao data)
  - Project maximum feasible dataset sizes

---

## Test Implementation Priority

### Phase 1: Core Functionality (Weeks 1-2)
- ✅ All Section 1 tests (Architecture)
- ✅ T10.1.1, T10.1.2 (Basic metrics)
- ✅ T12.1.1, T12.1.2, T12.1.3 (PyTorch integration)

### Phase 2: Key Features (Weeks 3-4)
- ⏳ Section 2 (Phylogenetic focus)
- ⏳ Section 3 (Temporal focus)
- ⏳ Section 4.1 (Interpretability)
- ⏳ Section 8 (Data processing)

### Phase 3: Performance Validation (Weeks 5-6)
- ⏳ Section 5 (All benchmarking)
- ⏳ Section 6 (Scalability)
- ⏳ Section 10 (Statistical analysis)

### Phase 4: Advanced Features (Weeks 7-8)
- ⏳ Section 7 (Optimization)
- ⏳ Section 9 (Cross-validation)
- ⏳ Section 11 (Case studies)
- ⏳ Section 13 (GUI)

### Phase 5: Robustness (Weeks 9-10)
- ⏳ Section 14 (Edge cases)
- ⏳ Section 15 (MITRE comparison)
- ⏳ Section 12.2, 12.3 (Deployment)

---

## Test Execution Framework

### Automated Testing
```python
# Structure for automated test suite
pytest test_mditre_comprehensive.py          # Current basic tests
pytest test_phylogenetic_focus.py            # Section 2
pytest test_temporal_focus.py                # Section 3
pytest test_interpretability.py              # Section 4
pytest test_benchmarking.py                  # Section 5 (long-running)
pytest test_scalability.py                   # Section 6 (GPU required)
pytest test_optimization.py                  # Section 7
pytest test_data_processing.py               # Section 8
pytest test_cross_validation.py              # Section 9
pytest test_edge_cases.py                    # Section 14
```

### Manual Testing
- Section 11: Biological case studies (requires manual inspection)
- Section 13: GUI testing (requires human interaction)
- Section 15: MITRE comparison (requires MITRE installation)

### Continuous Integration
- Run Phase 1 tests on every commit
- Run Phase 2-3 tests on pull requests
- Run Phase 4-5 tests nightly
- Generate coverage reports
- Track performance benchmarks over time

---

## Success Criteria

### Minimum Viable Testing
1. ✅ All Phase 1 tests pass
2. All Phase 2 tests pass
3. At least 6/8 real datasets replicate paper results (within 5%)
4. Runtime speedup >100x over MITRE on n=64
5. No critical bugs in core architecture

### Comprehensive Validation
1. All tests in Phases 1-4 pass
2. All 8 real datasets replicate paper results
3. Both case studies reproduce key findings
4. GUI functions correctly
5. Package installs on all platforms

### Publication-Ready
1. All 100+ tests pass
2. Statistical significance reproduced
3. All figures in paper can be regenerated
4. Complete documentation and tutorials
5. Independent validation on new dataset

---

## Test Data Requirements

### Synthetic Data
- 10 different random seeds
- Subject counts: 20, 24, 32, 48, 64, 128, 256, 512, 1024
- Time points: 6, 10, 15, 20, 25, 30
- Perturbations: 1 and 2 clades
- Noise levels: 1, 10, 100, 1000, 10000

### Real Data
- Bokulich et al. (2 tasks, n=37, 16S)
- David et al. (n=20, 16S)
- DiGiulio et al. (n=37, 16S, imbalanced)
- Vatanen et al. (n=117, 16S)
- Kostic et al. (n=17, metagenomics)
- Brooks et al. (n=30, metagenomics)
- Shao et al. (n=282, metagenomics)

### Required Preprocessing
- Raw abundance tables
- Phylogenetic trees (16S) or distances (metagenomics)
- Sample metadata with labels
- Time point information
- Quality filtering documentation

---

## Reporting and Documentation

### Test Report Contents
1. **Executive Summary**: Pass/fail counts, coverage metrics
2. **Performance Benchmarking**: Tables comparing to paper
3. **Statistical Analysis**: p-values, effect sizes, confidence intervals
4. **Failure Analysis**: Root causes, severity, recommendations
5. **Runtime Profiling**: Bottlenecks, optimization opportunities
6. **Biological Validation**: Case study reproductions, interpretations

### Deliverables
- Automated test suite (pytest)
- Test data repository
- Benchmark results database
- Performance regression tracking
- Bug tracking system
- Documentation updates

---

## Notes and Limitations

### Known Limitations from Paper
1. Binary classification only (not multiclass/regression)
2. No survival analysis support
3. Single-modality (microbiome only, no multi-omics)
4. MAP inference (no uncertainty quantification)
5. Limited generalization testing (no independent cohorts)

### Testing Gaps
1. True external validation not possible (no comparable studies)
2. MITRE comparison limited by runtime constraints
3. GUI testing requires manual verification
4. Biological interpretations are subjective

### Future Extensions to Test
1. Multiclass classification
2. Time-to-event prediction
3. Multi-omics integration
4. Variational inference for uncertainty
5. Transfer learning across studies

---

**Document Version:** 1.0  
**Last Updated:** November 1, 2025  
**Status:** Phase 1 Complete (9/9 tests passing)  
**Next Milestone:** Phase 2 implementation (Sections 2-4, 8)
