#!/usr/bin/env Rscript
# =============================================================================
# R MDITRE Setup Script
# Configures Python MDITRE backend and verifies R MDITRE frontend
# =============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("R MDITRE - Environment Setup and Verification\n")
cat(strrep("=", 80), "\n\n")

cat("MDITRE Two-Package Architecture:\n")
cat("  1. Python MDITRE (Backend): PyTorch models in MDITRE conda environment\n")
cat("  2. R MDITRE (Frontend): R interface bridging to Python via reticulate\n\n")

# Check reticulate
if (!requireNamespace("reticulate", quietly = TRUE)) {
  cat("ERROR: reticulate package not found.\n")
  cat("Install with: install.packages('reticulate')\n")
  quit(status = 1)
}

library(reticulate)

# Configure Python MDITRE backend
cat("Step 1: Configuring Python MDITRE backend...\n")
conda_env <- "MDITRE"

tryCatch({
  use_condaenv(conda_env, required = TRUE)
  cat(sprintf("  ✓ Using conda environment: %s\n", conda_env))
}, error = function(e) {
  cat(sprintf("  ✗ ERROR: Cannot find conda environment '%s'\n", conda_env))
  cat("\nTo create the Python MDITRE backend:\n")
  cat("  conda create -n MDITRE python=3.12\n")
  cat("  conda activate MDITRE\n")
  cat("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n")
  cat("  cd path/to/mditre/Python\n")
  cat("  pip install -e .\n")
  quit(status = 1)
})

# Install Python MDITRE package
cat("\nStep 2: Installing Python MDITRE package...\n")
python_dir <- normalizePath(file.path(getwd(), "..", "Python"), winslash = "/", mustWork = FALSE)

if (dir.exists(python_dir)) {
  result <- system2("conda", 
                   args = c("run", "-n", conda_env, "pip", "install", "-e", python_dir),
                   stdout = TRUE, stderr = TRUE)
  
  if (!is.null(attr(result, "status")) && attr(result, "status") != 0) {
    cat("  ✗ WARNING: Installation failed\n")
    cat("  Manual installation required:\n")
    cat(sprintf("    cd %s\n", python_dir))
    cat("    conda activate MDITRE\n")
    cat("    pip install -e .\n")
  } else {
    cat("  ✓ Python MDITRE installed/updated successfully\n")
  }
} else {
  cat(sprintf("  ✗ WARNING: Python directory not found at: %s\n", python_dir))
  cat("  Ensure Python MDITRE package is installed manually.\n")
}

# Verify Python MDITRE backend
cat("\nStep 3: Verifying Python MDITRE backend...\n")

# Check Python version
py_config <- py_config()
cat(sprintf("  Python: %s\n", py_config$version))
cat(sprintf("  Executable: %s\n", py_config$python))

# Check PyTorch
tryCatch({
  torch_py <- import("torch")
  torch_version <- torch_py$`__version__`
  cuda_available <- torch_py$cuda$is_available()
  device <- if (cuda_available) "cuda" else "cpu"
  
  cat(sprintf("  ✓ PyTorch: %s\n", torch_version))
  cat(sprintf("  ✓ CUDA: %s\n", ifelse(cuda_available, "Available", "Not available")))
  
  if (cuda_available) {
    gpu_name <- torch_py$cuda$get_device_name(0L)
    gpu_memory <- torch_py$cuda$get_device_properties(0L)$total_memory / 1e9
    cat(sprintf("    GPU: %s (%.1f GB)\n", gpu_name, gpu_memory))
  }
}, error = function(e) {
  cat("  ✗ ERROR: PyTorch not found\n")
  cat("  Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n")
  quit(status = 1)
})

# Check mditre
cat("\nStep 4: Verifying Python MDITRE package...\n")
tryCatch({
  mditre <- import("mditre")
  mditre_models <- import("mditre.models")
  mditre_version <- mditre$`__version__`
  
  cat(sprintf("  ✓ Python mditre: %s\n", mditre_version))
  cat("  ✓ mditre.models: Loaded successfully\n")
}, error = function(e) {
  cat("  ✗ ERROR: Python MDITRE not found or failed to load\n")
  cat("  Error:", e$message, "\n")
  quit(status = 1)
})

# Test basic functionality
cat("\nStep 5: Testing R → Python bridge functionality...\n")
tryCatch({
  # Create a simple model using Python MDITRE
  np <- import("numpy")
  
  embeddings <- matrix(rnorm(50 * 10), 50, 10)
  model <- mditre_models$MDITRE(
    num_rules = 3L,
    num_otus = 50L,
    num_otu_centers = 5L,
    num_time = 10L,
    num_time_centers = 3L,
    dist = embeddings,
    emb_dim = 10L
  )
  
  # Initialize model parameters
  init_args <- list(
    kappa_init = np$array(matrix(runif(3*5, 0.5, 2.0), 3, 5)),
    eta_init = np$array(array(rnorm(3*5*10) * 0.1, dim=c(3, 5, 10))),
    abun_a_init = np$array(matrix(runif(3*5, 0.2, 0.8), 3, 5)),
    abun_b_init = np$array(matrix(runif(3*5, 0.2, 0.8), 3, 5)),
    slope_a_init = np$array(matrix(runif(3*5, 0.2, 0.8), 3, 5)),
    slope_b_init = np$array(matrix(runif(3*5, 0.2, 0.8), 3, 5)),
    thresh_init = np$array(matrix(runif(3*5, 0.1, 0.5), 3, 5)),
    slope_init = np$array(matrix(runif(3*5, -0.1, 0.1), 3, 5)),
    alpha_init = np$array(matrix(runif(3*5, -1, 1), 3, 5)),
    w_init = np$array(matrix(rnorm(1*3) * 0.1, 1, 3)),
    bias_init = np$array(0.0),
    beta_init = np$array(runif(3, -1, 1))
  )
  
  model$init_params(init_args)
  
  cat("  ✓ Model creation via Python MDITRE: Success\n")
  
  # Test forward pass
  x <- torch_py$randn(4L, 50L, 10L)
  mask <- torch_py$ones(4L, 10L)
  output <- model(x, mask = mask)
  
  cat("  ✓ Forward pass via Python PyTorch: Success\n")
  cat(sprintf("    Input shape: [%s]\n", paste(py_to_r(x$shape), collapse = ", ")))
  cat(sprintf("    Output shape: [%s]\n", paste(py_to_r(output$shape), collapse = ", ")))
  
}, error = function(e) {
  cat("  ✗ ERROR: R → Python bridge test failed\n")
  cat("  Error:", e$message, "\n")
  quit(status = 1)
})

# Success summary
cat("\n")
cat(strrep("=", 80), "\n")
cat("✓✓✓ SETUP COMPLETE ✓✓✓\n")
cat(strrep("=", 80), "\n")
cat("\nYour R MDITRE environment is ready!\n\n")

cat("Architecture:\n")
py_ver <- py_config()$version
if (is.list(py_ver)) py_ver <- paste(unlist(py_ver), collapse=".")
cat("  R MDITRE (Frontend):     R", R.version$major, ".", R.version$minor, "\n", sep="")
cat("  Python MDITRE (Backend): Python", py_ver, "\n")
cat("  Bridge:                  reticulate", as.character(packageVersion("reticulate")), "\n")
cat("  Computation:             PyTorch", torch_py$`__version__`, "(", device, ")\n")
cat("\n")

cat("Quick Start:\n")
cat("  1. Run tests:        Rscript run_mditre_tests.R\n")
cat("  2. Try tutorials:    Open R/rmd/*.Rmd in RStudio\n")
cat("  3. Load in R:        library(reticulate); use_condaenv('MDITRE')\n")
cat("\n")

cat("For more information:\n")
cat("  - README: R/README.md\n")
cat("  - Tutorials: R/rmd/README.md\n")
cat("  - Tests: R/run_mditre_tests.R\n")
cat("\n")
