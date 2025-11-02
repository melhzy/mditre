# Fix torch loading issues on Windows
# This script diagnoses and attempts to fix torch installation problems

cat("=== Torch Installation Diagnostics ===\n\n")

# Check if torch package is installed
cat("1. Checking torch package installation...\n")
if (requireNamespace("torch", quietly = TRUE)) {
  cat("   ✓ torch package is installed (version", as.character(packageVersion("torch")), ")\n")
} else {
  cat("   ✗ torch package is NOT installed\n")
  cat("   Run: install.packages('torch')\n")
  quit(save = "no", status = 1)
}

# Check torch installation status
cat("\n2. Checking torch backend libraries...\n")
result <- try(torch::torch_is_installed(), silent = TRUE)
if (inherits(result, "try-error")) {
  cat("   ✗ torch backend check failed\n")
  cat("   Error:", as.character(result), "\n")
} else {
  cat("   Torch installed status:", result, "\n")
}

# Try to load torch
cat("\n3. Attempting to load torch...\n")
load_result <- try(library(torch), silent = TRUE)
if (inherits(load_result, "try-error")) {
  cat("   ✗ Failed to load torch\n")
  cat("   Error:", as.character(load_result), "\n")
} else {
  cat("   ✓ torch loaded successfully\n")
}

# Check for DLL dependencies
cat("\n4. Checking system dependencies...\n")
cat("   R Version:", R.version.string, "\n")
cat("   Platform:", R.version$platform, "\n")
cat("   OS:", Sys.info()["sysname"], Sys.info()["release"], "\n")

# Get torch installation directory
cat("\n5. Locating torch installation files...\n")
torch_home <- Sys.getenv("TORCH_HOME")
if (torch_home == "") {
  torch_home <- rappdirs::user_data_dir("torch")
}
cat("   TORCH_HOME:", torch_home, "\n")

if (dir.exists(torch_home)) {
  cat("   Contents:\n")
  files <- list.files(torch_home, recursive = FALSE)
  if (length(files) > 0) {
    for (f in files) {
      cat("     -", f, "\n")
    }
  } else {
    cat("     (empty directory)\n")
  }
} else {
  cat("   ✗ TORCH_HOME directory does not exist\n")
}

# Check R library path for torch
cat("\n6. Checking torch package location...\n")
torch_path <- system.file(package = "torch")
cat("   Package path:", torch_path, "\n")

if (dir.exists(file.path(torch_path, "deps"))) {
  cat("   deps/ directory exists\n")
  deps_files <- list.files(file.path(torch_path, "deps"), recursive = FALSE)
  if (length(deps_files) > 0) {
    cat("   deps/ contents:\n")
    for (f in head(deps_files, 10)) {
      cat("     -", f, "\n")
    }
  }
} else {
  cat("   ✗ deps/ directory not found\n")
}

# Attempt reinstallation
cat("\n=== Attempting Torch Reinstallation ===\n")
cat("\nThis will reinstall torch backend libraries.\n")
cat("Press Ctrl+C to cancel, or wait 5 seconds to continue...\n")
Sys.sleep(5)

cat("\nReinstalling torch backend...\n")
install_result <- try(torch::install_torch(reinstall = TRUE), silent = FALSE)

if (!inherits(install_result, "try-error")) {
  cat("\n=== Testing Installation ===\n")
  
  # Test if torch works now
  cat("\n1. Testing torch_is_installed()...\n")
  if (torch::torch_is_installed()) {
    cat("   ✓ Torch is now installed\n")
  } else {
    cat("   ✗ Torch still not installed\n")
  }
  
  cat("\n2. Testing tensor creation...\n")
  tensor_test <- try({
    t <- torch_tensor(c(1, 2, 3, 4, 5))
    print(t)
    cat("   ✓ Tensor creation successful\n")
  }, silent = TRUE)
  
  if (inherits(tensor_test, "try-error")) {
    cat("   ✗ Tensor creation failed\n")
    cat("   Error:", as.character(tensor_test), "\n")
  }
} else {
  cat("\n✗ Reinstallation failed\n")
  cat("Error:", as.character(install_result), "\n")
}

cat("\n=== Recommendations ===\n")
cat("
If torch is still not working, try these steps:

1. Install Visual C++ Redistributable:
   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   This is required for torch to load on Windows.

2. Restart R session after installing VC++ Redistributable

3. Try installing an older torch version:
   install.packages('torch', version='0.12.0')

4. Check Windows Event Viewer for DLL load errors:
   - Open Event Viewer
   - Go to Windows Logs > Application
   - Look for errors from R or RStudio

5. For more help, see:
   - https://torch.mlverse.org/docs/articles/installation.html
   - https://github.com/mlverse/torch/issues
")
