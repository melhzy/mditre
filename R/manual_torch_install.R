# Manual torch installation script
# This manually downloads and extracts torch dependencies

cat("=== Manual Torch Installation ===\n\n")

# Create necessary directories
torch_pkg <- system.file(package = "torch")
deps_dir <- file.path(torch_pkg, "deps")

cat("1. Creating deps directory at:", deps_dir, "\n")
if (!dir.exists(deps_dir)) {
  dir.create(deps_dir, recursive = TRUE, showWarnings = FALSE)
  cat("   ✓ Created deps directory\n")
} else {
  cat("   Directory already exists\n")
}

# Download URLs
libtorch_url <- "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.7.1%2Bcpu.zip"
lantern_url <- "https://torch-cdn.mlverse.org/binaries/refs/heads/cran/v0.16.2/latest/lantern-0.16.2+cpu-win64.zip"

# Download and extract libtorch
cat("\n2. Downloading libtorch (187 MB - this may take a few minutes)...\n")
libtorch_zip <- file.path(tempdir(), "libtorch.zip")

tryCatch({
  options(timeout = 600)  # 10 minute timeout
  download.file(libtorch_url, libtorch_zip, mode = "wb", quiet = FALSE)
  cat("   ✓ Downloaded libtorch\n")
  
  cat("\n3. Extracting libtorch...\n")
  unzip(libtorch_zip, exdir = deps_dir, overwrite = TRUE)
  cat("   ✓ Extracted libtorch to", deps_dir, "\n")
  
  # Clean up
  file.remove(libtorch_zip)
  
}, error = function(e) {
  cat("   ✗ Error downloading/extracting libtorch:", conditionMessage(e), "\n")
})

# Download and extract lantern
cat("\n4. Downloading lantern (2.3 MB)...\n")
lantern_zip <- file.path(tempdir(), "lantern.zip")

tryCatch({
  download.file(lantern_url, lantern_zip, mode = "wb", quiet = FALSE)
  cat("   ✓ Downloaded lantern\n")
  
  cat("\n5. Extracting lantern...\n")
  unzip(lantern_zip, exdir = deps_dir, overwrite = TRUE)
  cat("   ✓ Extracted lantern to", deps_dir, "\n")
  
  # Clean up
  file.remove(lantern_zip)
  
}, error = function(e) {
  cat("   ✗ Error downloading/extracting lantern:", conditionMessage(e), "\n")
})

# Verify installation
cat("\n6. Verifying installation...\n")
if (dir.exists(deps_dir)) {
  deps_contents <- list.files(deps_dir, recursive = FALSE)
  cat("   Contents of deps/:\n")
  for (item in deps_contents) {
    cat("     -", item, "\n")
  }
  
  # Look for critical files
  libtorch_dir <- file.path(deps_dir, "libtorch")
  if (dir.exists(libtorch_dir)) {
    cat("\n   ✓ libtorch directory found\n")
    lib_files <- list.files(file.path(libtorch_dir, "lib"), pattern = "\\.(dll|so|dylib)$")
    cat("   Found", length(lib_files), "library files\n")
  } else {
    cat("\n   ✗ libtorch directory NOT found\n")
  }
} else {
  cat("   ✗ deps directory still doesn't exist\n")
}

cat("\n=== Testing Torch ===\n")
cat("\nRestarting R is recommended. To test now, try:\n")
cat("  .rs.restartR()  # In RStudio\n")
cat("  # OR restart your R session manually\n\n")

cat("Then run:\n")
cat("  library(torch)\n")
cat("  torch_tensor(1:5)\n\n")

cat("If it still doesn't work, the issue may be:\n")
cat("- Antivirus blocking DLL loading\n")
cat("- Windows security policies\n")
cat("- Corrupted download\n")
