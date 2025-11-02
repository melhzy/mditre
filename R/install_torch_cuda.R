# Install torch with CUDA support manually

cat("=== Installing torch with CUDA 12.1 ===\n\n")

torch_pkg <- system.file(package = "torch")
deps_dir <- file.path(torch_pkg, "deps")

cat("1. Creating deps directory...\n")
dir.create(deps_dir, recursive = TRUE, showWarnings = FALSE)
cat("   ✓ Created\n\n")

# CUDA 12.1 URLs for torch 0.16.2
libtorch_url <- "https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.7.1%2Bcu121.zip"
lantern_url <- "https://torch-cdn.mlverse.org/binaries/refs/heads/cran/v0.16.2/latest/lantern-0.16.2+cu121-win64.zip"

# Download and extract libtorch with CUDA
cat("2. Downloading libtorch 2.7.1+cu121 (~2.5 GB - this will take several minutes)...\n")
libtorch_zip <- file.path(tempdir(), "libtorch_cuda.zip")
options(timeout = 1800)  # 30 minute timeout for large file

tryCatch({
  download.file(libtorch_url, libtorch_zip, mode = "wb", quiet = FALSE)
  cat("\n   ✓ Downloaded\n")
  
  cat("\n3. Extracting libtorch (this may take a few minutes)...\n")
  unzip(libtorch_zip, exdir = deps_dir, overwrite = TRUE)
  file.remove(libtorch_zip)
  cat("   ✓ Extracted\n")
}, error = function(e) {
  cat("\n   ✗ Error:", conditionMessage(e), "\n")
  cat("   Falling back to CPU version...\n")
  libtorch_cpu_url <- "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.7.1%2Bcpu.zip"
  download.file(libtorch_cpu_url, libtorch_zip, mode = "wb", quiet = FALSE)
  unzip(libtorch_zip, exdir = deps_dir, overwrite = TRUE)
  file.remove(libtorch_zip)
})

# Download and extract lantern with CUDA
cat("\n4. Downloading lantern 0.16.2+cu121...\n")
lantern_zip <- file.path(tempdir(), "lantern_cuda.zip")

tryCatch({
  download.file(lantern_url, lantern_zip, mode = "wb", quiet = FALSE)
  cat("\n   ✓ Downloaded\n")
  
  cat("\n5. Extracting lantern...\n")
  unzip(lantern_zip, exdir = deps_dir, overwrite = TRUE)
  file.remove(lantern_zip)
  cat("   ✓ Extracted\n")
}, error = function(e) {
  cat("\n   ✗ Error:", conditionMessage(e), "\n")
  cat("   Falling back to CPU version...\n")
  lantern_cpu_url <- "https://torch-cdn.mlverse.org/binaries/refs/heads/cran/v0.16.2/latest/lantern-0.16.2+cpu-win64.zip"
  download.file(lantern_cpu_url, lantern_zip, mode = "wb", quiet = FALSE)
  unzip(lantern_zip, exdir = deps_dir, overwrite = TRUE)
  file.remove(lantern_zip)
})

# Verify
cat("\n=== Verification ===\n")
cat("deps contents:\n")
contents <- list.files(deps_dir)
for (item in contents) {
  cat("  -", item, "\n")
}

# Check for CUDA libraries
libtorch_lib <- file.path(deps_dir, "libtorch", "lib")
if (dir.exists(libtorch_lib)) {
  cuda_dlls <- list.files(libtorch_lib, pattern = "cuda|cudnn|cublas|cufft", ignore.case = TRUE)
  if (length(cuda_dlls) > 0) {
    cat("\n✓ CUDA libraries found (", length(cuda_dlls), "files )\n")
  } else {
    cat("\n⚠ No CUDA libraries found (CPU version installed)\n")
  }
}

cat("\n=== Installation complete ===\n")
