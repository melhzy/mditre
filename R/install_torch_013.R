# Install torch 0.13.0 backend manually

cat("=== Installing torch 0.13.0 backend ===\n\n")

torch_pkg <- system.file(package = "torch")
deps_dir <- file.path(torch_pkg, "deps")

cat("Creating deps directory...\n")
dir.create(deps_dir, recursive = TRUE, showWarnings = FALSE)

# URLs for torch 0.13.0
libtorch_url <- "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.0.1%2Bcpu.zip"
lantern_url <- "https://torch-cdn.mlverse.org/binaries/refs/heads/cran/v0.13.0/latest/lantern-0.13.0+cpu-win64.zip"

# Download and extract libtorch
cat("\nDownloading libtorch 2.0.1 (154 MB)...\n")
libtorch_zip <- file.path(tempdir(), "libtorch.zip")
options(timeout = 600)
download.file(libtorch_url, libtorch_zip, mode = "wb", quiet = FALSE)
cat("Extracting...\n")
unzip(libtorch_zip, exdir = deps_dir, overwrite = TRUE)
file.remove(libtorch_zip)
cat("✓ libtorch installed\n")

# Download and extract lantern
cat("\nDownloading lantern 0.13.0 (2.6 MB)...\n")
lantern_zip <- file.path(tempdir(), "lantern.zip")
download.file(lantern_url, lantern_zip, mode = "wb", quiet = FALSE)
cat("Extracting...\n")
unzip(lantern_zip, exdir = deps_dir, overwrite = TRUE)
file.remove(lantern_zip)
cat("✓ lantern installed\n")

# Verify
cat("\n=== Verification ===\n")
cat("deps contents:\n")
list.files(deps_dir)
