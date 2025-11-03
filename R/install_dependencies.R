#!/usr/bin/env Rscript
# =============================================================================
# R MDITRE - Dependency Installation Script
# Installs all required R packages for MDITRE
# =============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("R MDITRE - Installing Dependencies\n")
cat("R version:", R.version.string, "\n")
cat(strrep("=", 80), "\n\n")

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Function to install package if not already installed
install_if_missing <- function(pkg, source = "CRAN") {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s from %s...\n", pkg, source))
    if (source == "CRAN") {
      install.packages(pkg, dependencies = TRUE)
    } else if (source == "Bioconductor") {
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager")
      }
      BiocManager::install(pkg, ask = FALSE, update = FALSE)
    }
    cat(sprintf("  ✓ %s installed\n", pkg))
  } else {
    cat(sprintf("  ✓ %s already installed\n", pkg))
  }
}

# Core dependencies
cat("\n1. Installing core dependencies...\n")
core_packages <- c(
  "reticulate",  # Python integration
  "ggplot2",     # Plotting
  "dplyr",       # Data manipulation
  "tidyr",       # Data tidying
  "patchwork",   # Plot composition
  "digest"       # Hashing
)

for (pkg in core_packages) {
  install_if_missing(pkg, "CRAN")
}

# Install BiocManager for Bioconductor packages
cat("\n2. Installing BiocManager...\n")
install_if_missing("BiocManager", "CRAN")

# Phylogenetic packages
cat("\n3. Installing phylogenetic packages...\n")
phylo_packages <- c(
  "ape",        # Phylogenetic analysis
  "phangorn"    # Phylogenetic reconstruction
)

for (pkg in phylo_packages) {
  install_if_missing(pkg, "CRAN")
}

# Bioconductor packages
cat("\n4. Installing Bioconductor packages...\n")
bioc_packages <- c(
  "phyloseq",    # Microbiome analysis
  "ggtree"       # Tree visualization
)

for (pkg in bioc_packages) {
  install_if_missing(pkg, "Bioconductor")
}

# Testing packages (suggested)
cat("\n5. Installing testing packages...\n")
test_packages <- c(
  "testthat",    # Unit testing
  "knitr",       # Dynamic reports
  "rmarkdown"    # R Markdown
)

for (pkg in test_packages) {
  install_if_missing(pkg, "CRAN")
}

# Additional suggested packages
cat("\n6. Installing additional packages (optional)...\n")
optional_packages <- c(
  "microbiome",  # Microbiome tools
  "vegan",       # Ecological analysis
  "covr"         # Code coverage
)

for (pkg in optional_packages) {
  tryCatch({
    install_if_missing(pkg, if (pkg == "microbiome") "Bioconductor" else "CRAN")
  }, error = function(e) {
    cat(sprintf("  ⚠ Warning: Could not install %s: %s\n", pkg, e$message))
  })
}

# Install seedhash from GitHub
cat("\n7. Installing seedhash package from GitHub...\n")
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

tryCatch({
  remotes::install_github("melhzy/seedhash", subdir = "R", upgrade = "never", quiet = TRUE)
  cat("  ✓ seedhash installed\n")
}, error = function(e) {
  cat(sprintf("  ⚠ Warning: Could not install seedhash: %s\n", e$message))
})

# Verify installations
cat("\n")
cat(strrep("=", 80), "\n")
cat("Verifying Installations\n")
cat(strrep("=", 80), "\n\n")

required_packages <- c(
  "reticulate", "ggplot2", "dplyr", "tidyr", "patchwork", 
  "digest", "ape", "phangorn", "phyloseq", "ggtree",
  "testthat", "knitr", "rmarkdown"
)

all_installed <- TRUE
for (pkg in required_packages) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("✓ %-20s installed\n", pkg))
  } else {
    cat(sprintf("✗ %-20s MISSING\n", pkg))
    all_installed <- FALSE
  }
}

cat("\n")
if (all_installed) {
  cat("🎉 All required packages installed successfully!\n\n")
  cat("Next steps:\n")
  cat("1. Configure Python MDITRE backend (conda environment)\n")
  cat("2. Run: Rscript R/setup_environment.R\n")
  cat("\n")
} else {
  cat("⚠ Some packages failed to install. Please install them manually.\n\n")
}
