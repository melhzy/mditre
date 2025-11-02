#!/usr/bin/env Rscript

# Simple Documentation Generator for MDITRE R Package
# Generates NAMESPACE and man/*.Rd files from roxygen2 comments
# without needing to load package dependencies

cat("=== Simple roxygen2 Documentation Generator ===\n\n")

# Check if roxygen2 is installed
if (!requireNamespace("roxygen2", quietly = TRUE)) {
  cat("ERROR: roxygen2 package is not installed.\n")
  cat("Install with: install.packages('roxygen2')\n")
  quit(status = 1)
}

cat("Step 1: Cleaning existing documentation...\n")
if (dir.exists("man")) {
  unlink("man", recursive = TRUE)
  cat("  - Removed existing man/ directory\n")
}

cat("\nStep 2: Generating documentation from roxygen2 comments...\n")

# Generate documentation without loading the package
tryCatch({
  roxygen2::roxygenise(
    package.dir = ".",
    roclets = c("rd", "namespace"),
    load_code = "source",  # Use source instead of loading compiled package
    clean = TRUE
  )
  cat("  ✓ Documentation generated successfully!\n")
}, error = function(e) {
  cat("  WARNING: Some documentation may not have been generated:\n")
  cat("  ", conditionMessage(e), "\n")
  cat("  This is expected if dependencies are not installed.\n")
  cat("  NAMESPACE and .Rd files should still be created.\n")
})

cat("\nStep 3: Verifying generated files...\n")

# Check NAMESPACE
if (file.exists("NAMESPACE")) {
  namespace_lines <- readLines("NAMESPACE")
  export_count <- sum(grepl("^export\\(", namespace_lines))
  cat("  ✓ NAMESPACE file created\n")
  cat("    - Exports:", export_count, "functions\n")
} else {
  cat("  ✗ NAMESPACE file not found\n")
}

# Check man/ directory
if (dir.exists("man")) {
  rd_files <- list.files("man", pattern = "\\.Rd$", full.names = FALSE)
  cat("  ✓ man/ directory created\n")
  cat("    - Generated", length(rd_files), ".Rd files\n")
  
  if (length(rd_files) > 0) {
    cat("\n  First 10 .Rd files:\n")
    for (f in head(rd_files, 10)) {
      cat("    -", f, "\n")
    }
    if (length(rd_files) > 10) {
      cat("    ... and", length(rd_files) - 10, "more\n")
    }
  }
} else {
  cat("  ✗ man/ directory not found\n")
}

cat("\n=== Documentation Generation Complete ===\n")
cat("\nNext steps:\n")
cat("1. Verify NAMESPACE exports the correct functions\n")
cat("2. Review .Rd files in man/ directory\n")
cat("3. Install package dependencies to build full documentation\n")
cat("4. Run pkgdown::build_site() to generate website\n")
cat("\nNote: To install dependencies, run:\n")
cat("  install.packages(c('torch', 'phyloseq', 'ggplot2', 'ggtree'))\n")
