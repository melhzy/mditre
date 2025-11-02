#!/usr/bin/env Rscript
#
# Generate roxygen2 documentation for MDITRE R package
#
# This script:
# 1. Processes all roxygen2 comments in R/*.R files
# 2. Generates NAMESPACE file
# 3. Generates man/*.Rd files
# 4. Builds package documentation
#
# Usage:
#   Rscript generate_docs.R
#   # Or from R:
#   source("generate_docs.R")

# Check if roxygen2 is installed
if (!requireNamespace("roxygen2", quietly = TRUE)) {
  message("Installing roxygen2...")
  install.packages("roxygen2")
}

# Check if devtools is installed
if (!requireNamespace("devtools", quietly = TRUE)) {
  message("Installing devtools...")
  install.packages("devtools")
}

library(roxygen2)
library(devtools)

# Set package root
pkg_root <- "."
if (!file.exists(file.path(pkg_root, "DESCRIPTION"))) {
  stop("DESCRIPTION file not found. Are you in the R package root directory?")
}

message("=== Generating roxygen2 documentation ===\n")

# 1. Clean existing man/ directory
man_dir <- file.path(pkg_root, "man")
if (dir.exists(man_dir)) {
  message("Removing existing man/ directory...")
  unlink(man_dir, recursive = TRUE)
}

# 2. Generate documentation
message("\nProcessing roxygen2 comments...")
roxygen2::roxygenise(
  package.dir = pkg_root,
  roclets = c("namespace", "rd", "collate")
)

message("\n=== Documentation generation complete ===\n")

# 3. List generated files
message("Generated files:")
message(sprintf("  NAMESPACE: %s", file.path(pkg_root, "NAMESPACE")))

if (dir.exists(man_dir)) {
  rd_files <- list.files(man_dir, pattern = "\\.Rd$", full.names = FALSE)
  message(sprintf("  man/ directory: %d .Rd files", length(rd_files)))
  
  if (length(rd_files) > 0) {
    message("\nDocumented functions:")
    for (f in sort(rd_files)) {
      message(sprintf("    %s", f))
    }
  }
} else {
  message("  WARNING: man/ directory not created")
}

# 4. Check for documentation issues
message("\n=== Checking documentation ===\n")
check_result <- tryCatch({
  devtools::document(pkg = pkg_root)
  message("✓ Documentation check passed")
  TRUE
}, error = function(e) {
  message("✗ Documentation check failed:")
  message(e$message)
  FALSE
})

# 5. Summary
message("\n=== Summary ===\n")
message("Package: mditre")
message(sprintf("Root: %s", normalizePath(pkg_root)))

if (file.exists(file.path(pkg_root, "NAMESPACE"))) {
  namespace_lines <- readLines(file.path(pkg_root, "NAMESPACE"))
  export_count <- sum(grepl("^export\\(", namespace_lines))
  import_count <- sum(grepl("^import\\(", namespace_lines))
  message(sprintf("Exports: %d functions/classes", export_count))
  message(sprintf("Imports: %d packages", import_count))
}

if (dir.exists(man_dir)) {
  rd_count <- length(list.files(man_dir, pattern = "\\.Rd$"))
  message(sprintf("Documentation files: %d .Rd files", rd_count))
}

message("\nNext steps:")
message("  1. Review generated NAMESPACE file")
message("  2. Check man/*.Rd files for completeness")
message("  3. Run R CMD check: devtools::check()")
message("  4. Build package: devtools::build()")
message("  5. Install: devtools::install()")

if (check_result) {
  message("\n✓ Documentation ready for use!")
} else {
  message("\n⚠ Please fix documentation issues before proceeding")
}
