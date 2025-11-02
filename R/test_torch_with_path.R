# Test torch with explicit PATH configuration

# Get torch package location
torch_pkg <- system.file(package = "torch")
cat("Torch package at:", torch_pkg, "\n")

# Set up paths
deps_dir <- file.path(torch_pkg, "deps")
cat("deps directory:", deps_dir, "\n")
cat("deps exists:", dir.exists(deps_dir), "\n\n")

# Find lantern and libtorch directories
lantern_dirs <- list.dirs(deps_dir, recursive = FALSE, full.names = TRUE)
lantern_dirs <- lantern_dirs[grepl("lantern", lantern_dirs)]
libtorch_dir <- file.path(deps_dir, "libtorch")

cat("Lantern directory:", lantern_dirs[1], "\n")
cat("Libtorch directory:", libtorch_dir, "\n\n")

# Add to PATH
if (length(lantern_dirs) > 0 && dir.exists(lantern_dirs[1])) {
  lantern_lib <- file.path(lantern_dirs[1], "lib")
  if (dir.exists(lantern_lib)) {
    cat("Adding to PATH:", lantern_lib, "\n")
    Sys.setenv(PATH = paste(lantern_lib, Sys.getenv("PATH"), sep = ";"))
  }
}

if (dir.exists(libtorch_dir)) {
  libtorch_lib <- file.path(libtorch_dir, "lib")
  if (dir.exists(libtorch_lib)) {
    cat("Adding to PATH:", libtorch_lib, "\n")
    Sys.setenv(PATH = paste(libtorch_lib, Sys.getenv("PATH"), sep = ";"))
  }
}

cat("\n=== Attempting to load torch ===\n")
# Unload torch if already loaded
if ("package:torch" %in% search()) {
  detach("package:torch", unload = TRUE)
}

# Try loading torch
result <- try({
  library(torch)
  cat("✓ Torch library loaded\n\n")
  
  cat("Testing torch_is_installed()...\n")
  installed <- torch_is_installed()
  cat("Result:", installed, "\n\n")
  
  cat("Testing tensor creation...\n")
  t <- torch_tensor(c(1.0, 2.0, 3.0, 4.0, 5.0))
  print(t)
  cat("\n✓✓✓ SUCCESS! Torch is working! ✓✓✓\n")
  
  # Test nn_module (critical for our package)
  cat("\nTesting nn_module (required for MDITRE)...\n")
  test_module <- nn_module(
    initialize = function() {
      self$linear <- nn_linear(10, 5)
    },
    forward = function(x) {
      self$linear(x)
    }
  )
  cat("✓ nn_module works!\n")
  
  TRUE
}, silent = FALSE)

if (inherits(result, "try-error")) {
  cat("\n✗ Torch still not working\n")
  cat("Error:", as.character(result), "\n\n")
  
  cat("Next steps:\n")
  cat("1. Try restarting your R session\n")
  cat("2. Check if antivirus is blocking DLL loading\n")
  cat("3. Try running R as administrator\n")
  cat("4. Check Windows Event Viewer for DLL load errors\n")
  FALSE
} else {
  result
}
