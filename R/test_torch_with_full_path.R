# Test torch with PATH configured from within R

cat("=== Setting up PATH for torch DLLs ===\n\n")

# Get torch package path
torch_pkg <- system.file(package = "torch")
deps_dir <- file.path(torch_pkg, "deps")

# Add torch DLL directories to PATH
libtorch_lib <- file.path(deps_dir, "libtorch", "lib")
lantern_lib <- file.path(deps_dir, "lantern-0.16.2+cpu-win64", "lib")

cat("Adding to PATH:\n")
cat("  -", libtorch_lib, "\n")
cat("  -", lantern_lib, "\n\n")

# Prepend to PATH so these are found first
current_path <- Sys.getenv("PATH")
new_path <- paste(lantern_lib, libtorch_lib, current_path, sep = ";")
Sys.setenv(PATH = new_path)

cat("PATH updated\n\n")

# Now try to load torch functions
cat("=== Testing torch functionality ===\n\n")

# Detach torch if loaded
if ("package:torch" %in% search()) {
  detach("package:torch", unload = TRUE)
}

# Load torch
cat("1. Loading torch...\n")
library(torch)
cat("   ✓ Loaded\n\n")

cat("2. Checking torch_is_installed()...\n")
installed <- torch_is_installed()
cat("   Result:", installed, "\n\n")

if (installed) {
  cat("3. Testing tensor creation...\n")
  t <- torch_tensor(c(1.0, 2.0, 3.0, 4.0, 5.0))
  print(t)
  cat("\n✓✓✓ SUCCESS! Torch is fully functional!\n\n")
  
  # Test nn_module (critical for MDITRE)
  cat("4. Testing nn_module...\n")
  test_mod <- nn_module(
    initialize = function() {
      self$fc <- nn_linear(10, 5)
    },
    forward = function(x) {
      self$fc(x)
    }
  )
  cat("   ✓ nn_module works!\n\n")
  
  cat("=== torch is ready for MDITRE testing! ===\n")
} else {
  cat("3. Still not installed. Trying manual DLL load...\n")
  
  # Add both lib directories
  libtorch_dlls <- list.files(libtorch_lib, pattern = "\\.dll$", full.names = TRUE)
  cat("   Found", length(libtorch_dlls), "libtorch DLLs\n")
  
  # Try loading key DLLs first
  for (dll in c("c10.dll", "torch_cpu.dll")) {
    dll_path <- file.path(libtorch_lib, dll)
    if (file.exists(dll_path)) {
      cat("   Loading", dll, "...")
      tryCatch({
        dyn.load(dll_path, local = FALSE, now = TRUE)
        cat(" ✓\n")
      }, error = function(e) {
        cat(" ✗ (", conditionMessage(e), ")\n")
      })
    }
  }
  
  # Now try lantern
  lantern_dll <- file.path(lantern_lib, "lantern.dll")
  cat("   Loading lantern.dll...")
  tryCatch({
    dyn.load(lantern_dll, local = FALSE, now = TRUE)
    cat(" ✓\n\n")
    
    cat("4. Testing after manual load...\n")
    t <- torch_tensor(c(1, 2, 3))
    print(t)
    cat("\n✓✓✓ SUCCESS after manual load!\n")
  }, error = function(e) {
    cat(" ✗\n")
    cat("   Error:", conditionMessage(e), "\n\n")
    cat("=== torch loading failed ===\n")
    cat("\nThis appears to be a system-level DLL dependency issue.\n")
    cat("Possible solutions:\n")
    cat("1. Install Visual Studio 2019 C++ Redistributable (x64)\n")
    cat("2. Check Windows Event Viewer for detailed DLL load errors\n")
    cat("3. Try torch version 0.13.0 or earlier\n")
    cat("4. Use reticulate with Python PyTorch as workaround\n")
  })
}
