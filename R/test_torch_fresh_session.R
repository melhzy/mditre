# Fresh session torch test with detailed debugging

cat("=== Testing torch in fresh R session ===\n\n")

# Clear any environment variables that might interfere
Sys.unsetenv("TORCH_HOME")

# Get torch paths
torch_pkg <- system.file(package = "torch")
deps_dir <- file.path(torch_pkg, "deps")

cat("1. Torch package path:", torch_pkg, "\n")
cat("2. deps directory:", deps_dir, "\n")
cat("3. deps exists:", dir.exists(deps_dir), "\n\n")

# Try loading torch with tracing
cat("4. Loading torch package...\n")
suppressPackageStartupMessages({
  result <- tryCatch({
    library(torch)
    cat("   ✓ Package loaded\n")
    TRUE
  }, error = function(e) {
    cat("   ✗ Package load failed:", conditionMessage(e), "\n")
    FALSE
  })
})

if (result) {
  cat("\n5. Checking torch_is_installed()...\n")
  installed <- torch_is_installed()
  cat("   Result:", installed, "\n")
  
  if (!installed) {
    cat("\n6. Attempting manual lantern load...\n")
    
    # Try to directly access the lantern loading function
    lantern_path <- file.path(deps_dir, "lantern-0.16.2+cpu-win64", "lib")
    cat("   Lantern path:", lantern_path, "\n")
    
    if (dir.exists(lantern_path)) {
      dll_file <- file.path(lantern_path, "lantern.dll")
      cat("   DLL file:", dll_file, "\n")
      cat("   DLL exists:", file.exists(dll_file), "\n")
      
      # Try manual DLL load
      cat("\n7. Attempting manual DLL load with dyn.load...\n")
      dll_result <- tryCatch({
        dyn.load(dll_file)
        cat("   ✓ DLL loaded successfully!\n")
        TRUE
      }, error = function(e) {
        cat("   ✗ DLL load failed:\n")
        cat("     Error:", conditionMessage(e), "\n")
        
        # Try to get more details
        if (grepl("cannot load", conditionMessage(e), ignore.case = TRUE)) {
          cat("\n   This error suggests missing dependencies.\n")
          cat("   Possible causes:\n")
          cat("   - Missing Visual C++ runtime libraries\n")
          cat("   - Missing CUDA libraries (even for CPU version)\n")
          cat("   - Incompatible DLL architecture\n")
        }
        FALSE
      })
      
      if (dll_result) {
        cat("\n8. Testing tensor creation after manual load...\n")
        tensor_result <- tryCatch({
          t <- torch_tensor(c(1, 2, 3, 4, 5))
          print(t)
          cat("\n   ✓✓✓ SUCCESS! Torch is working!\n")
          TRUE
        }, error = function(e) {
          cat("   ✗ Tensor creation still failed:", conditionMessage(e), "\n")
          FALSE
        })
      }
    }
  } else {
    cat("\n6. Testing tensor creation...\n")
    tensor_result <- tryCatch({
      t <- torch_tensor(c(1, 2, 3, 4, 5))
      print(t)
      cat("\n   ✓✓✓ SUCCESS! Torch is working!\n")
      TRUE
    }, error = function(e) {
      cat("   ✗ Tensor creation failed:", conditionMessage(e), "\n")
      FALSE
    })
  }
}

cat("\n=== Diagnosis Complete ===\n")
