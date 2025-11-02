# Force load torch DLLs before package initialization

cat("=== Forcing torch DLL loading ===\n\n")

# Get torch paths
torch_pkg <- system.file(package = "torch")
deps_dir <- file.path(torch_pkg, "deps")
libtorch_lib <- file.path(deps_dir, "libtorch", "lib")
lantern_dir <- list.dirs(deps_dir, recursive = FALSE)[grep("lantern", list.dirs(deps_dir, recursive = FALSE))]
lantern_lib <- file.path(lantern_dir, "lib")

cat("Torch package:", torch_pkg, "\n")
cat("Libtorch lib:", libtorch_lib, "\n")
cat("Lantern lib:", lantern_lib, "\n\n")

# Step 1: Add to system PATH
cat("1. Adding torch directories to PATH...\n")
old_path <- Sys.getenv("PATH")
new_path <- paste(lantern_lib, libtorch_lib, old_path, sep = ";")
Sys.setenv(PATH = new_path)
cat("   ✓ PATH updated\n\n")

# Step 2: Pre-load all libtorch DLLs in correct order
cat("2. Pre-loading libtorch DLLs...\n")
torch_dlls <- c(
  "c10.dll",
  "fbgemm.dll", 
  "asmjit.dll",
  "cpuinfo.dll",
  "torch_cpu.dll",
  "torch.dll"
)

for (dll_name in torch_dlls) {
  dll_path <- file.path(libtorch_lib, dll_name)
  if (file.exists(dll_path)) {
    result <- tryCatch({
      dyn.load(dll_path, local = FALSE, now = TRUE)
      cat("   ✓", dll_name, "\n")
      TRUE
    }, error = function(e) {
      cat("   ⚠", dll_name, "- ", conditionMessage(e), "\n")
      FALSE
    })
  }
}

# Step 3: Load lantern DLL
cat("\n3. Loading lantern DLL...\n")
lantern_dll <- file.path(lantern_lib, "lantern.dll")
if (file.exists(lantern_dll)) {
  result <- tryCatch({
    dyn.load(lantern_dll, local = FALSE, now = TRUE)
    cat("   ✓ lantern.dll loaded\n")
    TRUE
  }, error = function(e) {
    cat("   ✗ lantern.dll failed:", conditionMessage(e), "\n")
    FALSE
  })
}

# Step 4: Load torch package
cat("\n4. Loading torch package...\n")
if ("package:torch" %in% search()) {
  detach("package:torch", unload = TRUE)
  Sys.sleep(1)
}

suppressPackageStartupMessages({
  library(torch)
})
cat("   ✓ Package loaded\n\n")

# Step 5: Force initialize lantern
cat("5. Force initializing lantern...\n")

# Try to access internal torch initialization
tryCatch({
  # Access torch's internal namespace
  torch_ns <- asNamespace("torch")
  
  # Try to manually trigger lantern initialization
  if (exists("lantern_start", envir = torch_ns, inherits = FALSE)) {
    torch_ns$lantern_start()
    cat("   ✓ lantern_start() called\n")
  }
  
  # Set lantern as loaded
  if (exists(".lantern_loaded", envir = torch_ns, inherits = FALSE)) {
    assign(".lantern_loaded", TRUE, envir = torch_ns)
    cat("   ✓ .lantern_loaded set to TRUE\n")
  }
}, error = function(e) {
  cat("   ⚠ Could not access internal functions:", conditionMessage(e), "\n")
})

# Step 6: Test torch functionality
cat("\n6. Testing torch functionality...\n")

test1 <- tryCatch({
  result <- torch_is_installed()
  cat("   torch_is_installed():", result, "\n")
  result
}, error = function(e) {
  cat("   ✗ torch_is_installed() error:", conditionMessage(e), "\n")
  FALSE
})

test2 <- tryCatch({
  t <- torch_tensor(c(1.0, 2.0, 3.0, 4.0, 5.0))
  cat("   ✓ torch_tensor() works!\n")
  print(t)
  cat("\n")
  TRUE
}, error = function(e) {
  cat("   ✗ torch_tensor() error:", conditionMessage(e), "\n")
  FALSE
})

if (test2) {
  cat("\n✓✓✓ SUCCESS! torch is working! ✓✓✓\n\n")
  
  # Test nn_module
  cat("7. Testing nn_module...\n")
  tryCatch({
    test_mod <- nn_module(
      initialize = function() {
        self$fc <- nn_linear(10, 5)
      },
      forward = function(x) {
        self$fc(x)
      }
    )
    cat("   ✓ nn_module() works!\n\n")
    
    cat("=== torch is ready! Running devtools::test() ===\n\n")
    
    # Change to R package directory
    setwd("d:/Github/mditre/R")
    
    # Run tests
    devtools::test(reporter = "summary")
    
  }, error = function(e) {
    cat("   ✗ nn_module() error:", conditionMessage(e), "\n")
  })
} else {
  cat("\n✗✗✗ torch still not working ✗✗✗\n")
  cat("\nTrying alternative: Direct C function call...\n")
  
  # Last resort: Try to call lantern functions directly
  tryCatch({
    # List loaded DLLs
    loaded <- getLoadedDLLs()
    cat("Currently loaded DLLs with 'torch' or 'lantern':\n")
    for (name in names(loaded)) {
      if (grepl("torch|lantern", name, ignore.case = TRUE)) {
        cat("  -", name, ":", loaded[[name]][["path"]], "\n")
      }
    }
  }, error = function(e) {
    cat("Could not list DLLs\n")
  })
}
