#' Package Initialization
#'
#' Automatically configure the Python MDITRE backend when R MDITRE is loaded.
#'
#' @name zzz
#' @keywords internal
NULL

#' Configure Python MDITRE Backend
#'
#' Sets up the Python environment to use the MDITRE conda environment
#' and ensures the Python MDITRE package is available.
#'
#' @param conda_env Name of the conda environment (default: "MDITRE")
#' @param install_mditre Whether to install/update Python MDITRE in development mode
#' @param python_path Optional: Path to the Python MDITRE directory containing setup.py
#'
#' @return Invisible TRUE if successful
#'
#' @details
#' **Architecture**: R MDITRE is an R interface that bridges to Python MDITRE via reticulate.
#' 
#' This function:
#' - Configures reticulate to use the MDITRE conda environment (Python MDITRE backend)
#' - Optionally installs the Python MDITRE package in development mode
#' - Verifies that PyTorch is available
#' - Reports GPU availability
#'
#' **Two-Package System**:
#' - **Python MDITRE** (Backend): Native PyTorch models in MDITRE conda environment
#' - **R MDITRE** (Frontend): R interface providing R workflows and visualization
#' 
#' @examples
#' \dontrun{
#' # Use default MDITRE environment
#' setup_mditre_python()
#'
#' # Specify custom environment
#' setup_mditre_python(conda_env = "my_mditre_env")
#'
#' # Skip Python MDITRE installation
#' setup_mditre_python(install_mditre = FALSE)
#' }
#'
#' @export
setup_mditre_python <- function(conda_env = "MDITRE", 
                                 install_mditre = TRUE,
                                 python_path = NULL) {
  
  # Check if reticulate is available
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required but not installed. Install with: install.packages('reticulate')")
  }
  
  # Configure conda environment
  tryCatch({
    reticulate::use_condaenv(conda_env, required = TRUE)
    message(sprintf("✓ Using Python MDITRE backend: conda environment '%s'", conda_env))
  }, error = function(e) {
    stop(sprintf("Failed to configure conda environment '%s'. Error: %s", conda_env, e$message))
  })
  
  # Install Python MDITRE in development mode if requested
  if (install_mditre) {
    # Determine Python MDITRE directory path
    if (is.null(python_path)) {
      # Try to find Python directory relative to package installation
      pkg_path <- system.file(package = "rmditre")
      if (pkg_path != "") {
        # Installed package case
        python_path <- file.path(dirname(dirname(pkg_path)), "Python")
      } else {
        # Development case - assume we're in R/ directory
        python_path <- file.path(getwd(), "..", "Python")
      }
    }
    
    python_path <- normalizePath(python_path, winslash = "/", mustWork = FALSE)
    
    if (dir.exists(python_path)) {
      message("Installing Python MDITRE package in development mode...")
      result <- system2("conda", 
                       args = c("run", "-n", conda_env, "pip", "install", "-e", python_path),
                       stdout = TRUE, stderr = TRUE)
      
      if (!is.null(attr(result, "status")) && attr(result, "status") != 0) {
        warning("Failed to install Python MDITRE. You may need to install it manually.")
      } else {
        message("✓ Python MDITRE package installed/updated")
      }
    } else {
      message(sprintf("Python directory not found at: %s", python_path))
      message("Skipping Python MDITRE installation. Ensure it's installed manually.")
    }
  }
  
  # Verify PyTorch is available
  tryCatch({
    torch <- reticulate::import("torch", convert = FALSE)
    py_version <- reticulate::py_config()$version
    torch_version <- torch$`__version__`
    cuda_available <- torch$cuda$is_available()
    
    message(sprintf("✓ Python %s", py_version))
    message(sprintf("✓ PyTorch %s", torch_version))
    message(sprintf("✓ CUDA: %s", ifelse(cuda_available, "Available", "Not available")))
    
    if (cuda_available) {
      gpu_name <- torch$cuda$get_device_name(0L)
      message(sprintf("  GPU: %s", gpu_name))
    }
  }, error = function(e) {
    warning("PyTorch not available. Some functionality may not work.")
  })
  
  # Try to verify Python MDITRE is available
  tryCatch({
    mditre <- reticulate::import("mditre", convert = FALSE)
    mditre_version <- mditre$`__version__`
    message(sprintf("✓ Python MDITRE: %s", mditre_version))
  }, error = function(e) {
    warning("Python MDITRE package not found. Install with setup_mditre_python()")
  })
  
  invisible(TRUE)
}

#' @keywords internal
.onLoad <- function(libname, pkgname) {
  # Package load message
  packageStartupMessage(
    "R MDITRE - R interface to Python MDITRE\n",
    "Architecture: R 4.5.2+ → reticulate → Python MDITRE (MDITRE conda env)\n",
    "To configure Python backend, run: setup_mditre_python()\n",
    "For automatic setup, set options(rmditre.auto_setup = TRUE) before loading."
  )
  
  # Auto-setup if option is set
  if (isTRUE(getOption("rmditre.auto_setup", default = FALSE))) {
    tryCatch({
      setup_mditre_python()
    }, error = function(e) {
      packageStartupMessage("Auto-setup failed. Run setup_mditre_python() manually.")
    })
  }
}

#' @keywords internal
.onAttach <- function(libname, pkgname) {
  # Additional attach message if needed
  invisible(NULL)
}
