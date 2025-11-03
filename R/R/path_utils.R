#' Cross-Platform Path Utilities for R MDITRE
#'
#' Provides platform-independent path handling that works seamlessly
#' across Windows, macOS, and Linux/Ubuntu. Works both in development
#' mode and when installed via install.packages().
#'
#' @name path_utils
#' @keywords internal
NULL

#' Get MDITRE Package Installation Directory
#'
#' Returns the directory where the MDITRE R package is installed.
#' This could be in a user library (install.packages()) or a 
#' development directory (devtools::load_all()).
#'
#' @return Character string with absolute path to package directory,
#'   or NULL if package is not installed
#'
#' @examples
#' \dontrun{
#' pkg_dir <- get_package_root()
#' # Installed: "/usr/local/lib/R/site-library/rmditre"
#' # Dev mode: "/home/username/mditre/R"
#' }
#'
#' @export
get_package_root <- function() {
  tryCatch({
    # Try to find package installation directory
    pkg_path <- system.file(package = "rmditre")
    
    if (pkg_path != "") {
      return(normalizePath(pkg_path, winslash = "/", mustWork = FALSE))
    } else {
      # Not installed as package - assume development mode in R/ directory
      current_dir <- getwd()
      if (basename(current_dir) == "R" && dir.exists("R") && file.exists("DESCRIPTION")) {
        # We're in the R package root directory
        return(normalizePath(current_dir, winslash = "/", mustWork = FALSE))
      } else if (file.exists("DESCRIPTION") && file.exists("NAMESPACE")) {
        # Current directory is the package root
        return(normalizePath(current_dir, winslash = "/", mustWork = FALSE))
      } else {
        # Try sourced file location
        return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
      }
    }
  }, error = function(e) {
    stop(
      "Cannot determine MDITRE package location. ",
      "MDITRE may not be properly installed. ",
      "Please reinstall using: install.packages('mditre')\n",
      "Error details: ", e$message,
      call. = FALSE
    )
  })
}

#' Get MDITRE Project Root Directory (Development Mode Only)
#'
#' Returns the project root directory if in development mode (full
#' repository structure with Python/ and R/ directories). Returns NULL
#' when installed via install.packages().
#'
#' @return Character string with absolute path to project root,
#'   or NULL if in installed mode
#'
#' @examples
#' \dontrun{
#' root <- get_project_root()
#' if (!is.null(root)) {
#'   message("Development mode: ", root)
#' } else {
#'   message("Installed via install.packages()")
#' }
#' }
#'
#' @export
get_project_root <- function() {
  pkg_root <- get_package_root()
  
  if (is.null(pkg_root)) {
    return(NULL)
  }
  
  # Check if we're in development mode
  # Development: /path/to/mditre/R
  # Installed: /path/to/R/library/rmditre
  
  parent_dir <- dirname(pkg_root)
  
  # If parent directory contains both "Python" and "R", we're in dev mode
  if (dir.exists(file.path(parent_dir, "Python")) && 
      dir.exists(file.path(parent_dir, "R"))) {
    return(normalizePath(parent_dir, winslash = "/", mustWork = FALSE))
  }
  
  # Not in development mode
  return(NULL)
}

#' Find Project Root by Looking for Marker Files
#'
#' @param start_dir Starting directory for search
#' @return Path to project root or NULL if not found
#' @keywords internal
find_project_root_by_markers <- function(start_dir) {
  # Marker files that indicate project root
  markers <- c("README.md", "CHANGELOG.md", "Python", "R")
  
  current_dir <- normalizePath(start_dir, winslash = "/", mustWork = FALSE)
  max_levels <- 10  # Prevent infinite loop
  
  for (i in 1:max_levels) {
    # Check if current directory contains markers
    has_markers <- sapply(markers, function(m) {
      file.exists(file.path(current_dir, m)) || 
        dir.exists(file.path(current_dir, m))
    })
    
    # If we find multiple markers, likely the root
    if (sum(has_markers) >= 2) {
      return(current_dir)
    }
    
    # Move up one directory
    parent_dir <- dirname(current_dir)
    if (parent_dir == current_dir) {
      break  # Reached filesystem root
    }
    current_dir <- parent_dir
  }
  
  # Fallback to current working directory
  return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
}

#' Get Python Implementation Directory (Development Mode Only)
#'
#' Returns the Python/ directory in development mode, or NULL when
#' installed via install.packages().
#'
#' @return Character string with absolute path to Python/ directory,
#'   or NULL if not in development mode
#' @export
get_python_dir <- function() {
  project_root <- get_project_root()
  if (!is.null(project_root)) {
    python_dir <- file.path(project_root, "Python")
    if (dir.exists(python_dir)) {
      return(normalizePath(python_dir, winslash = "/", mustWork = FALSE))
    }
  }
  return(NULL)
}

#' Get R Implementation Directory
#'
#' Returns the R package directory location. In development mode, this is
#' the R/ subdirectory. When installed, this is the package library location.
#'
#' @return Character string with absolute path to R package directory
#' @export
get_r_dir <- function() {
  pkg_root <- get_package_root()
  if (!is.null(pkg_root)) {
    return(pkg_root)
  }
  return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
}

#' Get Data Directory Path
#'
#' In development mode, returns mditre_root/data/
#' When installed via install.packages(), returns base_path or current directory.
#'
#' @param subdirectory Optional subdirectory within data/ (e.g., 'raw', 'processed')
#' @param base_path Base path for data directory (defaults to project root or cwd)
#' @return Character string with absolute path to data directory
#'
#' @examples
#' \dontrun{
#' # Development mode
#' get_data_dir()  # Returns mditre/data/
#' get_data_dir('raw')  # Returns mditre/data/raw/
#' 
#' # Installed mode - specify your data location
#' get_data_dir(base_path = '/my/data/location')
#' }
#'
#' @export
get_data_dir <- function(subdirectory = NULL, base_path = NULL) {
  if (!is.null(base_path)) {
    data_dir <- base_path
  } else {
    project_root <- get_project_root()
    if (!is.null(project_root)) {
      data_dir <- file.path(project_root, "data")
    } else {
      # Installed via install.packages() - use current directory
      data_dir <- file.path(getwd(), "data")
    }
  }
  
  if (!is.null(subdirectory)) {
    data_dir <- file.path(data_dir, subdirectory)
  }
  return(normalizePath(data_dir, winslash = "/", mustWork = FALSE))
}

#' Get Output Directory for Results
#'
#' In development mode, returns mditre_root/outputs/
#' When installed via install.packages(), returns base_path or current directory.
#'
#' @param create Logical, if TRUE create directory if it doesn't exist
#' @param base_path Base path for output directory (defaults to project root or cwd)
#' @return Character string with absolute path to output directory
#'
#' @examples
#' \dontrun{
#' # Development mode
#' get_output_dir()  # Returns mditre/outputs/
#' 
#' # Installed mode - specify your output location
#' get_output_dir(base_path = '/my/output/location')
#' }
#'
#' @export
get_output_dir <- function(create = TRUE, base_path = NULL) {
  if (!is.null(base_path)) {
    output_dir <- base_path
  } else {
    project_root <- get_project_root()
    if (!is.null(project_root)) {
      output_dir <- file.path(project_root, "outputs")
    } else {
      # Installed via install.packages() - use current directory
      output_dir <- file.path(getwd(), "outputs")
    }
  }
  
  output_dir <- normalizePath(output_dir, winslash = "/", mustWork = FALSE)
  
  if (create && !dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  return(output_dir)
}

#' Normalize Path to Platform-Independent Format
#'
#' Converts any path to a platform-independent format using forward slashes.
#'
#' @param path Path to normalize (character string)
#' @param relative_to Optional base path to make relative to
#' @return Normalized path string
#'
#' @examples
#' \dontrun{
#' # Works on all platforms
#' normalize_path("~/data/file.txt")
#' normalize_path("C:\\Users\\data\\file.txt")  # On Windows
#' }
#'
#' @export
normalize_path <- function(path, relative_to = NULL) {
  # Expand ~ and environment variables
  path <- path.expand(path)
  
  # Normalize to platform format, using forward slashes
  path <- normalizePath(path, winslash = "/", mustWork = FALSE)
  
  # Make relative if requested
  if (!is.null(relative_to)) {
    relative_to <- normalizePath(path.expand(relative_to), 
                                 winslash = "/", mustWork = FALSE)
    tryCatch({
      # Try to compute relative path
      path <- gsub(paste0("^", relative_to, "/?"), "", path)
    }, error = function(e) {
      # If can't make relative, return absolute
    })
  }
  
  return(path)
}

#' Ensure Directory Exists
#'
#' Creates a directory and all parent directories if they don't exist.
#'
#' @param directory Directory path to create
#' @return Absolute path to the directory
#'
#' @examples
#' \dontrun{
#' ensure_dir_exists("outputs/models")
#' }
#'
#' @export
ensure_dir_exists <- function(directory) {
  tryCatch({
    directory <- normalizePath(path.expand(directory), 
                              winslash = "/", mustWork = FALSE)
    
    if (!dir.exists(directory)) {
      success <- dir.create(directory, recursive = TRUE, showWarnings = FALSE)
      if (!success) {
        stop("Directory creation failed")
      }
    }
    
    return(directory)
  }, error = function(e) {
    stop(
      "Cannot create directory '", directory, "'. ",
      "Please check directory permissions or choose a different location.\n",
      "Error details: ", e$message,
      call. = FALSE
    )
  })
}

#' Join Path Components
#'
#' Joins multiple path components in a platform-independent way.
#'
#' @param ... Path components to join
#' @return Combined path string
#'
#' @examples
#' \dontrun{
#' join_paths("data", "raw", "file.csv")
#' # Returns "data/raw/file.csv" (platform-independent)
#' }
#'
#' @export
join_paths <- function(...) {
  parts <- list(...)
  if (length(parts) == 0) {
    return("")
  }
  
  path <- file.path(...)
  # Use forward slashes for consistency
  path <- gsub("\\\\", "/", path)
  return(path)
}

#' Get Platform Information
#'
#' Returns information about the current platform.
#'
#' @return Named list with platform details
#'
#' @examples
#' \dontrun{
#' info <- get_platform_info()
#' # Returns list with: os, os_version, r_version, sep, home, cwd
#' }
#'
#' @export
get_platform_info <- function() {
  list(
    os = Sys.info()["sysname"],
    os_version = Sys.info()["release"],
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    sep = .Platform$file.sep,
    pathsep = .Platform$path.sep,
    home = normalizePath(path.expand("~"), winslash = "/", mustWork = FALSE),
    cwd = normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  )
}

#' Convert Path to Unix Style
#'
#' Converts any path to Unix-style with forward slashes.
#' Useful for cross-platform config files.
#'
#' @param path Path to convert
#' @return Path string with forward slashes
#'
#' @examples
#' \dontrun{
#' to_unix_path("C:\\Users\\data\\file.txt")
#' # Returns "C:/Users/data/file.txt"
#' }
#'
#' @export
to_unix_path <- function(path) {
  gsub("\\\\", "/", path)
}

#' Convert Path to Platform-Specific Style
#'
#' Converts path to use platform-appropriate separators.
#'
#' @param path Path to convert
#' @return Path string with platform separators
#'
#' @export
to_platform_path <- function(path) {
  if (.Platform$OS.type == "windows") {
    gsub("/", "\\\\", path)
  } else {
    gsub("\\\\", "/", path)
  }
}

#' Print Platform and Path Information
#'
#' Diagnostic function to display current platform and path configuration.
#'
#' @export
print_path_info <- function() {
  cat("MDITRE Cross-Platform Path Utilities\n")
  cat(strrep("=", 70), "\n\n")
  
  # Platform info
  info <- get_platform_info()
  cat("Platform Information:\n")
  cat(sprintf("  OS: %s %s\n", info$os, info$os_version))
  cat(sprintf("  R Version: %s\n", info$r_version))
  cat(sprintf("  Path Separator: %s\n", info$sep))
  cat(sprintf("  Home Directory: %s\n", info$home))
  cat(sprintf("  Current Directory: %s\n", info$cwd))
  
  # Installation mode
  cat("\n", strrep("=", 70), "\n")
  project_root <- get_project_root()
  if (!is.null(project_root)) {
    cat("Installation Mode: Development (devtools::load_all)\n")
    cat(sprintf("  Project Root: %s\n", project_root))
    python_dir <- get_python_dir()
    if (!is.null(python_dir)) {
      cat(sprintf("  Python Directory: %s\n", python_dir))
    }
    cat(sprintf("  R Directory: %s\n", get_r_dir()))
    cat(sprintf("  Data Directory: %s\n", get_data_dir()))
    cat(sprintf("  Output Directory: %s\n", get_output_dir(create = FALSE)))
  } else {
    cat("Installation Mode: Installed (install.packages)\n")
    cat(sprintf("  Package Location: %s\n", get_package_root()))
    cat(sprintf("  Data Directory: %s (defaults to cwd/data)\n", get_data_dir()))
    cat(sprintf("  Output Directory: %s (defaults to cwd/outputs)\n", get_output_dir(create = FALSE)))
    cat("\nNote: For installed mode, specify data/output paths in your code:\n")
    cat("  get_data_dir(base_path = '/path/to/your/data')\n")
    cat("  get_output_dir(base_path = '/path/to/your/output')\n")
  }
  
  # Path conversion example
  cat("\n", strrep("=", 70), "\n")
  cat("Path Conversion Examples:\n")
  test_path <- "data/raw/file.txt"
  cat(sprintf("  Original: %s\n", test_path))
  cat(sprintf("  Platform: %s\n", to_platform_path(test_path)))
  cat(sprintf("  Unix: %s\n", to_unix_path(test_path)))
  if (!is.null(project_root)) {
    cat(sprintf("  Normalized (abs): %s\n", normalize_path(test_path)))
  }
}
