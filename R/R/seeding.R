#' MDITRE Seeding Utilities
#'
#' @description
#' Deterministic seed generation using the seedhash library.
#' This ensures reproducibility across different runs and experiments.
#'
#' The seeding process works as follows:
#' 1. A seed_string (e.g., MDITRE master seed + experiment name) is hashed using MD5
#' 2. The MD5 hash is converted to a seed_number (master_seed) - a deterministic integer
#' 3. This master_seed is used to generate additional repeatable random seeds
#' 4. Seeds can be used to initialize random number generators (R base, torch)
#'
#' @name seeding
#' @examples
#' \dontrun{
#' library(mditre)
#' 
#' # Create generator with default MDITRE master seed
#' seed_gen <- mditre_seed_generator()
#' 
#' # Get the master_seed (first seed from hash)
#' master_seed <- seed_gen$generate_seeds(1)[1]
#' print(paste("Master seed:", master_seed))
#' 
#' # Use master_seed to set all RNGs
#' set_mditre_seeds(master_seed)
#' 
#' # Or generate multiple seeds for different components
#' seeds <- seed_gen$generate_seeds(5)
#' # seeds[1] for data splitting, seeds[2] for model init, etc.
#' 
#' # Create generator with custom experiment name
#' seed_gen_exp <- mditre_seed_generator(experiment_name = "experiment_v1")
#' exp_seeds <- seed_gen_exp$generate_seeds(3)
#' 
#' # Get the hash for reproducibility tracking
#' print(seed_gen_exp$get_hash())
#' }
NULL

# Master seed string for MDITRE project
# This string is hashed (MD5) to produce a deterministic seed_number
MDITRE_MASTER_SEED <- "MDITRE: Scalable and Interpretable Machine Learning for Predicting Host Status from Temporal Microbiome Dynamics"


#' MDITRE Seed Generator
#'
#' @description
#' MDITRE seed generator using deterministic hashing via the seedhash R package.
#' This provides MDITRE-specific seeding functionality with a standard master seed string.
#'
#' The seed generation process:
#' 1. seed_string (master seed + optional experiment name) → MD5 hash
#' 2. MD5 hash → seed_number (master_seed) - a deterministic integer
#' 3. seed_number → sequence of random seeds via internal PRNG
#'
#' All seeds are deterministic and reproducible given the same seed_string.
#'
#' @param experiment_name Character string; optional experiment identifier to append to master seed.
#'   If NULL, uses only the master seed.
#' @param min_value Integer; minimum value for random seed range (default: 0)
#' @param max_value Integer; maximum value for random seed range (default: 2^31-1)
#'
#' @return An R6 object with methods:
#'   \describe{
#'     \item{generate_seeds(count)}{Generate a list of deterministic random seeds}
#'     \item{get_hash()}{Get the MD5 hash of the seed string}
#'   }
#'
#' @export
#' @examples
#' \dontrun{
#' # Create generator with default MDITRE master seed
#' seed_gen <- mditre_seed_generator()
#' 
#' # Get the master_seed (first seed from hash)
#' master_seed <- seed_gen$generate_seeds(1)[1]
#' print(paste("Master seed:", master_seed))
#' 
#' # Use master_seed to set all RNGs
#' set_mditre_seeds(master_seed)
#' 
#' # Generate multiple seeds for different components
#' seeds <- seed_gen$generate_seeds(5)
#' 
#' # Create generator with custom experiment name
#' seed_gen_exp <- mditre_seed_generator(experiment_name = "experiment_v1")
#' exp_seeds <- seed_gen_exp$generate_seeds(3)
#' }
mditre_seed_generator <- function(
  experiment_name = NULL,
  min_value = 0,
  max_value = 2147483646  # R integer max minus 1 to avoid range limit
) {
  # Check if seedhash is installed
  if (!requireNamespace("seedhash", quietly = TRUE)) {
    stop(
      "The 'seedhash' package is required for MDITRE seeding.\n",
      "Install it with:\n",
      "  devtools::install_github('melhzy/seedhash', subdir = 'R')\n",
      "Or:\n",
      "  remotes::install_github('melhzy/seedhash', subdir = 'R')"
    )
  }
  
  # Validate inputs
  if (!is.null(experiment_name)) {
    if (!is.character(experiment_name) || length(experiment_name) != 1) {
      stop("experiment_name must be a single character string or NULL")
    }
    if (nchar(trimws(experiment_name)) == 0) {
      stop("experiment_name cannot be empty")
    }
  }
  
  if (!is.numeric(min_value) || !is.numeric(max_value)) {
    stop("min_value and max_value must be numeric")
  }
  
  if (min_value >= max_value) {
    stop(sprintf("min_value (%d) must be less than max_value (%d)", min_value, max_value))
  }
  
  # Construct the full seed string
  if (!is.null(experiment_name)) {
    seed_string <- paste0(MDITRE_MASTER_SEED, "::", experiment_name)
  } else {
    seed_string <- MDITRE_MASTER_SEED
  }
  
  # Initialize the seedhash generator
  generator <- seedhash::SeedHashGenerator$new(
    input_string = seed_string,
    min_value = min_value,
    max_value = max_value
  )
  
  # Return the generator with additional metadata
  structure(
    list(
      generator = generator,
      master_seed = MDITRE_MASTER_SEED,
      seed_string = seed_string,
      experiment_name = experiment_name,
      generate_seeds = function(count) {
        generator$generate_seeds(count)
      },
      get_hash = function() {
        generator$get_hash()
      }
    ),
    class = "mditre_seed_generator"
  )
}


#' Set All Random Seeds for MDITRE
#'
#' @description
#' Set all random seeds for reproducibility across R base random functions,
#' torch, and other packages. This ensures deterministic behavior for
#' MDITRE model training and evaluation.
#'
#' @param seed Integer; seed value to set (typically obtained from mditre_seed_generator)
#' @param verbose Logical; whether to print confirmation message (default: TRUE)
#'
#' @details
#' This function sets:
#' - R base random seed (`set.seed()`)
#' - torch manual seed (if torch is loaded)
#' - torch CUDA seed (if CUDA is available)
#' - torch deterministic operations (cudnn)
#'
#' @export
#' @examples
#' \dontrun{
#' # Set seeds using a fixed integer
#' set_mditre_seeds(42)
#' 
#' # Set seeds from generator
#' seed_gen <- mditre_seed_generator()
#' master_seed <- seed_gen$generate_seeds(1)[1]
#' set_mditre_seeds(master_seed)
#' 
#' # Verify reproducibility
#' set_mditre_seeds(123)
#' x1 <- rnorm(10)
#' set_mditre_seeds(123)
#' x2 <- rnorm(10)
#' all.equal(x1, x2)  # TRUE
#' }
set_mditre_seeds <- function(seed, verbose = TRUE) {
  # Validate seed
  if (!is.numeric(seed) || length(seed) != 1) {
    stop("seed must be a single numeric value")
  }
  
  seed <- as.integer(seed)
  
  # Set R base random seed
  set.seed(seed)
  
  # Set torch random seed if available
  if (requireNamespace("torch", quietly = TRUE)) {
    torch::torch_manual_seed(seed)
    
    # Set CUDA seed if available
    if (torch::cuda_is_available()) {
      torch::cuda_manual_seed_all(seed)
    }
    
    # Make torch operations deterministic
    # Note: This may impact performance but ensures reproducibility
    torch::torch_set_deterministic(TRUE)
  }
  
  if (verbose) {
    message(sprintf("All random seeds set to: %d", seed))
    if (requireNamespace("torch", quietly = TRUE)) {
      message("  - R base random seed: set")
      message("  - torch manual seed: set")
      if (torch::cuda_is_available()) {
        message("  - torch CUDA seed: set")
      }
      message("  - torch deterministic mode: enabled")
    } else {
      message("  - R base random seed: set")
      message("  - torch: not loaded (will be set when torch is loaded)")
    }
  }
  
  invisible(seed)
}


#' Get MDITRE Seed Generator Function
#'
#' @description
#' Create a function that generates deterministic seeds from a base seed.
#' This is useful for generating multiple related but distinct seeds
#' for different components of an experiment.
#'
#' @param base_seed Integer; base seed to initialize the generator
#'
#' @return A function that returns a new deterministic seed each time it's called
#'
#' @export
#' @examples
#' \dontrun{
#' # Create a seed generator function
#' get_seed <- get_mditre_seed_generator(42)
#' 
#' # Generate seeds for different components
#' seed_data_split <- get_seed()    # 123456789
#' seed_model_init <- get_seed()    # 987654321
#' seed_training <- get_seed()      # 555555555
#' 
#' # Same base seed always produces same sequence
#' get_seed2 <- get_mditre_seed_generator(42)
#' identical(get_seed2(), seed_data_split)  # TRUE
#' }
get_mditre_seed_generator <- function(base_seed) {
  if (!is.numeric(base_seed) || length(base_seed) != 1) {
    stop("base_seed must be a single numeric value")
  }
  
  base_seed <- as.integer(base_seed)
  counter <- 0
  
  function() {
    counter <<- counter + 1
    
    # Use digest to create deterministic seed from base_seed + counter
    if (!requireNamespace("digest", quietly = TRUE)) {
      stop("The 'digest' package is required. Install it with: install.packages('digest')")
    }
    
    seed_string <- paste0(base_seed, "_", counter)
    hash_value <- digest::digest(seed_string, algo = "md5")
    
    # Convert first 8 hex characters to integer
    seed_int <- strtoi(substr(hash_value, 1, 8), base = 16L)
    
    # Ensure it's in valid range for R (32-bit integer)
    seed_int <- seed_int %% 2147483647L
    
    return(as.integer(seed_int))
  }
}


#' Print Method for MDITRE Seed Generator
#'
#' @param x An mditre_seed_generator object
#' @param ... Additional arguments (ignored)
#' @export
print.mditre_seed_generator <- function(x, ...) {
  cat("MDITRE Seed Generator\n")
  cat("=====================\n")
  cat(sprintf("Master Seed String: %s\n", substr(x$master_seed, 1, 50)))
  if (nchar(x$master_seed) > 50) {
    cat("                    ...\n")
  }
  
  if (!is.null(x$experiment_name)) {
    cat(sprintf("Experiment Name: %s\n", x$experiment_name))
  } else {
    cat("Experiment Name: <default>\n")
  }
  
  cat(sprintf("Full Seed String: %s\n", substr(x$seed_string, 1, 60)))
  if (nchar(x$seed_string) > 60) {
    cat("                  ...\n")
  }
  
  cat(sprintf("MD5 Hash: %s\n", x$get_hash()))
  cat("\nUsage:\n")
  cat("  seeds <- x$generate_seeds(count)\n")
  cat("  hash <- x$get_hash()\n")
  
  invisible(x)
}


#' Get Default MDITRE Seeds
#'
#' @description
#' Convenience function to get a set of default seeds for common MDITRE tasks.
#' Returns a named list of seeds for different purposes.
#'
#' @param experiment_name Character string; optional experiment identifier
#'
#' @return Named list with seeds for:
#'   \describe{
#'     \item{master}{Master seed for overall reproducibility}
#'     \item{data_split}{Seed for train/test/validation splitting}
#'     \item{model_init}{Seed for model parameter initialization}
#'     \item{training}{Seed for training process}
#'     \item{evaluation}{Seed for evaluation}
#'   }
#'
#' @export
#' @examples
#' \dontrun{
#' # Get default seeds
#' seeds <- get_default_mditre_seeds()
#' 
#' # Use for data splitting
#' set_mditre_seeds(seeds$data_split)
#' # ... split data ...
#' 
#' # Use for model initialization
#' set_mditre_seeds(seeds$model_init)
#' # ... initialize model ...
#' 
#' # Get seeds for specific experiment
#' exp_seeds <- get_default_mditre_seeds("experiment_v1")
#' }
get_default_mditre_seeds <- function(experiment_name = NULL) {
  seed_gen <- mditre_seed_generator(experiment_name = experiment_name)
  seeds <- seed_gen$generate_seeds(5)
  
  list(
    master = seeds[1],
    data_split = seeds[2],
    model_init = seeds[3],
    training = seeds[4],
    evaluation = seeds[5]
  )
}
