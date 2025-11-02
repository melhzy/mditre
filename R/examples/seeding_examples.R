# MDITRE R Seeding Examples
# Using seedhash for deterministic seed generation

library(mditre)
library(torch)

# ==============================================================================
# Example 1: Basic Seed Generation
# ==============================================================================

cat("\n=== Example 1: Basic Seed Generation ===\n")

# Create a seed generator with default MDITRE master seed
seed_gen <- mditre_seed_generator()

# Generate a single master seed
master_seed <- seed_gen$generate_seeds(1)[1]
cat(sprintf("Master seed: %d\n", master_seed))

# Get the MD5 hash (useful for reproducibility tracking)
hash_value <- seed_gen$get_hash()
cat(sprintf("MD5 hash: %s\n", hash_value))

# Print generator info
print(seed_gen)


# ==============================================================================
# Example 2: Setting All Random Seeds
# ==============================================================================

cat("\n\n=== Example 2: Setting All Random Seeds ===\n")

# Set all RNGs (R base, torch, CUDA if available)
set_mditre_seeds(12345)

# Test reproducibility
x1 <- rnorm(5)
cat("Random numbers (run 1):", x1, "\n")

# Reset with same seed
set_mditre_seeds(12345, verbose = FALSE)
x2 <- rnorm(5)
cat("Random numbers (run 2):", x2, "\n")

# Verify identical
cat(sprintf("Identical: %s\n", identical(x1, x2)))


# ==============================================================================
# Example 3: Custom Experiment Names
# ==============================================================================

cat("\n\n=== Example 3: Custom Experiment Names ===\n")

# Create generators for different experiments
experiments <- c("exp_A", "exp_B", "exp_C")

for (exp_name in experiments) {
  gen <- mditre_seed_generator(experiment_name = exp_name)
  seeds <- gen$generate_seeds(3)
  cat(sprintf("Experiment %s: %s\n", 
              exp_name, 
              paste(seeds, collapse = ", ")))
}


# ==============================================================================
# Example 4: Multiple Seeds for Different Components
# ==============================================================================

cat("\n\n=== Example 4: Multiple Seeds for Different Components ===\n")

# Generate multiple seeds for different tasks
seed_gen <- mditre_seed_generator(experiment_name = "my_analysis")
seeds <- seed_gen$generate_seeds(5)

# Assign seeds to different components
seed_data_split <- seeds[1]
seed_model_init <- seeds[2]
seed_training <- seeds[3]
seed_validation <- seeds[4]
seed_evaluation <- seeds[5]

cat(sprintf("Data split seed: %d\n", seed_data_split))
cat(sprintf("Model init seed: %d\n", seed_model_init))
cat(sprintf("Training seed: %d\n", seed_training))
cat(sprintf("Validation seed: %d\n", seed_validation))
cat(sprintf("Evaluation seed: %d\n", seed_evaluation))


# ==============================================================================
# Example 5: Using Default Seeds
# ==============================================================================

cat("\n\n=== Example 5: Using Default Seeds ===\n")

# Get default seeds for common tasks
default_seeds <- get_default_mditre_seeds("my_experiment")

cat("Default seeds:\n")
for (name in names(default_seeds)) {
  cat(sprintf("  %s: %d\n", name, default_seeds[[name]]))
}

# Use for specific tasks
set_mditre_seeds(default_seeds$data_split, verbose = FALSE)
cat("\nSeeds set for data splitting\n")

set_mditre_seeds(default_seeds$model_init, verbose = FALSE)
cat("Seeds set for model initialization\n")


# ==============================================================================
# Example 6: Seed Function Generator
# ==============================================================================

cat("\n\n=== Example 6: Seed Function Generator ===\n")

# Create a function that generates deterministic seeds
get_seed <- get_mditre_seed_generator(base_seed = 42)

# Generate seeds on demand
seed1 <- get_seed()
seed2 <- get_seed()
seed3 <- get_seed()

cat(sprintf("Seed 1: %d\n", seed1))
cat(sprintf("Seed 2: %d\n", seed2))
cat(sprintf("Seed 3: %d\n", seed3))

# Verify reproducibility: same base seed produces same sequence
get_seed2 <- get_mditre_seed_generator(base_seed = 42)
seed1_again <- get_seed2()

cat(sprintf("\nReproducibility check: %s\n", identical(seed1, seed1_again)))


# ==============================================================================
# Example 7: Integration with MDITRE Workflow
# ==============================================================================

cat("\n\n=== Example 7: Integration with MDITRE Workflow ===\n")

# Complete workflow example
workflow_seeds <- function(experiment_name) {
  # Create seed generator
  gen <- mditre_seed_generator(experiment_name = experiment_name)
  seeds <- gen$generate_seeds(5)
  
  # Set master seed
  cat(sprintf("\n--- Experiment: %s ---\n", experiment_name))
  set_mditre_seeds(seeds[1], verbose = FALSE)
  cat(sprintf("Master seed: %d\n", seeds[1]))
  
  # Simulate different stages
  cat("\n1. Data splitting:\n")
  set_mditre_seeds(seeds[2], verbose = FALSE)
  train_indices <- sample(1:100, 70)
  cat(sprintf("   First 5 train indices: %s\n", 
              paste(head(train_indices, 5), collapse = ", ")))
  
  cat("\n2. Model initialization:\n")
  set_mditre_seeds(seeds[3], verbose = FALSE)
  # Simulate parameter initialization
  init_params <- rnorm(3)
  cat(sprintf("   Initial parameters: %s\n", 
              paste(round(init_params, 4), collapse = ", ")))
  
  cat("\n3. Training:\n")
  set_mditre_seeds(seeds[4], verbose = FALSE)
  cat("   Training with seed:", seeds[4], "\n")
  
  # Return hash for reproducibility tracking
  invisible(gen$get_hash())
}

# Run workflow
hash1 <- workflow_seeds("production_run_v1")
cat(sprintf("\nReproducibility hash: %s\n", hash1))


# ==============================================================================
# Example 8: Cross-Validation with Reproducible Folds
# ==============================================================================

cat("\n\n=== Example 8: Cross-Validation with Reproducible Folds ===\n")

# Create reproducible CV folds
create_cv_folds <- function(n_samples, n_folds, experiment_name) {
  # Generate seeds for each fold
  gen <- mditre_seed_generator(
    experiment_name = paste0(experiment_name, "_cv")
  )
  fold_seeds <- gen$generate_seeds(n_folds)
  
  cat(sprintf("Creating %d-fold CV for %d samples\n", n_folds, n_samples))
  
  folds <- list()
  for (i in 1:n_folds) {
    set_mditre_seeds(fold_seeds[i], verbose = FALSE)
    fold_size <- floor(n_samples / n_folds)
    test_idx <- sample(1:n_samples, fold_size)
    folds[[i]] <- list(
      test = test_idx,
      train = setdiff(1:n_samples, test_idx),
      seed = fold_seeds[i]
    )
    cat(sprintf("  Fold %d: %d test samples (seed: %d)\n", 
                i, length(test_idx), fold_seeds[i]))
  }
  
  return(folds)
}

# Create folds
folds <- create_cv_folds(
  n_samples = 100, 
  n_folds = 5, 
  experiment_name = "my_cv_experiment"
)


# ==============================================================================
# Example 9: Comparing Python and R Seeds
# ==============================================================================

cat("\n\n=== Example 9: Comparing Python and R Seeds ===\n")

cat("Note: When using the same experiment name in Python and R,\n")
cat("the MD5 hashes will be identical, ensuring reproducibility\n")
cat("across both implementations.\n\n")

# R implementation
r_gen <- mditre_seed_generator(experiment_name = "shared_experiment")
r_hash <- r_gen$get_hash()
r_seeds <- r_gen$generate_seeds(3)

cat(sprintf("R Implementation:\n"))
cat(sprintf("  Hash: %s\n", r_hash))
cat(sprintf("  Seeds: %s\n", paste(r_seeds, collapse = ", ")))

cat("\nPython equivalent:\n")
cat("  from mditre.seeding import MDITRESeedGenerator\n")
cat("  gen = MDITRESeedGenerator(experiment_name='shared_experiment')\n")
cat("  hash = gen.get_hash()\n")
cat("  seeds = gen.generate_seeds(3)\n")
cat("\nThe hash values will match, enabling cross-language reproducibility!\n")


cat("\n\n=== All Examples Complete ===\n")
