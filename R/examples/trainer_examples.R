#' MDITRE Training Examples
#'
#' @description
#' Comprehensive examples demonstrating training MDITRE models with various
#' configurations, hyperparameters, and evaluation strategies.
#'
#' @examples
#' \dontrun{
#' # Load required libraries
#' library(torch)
#' library(phyloseq)
#' library(mditre)
#' 
#' # These examples assume you have phyloseq data loaded
#' # data(GlobalPatterns)  # Example phyloseq dataset
#' }


# ============================================================================
# Example 1: Basic Training Workflow
# ============================================================================

#' Basic training with default hyperparameters
basic_training_example <- function() {
  cat("Example 1: Basic Training Workflow\n")
  cat("===================================\n\n")
  
  # 1. Load and prepare data (assuming phyloseq object 'ps' exists)
  # ps <- GlobalPatterns  # Example data
  
  # Convert phyloseq to MDITRE format
  mditre_data <- phyloseq_to_mditre(
    ps_data = ps,
    subject_col = "Subject",
    time_col = "Time",
    label_col = "Disease",
    normalize = TRUE,
    log_transform = TRUE
  )
  
  # 2. Split into train/test
  split_data <- split_train_test(
    mditre_data,
    test_fraction = 0.2,
    stratified = TRUE,
    seed = 42
  )
  
  # 3. Create dataloaders
  train_loader <- create_dataloader(
    split_data$train,
    batch_size = 16,
    shuffle = TRUE
  )
  
  val_loader <- create_dataloader(
    split_data$test,
    batch_size = 16,
    shuffle = FALSE
  )
  
  # 4. Create model
  model <- mditre_model(
    num_rules = 5,
    num_otus = mditre_data$metadata$n_otus,
    num_time = mditre_data$metadata$n_timepoints,
    dist = mditre_data$phylo_dist
  )
  
  cat("Model created with:\n")
  cat(sprintf("  - %d OTUs\n", mditre_data$metadata$n_otus))
  cat(sprintf("  - %d timepoints\n", mditre_data$metadata$n_timepoints))
  cat(sprintf("  - %d rules\n", 5))
  cat("\n")
  
  # 5. Train model with default settings
  result <- train_mditre(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    epochs = 100,
    verbose = TRUE
  )
  
  # 6. Access results
  trained_model <- result$model
  history <- result$history
  
  cat("\nTraining completed!\n")
  cat(sprintf("Best epoch: %d\n", result$best_epoch))
  cat(sprintf("Best validation loss: %.4f\n", min(history$val_loss, na.rm = TRUE)))
  
  # 7. Plot training history
  plot(history$train_loss, type = "l", col = "blue",
       main = "Training History", xlab = "Epoch", ylab = "Loss",
       ylim = range(c(history$train_loss, history$val_loss), na.rm = TRUE))
  lines(history$val_loss, col = "red")
  legend("topright", legend = c("Train", "Validation"),
         col = c("blue", "red"), lty = 1)
  
  return(result)
}


# ============================================================================
# Example 2: Custom Hyperparameters
# ============================================================================

#' Training with custom learning rates and temperature schedule
custom_hyperparameters_example <- function() {
  cat("Example 2: Custom Hyperparameters\n")
  cat("==================================\n\n")
  
  # Prepare data (same as Example 1)
  mditre_data <- phyloseq_to_mditre(ps, "Subject", "Time", "Disease")
  split_data <- split_train_test(mditre_data, test_fraction = 0.2)
  train_loader <- create_dataloader(split_data$train, batch_size = 32)
  val_loader <- create_dataloader(split_data$test, batch_size = 32)
  
  # Create model
  model <- mditre_model(
    num_rules = 10,  # More rules for complex patterns
    num_otus = mditre_data$metadata$n_otus,
    num_time = mditre_data$metadata$n_timepoints,
    dist = mditre_data$phylo_dist
  )
  
  # Custom learning rates
  custom_lr <- list(
    kappa = 0.002,    # Higher learning rate for phylogenetic focus
    eta = 0.002,
    time = 0.02,      # Higher for temporal parameters
    mu = 0.02,
    thresh = 0.0005,  # Higher for thresholds
    slope = 0.0001,
    alpha = 0.01,     # Higher for detector selection
    beta = 0.01,      # Higher for rule selection
    fc = 0.002,
    bias = 0.002
  )
  
  # Custom temperature schedule (slower annealing)
  custom_temp <- list(
    k_bc_max = 15,     # Higher initial temperature
    k_bc_min = 0.5,    # Lower final temperature
    k_otu_max = 2000,
    k_otu_min = 50,
    k_time_max = 20,
    k_time_min = 0.5,
    k_thresh_max = 2000,
    k_thresh_min = 50,
    k_slope_max = 20000,
    k_slope_min = 500
  )
  
  # Custom priors (more sparsity)
  custom_priors <- list(
    z_mean = 0.5,      # Expect fewer active detectors
    z_var = 3,         # Less variance
    z_r_mean = 0.5,    # Expect fewer active rules
    z_r_var = 3,
    w_var = 1e6        # Weaker weight prior
  )
  
  cat("Training with custom hyperparameters:\n")
  cat("  - Learning rates: 2x higher\n")
  cat("  - Temperature schedule: slower annealing\n")
  cat("  - Priors: more sparsity (fewer active components)\n\n")
  
  # Train with custom settings
  result <- train_mditre(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    epochs = 300,
    learning_rates = custom_lr,
    temperature_schedule = custom_temp,
    priors = custom_priors,
    verbose = TRUE,
    log_every = 20
  )
  
  cat(sprintf("\nFinal validation F1: %.4f\n", result$history$val_f1[result$best_epoch]))
  
  return(result)
}


# ============================================================================
# Example 3: Model Checkpointing
# ============================================================================

#' Training with model checkpointing
checkpointing_example <- function() {
  cat("Example 3: Model Checkpointing\n")
  cat("==============================\n\n")
  
  # Prepare data
  mditre_data <- phyloseq_to_mditre(ps, "Subject", "Time", "Disease")
  split_data <- split_train_test(mditre_data, test_fraction = 0.2)
  train_loader <- create_dataloader(split_data$train, batch_size = 16)
  val_loader <- create_dataloader(split_data$test, batch_size = 16)
  
  # Create model
  model <- mditre_model(
    num_rules = 5,
    num_otus = mditre_data$metadata$n_otus,
    num_time = mditre_data$metadata$n_timepoints,
    dist = mditre_data$phylo_dist
  )
  
  # Create checkpoint directory
  checkpoint_dir <- "checkpoints/experiment_1"
  
  cat(sprintf("Checkpoints will be saved to: %s\n", checkpoint_dir))
  cat("Saving every 25 epochs\n\n")
  
  # Train with checkpointing
  result <- train_mditre(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    epochs = 200,
    checkpoint_dir = checkpoint_dir,
    checkpoint_every = 25,
    verbose = TRUE
  )
  
  cat("\nCheckpoints saved:\n")
  checkpoints <- list.files(checkpoint_dir, pattern = "\\.pt$", full.names = TRUE)
  for (ckpt in checkpoints) {
    cat(sprintf("  - %s\n", basename(ckpt)))
  }
  
  # Load a specific checkpoint
  cat("\nLoading checkpoint from epoch 100...\n")
  model_epoch_100 <- mditre_model(
    num_rules = 5,
    num_otus = mditre_data$metadata$n_otus,
    num_time = mditre_data$metadata$n_timepoints,
    dist = mditre_data$phylo_dist
  )
  model_epoch_100 <- load_checkpoint(
    model_epoch_100,
    file.path(checkpoint_dir, "model_epoch_100.pt")
  )
  
  return(list(result = result, model_epoch_100 = model_epoch_100))
}


# ============================================================================
# Example 4: Early Stopping
# ============================================================================

#' Training with early stopping
early_stopping_example <- function() {
  cat("Example 4: Early Stopping\n")
  cat("=========================\n\n")
  
  # Prepare data
  mditre_data <- phyloseq_to_mditre(ps, "Subject", "Time", "Disease")
  split_data <- split_train_test(mditre_data, test_fraction = 0.2, seed = 123)
  train_loader <- create_dataloader(split_data$train, batch_size = 16)
  val_loader <- create_dataloader(split_data$test, batch_size = 16)
  
  # Create model
  model <- mditre_model(
    num_rules = 5,
    num_otus = mditre_data$metadata$n_otus,
    num_time = mditre_data$metadata$n_timepoints,
    dist = mditre_data$phylo_dist
  )
  
  cat("Training with early stopping (patience = 30 epochs)\n")
  cat("Will stop if validation loss doesn't improve for 30 epochs\n\n")
  
  # Train with early stopping
  result <- train_mditre(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    epochs = 500,  # Maximum epochs
    early_stopping_patience = 30,
    verbose = TRUE
  )
  
  actual_epochs <- sum(!is.na(result$history$train_loss))
  
  cat(sprintf("\nTraining stopped at epoch %d (out of 500 max)\n", actual_epochs))
  cat(sprintf("Best model from epoch %d\n", result$best_epoch))
  
  return(result)
}


# ============================================================================
# Example 5: Abundance-Only Model Training
# ============================================================================

#' Training MDITREAbun (abundance-only model)
abundance_only_training_example <- function() {
  cat("Example 5: Abundance-Only Model Training\n")
  cat("=========================================\n\n")
  
  # Prepare data
  mditre_data <- phyloseq_to_mditre(ps, "Subject", "Time", "Disease")
  split_data <- split_train_test(mditre_data, test_fraction = 0.2)
  train_loader <- create_dataloader(split_data$train, batch_size = 16)
  val_loader <- create_dataloader(split_data$test, batch_size = 16)
  
  # Create abundance-only model (no slope detectors)
  model <- mditre_abun_model(
    num_rules = 5,
    num_otus = mditre_data$metadata$n_otus,
    num_time = mditre_data$metadata$n_timepoints,
    dist = mditre_data$phylo_dist
  )
  
  cat("Training MDITREAbun (abundance-only, no slope detectors)\n\n")
  
  # Train (learning rates for slope parameters will be ignored)
  result <- train_mditre(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    epochs = 200,
    verbose = TRUE
  )
  
  cat("\nMDITREAbun training completed!\n")
  cat("This model uses only abundance thresholds (no temporal slopes)\n")
  
  return(result)
}


# ============================================================================
# Example 6: GPU Training
# ============================================================================

#' Training on GPU (if available)
gpu_training_example <- function() {
  cat("Example 6: GPU Training\n")
  cat("=======================\n\n")
  
  # Check GPU availability
  if (!cuda_is_available()) {
    cat("CUDA not available. Falling back to CPU training.\n")
    device <- torch_device("cpu")
  } else {
    cat("CUDA available! Training on GPU.\n")
    device <- torch_device("cuda")
  }
  
  # Prepare data
  mditre_data <- phyloseq_to_mditre(ps, "Subject", "Time", "Disease")
  split_data <- split_train_test(mditre_data, test_fraction = 0.2)
  train_loader <- create_dataloader(split_data$train, batch_size = 32)
  val_loader <- create_dataloader(split_data$test, batch_size = 32)
  
  # Create model
  model <- mditre_model(
    num_rules = 10,
    num_otus = mditre_data$metadata$n_otus,
    num_time = mditre_data$metadata$n_timepoints,
    dist = mditre_data$phylo_dist
  )
  
  # Train on specified device
  result <- train_mditre(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    epochs = 300,
    device = device,
    verbose = TRUE
  )
  
  cat(sprintf("\nTraining completed on: %s\n", device$type))
  
  return(result)
}


# ============================================================================
# Example 7: Training History Analysis
# ============================================================================

#' Detailed training history analysis and plotting
training_history_analysis <- function() {
  cat("Example 7: Training History Analysis\n")
  cat("=====================================\n\n")
  
  # Train model (reuse Example 1 setup)
  mditre_data <- phyloseq_to_mditre(ps, "Subject", "Time", "Disease")
  split_data <- split_train_test(mditre_data, test_fraction = 0.2)
  train_loader <- create_dataloader(split_data$train, batch_size = 16)
  val_loader <- create_dataloader(split_data$test, batch_size = 16)
  
  model <- mditre_model(
    num_rules = 5,
    num_otus = mditre_data$metadata$n_otus,
    num_time = mditre_data$metadata$n_timepoints,
    dist = mditre_data$phylo_dist
  )
  
  result <- train_mditre(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    epochs = 200,
    verbose = FALSE  # Suppress training logs for cleaner output
  )
  
  history <- result$history
  
  # Create comprehensive plots
  par(mfrow = c(2, 2))
  
  # Plot 1: Total Loss
  plot(history$train_loss, type = "l", col = "blue",
       main = "Total Loss", xlab = "Epoch", ylab = "Loss",
       ylim = range(c(history$train_loss, history$val_loss), na.rm = TRUE))
  lines(history$val_loss, col = "red")
  legend("topright", legend = c("Train", "Val"), col = c("blue", "red"), lty = 1)
  abline(v = result$best_epoch, col = "green", lty = 2)
  
  # Plot 2: Cross-Entropy Loss
  plot(history$train_ce_loss, type = "l", col = "blue",
       main = "Cross-Entropy Loss", xlab = "Epoch", ylab = "CE Loss",
       ylim = range(c(history$train_ce_loss, history$val_ce_loss), na.rm = TRUE))
  lines(history$val_ce_loss, col = "red")
  legend("topright", legend = c("Train", "Val"), col = c("blue", "red"), lty = 1)
  
  # Plot 3: F1 Score
  plot(history$train_f1, type = "l", col = "blue",
       main = "F1 Score", xlab = "Epoch", ylab = "F1",
       ylim = range(c(history$train_f1, history$val_f1), na.rm = TRUE))
  lines(history$val_f1, col = "red")
  legend("bottomright", legend = c("Train", "Val"), col = c("blue", "red"), lty = 1)
  abline(v = result$best_epoch, col = "green", lty = 2)
  
  # Plot 4: Overfitting Detection
  gap <- history$train_loss - history$val_loss
  plot(gap, type = "l", col = "purple",
       main = "Train-Val Gap (Overfitting Indicator)",
       xlab = "Epoch", ylab = "Loss Gap")
  abline(h = 0, col = "gray", lty = 2)
  
  par(mfrow = c(1, 1))
  
  # Print summary statistics
  cat("\nTraining Summary:\n")
  cat(sprintf("  Best epoch: %d\n", result$best_epoch))
  cat(sprintf("  Best val loss: %.4f\n", history$val_loss[result$best_epoch]))
  cat(sprintf("  Best val F1: %.4f\n", history$val_f1[result$best_epoch]))
  cat(sprintf("  Final train loss: %.4f\n", tail(na.omit(history$train_loss), 1)))
  cat(sprintf("  Final train F1: %.4f\n", tail(na.omit(history$train_f1), 1)))
  
  return(result)
}


# ============================================================================
# Example 8: Resuming Training from Checkpoint
# ============================================================================

#' Resume training from a saved checkpoint
resume_training_example <- function() {
  cat("Example 8: Resume Training from Checkpoint\n")
  cat("===========================================\n\n")
  
  # Initial training
  cat("Phase 1: Initial training (100 epochs)\n")
  mditre_data <- phyloseq_to_mditre(ps, "Subject", "Time", "Disease")
  split_data <- split_train_test(mditre_data, test_fraction = 0.2, seed = 42)
  train_loader <- create_dataloader(split_data$train, batch_size = 16)
  val_loader <- create_dataloader(split_data$test, batch_size = 16)
  
  model <- mditre_model(
    num_rules = 5,
    num_otus = mditre_data$metadata$n_otus,
    num_time = mditre_data$metadata$n_timepoints,
    dist = mditre_data$phylo_dist
  )
  
  result1 <- train_mditre(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    epochs = 100,
    checkpoint_dir = "checkpoints/resume_example",
    checkpoint_every = 50,
    verbose = TRUE
  )
  
  # Save final model
  save_checkpoint(result1$model, "checkpoints/resume_example/model_final.pt")
  
  cat("\n" + paste(rep("=", 50), collapse = "") + "\n\n")
  
  # Resume training
  cat("Phase 2: Resuming training (additional 100 epochs)\n")
  
  # Load the model
  model_resumed <- mditre_model(
    num_rules = 5,
    num_otus = mditre_data$metadata$n_otus,
    num_time = mditre_data$metadata$n_timepoints,
    dist = mditre_data$phylo_dist
  )
  model_resumed <- load_checkpoint(model_resumed, "checkpoints/resume_example/model_final.pt")
  
  # Continue training
  result2 <- train_mditre(
    model = model_resumed,
    train_loader = train_loader,
    val_loader = val_loader,
    epochs = 100,
    verbose = TRUE
  )
  
  cat("\nTraining resumed and completed!\n")
  cat(sprintf("Total effective epochs: 200\n"))
  
  return(list(phase1 = result1, phase2 = result2))
}


# ============================================================================
# Run All Examples
# ============================================================================

#' Run all training examples (for demonstration)
run_all_training_examples <- function() {
  cat("\n")
  cat(paste(rep("=", 70), collapse = ""))
  cat("\nMDITRE Training Examples\n")
  cat(paste(rep("=", 70), collapse = ""))
  cat("\n\n")
  
  cat("These examples demonstrate various training scenarios.\n")
  cat("Make sure you have a phyloseq object loaded as 'ps'.\n\n")
  
  cat("Available examples:\n")
  cat("  1. basic_training_example()\n")
  cat("  2. custom_hyperparameters_example()\n")
  cat("  3. checkpointing_example()\n")
  cat("  4. early_stopping_example()\n")
  cat("  5. abundance_only_training_example()\n")
  cat("  6. gpu_training_example()\n")
  cat("  7. training_history_analysis()\n")
  cat("  8. resume_training_example()\n")
  cat("\n")
  
  cat("To run an example: result <- basic_training_example()\n\n")
}

# Print examples summary
run_all_training_examples()
