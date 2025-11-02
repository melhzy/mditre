#' MDITRE Training Infrastructure
#'
#' @description
#' Functions for training MDITRE models with customizable hyperparameters,
#' learning rate schedules, loss functions, and model checkpointing.
#'
#' @name trainer
#' @keywords internal
NULL

#' Train MDITRE Model
#'
#' @description
#' Main training function for MDITRE models. Implements the complete training
#' loop with optimizer configuration, learning rate scheduling, validation,
#' and model checkpointing.
#'
#' @param model MDITRE or MDITREAbun model created with mditre_model() or mditre_abun_model()
#' @param train_loader torch dataloader for training data (from create_dataloader())
#' @param val_loader torch dataloader for validation data (optional)
#' @param test_loader torch dataloader for test data (optional)
#' @param epochs Number of training epochs (default: 500)
#' @param learning_rates Named list of learning rates for each parameter group:
#'   \itemize{
#'     \item kappa: Learning rate for phylogenetic focus kappa (default: 0.001)
#'     \item eta: Learning rate for phylogenetic focus eta (default: 0.001)
#'     \item time: Learning rate for temporal focus sigma (default: 0.01)
#'     \item mu: Learning rate for temporal focus mu (default: 0.01)
#'     \item thresh: Learning rate for threshold detectors (default: 0.0001)
#'     \item slope: Learning rate for slope detectors (default: 0.00001)
#'     \item alpha: Learning rate for detector selection (default: 0.005)
#'     \item beta: Learning rate for rule selection (default: 0.005)
#'     \item fc: Learning rate for classifier weights (default: 0.001)
#'     \item bias: Learning rate for classifier bias (default: 0.001)
#'   }
#' @param temperature_schedule Named list with temperature annealing parameters:
#'   \itemize{
#'     \item k_bc_max: Max temperature for binary concrete (default: 10)
#'     \item k_bc_min: Min temperature for binary concrete (default: 1)
#'     \item k_otu_max: Max temperature for OTU selection (default: 1000)
#'     \item k_otu_min: Min temperature for OTU selection (default: 100)
#'     \item k_time_max: Max temperature for time window (default: 10)
#'     \item k_time_min: Min temperature for time window (default: 1)
#'     \item k_thresh_max: Max temperature for threshold (default: 1000)
#'     \item k_thresh_min: Min temperature for threshold (default: 100)
#'     \item k_slope_max: Max temperature for slope (default: 10000)
#'     \item k_slope_min: Min temperature for slope (default: 1000)
#'   }
#' @param priors Named list with prior parameters:
#'   \itemize{
#'     \item z_mean: Mean active detectors per rule (default: 1)
#'     \item z_var: Variance of active detectors per rule (default: 5)
#'     \item z_r_mean: Mean active rules (default: 1)
#'     \item z_r_var: Variance of active rules (default: 5)
#'     \item w_var: Normal prior variance on weights (default: 1e5)
#'   }
#' @param use_scheduler Whether to use cosine annealing learning rate scheduler (default: TRUE)
#' @param checkpoint_dir Directory to save model checkpoints (default: NULL, no saving)
#' @param checkpoint_every Save checkpoint every N epochs (default: 50)
#' @param early_stopping_patience Stop training if validation loss doesn't improve for N epochs (default: NULL, disabled)
#' @param device torch device ("cuda" or "cpu", default: auto-detect)
#' @param verbose Print training progress (default: TRUE)
#' @param log_every Print log every N epochs (default: 10)
#'
#' @return A list containing:
#'   \itemize{
#'     \item model: Trained model
#'     \item history: Training history (losses, metrics per epoch)
#'     \item best_model_state: State dict of best model (by validation loss)
#'     \item best_epoch: Epoch with best validation performance
#'   }
#'
#' @examples
#' \dontrun{
#' # Load data
#' library(torch)
#' mditre_data <- phyloseq_to_mditre(ps_data, "Subject", "Time", "Disease")
#' split_data <- split_train_test(mditre_data, test_fraction = 0.2)
#' train_loader <- create_dataloader(split_data$train, batch_size = 16, shuffle = TRUE)
#' val_loader <- create_dataloader(split_data$test, batch_size = 16, shuffle = FALSE)
#'
#' # Create model
#' model <- mditre_model(
#'   num_rules = 5,
#'   num_otus = mditre_data$metadata$n_otus,
#'   num_time = mditre_data$metadata$n_timepoints,
#'   dist = mditre_data$phylo_dist
#' )
#'
#' # Train model
#' result <- train_mditre(
#'   model = model,
#'   train_loader = train_loader,
#'   val_loader = val_loader,
#'   epochs = 200,
#'   checkpoint_dir = "checkpoints/",
#'   early_stopping_patience = 20
#' )
#'
#' # Access results
#' trained_model <- result$model
#' history <- result$history
#' plot(history$train_loss, type = "l", main = "Training Loss")
#' }
#'
#' @export
train_mditre <- function(model,
                        train_loader,
                        val_loader = NULL,
                        test_loader = NULL,
                        epochs = 500,
                        learning_rates = list(),
                        temperature_schedule = list(),
                        priors = list(),
                        use_scheduler = TRUE,
                        checkpoint_dir = NULL,
                        checkpoint_every = 50,
                        early_stopping_patience = NULL,
                        device = NULL,
                        verbose = TRUE,
                        log_every = 10) {
  
  # Auto-detect device
  if (is.null(device)) {
    device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
  }
  
  # Move model to device
  model$to(device = device)
  
  # Set default learning rates
  lr <- modifyList(list(
    kappa = 0.001, eta = 0.001, time = 0.01, mu = 0.01,
    thresh = 0.0001, slope = 0.00001, alpha = 0.005,
    beta = 0.005, fc = 0.001, bias = 0.001
  ), learning_rates)
  
  # Set default temperature schedule
  temp <- modifyList(list(
    k_bc_max = 10, k_bc_min = 1,
    k_otu_max = 1000, k_otu_min = 100,
    k_time_max = 10, k_time_min = 1,
    k_thresh_max = 1000, k_thresh_min = 100,
    k_slope_max = 10000, k_slope_min = 1000
  ), temperature_schedule)
  
  # Set default priors
  prior <- modifyList(list(
    z_mean = 1, z_var = 5,
    z_r_mean = 1, z_r_var = 5,
    w_var = 1e5
  ), priors)
  
  # Check if model is abundance-only
  is_abun_only <- !("slope_detector" %in% names(model))
  
  # Configure optimizer
  if (is_abun_only) {
    # MDITREAbun: No slope parameters
    optimizer <- optim_rmsprop(list(
      list(params = list(model$phylo_focus$kappa), lr = lr$kappa),
      list(params = list(model$phylo_focus$eta), lr = lr$eta),
      list(params = list(model$temporal_focus$abun_a), lr = lr$time),
      list(params = list(model$temporal_focus$abun_b), lr = lr$mu),
      list(params = list(model$threshold_detector$thresh), lr = lr$thresh),
      list(params = list(model$rules$alpha), lr = lr$alpha),
      list(params = list(model$classifier$weight), lr = lr$fc),
      list(params = list(model$classifier$bias), lr = lr$bias),
      list(params = list(model$classifier$beta), lr = lr$beta)
    ))
  } else {
    # Full MDITRE: With slope parameters
    optimizer <- optim_rmsprop(list(
      list(params = list(model$phylo_focus$kappa), lr = lr$kappa),
      list(params = list(model$phylo_focus$eta), lr = lr$eta),
      list(params = list(model$temporal_focus$abun_a), lr = lr$time),
      list(params = list(model$temporal_focus$abun_b), lr = lr$mu),
      list(params = list(model$temporal_focus$slope_a), lr = lr$time),
      list(params = list(model$temporal_focus$slope_b), lr = lr$mu),
      list(params = list(model$threshold_detector$thresh), lr = lr$thresh),
      list(params = list(model$slope_detector$slope), lr = lr$slope),
      list(params = list(model$rules$alpha), lr = lr$alpha),
      list(params = list(model$rules_slope$alpha), lr = lr$alpha),
      list(params = list(model$classifier$weight), lr = lr$fc),
      list(params = list(model$classifier$bias), lr = lr$bias),
      list(params = list(model$classifier$beta), lr = lr$beta)
    ))
  }
  
  # Configure learning rate scheduler
  if (use_scheduler) {
    scheduler <- lr_cosine_annealing(optimizer, T_max = epochs)
  }
  
  # Create checkpoint directory if needed
  if (!is.null(checkpoint_dir) && !dir.exists(checkpoint_dir)) {
    dir.create(checkpoint_dir, recursive = TRUE)
  }
  
  # Initialize tracking variables
  history <- list(
    train_loss = numeric(epochs),
    train_ce_loss = numeric(epochs),
    train_f1 = numeric(epochs),
    val_loss = numeric(epochs),
    val_ce_loss = numeric(epochs),
    val_f1 = numeric(epochs),
    test_loss = numeric(epochs),
    test_ce_loss = numeric(epochs),
    test_f1 = numeric(epochs)
  )
  
  best_val_loss <- Inf
  best_epoch <- 0
  best_model_state <- NULL
  epochs_without_improvement <- 0
  
  # Training loop
  for (epoch in 1:epochs) {
    
    # Compute annealed temperatures
    k_alpha <- linear_anneal(epoch, temp$k_bc_max, temp$k_bc_min, epochs)
    k_beta <- linear_anneal(epoch, temp$k_bc_max, temp$k_bc_min, epochs)
    k_otu <- linear_anneal(epoch, temp$k_otu_max, temp$k_otu_min, epochs)
    k_time <- linear_anneal(epoch, temp$k_time_max, temp$k_time_min, epochs)
    k_thresh <- linear_anneal(epoch, temp$k_thresh_max, temp$k_thresh_min, epochs)
    k_slope <- linear_anneal(epoch, temp$k_slope_max, temp$k_slope_min, epochs)
    
    # Training phase
    model$train()
    train_losses <- list()
    train_ce_losses <- list()
    train_preds <- list()
    train_labels <- list()
    
    # Iterate over training batches
    coro::loop(for (batch in train_loader) {
      # Move data to device
      data <- batch$data$to(device = device)
      labels <- batch$label$to(device = device)
      mask <- batch$mask$to(device = device)
      
      # Forward pass
      outputs <- model(
        data,
        mask = mask,
        k_alpha = k_alpha,
        k_beta = k_beta,
        k_otu = k_otu,
        k_time = k_time,
        k_thresh = k_thresh,
        k_slope = k_slope,
        use_noise = FALSE,
        hard = FALSE
      )
      
      # Compute cross-entropy loss
      ce_loss <- nnf_binary_cross_entropy_with_logits(outputs, labels$unsqueeze(2))
      
      # Compute priors/regularizers
      loss_components <- compute_loss_components(
        model = model,
        ce_loss = ce_loss,
        priors = prior,
        is_abun_only = is_abun_only
      )
      
      total_loss <- loss_components$total_loss
      
      # Backward pass
      optimizer$zero_grad()
      total_loss$backward()
      optimizer$step()
      
      # Store metrics
      train_losses[[length(train_losses) + 1]] <- total_loss$item()
      train_ce_losses[[length(train_ce_losses) + 1]] <- ce_loss$item()
      train_preds[[length(train_preds) + 1]] <- torch_sigmoid(outputs)$cpu()
      train_labels[[length(train_labels) + 1]] <- labels$cpu()
    })
    
    # Average training metrics
    history$train_loss[epoch] <- mean(unlist(train_losses))
    history$train_ce_loss[epoch] <- mean(unlist(train_ce_losses))
    
    # Compute F1 score
    all_preds <- torch_cat(train_preds, dim = 1)
    all_labels <- torch_cat(train_labels, dim = 1)
    history$train_f1[epoch] <- compute_f1(all_preds, all_labels)
    
    # Validation phase
    if (!is.null(val_loader)) {
      val_metrics <- evaluate_model(
        model = model,
        data_loader = val_loader,
        device = device,
        k_alpha = k_alpha,
        k_beta = k_beta,
        k_otu = k_otu,
        k_time = k_time,
        k_thresh = k_thresh,
        k_slope = k_slope,
        priors = prior,
        is_abun_only = is_abun_only
      )
      
      history$val_loss[epoch] <- val_metrics$total_loss
      history$val_ce_loss[epoch] <- val_metrics$ce_loss
      history$val_f1[epoch] <- val_metrics$f1
      
      # Check for best model
      if (val_metrics$total_loss < best_val_loss) {
        best_val_loss <- val_metrics$total_loss
        best_epoch <- epoch
        best_model_state <- model$state_dict()
        epochs_without_improvement <- 0
      } else {
        epochs_without_improvement <- epochs_without_improvement + 1
      }
    }
    
    # Test phase (optional, for monitoring only)
    if (!is.null(test_loader)) {
      test_metrics <- evaluate_model(
        model = model,
        data_loader = test_loader,
        device = device,
        k_alpha = k_alpha,
        k_beta = k_beta,
        k_otu = k_otu,
        k_time = k_time,
        k_thresh = k_thresh,
        k_slope = k_slope,
        priors = prior,
        is_abun_only = is_abun_only
      )
      
      history$test_loss[epoch] <- test_metrics$total_loss
      history$test_ce_loss[epoch] <- test_metrics$ce_loss
      history$test_f1[epoch] <- test_metrics$f1
    }
    
    # Learning rate scheduling
    if (use_scheduler) {
      scheduler$step()
    }
    
    # Logging
    if (verbose && (epoch %% log_every == 0 || epoch == 1)) {
      log_msg <- sprintf(
        "Epoch %d/%d | Train Loss: %.4f, CE: %.4f, F1: %.4f",
        epoch, epochs, history$train_loss[epoch],
        history$train_ce_loss[epoch], history$train_f1[epoch]
      )
      
      if (!is.null(val_loader)) {
        log_msg <- paste0(
          log_msg,
          sprintf(" | Val Loss: %.4f, CE: %.4f, F1: %.4f",
                  history$val_loss[epoch], history$val_ce_loss[epoch],
                  history$val_f1[epoch])
        )
      }
      
      cat(log_msg, "\n")
    }
    
    # Checkpointing
    if (!is.null(checkpoint_dir) && epoch %% checkpoint_every == 0) {
      checkpoint_path <- file.path(checkpoint_dir, sprintf("model_epoch_%d.pt", epoch))
      torch_save(model$state_dict(), checkpoint_path)
      if (verbose) {
        cat(sprintf("Checkpoint saved: %s\n", checkpoint_path))
      }
    }
    
    # Early stopping
    if (!is.null(early_stopping_patience) && epochs_without_improvement >= early_stopping_patience) {
      if (verbose) {
        cat(sprintf("Early stopping at epoch %d (best epoch: %d)\n", epoch, best_epoch))
      }
      break
    }
  }
  
  # Load best model if validation was used
  if (!is.null(best_model_state)) {
    model$load_state_dict(best_model_state)
    if (verbose) {
      cat(sprintf("Loaded best model from epoch %d\n", best_epoch))
    }
  }
  
  # Return results
  return(list(
    model = model,
    history = history,
    best_model_state = best_model_state,
    best_epoch = best_epoch
  ))
}


#' Linear Annealing Schedule
#'
#' @description
#' Linearly anneal a value from max to min over a specified number of steps.
#'
#' @param step Current step
#' @param max_val Maximum value (at step 1)
#' @param min_val Minimum value (at final step)
#' @param total_steps Total number of steps
#'
#' @return Annealed value at current step
#'
#' @keywords internal
linear_anneal <- function(step, max_val, min_val, total_steps) {
  slope <- (min_val - max_val) / total_steps
  return(max(min_val, max_val + slope * step))
}


#' Compute Loss Components
#'
#' @description
#' Compute all loss components including cross-entropy, priors, and regularizers.
#'
#' @param model MDITRE model
#' @param ce_loss Cross-entropy loss
#' @param priors Named list of prior parameters
#' @param is_abun_only Whether model is abundance-only
#'
#' @return List with total_loss and individual components
#'
#' @keywords internal
compute_loss_components <- function(model, ce_loss, priors, is_abun_only) {
  
  # Extract selection probabilities
  rules <- model$classifier$z  # Rule selection probabilities
  detectors <- model$rules$z   # Detector selection probabilities (threshold/abundance)
  
  # Negative binomial priors on number of active components
  # Encourage sparsity: prefer fewer active rules and detectors
  negbin_zr_loss <- negbin_loss(rules$sum(), priors$z_r_mean, priors$z_r_var)
  
  if (is_abun_only) {
    # Abundance-only: only threshold detectors
    negbin_z_loss <- negbin_loss(
      detectors$sum(dim = -1),
      priors$z_mean,
      priors$z_var
    )$sum()
  } else {
    # Full model: threshold + slope detectors
    detectors_slope <- model$rules_slope$z
    negbin_z_loss <- negbin_loss(
      detectors$sum(dim = -1) + detectors_slope$sum(dim = -1),
      priors$z_mean,
      priors$z_var
    )$sum()
  }
  
  # L2 regularization on classifier weights
  l2_wts_loss <- (model$classifier$weight$pow(2)$sum()) / (2 * priors$w_var)
  
  # Total loss
  total_loss <- ce_loss + negbin_zr_loss + negbin_z_loss + l2_wts_loss
  
  return(list(
    total_loss = total_loss,
    ce_loss = ce_loss,
    negbin_zr_loss = negbin_zr_loss,
    negbin_z_loss = negbin_z_loss,
    l2_wts_loss = l2_wts_loss
  ))
}


#' Negative Binomial Loss
#'
#' @description
#' Compute negative log-likelihood of negative binomial distribution.
#' Used as a prior to encourage sparsity in the number of active components.
#'
#' @param x Observed count
#' @param mean Mean parameter
#' @param variance Variance parameter
#'
#' @return Negative log-likelihood
#'
#' @keywords internal
negbin_loss <- function(x, mean, variance) {
  # Negative binomial parameterization: mean and variance
  # r = mean^2 / (variance - mean)
  # p = mean / variance
  
  r <- mean^2 / (variance - mean)
  p <- mean / variance
  
  # Negative log-likelihood
  nll <- -torch_lgamma(x + r) + torch_lgamma(r) + torch_lgamma(x + 1) -
    r * torch_log(1 - p) - x * torch_log(p)
  
  return(nll)
}


#' Evaluate Model
#'
#' @description
#' Evaluate model on a dataset (validation or test).
#'
#' @param model MDITRE model
#' @param data_loader torch dataloader
#' @param device torch device
#' @param k_alpha Temperature for alpha binary concrete
#' @param k_beta Temperature for beta binary concrete
#' @param k_otu Temperature for OTU selection
#' @param k_time Temperature for time window
#' @param k_thresh Temperature for threshold
#' @param k_slope Temperature for slope
#' @param priors Named list of prior parameters
#' @param is_abun_only Whether model is abundance-only
#'
#' @return List with metrics (total_loss, ce_loss, f1)
#'
#' @keywords internal
evaluate_model <- function(model, data_loader, device,
                          k_alpha, k_beta, k_otu, k_time,
                          k_thresh, k_slope, priors, is_abun_only) {
  
  model$eval()
  
  losses <- list()
  ce_losses <- list()
  preds <- list()
  labels_list <- list()
  
  with_no_grad({
    coro::loop(for (batch in data_loader) {
      # Move data to device
      data <- batch$data$to(device = device)
      labels <- batch$label$to(device = device)
      mask <- batch$mask$to(device = device)
      
      # Forward pass
      outputs <- model(
        data,
        mask = mask,
        k_alpha = k_alpha,
        k_beta = k_beta,
        k_otu = k_otu,
        k_time = k_time,
        k_thresh = k_thresh,
        k_slope = k_slope,
        use_noise = FALSE,
        hard = TRUE  # Use hard selections for evaluation
      )
      
      # Compute loss
      ce_loss <- nnf_binary_cross_entropy_with_logits(outputs, labels$unsqueeze(2))
      loss_components <- compute_loss_components(model, ce_loss, priors, is_abun_only)
      
      # Store metrics
      losses[[length(losses) + 1]] <- loss_components$total_loss$item()
      ce_losses[[length(ce_losses) + 1]] <- ce_loss$item()
      preds[[length(preds) + 1]] <- torch_sigmoid(outputs)$cpu()
      labels_list[[length(labels_list) + 1]] <- labels$cpu()
    })
  })
  
  # Average metrics
  avg_loss <- mean(unlist(losses))
  avg_ce_loss <- mean(unlist(ce_losses))
  
  # Compute F1
  all_preds <- torch_cat(preds, dim = 1)
  all_labels <- torch_cat(labels_list, dim = 1)
  f1 <- compute_f1(all_preds, all_labels)
  
  return(list(
    total_loss = avg_loss,
    ce_loss = avg_ce_loss,
    f1 = f1
  ))
}


#' Compute F1 Score
#'
#' @description
#' Compute F1 score from predictions and labels.
#'
#' @param preds Predicted probabilities (torch tensor)
#' @param labels True labels (torch tensor)
#' @param threshold Classification threshold (default: 0.5)
#'
#' @return F1 score
#'
#' @keywords internal
compute_f1 <- function(preds, labels, threshold = 0.5) {
  # Convert to binary predictions
  pred_binary <- (preds > threshold)$to(dtype = torch_float())
  labels_binary <- labels$to(dtype = torch_float())
  
  # Compute TP, FP, FN
  tp <- (pred_binary * labels_binary)$sum()$item()
  fp <- (pred_binary * (1 - labels_binary))$sum()$item()
  fn <- ((1 - pred_binary) * labels_binary)$sum()$item()
  
  # Compute F1
  if (tp + fp == 0 || tp + fn == 0) {
    return(0)
  }
  
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  
  if (precision + recall == 0) {
    return(0)
  }
  
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  return(f1)
}


#' Load Model Checkpoint
#'
#' @description
#' Load a saved model checkpoint.
#'
#' @param model MDITRE model
#' @param checkpoint_path Path to checkpoint file
#'
#' @return Model with loaded state
#'
#' @examples
#' \dontrun{
#' model <- mditre_model(num_rules = 5, num_otus = 100, num_time = 10, dist = dist_matrix)
#' model <- load_checkpoint(model, "checkpoints/best_model.pt")
#' }
#'
#' @export
load_checkpoint <- function(model, checkpoint_path) {
  if (!file.exists(checkpoint_path)) {
    stop(sprintf("Checkpoint file not found: %s", checkpoint_path))
  }
  
  state_dict <- torch_load(checkpoint_path)
  model$load_state_dict(state_dict)
  
  cat(sprintf("Loaded checkpoint from: %s\n", checkpoint_path))
  
  return(model)
}


#' Save Model Checkpoint
#'
#' @description
#' Save model state to a checkpoint file.
#'
#' @param model MDITRE model
#' @param checkpoint_path Path to save checkpoint
#'
#' @examples
#' \dontrun{
#' save_checkpoint(model, "checkpoints/my_model.pt")
#' }
#'
#' @export
save_checkpoint <- function(model, checkpoint_path) {
  # Create directory if needed
  checkpoint_dir <- dirname(checkpoint_path)
  if (!dir.exists(checkpoint_dir)) {
    dir.create(checkpoint_dir, recursive = TRUE)
  }
  
  torch_save(model$state_dict(), checkpoint_path)
  cat(sprintf("Saved checkpoint to: %s\n", checkpoint_path))
}
