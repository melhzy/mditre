#' MDITRE Evaluation Utilities
#'
#' @description
#' Functions for evaluating MDITRE model performance including various metrics,
#' cross-validation, ROC curves, and model comparison utilities.
#'
#' @name evaluation
#' @keywords internal
NULL

#' Compute Classification Metrics
#'
#' @description
#' Compute comprehensive classification metrics including accuracy, precision,
#' recall, F1 score, sensitivity, specificity, and AUC-ROC.
#'
#' @param predictions Predicted probabilities (numeric vector or torch tensor)
#' @param labels True binary labels (numeric vector or torch tensor)
#' @param threshold Classification threshold (default: 0.5)
#'
#' @return Named list with metrics:
#'   \itemize{
#'     \item accuracy: Overall accuracy
#'     \item precision: Positive predictive value
#'     \item recall: Sensitivity / True positive rate
#'     \item f1: F1 score (harmonic mean of precision and recall)
#'     \item sensitivity: Same as recall
#'     \item specificity: True negative rate
#'     \item auc: Area under ROC curve
#'     \item confusion_matrix: 2x2 confusion matrix
#'   }
#'
#' @examples
#' \dontrun{
#' # Generate example predictions
#' predictions <- c(0.9, 0.8, 0.3, 0.2, 0.7, 0.1)
#' labels <- c(1, 1, 0, 0, 1, 0)
#' 
#' # Compute metrics
#' metrics <- compute_metrics(predictions, labels)
#' print(metrics$f1)
#' print(metrics$auc)
#' }
#'
#' @export
compute_metrics <- function(predictions, labels, threshold = 0.5) {
  
  # Convert torch tensors to R vectors if needed
  if (inherits(predictions, "torch_tensor")) {
    predictions <- as.numeric(predictions$cpu())
  }
  if (inherits(labels, "torch_tensor")) {
    labels <- as.numeric(labels$cpu())
  }
  
  # Ensure numeric vectors
  predictions <- as.numeric(predictions)
  labels <- as.numeric(labels)
  
  # Binary predictions
  pred_binary <- as.numeric(predictions > threshold)
  
  # Confusion matrix
  tp <- sum(pred_binary == 1 & labels == 1)
  tn <- sum(pred_binary == 0 & labels == 0)
  fp <- sum(pred_binary == 1 & labels == 0)
  fn <- sum(pred_binary == 0 & labels == 1)
  
  conf_matrix <- matrix(c(tn, fp, fn, tp), nrow = 2, byrow = TRUE,
                        dimnames = list(c("Pred 0", "Pred 1"),
                                       c("True 0", "True 1")))
  
  # Metrics
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  
  precision <- if (tp + fp > 0) tp / (tp + fp) else 0
  recall <- if (tp + fn > 0) tp / (tp + fn) else 0
  sensitivity <- recall
  specificity <- if (tn + fp > 0) tn / (tn + fp) else 0
  
  f1 <- if (precision + recall > 0) {
    2 * (precision * recall) / (precision + recall)
  } else {
    0
  }
  
  # AUC-ROC
  auc <- compute_auc_roc(predictions, labels)
  
  return(list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1 = f1,
    sensitivity = sensitivity,
    specificity = specificity,
    auc = auc,
    confusion_matrix = conf_matrix,
    tp = tp,
    tn = tn,
    fp = fp,
    fn = fn
  ))
}


#' Compute AUC-ROC
#'
#' @description
#' Compute Area Under the Receiver Operating Characteristic curve.
#'
#' @param predictions Predicted probabilities (numeric vector)
#' @param labels True binary labels (numeric vector)
#'
#' @return AUC-ROC score
#'
#' @keywords internal
compute_auc_roc <- function(predictions, labels) {
  
  # Handle edge cases
  if (length(unique(labels)) < 2) {
    warning("Only one class present in labels. AUC-ROC undefined.")
    return(NA)
  }
  
  # Sort by predictions
  order_idx <- order(predictions, decreasing = TRUE)
  sorted_labels <- labels[order_idx]
  
  # Compute TPR and FPR at each threshold
  n_pos <- sum(labels == 1)
  n_neg <- sum(labels == 0)
  
  if (n_pos == 0 || n_neg == 0) {
    warning("No positive or negative examples. AUC-ROC undefined.")
    return(NA)
  }
  
  tpr <- cumsum(sorted_labels) / n_pos
  fpr <- cumsum(1 - sorted_labels) / n_neg
  
  # Add endpoints
  tpr <- c(0, tpr, 1)
  fpr <- c(0, fpr, 1)
  
  # Compute AUC using trapezoidal rule
  auc <- sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
  
  return(auc)
}


#' Compute ROC Curve
#'
#' @description
#' Compute ROC curve coordinates (FPR, TPR, thresholds).
#'
#' @param predictions Predicted probabilities (numeric vector)
#' @param labels True binary labels (numeric vector)
#' @param n_thresholds Number of threshold points to compute (default: 100)
#'
#' @return List with:
#'   \itemize{
#'     \item fpr: False positive rates
#'     \item tpr: True positive rates
#'     \item thresholds: Threshold values
#'     \item auc: Area under curve
#'   }
#'
#' @examples
#' \dontrun{
#' predictions <- runif(100)
#' labels <- rbinom(100, 1, 0.5)
#' roc <- compute_roc_curve(predictions, labels)
#' plot(roc$fpr, roc$tpr, type = "l", main = sprintf("ROC (AUC = %.3f)", roc$auc))
#' }
#'
#' @export
compute_roc_curve <- function(predictions, labels, n_thresholds = 100) {
  
  # Convert torch tensors if needed
  if (inherits(predictions, "torch_tensor")) {
    predictions <- as.numeric(predictions$cpu())
  }
  if (inherits(labels, "torch_tensor")) {
    labels <- as.numeric(labels$cpu())
  }
  
  # Generate thresholds
  thresholds <- seq(0, 1, length.out = n_thresholds)
  
  # Compute TPR and FPR for each threshold
  fpr <- numeric(n_thresholds)
  tpr <- numeric(n_thresholds)
  
  n_pos <- sum(labels == 1)
  n_neg <- sum(labels == 0)
  
  for (i in seq_along(thresholds)) {
    pred_binary <- as.numeric(predictions >= thresholds[i])
    
    tp <- sum(pred_binary == 1 & labels == 1)
    fp <- sum(pred_binary == 1 & labels == 0)
    
    tpr[i] <- if (n_pos > 0) tp / n_pos else 0
    fpr[i] <- if (n_neg > 0) fp / n_neg else 0
  }
  
  # Compute AUC
  auc <- compute_auc_roc(predictions, labels)
  
  return(list(
    fpr = fpr,
    tpr = tpr,
    thresholds = thresholds,
    auc = auc
  ))
}


#' Evaluate Model on Dataset
#'
#' @description
#' Comprehensive evaluation of a trained model on a dataset.
#'
#' @param model Trained MDITRE model
#' @param data_loader torch dataloader
#' @param device torch device ("cuda" or "cpu")
#' @param return_predictions Whether to return predictions and labels (default: FALSE)
#'
#' @return List with:
#'   \itemize{
#'     \item metrics: All classification metrics
#'     \item predictions: Predicted probabilities (if return_predictions = TRUE)
#'     \item labels: True labels (if return_predictions = TRUE)
#'   }
#'
#' @examples
#' \dontrun{
#' # Evaluate on test set
#' results <- evaluate_model_on_data(model, test_loader, device = torch_device("cpu"))
#' print(results$metrics$f1)
#' print(results$metrics$auc)
#' }
#'
#' @export
evaluate_model_on_data <- function(model, data_loader, device = NULL,
                                   return_predictions = FALSE) {
  
  # Auto-detect device
  if (is.null(device)) {
    device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
  }
  
  model$to(device = device)
  model$eval()
  
  all_preds <- list()
  all_labels <- list()
  
  with_no_grad({
    coro::loop(for (batch in data_loader) {
      # Move data to device
      data <- batch$data$to(device = device)
      labels <- batch$label$to(device = device)
      mask <- batch$mask$to(device = device)
      
      # Forward pass (use hard selections for evaluation)
      outputs <- model(
        data,
        mask = mask,
        k_alpha = 1,
        k_beta = 1,
        k_otu = 100,
        k_time = 1,
        k_thresh = 100,
        k_slope = 1000,
        use_noise = FALSE,
        hard = TRUE
      )
      
      # Convert logits to probabilities
      probs <- torch_sigmoid(outputs)
      
      # Store predictions and labels
      all_preds[[length(all_preds) + 1]] <- probs$cpu()
      all_labels[[length(all_labels) + 1]] <- labels$cpu()
    })
  })
  
  # Concatenate all batches
  predictions <- torch_cat(all_preds, dim = 1)
  labels <- torch_cat(all_labels, dim = 1)
  
  # Convert to vectors
  pred_vec <- as.numeric(predictions$squeeze())
  label_vec <- as.numeric(labels$squeeze())
  
  # Compute metrics
  metrics <- compute_metrics(pred_vec, label_vec)
  
  result <- list(metrics = metrics)
  
  if (return_predictions) {
    result$predictions <- pred_vec
    result$labels <- label_vec
  }
  
  return(result)
}


#' K-Fold Cross-Validation
#'
#' @description
#' Perform k-fold cross-validation for MDITRE model training and evaluation.
#'
#' @param mditre_data Full MDITRE data object (from phyloseq_to_mditre)
#' @param k Number of folds (default: 5)
#' @param model_params Named list of model parameters (num_rules, num_otus, num_time, dist)
#' @param train_params Named list of training parameters (epochs, learning_rates, etc.)
#' @param stratified Whether to use stratified splitting (default: TRUE)
#' @param seed Random seed for reproducibility (default: 42)
#' @param verbose Print progress (default: TRUE)
#'
#' @return List with:
#'   \itemize{
#'     \item fold_results: List of results for each fold
#'     \item mean_metrics: Average metrics across folds
#'     \item std_metrics: Standard deviation of metrics
#'     \item all_predictions: All predictions (if requested)
#'     \item all_labels: All true labels (if requested)
#'   }
#'
#' @examples
#' \dontrun{
#' # Prepare data
#' mditre_data <- phyloseq_to_mditre(ps, "Subject", "Time", "Disease")
#' 
#' # Model parameters
#' model_params <- list(
#'   num_rules = 5,
#'   num_otus = mditre_data$metadata$n_otus,
#'   num_time = mditre_data$metadata$n_timepoints,
#'   dist = mditre_data$phylo_dist
#' )
#' 
#' # Training parameters
#' train_params <- list(epochs = 200, early_stopping_patience = 20)
#' 
#' # Run cross-validation
#' cv_results <- cross_validate_mditre(
#'   mditre_data,
#'   k = 5,
#'   model_params = model_params,
#'   train_params = train_params
#' )
#' 
#' # Print results
#' print(cv_results$mean_metrics)
#' }
#'
#' @export
cross_validate_mditre <- function(mditre_data,
                                  k = 5,
                                  model_params,
                                  train_params = list(),
                                  stratified = TRUE,
                                  seed = 42,
                                  verbose = TRUE) {
  
  set.seed(seed)
  
  n_subjects <- dim(mditre_data$X)[1]
  labels <- as.numeric(mditre_data$y)
  
  # Create fold indices
  if (stratified) {
    # Stratified k-fold
    pos_indices <- which(labels == 1)
    neg_indices <- which(labels == 0)
    
    pos_folds <- split(sample(pos_indices), rep(1:k, length.out = length(pos_indices)))
    neg_folds <- split(sample(neg_indices), rep(1:k, length.out = length(neg_indices)))
    
    folds <- lapply(1:k, function(i) c(pos_folds[[i]], neg_folds[[i]]))
  } else {
    # Random k-fold
    indices <- sample(1:n_subjects)
    folds <- split(indices, rep(1:k, length.out = n_subjects))
  }
  
  # Store results for each fold
  fold_results <- list()
  all_predictions <- numeric()
  all_labels <- numeric()
  
  for (fold_idx in 1:k) {
    if (verbose) {
      cat(sprintf("\n=== Fold %d/%d ===\n", fold_idx, k))
    }
    
    # Split data
    test_indices <- folds[[fold_idx]]
    train_indices <- setdiff(1:n_subjects, test_indices)
    
    # Create train/test splits
    train_data <- list(
      X = mditre_data$X[train_indices, , , drop = FALSE],
      y = mditre_data$y[train_indices],
      times = mditre_data$times[train_indices, , drop = FALSE],
      mask = mditre_data$mask[train_indices, , drop = FALSE],
      phylo_dist = mditre_data$phylo_dist,
      phylo_tree = mditre_data$phylo_tree,
      metadata = mditre_data$metadata
    )
    
    test_data <- list(
      X = mditre_data$X[test_indices, , , drop = FALSE],
      y = mditre_data$y[test_indices],
      times = mditre_data$times[test_indices, , drop = FALSE],
      mask = mditre_data$mask[test_indices, , drop = FALSE],
      phylo_dist = mditre_data$phylo_dist,
      phylo_tree = mditre_data$phylo_tree,
      metadata = mditre_data$metadata
    )
    
    # Create dataloaders
    train_loader <- create_dataloader(train_data, batch_size = 16, shuffle = TRUE)
    test_loader <- create_dataloader(test_data, batch_size = 16, shuffle = FALSE)
    
    # Create model
    model <- do.call(mditre_model, model_params)
    
    # Train model
    train_result <- do.call(train_mditre, c(
      list(
        model = model,
        train_loader = train_loader,
        val_loader = NULL,
        verbose = verbose
      ),
      train_params
    ))
    
    # Evaluate on test set
    eval_result <- evaluate_model_on_data(
      train_result$model,
      test_loader,
      return_predictions = TRUE
    )
    
    # Store results
    fold_results[[fold_idx]] <- list(
      train_history = train_result$history,
      test_metrics = eval_result$metrics,
      predictions = eval_result$predictions,
      labels = eval_result$labels
    )
    
    all_predictions <- c(all_predictions, eval_result$predictions)
    all_labels <- c(all_labels, eval_result$labels)
    
    if (verbose) {
      cat(sprintf("Fold %d - Test F1: %.4f, AUC: %.4f\n",
                  fold_idx, eval_result$metrics$f1, eval_result$metrics$auc))
    }
  }
  
  # Aggregate metrics across folds
  metric_names <- c("accuracy", "precision", "recall", "f1", 
                   "sensitivity", "specificity", "auc")
  
  mean_metrics <- list()
  std_metrics <- list()
  
  for (metric in metric_names) {
    values <- sapply(fold_results, function(x) x$test_metrics[[metric]])
    mean_metrics[[metric]] <- mean(values, na.rm = TRUE)
    std_metrics[[metric]] <- sd(values, na.rm = TRUE)
  }
  
  if (verbose) {
    cat("\n=== Cross-Validation Results ===\n")
    cat(sprintf("Mean F1: %.4f ± %.4f\n", mean_metrics$f1, std_metrics$f1))
    cat(sprintf("Mean AUC: %.4f ± %.4f\n", mean_metrics$auc, std_metrics$auc))
    cat(sprintf("Mean Accuracy: %.4f ± %.4f\n", mean_metrics$accuracy, std_metrics$accuracy))
  }
  
  return(list(
    fold_results = fold_results,
    mean_metrics = mean_metrics,
    std_metrics = std_metrics,
    all_predictions = all_predictions,
    all_labels = all_labels,
    k = k
  ))
}


#' Compare Multiple Models
#'
#' @description
#' Compare performance of multiple MDITRE models with different configurations.
#'
#' @param mditre_data MDITRE data object
#' @param model_configs List of model configurations (each with model_params and train_params)
#' @param test_fraction Fraction of data for testing (default: 0.2)
#' @param seed Random seed (default: 42)
#' @param verbose Print progress (default: TRUE)
#'
#' @return List with:
#'   \itemize{
#'     \item results: Results for each model
#'     \item comparison: Data frame comparing metrics
#'     \item best_model_idx: Index of best model (by F1 score)
#'   }
#'
#' @examples
#' \dontrun{
#' # Define model configurations
#' configs <- list(
#'   list(
#'     name = "MDITRE_5rules",
#'     model_params = list(num_rules = 5, num_otus = n_otus, 
#'                        num_time = n_time, dist = dist),
#'     train_params = list(epochs = 200)
#'   ),
#'   list(
#'     name = "MDITRE_10rules",
#'     model_params = list(num_rules = 10, num_otus = n_otus,
#'                        num_time = n_time, dist = dist),
#'     train_params = list(epochs = 200)
#'   )
#' )
#' 
#' # Compare models
#' comparison <- compare_models(mditre_data, configs)
#' print(comparison$comparison)
#' }
#'
#' @export
compare_models <- function(mditre_data,
                          model_configs,
                          test_fraction = 0.2,
                          seed = 42,
                          verbose = TRUE) {
  
  # Split data once
  split_data <- split_train_test(mditre_data, test_fraction = test_fraction, 
                                 stratified = TRUE, seed = seed)
  
  train_loader <- create_dataloader(split_data$train, batch_size = 16, shuffle = TRUE)
  test_loader <- create_dataloader(split_data$test, batch_size = 16, shuffle = FALSE)
  
  # Train and evaluate each model
  results <- list()
  comparison_df <- data.frame()
  
  for (i in seq_along(model_configs)) {
    config <- model_configs[[i]]
    model_name <- if (!is.null(config$name)) config$name else sprintf("Model_%d", i)
    
    if (verbose) {
      cat(sprintf("\n=== Training %s ===\n", model_name))
    }
    
    # Create model
    model <- do.call(mditre_model, config$model_params)
    
    # Train model
    train_result <- do.call(train_mditre, c(
      list(
        model = model,
        train_loader = train_loader,
        val_loader = test_loader,
        verbose = verbose
      ),
      config$train_params
    ))
    
    # Evaluate on test set
    eval_result <- evaluate_model_on_data(
      train_result$model,
      test_loader,
      return_predictions = TRUE
    )
    
    # Store results
    results[[i]] <- list(
      name = model_name,
      config = config,
      train_result = train_result,
      eval_result = eval_result
    )
    
    # Add to comparison
    comparison_df <- rbind(
      comparison_df,
      data.frame(
        Model = model_name,
        F1 = eval_result$metrics$f1,
        AUC = eval_result$metrics$auc,
        Accuracy = eval_result$metrics$accuracy,
        Precision = eval_result$metrics$precision,
        Recall = eval_result$metrics$recall,
        Specificity = eval_result$metrics$specificity,
        stringsAsFactors = FALSE
      )
    )
    
    if (verbose) {
      cat(sprintf("%s - F1: %.4f, AUC: %.4f\n",
                  model_name, eval_result$metrics$f1, eval_result$metrics$auc))
    }
  }
  
  # Find best model
  best_idx <- which.max(comparison_df$F1)
  
  if (verbose) {
    cat(sprintf("\nBest model: %s (F1: %.4f)\n",
                comparison_df$Model[best_idx], comparison_df$F1[best_idx]))
  }
  
  return(list(
    results = results,
    comparison = comparison_df,
    best_model_idx = best_idx
  ))
}


#' Print Metrics Summary
#'
#' @description
#' Pretty print classification metrics.
#'
#' @param metrics Metrics object from compute_metrics()
#'
#' @export
print_metrics <- function(metrics) {
  cat("\n")
  cat("========================================\n")
  cat("Classification Metrics\n")
  cat("========================================\n")
  cat(sprintf("Accuracy:     %.4f\n", metrics$accuracy))
  cat(sprintf("Precision:    %.4f\n", metrics$precision))
  cat(sprintf("Recall:       %.4f\n", metrics$recall))
  cat(sprintf("F1 Score:     %.4f\n", metrics$f1))
  cat(sprintf("Sensitivity:  %.4f\n", metrics$sensitivity))
  cat(sprintf("Specificity:  %.4f\n", metrics$specificity))
  cat(sprintf("AUC-ROC:      %.4f\n", metrics$auc))
  cat("========================================\n")
  cat("Confusion Matrix:\n")
  print(metrics$confusion_matrix)
  cat("========================================\n\n")
}
