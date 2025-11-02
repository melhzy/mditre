#' MDITRE Visualization Examples
#'
#' @description
#' Comprehensive examples demonstrating all visualization functions in MDITRE.
#'
#' @examples
#' \dontrun{
#' # Source this file to run all examples
#' source("examples/visualize_examples.R")
#' }

library(torch)
library(ggplot2)
library(ggtree)
library(patchwork)

# Load MDITRE functions
source("R/models.R")
source("R/phyloseq_loader.R")
source("R/trainer.R")
source("R/evaluation.R")
source("R/visualize.R")

cat("MDITRE Visualization Examples\n")
cat("==============================\n\n")

# ============================================================================
# Example 1: Training History Visualization
# ============================================================================
cat("Example 1: Plot Training History\n")
cat("---------------------------------\n")

# Simulate training history
set.seed(42)
epochs <- 100

history <- list(
  train_loss = 2.0 * exp(-seq(0, 4, length.out = epochs)) + rnorm(epochs, 0, 0.05),
  val_loss = 2.2 * exp(-seq(0, 3.8, length.out = epochs)) + rnorm(epochs, 0, 0.08),
  train_f1 = 1 - 0.9 * exp(-seq(0, 4, length.out = epochs)) + rnorm(epochs, 0, 0.02),
  val_f1 = 1 - 0.95 * exp(-seq(0, 3.5, length.out = epochs)) + rnorm(epochs, 0, 0.03),
  train_ce_loss = 1.8 * exp(-seq(0, 4.2, length.out = epochs)) + rnorm(epochs, 0, 0.04),
  val_ce_loss = 2.0 * exp(-seq(0, 3.9, length.out = epochs)) + rnorm(epochs, 0, 0.06)
)

# Plot default metrics (loss and F1)
plot_training_history(history)

# Plot multiple metrics
plot_training_history(history, metrics = c("loss", "ce_loss", "f1"))

# Save to file
# plot_training_history(history, save_path = "training_history.pdf")

cat("✓ Training history plots created\n\n")


# ============================================================================
# Example 2: ROC Curve Visualization
# ============================================================================
cat("Example 2: Plot ROC Curve\n")
cat("-------------------------\n")

# Simulate predictions and labels
set.seed(123)
n_samples <- 500

# Generate predictions (positive class has higher scores)
predictions <- c(
  rnorm(250, mean = 0.7, sd = 0.2),  # True positives
  rnorm(250, mean = 0.3, sd = 0.2)   # True negatives
)
predictions <- pmax(0, pmin(1, predictions))  # Clip to [0, 1]

labels <- c(rep(1, 250), rep(0, 250))

# Plot ROC curve
plot_roc_curve(predictions, labels)

# Custom title
plot_roc_curve(predictions, labels, title = "MDITRE Test Set ROC Curve")

# Save to file
# plot_roc_curve(predictions, labels, save_path = "roc_curve.pdf")

cat("✓ ROC curve created\n\n")


# ============================================================================
# Example 3: Confusion Matrix Visualization
# ============================================================================
cat("Example 3: Plot Confusion Matrix\n")
cat("---------------------------------\n")

# Compute metrics from predictions
metrics <- compute_metrics(predictions, labels)

# Plot confusion matrix
plot_confusion_matrix(metrics)

# Save to file
# plot_confusion_matrix(metrics, save_path = "confusion_matrix.pdf")

cat("✓ Confusion matrix created\n")
cat(sprintf("  Accuracy: %.3f\n", metrics$accuracy))
cat(sprintf("  F1 Score: %.3f\n", metrics$f1))
cat(sprintf("  AUC: %.3f\n\n", metrics$auc))


# ============================================================================
# Example 4: Cross-Validation Results Visualization
# ============================================================================
cat("Example 4: Plot Cross-Validation Results\n")
cat("-----------------------------------------\n")

# Simulate cross-validation results
cv_results <- list(
  k = 5,
  mean_metrics = list(
    accuracy = 0.8523,
    precision = 0.8412,
    recall = 0.8634,
    f1 = 0.8521,
    sensitivity = 0.8634,
    specificity = 0.8412,
    auc = 0.9156
  ),
  std_metrics = list(
    accuracy = 0.0234,
    precision = 0.0312,
    recall = 0.0287,
    f1 = 0.0256,
    sensitivity = 0.0287,
    specificity = 0.0312,
    auc = 0.0189
  )
)

# Plot default metrics
plot_cv_results(cv_results)

# Plot specific metrics
plot_cv_results(cv_results, metrics = c("f1", "auc", "accuracy", "precision"))

# Save to file
# plot_cv_results(cv_results, save_path = "cv_results.pdf")

cat("✓ Cross-validation results plotted\n\n")


# ============================================================================
# Example 5: Model Comparison Visualization
# ============================================================================
cat("Example 5: Plot Model Comparison\n")
cat("---------------------------------\n")

# Simulate model comparison results
comparison_results <- list(
  comparison = data.frame(
    Model = c("MDITRE-5rules", "MDITRE-3rules", "MDITRE-7rules", "MDITREAbun"),
    Accuracy = c(0.8523, 0.8312, 0.8645, 0.8234),
    Precision = c(0.8412, 0.8201, 0.8534, 0.8123),
    Recall = c(0.8634, 0.8423, 0.8756, 0.8345),
    F1 = c(0.8521, 0.8310, 0.8643, 0.8232),
    Auc = c(0.9156, 0.8945, 0.9278, 0.8867)
  ),
  best_model_idx = 3
)

# Plot F1 comparison
plot_model_comparison(comparison_results, metric = "f1")

# Plot AUC comparison
plot_model_comparison(comparison_results, metric = "auc")

# Save to file
# plot_model_comparison(comparison_results, save_path = "model_comparison.pdf")

cat("✓ Model comparison created\n")
cat(sprintf("  Best model: %s (F1=%.3f)\n\n", 
            comparison_results$comparison$Model[comparison_results$best_model_idx],
            comparison_results$comparison$F1[comparison_results$best_model_idx]))


# ============================================================================
# Example 6: Phylogenetic Tree Visualization
# ============================================================================
cat("Example 6: Plot Phylogenetic Tree\n")
cat("----------------------------------\n")

if (requireNamespace("ape", quietly = TRUE)) {
  library(ape)
  
  # Create example phylogenetic tree
  set.seed(456)
  n_otus <- 20
  tree <- rtree(n_otus)
  tree$tip.label <- paste0("OTU_", 1:n_otus)
  
  # Basic tree plot
  plot_phylogenetic_tree(tree)
  
  # Tree with selection weights
  weights <- runif(n_otus)
  names(weights) <- tree$tip.label
  plot_phylogenetic_tree(tree, weights = weights)
  
  # Highlight specific OTUs
  highlight_otus <- c("OTU_5", "OTU_12", "OTU_18")
  plot_phylogenetic_tree(tree, weights = weights, highlight_tips = highlight_otus)
  
  # Save to file
  # plot_phylogenetic_tree(tree, weights = weights, save_path = "phylo_tree.pdf")
  
  cat("✓ Phylogenetic tree visualizations created\n\n")
} else {
  cat("⚠ Skipping tree visualization (ape package not available)\n\n")
}


# ============================================================================
# Example 7: Parameter Distribution Visualization
# ============================================================================
cat("Example 7: Plot Parameter Distributions\n")
cat("----------------------------------------\n")

# Create a small MDITRE model
set.seed(789)
n_otus <- 50
n_time <- 5
num_rules <- 3

# Create random phylogenetic distances
dist_matrix <- matrix(runif(n_otus * n_otus), n_otus, n_otus)
dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
diag(dist_matrix) <- 0
dist <- torch_tensor(dist_matrix)

# Create model
model <- mditre_model(
  num_rules = num_rules,
  num_otus = n_otus,
  num_time = n_time,
  dist = dist
)

# Initialize with some variability
with_no_grad({
  model$phylo_focus$kappa$uniform_(0.1, 2.0)
  model$phylo_focus$eta$uniform_(-1.0, 1.0)
  model$threshold_detector$thresh$uniform_(0.2, 0.8)
})

# Plot parameter distributions
plot_parameter_distributions(model, parameters = c("kappa", "eta", "thresh"))

# Save to file
# plot_parameter_distributions(model, save_path = "param_distributions.pdf")

cat("✓ Parameter distributions plotted\n")
cat(sprintf("  Kappa range: [%.3f, %.3f]\n", 
            min(as.numeric(model$phylo_focus$kappa$cpu())),
            max(as.numeric(model$phylo_focus$kappa$cpu()))))
cat(sprintf("  Eta range: [%.3f, %.3f]\n", 
            min(as.numeric(model$phylo_focus$eta$cpu())),
            max(as.numeric(model$phylo_focus$eta$cpu()))))
cat(sprintf("  Thresh range: [%.3f, %.3f]\n\n", 
            min(as.numeric(model$threshold_detector$thresh$cpu())),
            max(as.numeric(model$threshold_detector$thresh$cpu()))))


# ============================================================================
# Example 8: Complete Evaluation Report
# ============================================================================
cat("Example 8: Create Comprehensive Evaluation Report\n")
cat("--------------------------------------------------\n")

# Prepare mock training result
training_result <- list(
  model = model,
  history = history,
  best_f1 = 0.8521,
  best_epoch = 87
)

# Prepare mock evaluation result
eval_result <- list(
  metrics = metrics,
  predictions = predictions,
  labels = labels
)

# Create comprehensive report
# create_evaluation_report(
#   training_result,
#   eval_result,
#   save_path = "mditre_evaluation_report.pdf",
#   width = 14,
#   height = 10
# )

cat("✓ Comprehensive evaluation report structure ready\n")
cat("  (Uncomment save_path to generate PDF)\n\n")


# ============================================================================
# Example 9: Custom Multi-Panel Visualization
# ============================================================================
cat("Example 9: Custom Multi-Panel Layout\n")
cat("-------------------------------------\n")

if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  
  # Create individual plots
  p1 <- plot_training_history(history, metrics = "loss", save_path = NULL)
  p2 <- plot_roc_curve(predictions, labels, save_path = NULL)
  p3 <- plot_confusion_matrix(metrics, save_path = NULL)
  p4 <- plot_cv_results(cv_results, metrics = c("f1", "auc"), save_path = NULL)
  
  # Combine with patchwork
  combined <- (p1 | p2) / (p3 | p4) +
    plot_annotation(
      title = "MDITRE Complete Analysis Dashboard",
      theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16))
    )
  
  print(combined)
  
  # Save combined plot
  # ggsave("mditre_dashboard.pdf", combined, width = 14, height = 10)
  
  cat("✓ Multi-panel dashboard created\n\n")
} else {
  cat("⚠ Skipping multi-panel example (patchwork package not available)\n\n")
}


# ============================================================================
# Example 10: Side-by-Side Model Performance
# ============================================================================
cat("Example 10: Side-by-Side Model Performance Comparison\n")
cat("------------------------------------------------------\n")

# Create multiple ROC curves for different models
set.seed(321)

models_roc_data <- list(
  list(
    name = "MDITRE-5rules",
    predictions = pmax(0, pmin(1, c(rnorm(250, 0.72, 0.18), rnorm(250, 0.28, 0.18)))),
    labels = c(rep(1, 250), rep(0, 250))
  ),
  list(
    name = "MDITRE-3rules",
    predictions = pmax(0, pmin(1, c(rnorm(250, 0.68, 0.20), rnorm(250, 0.32, 0.20)))),
    labels = c(rep(1, 250), rep(0, 250))
  ),
  list(
    name = "MDITREAbun",
    predictions = pmax(0, pmin(1, c(rnorm(250, 0.65, 0.22), rnorm(250, 0.35, 0.22)))),
    labels = c(rep(1, 250), rep(0, 250))
  )
)

# Create ROC curves for each model
roc_plots <- lapply(models_roc_data, function(m) {
  plot_roc_curve(m$predictions, m$labels, title = m$name, save_path = NULL)
})

# Combine plots
if (requireNamespace("patchwork", quietly = TRUE)) {
  combined_roc <- wrap_plots(roc_plots, ncol = 3) +
    plot_annotation(
      title = "ROC Curves: Model Comparison",
      theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))
    )
  
  print(combined_roc)
  
  cat("✓ Side-by-side ROC comparison created\n\n")
}


# ============================================================================
# Summary
# ============================================================================
cat("\n")
cat("=" %R% paste(rep("=", 70), collapse = "") %R% "\n")
cat("VISUALIZATION EXAMPLES COMPLETE\n")
cat("=" %R% paste(rep("=", 70), collapse = "") %R% "\n\n")

cat("Examples Demonstrated:\n")
cat("1. ✓ Training history plots (loss, F1, multiple metrics)\n")
cat("2. ✓ ROC curves with AUC\n")
cat("3. ✓ Confusion matrix heatmaps\n")
cat("4. ✓ Cross-validation results with error bars\n")
cat("5. ✓ Model comparison bar plots\n")
cat("6. ✓ Phylogenetic tree visualization with weights\n")
cat("7. ✓ Parameter distribution histograms\n")
cat("8. ✓ Comprehensive evaluation reports\n")
cat("9. ✓ Custom multi-panel layouts\n")
cat("10. ✓ Side-by-side model performance\n\n")

cat("Key Features:\n")
cat("• All plots use ggplot2 for consistent styling\n")
cat("• PDF export support for publications\n")
cat("• Multi-panel layouts with patchwork\n")
cat("• Phylogenetic tree integration with ggtree\n")
cat("• Customizable colors, sizes, and themes\n")
cat("• Automatic AUC/metric annotations\n\n")

cat("Next Steps:\n")
cat("• Uncomment save_path arguments to generate PDF files\n")
cat("• Customize colors and themes for your publication\n")
cat("• Integrate with real MDITRE training workflows\n")
cat("• Create custom multi-panel reports\n\n")

cat("Visualization toolkit ready for production use!\n")
