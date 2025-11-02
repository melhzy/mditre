#' MDITRE Visualization Functions
#'
#' @description
#' Functions for visualizing MDITRE models, training progress, ROC curves,
#' and interpretable rules using ggplot2, ggtree, and patchwork.
#'
#' @name visualization
#' @keywords internal
NULL

#' Plot Training History
#'
#' @description
#' Plot training and validation metrics over epochs.
#'
#' @param history Training history from train_mditre()
#' @param metrics Vector of metric names to plot (default: c("loss", "f1"))
#' @param save_path Optional path to save plot (default: NULL, display only)
#' @param width Plot width in inches (default: 10)
#' @param height Plot height in inches (default: 6)
#'
#' @return ggplot object (invisibly)
#'
#' @examples
#' \dontrun{
#' # After training
#' result <- train_mditre(model, train_loader, val_loader, epochs = 200)
#' plot_training_history(result$history)
#' plot_training_history(result$history, metrics = c("loss", "ce_loss", "f1"))
#' }
#'
#' @export
plot_training_history <- function(history,
                                  metrics = c("loss", "f1"),
                                  save_path = NULL,
                                  width = 10,
                                  height = 6) {
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' required for plotting. Please install it.")
  }
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop("Package 'patchwork' required for plotting. Please install it.")
  }
  
  library(ggplot2)
  library(patchwork)
  
  plots <- list()
  
  for (metric in metrics) {
    # Prepare data
    train_col <- paste0("train_", metric)
    val_col <- paste0("val_", metric)
    
    if (!(train_col %in% names(history))) {
      warning(sprintf("Metric '%s' not found in history", metric))
      next
    }
    
    epochs <- seq_along(history[[train_col]])
    
    df <- data.frame(
      Epoch = rep(epochs, 2),
      Value = c(history[[train_col]], history[[val_col]]),
      Set = rep(c("Train", "Validation"), each = length(epochs))
    )
    
    # Remove NA values
    df <- df[!is.na(df$Value), ]
    
    # Create plot
    p <- ggplot(df, aes(x = Epoch, y = Value, color = Set)) +
      geom_line(linewidth = 1) +
      geom_point(size = 0.5, alpha = 0.5) +
      scale_color_manual(values = c("Train" = "#2E86AB", "Validation" = "#A23B72")) +
      labs(
        title = sprintf("%s over Epochs", tools::toTitleCase(metric)),
        x = "Epoch",
        y = tools::toTitleCase(metric)
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "top",
        legend.title = element_blank()
      )
    
    plots[[metric]] <- p
  }
  
  # Combine plots
  if (length(plots) == 0) {
    stop("No valid metrics to plot")
  } else if (length(plots) == 1) {
    combined_plot <- plots[[1]]
  } else {
    combined_plot <- wrap_plots(plots, ncol = 2)
  }
  
  # Save or display
  if (!is.null(save_path)) {
    ggsave(save_path, combined_plot, width = width, height = height)
    cat(sprintf("Plot saved to: %s\n", save_path))
  }
  
  print(combined_plot)
  invisible(combined_plot)
}


#' Plot ROC Curve
#'
#' @description
#' Plot ROC curve with AUC.
#'
#' @param predictions Predicted probabilities
#' @param labels True binary labels
#' @param title Plot title (default: "ROC Curve")
#' @param save_path Optional path to save plot
#' @param width Plot width in inches (default: 6)
#' @param height Plot height in inches (default: 6)
#'
#' @return ggplot object (invisibly)
#'
#' @examples
#' \dontrun{
#' # After evaluation
#' eval_result <- evaluate_model_on_data(model, test_loader, return_predictions = TRUE)
#' plot_roc_curve(eval_result$predictions, eval_result$labels)
#' }
#'
#' @export
plot_roc_curve <- function(predictions,
                          labels,
                          title = "ROC Curve",
                          save_path = NULL,
                          width = 6,
                          height = 6) {
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' required. Please install it.")
  }
  
  library(ggplot2)
  
  # Compute ROC curve
  roc_data <- compute_roc_curve(predictions, labels)
  
  # Create dataframe
  df <- data.frame(
    FPR = roc_data$fpr,
    TPR = roc_data$tpr
  )
  
  # Create plot
  p <- ggplot(df, aes(x = FPR, y = TPR)) +
    geom_line(color = "#2E86AB", linewidth = 1.2) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
                color = "gray50", linewidth = 0.8) +
    annotate("text", x = 0.75, y = 0.25, 
             label = sprintf("AUC = %.3f", roc_data$auc),
             size = 5, fontface = "bold") +
    labs(
      title = title,
      x = "False Positive Rate",
      y = "True Positive Rate"
    ) +
    coord_fixed() +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10)
    )
  
  # Save or display
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = width, height = height)
    cat(sprintf("Plot saved to: %s\n", save_path))
  }
  
  print(p)
  invisible(p)
}


#' Plot Confusion Matrix
#'
#' @description
#' Visualize confusion matrix as a heatmap.
#'
#' @param metrics Metrics object from compute_metrics()
#' @param save_path Optional path to save plot
#' @param width Plot width in inches (default: 6)
#' @param height Plot height in inches (default: 6)
#'
#' @return ggplot object (invisibly)
#'
#' @examples
#' \dontrun{
#' metrics <- compute_metrics(predictions, labels)
#' plot_confusion_matrix(metrics)
#' }
#'
#' @export
plot_confusion_matrix <- function(metrics,
                                 save_path = NULL,
                                 width = 6,
                                 height = 6) {
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' required. Please install it.")
  }
  
  library(ggplot2)
  
  # Extract confusion matrix
  cm <- metrics$confusion_matrix
  
  # Convert to long format
  df <- data.frame(
    Predicted = rep(rownames(cm), times = ncol(cm)),
    True = rep(colnames(cm), each = nrow(cm)),
    Count = as.vector(cm)
  )
  
  # Create plot
  p <- ggplot(df, aes(x = True, y = Predicted, fill = Count)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = Count), size = 8, fontface = "bold", color = "white") +
    scale_fill_gradient(low = "#3B4371", high = "#F72C25", 
                       name = "Count") +
    labs(
      title = "Confusion Matrix",
      x = "True Class",
      y = "Predicted Class"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      legend.position = "right"
    )
  
  # Save or display
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = width, height = height)
    cat(sprintf("Plot saved to: %s\n", save_path))
  }
  
  print(p)
  invisible(p)
}


#' Plot Cross-Validation Results
#'
#' @description
#' Visualize cross-validation results with error bars.
#'
#' @param cv_results Cross-validation results from cross_validate_mditre()
#' @param metrics Vector of metrics to plot (default: c("f1", "auc", "accuracy"))
#' @param save_path Optional path to save plot
#' @param width Plot width in inches (default: 8)
#' @param height Plot height in inches (default: 6)
#'
#' @return ggplot object (invisibly)
#'
#' @examples
#' \dontrun{
#' cv_results <- cross_validate_mditre(mditre_data, k = 5, ...)
#' plot_cv_results(cv_results)
#' }
#'
#' @export
plot_cv_results <- function(cv_results,
                           metrics = c("f1", "auc", "accuracy"),
                           save_path = NULL,
                           width = 8,
                           height = 6) {
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' required. Please install it.")
  }
  
  library(ggplot2)
  
  # Prepare data
  df_list <- list()
  
  for (metric in metrics) {
    if (!(metric %in% names(cv_results$mean_metrics))) {
      warning(sprintf("Metric '%s' not found in results", metric))
      next
    }
    
    mean_val <- cv_results$mean_metrics[[metric]]
    std_val <- cv_results$std_metrics[[metric]]
    
    df_list[[metric]] <- data.frame(
      Metric = tools::toTitleCase(metric),
      Mean = mean_val,
      SD = std_val
    )
  }
  
  if (length(df_list) == 0) {
    stop("No valid metrics to plot")
  }
  
  df <- do.call(rbind, df_list)
  
  # Create plot
  p <- ggplot(df, aes(x = Metric, y = Mean)) +
    geom_bar(stat = "identity", fill = "#2E86AB", alpha = 0.7, width = 0.6) +
    geom_errorbar(aes(ymin = Mean - SD, ymax = Mean + SD),
                  width = 0.2, linewidth = 0.8) +
    geom_text(aes(label = sprintf("%.3f ± %.3f", Mean, SD)),
              vjust = -0.5, size = 4, fontface = "bold") +
    labs(
      title = sprintf("%d-Fold Cross-Validation Results", cv_results$k),
      x = NULL,
      y = "Score"
    ) +
    ylim(0, max(df$Mean + df$SD) * 1.15) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      axis.text.x = element_text(angle = 0)
    )
  
  # Save or display
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = width, height = height)
    cat(sprintf("Plot saved to: %s\n", save_path))
  }
  
  print(p)
  invisible(p)
}


#' Plot Model Comparison
#'
#' @description
#' Compare performance of multiple models.
#'
#' @param comparison_results Results from compare_models()
#' @param metric Metric to compare (default: "f1")
#' @param save_path Optional path to save plot
#' @param width Plot width in inches (default: 10)
#' @param height Plot height in inches (default: 6)
#'
#' @return ggplot object (invisibly)
#'
#' @examples
#' \dontrun{
#' comparison <- compare_models(mditre_data, model_configs)
#' plot_model_comparison(comparison)
#' }
#'
#' @export
plot_model_comparison <- function(comparison_results,
                                 metric = "f1",
                                 save_path = NULL,
                                 width = 10,
                                 height = 6) {
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' required. Please install it.")
  }
  
  library(ggplot2)
  
  df <- comparison_results$comparison
  
  # Capitalize metric name
  metric_col <- tools::toTitleCase(metric)
  
  if (!(metric_col %in% names(df))) {
    stop(sprintf("Metric '%s' not found in comparison results", metric))
  }
  
  # Reorder by metric value
  df$Model <- factor(df$Model, levels = df$Model[order(df[[metric_col]], decreasing = TRUE)])
  
  # Highlight best model
  best_idx <- which.max(df[[metric_col]])
  df$IsBest <- seq_len(nrow(df)) == best_idx
  
  # Create plot
  p <- ggplot(df, aes(x = Model, y = .data[[metric_col]], fill = IsBest)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    scale_fill_manual(values = c("TRUE" = "#A23B72", "FALSE" = "#2E86AB"),
                     guide = "none") +
    geom_text(aes(label = sprintf("%.3f", .data[[metric_col]])),
              vjust = -0.5, size = 4, fontface = "bold") +
    labs(
      title = sprintf("Model Comparison (%s)", metric_col),
      x = "Model",
      y = metric_col
    ) +
    ylim(0, max(df[[metric_col]]) * 1.15) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  # Save or display
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = width, height = height)
    cat(sprintf("Plot saved to: %s\n", save_path))
  }
  
  print(p)
  invisible(p)
}


#' Plot Phylogenetic Tree
#'
#' @description
#' Visualize phylogenetic tree with OTU selection weights.
#'
#' @param phylo_tree phylo object (from ape package)
#' @param weights Optional OTU selection weights to display (numeric vector)
#' @param highlight_tips Optional tip names to highlight
#' @param save_path Optional path to save plot
#' @param width Plot width in inches (default: 8)
#' @param height Plot height in inches (default: 10)
#'
#' @return ggtree plot object (invisibly)
#'
#' @examples
#' \dontrun{
#' library(ape)
#' tree <- rtree(20)
#' plot_phylogenetic_tree(tree)
#' 
#' # With weights
#' weights <- runif(20)
#' names(weights) <- tree$tip.label
#' plot_phylogenetic_tree(tree, weights = weights)
#' }
#'
#' @export
plot_phylogenetic_tree <- function(phylo_tree,
                                   weights = NULL,
                                   highlight_tips = NULL,
                                   save_path = NULL,
                                   width = 8,
                                   height = 10) {
  
  if (!requireNamespace("ggtree", quietly = TRUE)) {
    stop("Package 'ggtree' required. Please install from Bioconductor.")
  }
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' required. Please install it.")
  }
  
  library(ggtree)
  library(ggplot2)
  
  # Basic tree plot
  p <- ggtree(phylo_tree) +
    geom_tiplab(size = 3, align = TRUE) +
    theme_tree2()
  
  # Add weights if provided
  if (!is.null(weights)) {
    # Create data frame
    tip_data <- data.frame(
      label = names(weights),
      weight = as.numeric(weights)
    )
    
    p <- p %<+% tip_data +
      geom_tippoint(aes(color = weight), size = 3) +
      scale_color_gradient(low = "lightblue", high = "red",
                          name = "Selection\nWeight") +
      theme(legend.position = "right")
  }
  
  # Highlight specific tips
  if (!is.null(highlight_tips)) {
    p <- p +
      geom_tiplab(aes(subset = label %in% highlight_tips),
                  color = "red", fontface = "bold", size = 3.5)
  }
  
  # Save or display
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = width, height = height)
    cat(sprintf("Plot saved to: %s\n", save_path))
  }
  
  print(p)
  invisible(p)
}


#' Plot Parameter Distributions
#'
#' @description
#' Visualize distributions of learned model parameters.
#'
#' @param model Trained MDITRE model
#' @param parameters Vector of parameter names to plot 
#'   (e.g., c("kappa", "eta", "thresh"))
#' @param save_path Optional path to save plot
#' @param width Plot width in inches (default: 10)
#' @param height Plot height in inches (default: 6)
#'
#' @return ggplot object (invisibly)
#'
#' @examples
#' \dontrun{
#' # After training
#' plot_parameter_distributions(trained_model, c("kappa", "eta", "thresh"))
#' }
#'
#' @export
plot_parameter_distributions <- function(model,
                                        parameters = c("kappa", "eta", "thresh"),
                                        save_path = NULL,
                                        width = 10,
                                        height = 6) {
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' required. Please install it.")
  }
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop("Package 'patchwork' required. Please install it.")
  }
  
  library(ggplot2)
  library(patchwork)
  
  plots <- list()
  
  for (param in parameters) {
    # Extract parameter values
    param_values <- tryCatch({
      if (param == "kappa") {
        as.numeric(model$phylo_focus$kappa$cpu())
      } else if (param == "eta") {
        as.numeric(model$phylo_focus$eta$cpu())
      } else if (param == "thresh") {
        as.numeric(model$threshold_detector$thresh$cpu())
      } else if (param == "slope") {
        if ("slope_detector" %in% names(model)) {
          as.numeric(model$slope_detector$slope$cpu())
        } else {
          NULL
        }
      } else {
        NULL
      }
    }, error = function(e) NULL)
    
    if (is.null(param_values)) {
      warning(sprintf("Parameter '%s' not found in model", param))
      next
    }
    
    # Create histogram
    df <- data.frame(Value = param_values)
    
    p <- ggplot(df, aes(x = Value)) +
      geom_histogram(bins = 30, fill = "#2E86AB", alpha = 0.7, color = "white") +
      geom_vline(xintercept = mean(param_values), color = "red", 
                 linetype = "dashed", linewidth = 1) +
      labs(
        title = sprintf("%s Distribution", tools::toTitleCase(param)),
        subtitle = sprintf("Mean: %.3f, SD: %.3f", 
                          mean(param_values), sd(param_values)),
        x = tools::toTitleCase(param),
        y = "Count"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 9)
      )
    
    plots[[param]] <- p
  }
  
  # Combine plots
  if (length(plots) == 0) {
    stop("No valid parameters to plot")
  } else if (length(plots) == 1) {
    combined_plot <- plots[[1]]
  } else {
    combined_plot <- wrap_plots(plots, ncol = 2)
  }
  
  # Save or display
  if (!is.null(save_path)) {
    ggsave(save_path, combined_plot, width = width, height = height)
    cat(sprintf("Plot saved to: %s\n", save_path))
  }
  
  print(combined_plot)
  invisible(combined_plot)
}


#' Create Comprehensive Evaluation Report
#'
#' @description
#' Generate a multi-panel evaluation report with training history, ROC curve,
#' confusion matrix, and metrics.
#'
#' @param training_result Result from train_mditre()
#' @param eval_result Result from evaluate_model_on_data()
#' @param save_path Path to save the report
#' @param width Plot width in inches (default: 14)
#' @param height Plot height in inches (default: 10)
#'
#' @return Combined plot object (invisibly)
#'
#' @examples
#' \dontrun{
#' # Complete workflow
#' train_result <- train_mditre(model, train_loader, val_loader, epochs = 200)
#' eval_result <- evaluate_model_on_data(train_result$model, test_loader,
#'                                       return_predictions = TRUE)
#' create_evaluation_report(train_result, eval_result, "report.pdf")
#' }
#'
#' @export
create_evaluation_report <- function(training_result,
                                    eval_result,
                                    save_path,
                                    width = 14,
                                    height = 10) {
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' required. Please install it.")
  }
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop("Package 'patchwork' required. Please install it.")
  }
  
  library(ggplot2)
  library(patchwork)
  
  # Create individual plots
  cat("Generating plots...\n")
  
  # 1. Training history
  history_plot <- plot_training_history(training_result$history, 
                                        metrics = c("loss", "f1"),
                                        save_path = NULL)
  
  # 2. ROC curve
  roc_plot <- plot_roc_curve(eval_result$predictions, eval_result$labels,
                             save_path = NULL)
  
  # 3. Confusion matrix
  cm_plot <- plot_confusion_matrix(eval_result$metrics, save_path = NULL)
  
  # Combine plots
  combined <- (history_plot | (roc_plot / cm_plot)) +
    plot_annotation(
      title = "MDITRE Model Evaluation Report",
      theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 16))
    )
  
  # Save
  ggsave(save_path, combined, width = width, height = height)
  cat(sprintf("Report saved to: %s\n", save_path))
  
  # Print summary metrics
  cat("\nModel Performance Summary:\n")
  cat("==========================\n")
  print_metrics(eval_result$metrics)
  
  invisible(combined)
}
