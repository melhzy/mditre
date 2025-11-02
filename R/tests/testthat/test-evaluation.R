# Test Evaluation Utilities
# Tests for metrics, cross-validation, and model comparison

test_that("compute_metrics calculates accuracy correctly", {
  # Perfect predictions
  predictions <- c(0.1, 0.2, 0.9, 0.95, 0.85)
  labels <- c(0, 0, 1, 1, 1)
  
  metrics <- compute_metrics(predictions, labels, threshold = 0.5)
  
  expect_equal(metrics$accuracy, 1.0)
  expect_equal(metrics$f1, 1.0)
})

test_that("compute_metrics handles mixed predictions", {
  predictions <- c(0.2, 0.3, 0.7, 0.8, 0.4, 0.9)
  labels <- c(0, 0, 1, 1, 0, 1)
  
  metrics <- compute_metrics(predictions, labels, threshold = 0.5)
  
  # Should have reasonable values
  expect_true(metrics$accuracy >= 0 && metrics$accuracy <= 1)
  expect_true(metrics$precision >= 0 && metrics$precision <= 1)
  expect_true(metrics$recall >= 0 && metrics$recall <= 1)
  expect_true(metrics$f1 >= 0 && metrics$f1 <= 1)
})

test_that("compute_metrics calculates confusion matrix correctly", {
  predictions <- c(0.2, 0.3, 0.7, 0.8)
  labels <- c(0, 1, 0, 1)
  
  metrics <- compute_metrics(predictions, labels, threshold = 0.5)
  cm <- metrics$confusion_matrix
  
  # TP = 1 (pred=0.8, label=1)
  # TN = 1 (pred=0.2, label=0)
  # FP = 1 (pred=0.7, label=0)
  # FN = 1 (pred=0.3, label=1)
  
  expect_equal(cm["0", "0"], 1)  # TN
  expect_equal(cm["1", "1"], 1)  # TP
  expect_equal(sum(cm), 4)
})

test_that("compute_auc_roc calculates AUC correctly", {
  # Perfect separation
  predictions <- c(0.1, 0.2, 0.3, 0.7, 0.8, 0.9)
  labels <- c(0, 0, 0, 1, 1, 1)
  
  auc <- compute_auc_roc(predictions, labels)
  
  expect_equal(auc, 1.0)
})

test_that("compute_auc_roc handles random predictions", {
  set.seed(42)
  predictions <- runif(100)
  labels <- sample(c(0, 1), 100, replace = TRUE)
  
  auc <- compute_auc_roc(predictions, labels)
  
  # Random predictions should give AUC around 0.5
  expect_true(auc >= 0.3 && auc <= 0.7)
})

test_that("compute_roc_curve returns valid curve", {
  predictions <- c(0.1, 0.2, 0.7, 0.8, 0.9)
  labels <- c(0, 0, 1, 1, 1)
  
  roc <- compute_roc_curve(predictions, labels, n_thresholds = 10)
  
  expect_true(!is.null(roc$fpr))
  expect_true(!is.null(roc$tpr))
  expect_true(!is.null(roc$thresholds))
  expect_true(!is.null(roc$auc))
  
  # TPR and FPR should be in [0, 1]
  expect_true(all(roc$fpr >= 0 & roc$fpr <= 1))
  expect_true(all(roc$tpr >= 0 & roc$tpr <= 1))
})

test_that("compute_metrics works with torch tensors", {
  predictions <- torch_tensor(c(0.2, 0.8, 0.3, 0.9))
  labels <- torch_tensor(c(0, 1, 0, 1))
  
  metrics <- compute_metrics(predictions, labels)
  
  expect_true(!is.null(metrics$accuracy))
  expect_true(!is.null(metrics$f1))
})

test_that("compute_metrics handles edge case - all same class", {
  # All predictions for class 1
  predictions <- c(0.9, 0.95, 0.85, 0.92)
  labels <- c(1, 1, 1, 1)
  
  metrics <- compute_metrics(predictions, labels, threshold = 0.5)
  
  # Should handle without error
  expect_equal(metrics$accuracy, 1.0)
})

test_that("compute_metrics handles edge case - no positive predictions", {
  predictions <- c(0.1, 0.2, 0.3, 0.4)
  labels <- c(0, 0, 1, 1)
  
  metrics <- compute_metrics(predictions, labels, threshold = 0.5)
  
  # Should handle without error (precision/recall may be NA or 0)
  expect_true(!is.null(metrics$accuracy))
})

test_that("print_metrics displays output", {
  predictions <- c(0.2, 0.8, 0.3, 0.9)
  labels <- c(0, 1, 0, 1)
  
  metrics <- compute_metrics(predictions, labels)
  
  # Should not error
  expect_output(print_metrics(metrics), "Classification Metrics")
})
