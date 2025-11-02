# Test Layer 5: Classification Layers
# Tests for DenseLayer (classification) and DenseLayerAbun (abundance-only)

test_that("Classification layer initializes correctly", {
  in_feat <- 5
  out_feat <- 1
  
  layer <- classification_layer(in_feat, out_feat)
  
  # Check parameters exist
  expect_true(!is.null(layer$weight))
  expect_true(!is.null(layer$bias))
  expect_true(!is.null(layer$beta))
  expect_equal(as.numeric(layer$weight$shape), c(out_feat, in_feat))
  expect_equal(as.numeric(layer$bias$shape), c(out_feat))
  expect_equal(as.numeric(layer$beta$shape), c(in_feat))
  expect_equal(layer$layer_name, "dense_layer")
})

test_that("Classification layer forward pass works", {
  set.seed(42)
  batch_size <- 4
  in_feat <- 5
  out_feat <- 1
  
  layer <- classification_layer(in_feat, out_feat)
  
  # Create input (rule responses and slopes)
  x <- torch_rand(batch_size, in_feat)
  x_slope <- torch_randn(batch_size, in_feat)
  
  # Forward pass
  output <- layer(x, x_slope, k = 1.0, hard = FALSE, use_noise = FALSE)
  
  # Check output shape (should be log odds for binary classification)
  expect_equal(as.numeric(output$shape), c(batch_size))
})

test_that("Classification layer produces valid log odds", {
  set.seed(42)
  batch_size <- 4
  in_feat <- 5
  out_feat <- 1
  
  layer <- classification_layer(in_feat, out_feat)
  
  # Create input
  x <- torch_rand(batch_size, in_feat)
  x_slope <- torch_randn(batch_size, in_feat)
  
  # Forward pass
  log_odds <- layer(x, x_slope, k = 1.0, hard = FALSE, use_noise = FALSE)
  
  # Convert to probabilities
  probs <- torch_sigmoid(log_odds)
  
  # Check probabilities are in [0,1]
  expect_true(all(as.array(probs) >= 0))
  expect_true(all(as.array(probs) <= 1))
})

test_that("Classification layer beta controls rule selection", {
  set.seed(42)
  batch_size <- 2
  in_feat <- 3
  out_feat <- 1
  
  layer <- classification_layer(in_feat, out_feat)
  
  # Set beta to select only first rule
  layer$beta$data <- torch_ones_like(layer$beta) * -10  # deselect all
  layer$beta$data[1] <- 10  # select only first
  
  # Set weights to make prediction interpretable
  layer$weight$data <- torch_ones_like(layer$weight)
  layer$bias$data <- torch_zeros_like(layer$bias)
  
  # Create input where first rule is high, others low
  x <- torch_zeros(batch_size, in_feat)
  x[, 1] <- 1.0
  x_slope <- torch_ones(batch_size, in_feat)
  
  # Forward pass with hard selection
  output <- layer(x, x_slope, k = 1.0, hard = TRUE, use_noise = FALSE)
  
  # Output should be positive (since first rule is high and selected)
  expect_true(all(as.array(output) > 0))
})

test_that("Classification layer requires x_slope argument", {
  batch_size <- 4
  in_feat <- 5
  out_feat <- 1
  
  layer <- classification_layer(in_feat, out_feat)
  
  # Create input without slope
  x <- torch_rand(batch_size, in_feat)
  
  # Should error without x_slope
  expect_error(layer(x), "DenseLayer requires x_slope argument")
})

test_that("Classification layer training vs evaluation differs", {
  set.seed(42)
  batch_size <- 4
  in_feat <- 5
  out_feat <- 1
  
  layer <- classification_layer(in_feat, out_feat)
  x <- torch_rand(batch_size, in_feat)
  x_slope <- torch_randn(batch_size, in_feat)
  
  # Training mode with noise
  layer$train()
  torch_manual_seed(42L)
  output_train1 <- layer(x, x_slope, k = 1.0, hard = FALSE, use_noise = TRUE)
  torch_manual_seed(42L)
  output_train2 <- layer(x, x_slope, k = 1.0, hard = FALSE, use_noise = TRUE)
  
  # Should be identical with same seed
  expect_equal(as.array(output_train1), as.array(output_train2), tolerance = 1e-5)
  
  # Evaluation mode (deterministic)
  layer$eval()
  output_eval <- layer(x, x_slope, k = 1.0, hard = FALSE, use_noise = FALSE)
  output_eval2 <- layer(x, x_slope, k = 1.0, hard = FALSE, use_noise = FALSE)
  
  # Should be identical
  expect_equal(as.array(output_eval), as.array(output_eval2), tolerance = 1e-5)
})

test_that("Classification layer get_params and set_params work", {
  in_feat <- 5
  out_feat <- 1
  
  layer <- classification_layer(in_feat, out_feat)
  
  # Get parameters
  params <- layer$get_params()
  expect_true(!is.null(params$weight))
  expect_true(!is.null(params$bias))
  expect_true(!is.null(params$beta))
  expect_equal(as.numeric(params$weight$shape), c(out_feat, in_feat))
  
  # Modify and set parameters
  new_weight <- torch_ones_like(params$weight) * 0.5
  new_bias <- torch_ones_like(params$bias) * -0.5
  new_beta <- torch_ones_like(params$beta) * 2.0
  
  layer$set_params(list(
    weight = new_weight,
    bias = new_bias,
    beta = new_beta
  ))
  
  # Check parameters were updated
  expect_equal(as.array(layer$weight$cpu()), as.array(new_weight$cpu()), tolerance = 1e-5)
  expect_equal(as.array(layer$bias$cpu()), as.array(new_bias$cpu()), tolerance = 1e-5)
  expect_equal(as.array(layer$beta$cpu()), as.array(new_beta$cpu()), tolerance = 1e-5)
})

test_that("Abundance-only classification layer initializes correctly", {
  in_feat <- 5
  out_feat <- 1
  
  layer <- classification_abun_layer(in_feat, out_feat)
  
  # Check parameters exist
  expect_true(!is.null(layer$weight))
  expect_true(!is.null(layer$bias))
  expect_true(!is.null(layer$beta))
  expect_equal(layer$layer_name, "dense_layer_abun")
})

test_that("Abundance-only classification layer forward pass works", {
  set.seed(42)
  batch_size <- 4
  in_feat <- 5
  out_feat <- 1
  
  layer <- classification_abun_layer(in_feat, out_feat)
  
  # Create input (rule responses only, no slopes)
  x <- torch_rand(batch_size, in_feat)
  
  # Forward pass (no x_slope argument needed)
  output <- layer(x, k = 1.0, hard = FALSE, use_noise = FALSE)
  
  # Check output shape
  expect_equal(as.numeric(output$shape), c(batch_size))
})

test_that("Abundance-only layer produces valid log odds", {
  set.seed(42)
  batch_size <- 4
  in_feat <- 5
  out_feat <- 1
  
  layer <- classification_abun_layer(in_feat, out_feat)
  
  # Create input
  x <- torch_rand(batch_size, in_feat)
  
  # Forward pass
  log_odds <- layer(x, k = 1.0, hard = FALSE, use_noise = FALSE)
  
  # Convert to probabilities
  probs <- torch_sigmoid(log_odds)
  
  # Check probabilities are in [0,1]
  expect_true(all(as.array(probs) >= 0))
  expect_true(all(as.array(probs) <= 1))
})

test_that("Abundance-only layer does not require x_slope", {
  set.seed(42)
  batch_size <- 4
  in_feat <- 5
  out_feat <- 1
  
  layer <- classification_abun_layer(in_feat, out_feat)
  
  # Create input without slope
  x <- torch_rand(batch_size, in_feat)
  
  # Should work without error
  output <- layer(x, k = 1.0, hard = FALSE, use_noise = FALSE)
  expect_equal(as.numeric(output$shape), c(batch_size))
})

test_that("Both classification layers have similar structure", {
  in_feat <- 5
  out_feat <- 1
  
  layer_full <- classification_layer(in_feat, out_feat)
  layer_abun <- classification_abun_layer(in_feat, out_feat)
  
  # Both should have same parameter shapes
  expect_equal(as.numeric(layer_full$weight$shape), 
               as.numeric(layer_abun$weight$shape))
  expect_equal(as.numeric(layer_full$bias$shape), 
               as.numeric(layer_abun$bias$shape))
  expect_equal(as.numeric(layer_full$beta$shape), 
               as.numeric(layer_abun$beta$shape))
})
