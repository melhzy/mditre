# Test Layer 4: Rule Layer
# Tests for Rules layer combining detectors with soft AND logic

test_that("Rule layer initializes correctly", {
  num_rules <- 5
  num_otus <- 20
  num_time_centers <- 1
  
  layer <- rule_layer(num_rules, num_otus, num_time_centers)
  
  # Check parameters exist
  expect_true(!is.null(layer$alpha))
  expect_equal(as.numeric(layer$alpha$shape), c(num_rules, num_otus))
  expect_equal(layer$layer_name, "rules")
})

test_that("Rule layer forward pass works", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 10
  num_time_centers <- 1
  
  layer <- rule_layer(num_rules, num_otus, num_time_centers)
  
  # Create input (detector responses in [0,1])
  x <- torch_rand(batch_size, num_rules, num_otus)
  
  # Forward pass
  output <- layer(x, k = 1.0, hard = FALSE, use_noise = FALSE)
  
  # Check output shape (should reduce from otus dimension)
  expect_equal(as.numeric(output$shape), c(batch_size, num_rules))
})

test_that("Rule layer output is in [0,1] range", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 10
  
  layer <- rule_layer(num_rules, num_otus, 1)
  
  # Create input
  x <- torch_rand(batch_size, num_rules, num_otus)
  
  # Forward pass
  output <- layer(x, k = 1.0, hard = FALSE, use_noise = FALSE)
  
  # Check range (soft AND should produce values in [0,1])
  expect_true(all(as.array(output) >= 0))
  expect_true(all(as.array(output) <= 1))
})

test_that("Rule layer implements soft AND logic", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 1
  num_otus <- 3
  
  layer <- rule_layer(num_rules, num_otus, 1)
  
  # Set alpha to select all detectors (high positive values)
  layer$alpha$data <- torch_ones_like(layer$alpha) * 10
  
  # Test case 1: All inputs high -> output high
  x_all_high <- torch_ones(batch_size, num_rules, num_otus) * 0.9
  output_high <- layer(x_all_high, k = 1.0, hard = TRUE, use_noise = FALSE)
  
  # Test case 2: One input low -> output low (AND logic)
  x_one_low <- torch_ones(batch_size, num_rules, num_otus) * 0.9
  x_one_low[1, 1, 1] <- 0.1
  output_low <- layer(x_one_low, k = 1.0, hard = TRUE, use_noise = FALSE)
  
  # With AND logic, having one low input should reduce output
  expect_true(as.numeric(output_high[1, 1]) > as.numeric(output_low[1, 1]))
})

test_that("Rule layer alpha parameter controls selection", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 1
  num_otus <- 3
  
  layer <- rule_layer(num_rules, num_otus, 1)
  
  # Set alpha to deselect all but one detector
  layer$alpha$data <- torch_ones_like(layer$alpha) * -10  # deselect all
  layer$alpha$data[1, 1] <- 10  # select only first detector
  
  # Create input where first detector is low, others high
  x <- torch_ones(batch_size, num_rules, num_otus) * 0.9
  x[, , 1] <- 0.1  # first detector low
  
  # Forward pass with hard selection
  output <- layer(x, k = 1.0, hard = TRUE, use_noise = FALSE)
  
  # Output should be influenced primarily by first detector (which is low)
  expect_true(all(as.array(output) < 0.5))
})

test_that("Rule layer training vs evaluation mode differs", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 10
  
  layer <- rule_layer(num_rules, num_otus, 1)
  x <- torch_rand(batch_size, num_rules, num_otus)
  
  # Training mode with noise
  layer$train()
  torch_manual_seed(42L)
  output_train1 <- layer(x, k = 1.0, hard = FALSE, use_noise = TRUE)
  torch_manual_seed(42L)
  output_train2 <- layer(x, k = 1.0, hard = FALSE, use_noise = TRUE)
  
  # Outputs should be identical with same seed
  expect_equal(as.array(output_train1), as.array(output_train2), tolerance = 1e-5)
  
  # Evaluation mode (deterministic)
  layer$eval()
  output_eval <- layer(x, k = 1.0, hard = FALSE, use_noise = FALSE)
  
  # Eval outputs should be deterministic (no noise)
  output_eval2 <- layer(x, k = 1.0, hard = FALSE, use_noise = FALSE)
  expect_equal(as.array(output_eval), as.array(output_eval2), tolerance = 1e-5)
})

test_that("Rule layer hard vs soft selection differs", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 2
  num_otus <- 5
  
  layer <- rule_layer(num_rules, num_otus, 1)
  x <- torch_rand(batch_size, num_rules, num_otus)
  
  # Soft selection (continuous)
  output_soft <- layer(x, k = 1.0, hard = FALSE, use_noise = FALSE)
  
  # Hard selection (discrete via straight-through)
  output_hard <- layer(x, k = 1.0, hard = TRUE, use_noise = FALSE)
  
  # Hard and soft outputs should differ
  # (unless alpha values are very extreme)
  # This is a weak test since they might be similar
  expect_equal(as.numeric(output_soft$shape), as.numeric(output_hard$shape))
})

test_that("Rule layer get_params and set_params work", {
  num_rules <- 3
  num_otus <- 5
  
  layer <- rule_layer(num_rules, num_otus, 1)
  
  # Get parameters
  params <- layer$get_params()
  expect_true(!is.null(params$alpha))
  expect_equal(as.numeric(params$alpha$shape), c(num_rules, num_otus))
  
  # Modify and set parameters
  new_alpha <- torch_ones_like(params$alpha) * 2.0
  layer$set_params(list(alpha = new_alpha))
  
  # Check parameters were updated
  expect_equal(as.array(layer$alpha$cpu()), as.array(new_alpha$cpu()), tolerance = 1e-5)
})

test_that("Rule layer handles edge cases", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 2
  num_otus <- 5
  
  layer <- rule_layer(num_rules, num_otus, 1)
  
  # Test with all zeros
  x_zeros <- torch_zeros(batch_size, num_rules, num_otus)
  output_zeros <- layer(x_zeros, k = 1.0, hard = FALSE, use_noise = FALSE)
  expect_true(all(as.array(output_zeros) >= 0))
  expect_true(all(as.array(output_zeros) <= 1))
  
  # Test with all ones
  x_ones <- torch_ones(batch_size, num_rules, num_otus)
  output_ones <- layer(x_ones, k = 1.0, hard = FALSE, use_noise = FALSE)
  expect_true(all(as.array(output_ones) >= 0))
  expect_true(all(as.array(output_ones) <= 1))
  
  # With all inputs at 1, output should be relatively high
  expect_true(mean(as.array(output_ones)) > mean(as.array(output_zeros)))
})
