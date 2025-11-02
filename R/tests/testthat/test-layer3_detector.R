# Test Layer 3: Detector Layers
# Tests for Threshold and Slope detector layers

test_that("Threshold layer initializes correctly", {
  num_rules <- 5
  num_otus <- 20
  num_time_centers <- 1
  
  layer <- threshold_layer(num_rules, num_otus, num_time_centers)
  
  # Check parameters exist
  expect_true(!is.null(layer$thresh))
  expect_equal(as.numeric(layer$thresh$shape), c(num_rules, num_otus))
  expect_equal(layer$layer_name, "threshold")
})

test_that("Threshold layer forward pass works", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 10
  num_time_centers <- 1
  
  layer <- threshold_layer(num_rules, num_otus, num_time_centers)
  
  # Create input (aggregated abundance)
  x <- torch_rand(batch_size, num_rules, num_otus)
  
  # Forward pass
  output <- layer(x, k = 1.0)
  
  # Check output shape
  expect_equal(as.numeric(output$shape), c(batch_size, num_rules, num_otus))
})

test_that("Threshold layer output is in [0,1] range", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 10
  
  layer <- threshold_layer(num_rules, num_otus, 1)
  
  # Create input
  x <- torch_randn(batch_size, num_rules, num_otus)
  
  # Forward pass
  output <- layer(x, k = 1.0)
  
  # Check range (should be sigmoid output)
  expect_true(all(as.array(output) >= 0))
  expect_true(all(as.array(output) <= 1))
})

test_that("Threshold layer sharpness parameter k works", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 2
  num_otus <- 5
  
  layer <- threshold_layer(num_rules, num_otus, 1)
  
  # Set a specific threshold
  layer$thresh$data <- torch_zeros_like(layer$thresh)
  
  # Create input centered around zero
  x <- torch_tensor(array(c(-1, 0, 1), dim = c(1, 1, 3)))$expand(c(batch_size, num_rules, 3))
  
  # Test with different k values
  output_soft <- layer(x, k = 1.0)
  output_sharp <- layer(x, k = 10.0)
  
  # Sharp version should be closer to 0 or 1
  expect_true(as.numeric(torch_mean(torch_abs(output_sharp - 0.5))) > 
              as.numeric(torch_mean(torch_abs(output_soft - 0.5))))
})

test_that("Threshold layer init_params works", {
  num_rules <- 3
  num_otus <- 5
  
  layer <- threshold_layer(num_rules, num_otus, 1)
  
  # Set custom initial values
  thresh_init <- matrix(runif(num_rules * num_otus), num_rules, num_otus)
  layer$init_params(list(thresh_init = thresh_init))
  
  # Check values were set
  expect_equal(as.array(layer$thresh$cpu()), thresh_init, tolerance = 1e-5)
})

test_that("Threshold layer get_params and set_params work", {
  num_rules <- 3
  num_otus <- 5
  
  layer <- threshold_layer(num_rules, num_otus, 1)
  
  # Get parameters
  params <- layer$get_params()
  expect_true(!is.null(params$thresh))
  expect_equal(as.numeric(params$thresh$shape), c(num_rules, num_otus))
  
  # Modify and set parameters
  new_thresh <- torch_ones_like(params$thresh) * 0.5
  layer$set_params(list(thresh = new_thresh))
  
  # Check parameters were updated
  expect_equal(as.array(layer$thresh$cpu()), as.array(new_thresh$cpu()), tolerance = 1e-5)
})

test_that("Slope layer initializes correctly", {
  num_rules <- 5
  num_otus <- 20
  num_time_centers <- 1
  
  layer <- slope_layer(num_rules, num_otus, num_time_centers)
  
  # Check parameters exist
  expect_true(!is.null(layer$thresh_slope))
  expect_equal(as.numeric(layer$thresh_slope$shape), c(num_rules, num_otus))
  expect_equal(layer$layer_name, "slope")
})

test_that("Slope layer forward pass works", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 10
  num_time_centers <- 1
  
  layer <- slope_layer(num_rules, num_otus, num_time_centers)
  
  # Create input (slope values can be negative)
  x <- torch_randn(batch_size, num_rules, num_otus)
  
  # Forward pass
  output <- layer(x, k = 1.0)
  
  # Check output shape
  expect_equal(as.numeric(output$shape), c(batch_size, num_rules, num_otus))
})

test_that("Slope layer output is in [0,1] range", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 10
  
  layer <- slope_layer(num_rules, num_otus, 1)
  
  # Create input with both positive and negative slopes
  x <- torch_randn(batch_size, num_rules, num_otus) * 2
  
  # Forward pass
  output <- layer(x, k = 1.0)
  
  # Check range (should be sigmoid output)
  expect_true(all(as.array(output) >= 0))
  expect_true(all(as.array(output) <= 1))
})

test_that("Slope layer detects positive and negative slopes", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 1
  num_otus <- 3
  
  layer <- slope_layer(num_rules, num_otus, 1)
  
  # Set threshold to zero
  layer$thresh_slope$data <- torch_zeros_like(layer$thresh_slope)
  
  # Create input with negative, zero, and positive slopes
  x <- torch_tensor(array(c(-1, 0, 1), dim = c(batch_size, num_rules, 3)))
  
  # Forward pass with high sharpness
  output <- layer(x, k = 10.0)
  
  # With threshold at 0, negative slopes should give ~0, positive ~1
  output_array <- as.array(output$cpu())
  expect_true(output_array[1, 1, 1] < 0.1)  # negative slope
  expect_true(output_array[1, 1, 2] > 0.4 && output_array[1, 1, 2] < 0.6)  # zero slope
  expect_true(output_array[1, 1, 3] > 0.9)  # positive slope
})

test_that("Slope layer get_params and set_params work", {
  num_rules <- 3
  num_otus <- 5
  
  layer <- slope_layer(num_rules, num_otus, 1)
  
  # Get parameters
  params <- layer$get_params()
  expect_true(!is.null(params$thresh_slope))
  expect_equal(as.numeric(params$thresh_slope$shape), c(num_rules, num_otus))
  
  # Modify and set parameters
  new_thresh <- torch_ones_like(params$thresh_slope) * -0.5
  layer$set_params(list(thresh_slope = new_thresh))
  
  # Check parameters were updated
  expect_equal(as.array(layer$thresh_slope$cpu()), 
               as.array(new_thresh$cpu()), tolerance = 1e-5)
})
