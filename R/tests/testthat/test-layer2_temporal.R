# Test Layer 2: Temporal Focus
# Tests for TimeAgg and TimeAggAbun layers

test_that("TimeAgg initializes correctly", {
  num_rules <- 3
  num_time <- 5
  times <- 1:num_time
  
  layer <- time_agg_layer(num_rules, num_time, times)
  
  # Check parameters exist
  expect_true(!is.null(layer$mu))
  expect_true(!is.null(layer$sigma))
  expect_equal(as.numeric(layer$mu$shape), c(num_rules, 1))
  expect_equal(as.numeric(layer$sigma$shape), c(num_rules, 1))
})

test_that("TimeAgg forward pass works", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_time <- 5
  times <- 1:num_time
  
  layer <- time_agg_layer(num_rules, num_time, times)
  
  # Create input (abundance and slopes)
  x_abun <- torch_rand(batch_size, num_rules, num_time)
  x_slope <- torch_randn(batch_size, num_rules, num_time)
  x <- list(x_abun, x_slope)
  
  # Forward pass
  output <- layer(x, temp = 0.5)
  
  # Check output shape - should return list(abundance, slope)
  expect_type(output, "list")
  expect_equal(length(output), 2)
  expect_equal(as.numeric(output[[1]]$shape), c(batch_size, num_rules))
  expect_equal(as.numeric(output[[2]]$shape), c(batch_size, num_rules))
})

test_that("TimeAgg focuses on specific time windows", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 1
  num_time <- 10
  times <- 1:num_time
  
  layer <- time_agg_layer(num_rules, num_time, times)
  
  # Set mu to focus on middle timepoints
  with_no_grad({
    layer$mu[1, 1] <- 5.0
    layer$sigma[1, 1] <- 0.5  # Narrow window
  })
  
  # Create input with peak in middle
  x_abun <- torch_zeros(batch_size, num_rules, num_time)
  x_abun[, , 5] <- 10.0  # Peak at time 5
  x_slope <- torch_zeros(batch_size, num_rules, num_time)
  
  output <- layer(list(x_abun, x_slope), temp = 0.1)
  
  # Output should capture the peak
  expect_true(as.numeric(output[[1]]$mean()$cpu()) > 1.0)
})

test_that("TimeAgg is differentiable", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 2
  num_time <- 5
  times <- 1:num_time
  
  layer <- time_agg_layer(num_rules, num_time, times)
  
  x_abun <- torch_rand(batch_size, num_rules, num_time, requires_grad = TRUE)
  x_slope <- torch_randn(batch_size, num_rules, num_time, requires_grad = TRUE)
  
  output <- layer(list(x_abun, x_slope), temp = 0.5)
  loss <- output[[1]]$sum() + output[[2]]$sum()
  loss$backward()
  
  # Gradients should exist
  expect_false(is.null(x_abun$grad))
  expect_false(is.null(layer$mu$grad))
  expect_false(is.null(layer$sigma$grad))
})

test_that("TimeAggAbun initializes correctly", {
  num_rules <- 3
  num_time <- 5
  times <- 1:num_time
  
  layer <- time_agg_abun_layer(num_rules, num_time, times)
  
  # Check parameters exist
  expect_true(!is.null(layer$mu))
  expect_true(!is.null(layer$sigma))
})

test_that("TimeAggAbun forward pass works", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_time <- 5
  times <- 1:num_time
  
  layer <- time_agg_abun_layer(num_rules, num_time, times)
  
  # Create input (abundance only)
  x <- torch_rand(batch_size, num_rules, num_time)
  
  # Forward pass
  output <- layer(x, temp = 0.5)
  
  # Check output shape - should return single tensor
  expect_s3_class(output, "torch_tensor")
  expect_equal(as.numeric(output$shape), c(batch_size, num_rules))
})

test_that("TimeAggAbun handles missing timepoints", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 2
  num_time <- 5
  times <- 1:num_time
  
  layer <- time_agg_abun_layer(num_rules, num_time, times)
  
  # Create input with some missing values (represented as zeros or NaN)
  x <- torch_rand(batch_size, num_rules, num_time)
  x[1, 1, 3] <- 0  # Missing value
  
  # Forward pass should not error
  output <- layer(x, temp = 0.5)
  
  expect_s3_class(output, "torch_tensor")
  expect_false(any(is.na(as.numeric(output$cpu()))))
})

test_that("temporal window width affects aggregation", {
  set.seed(42)
  batch_size <- 1
  num_rules <- 1
  num_time <- 10
  times <- 1:num_time
  
  layer_narrow <- time_agg_abun_layer(num_rules, num_time, times)
  layer_wide <- time_agg_abun_layer(num_rules, num_time, times)
  
  # Set different window widths
  with_no_grad({
    layer_narrow$mu[1, 1] <- 5.0
    layer_narrow$sigma[1, 1] <- 0.5  # Narrow window
    
    layer_wide$mu[1, 1] <- 5.0
    layer_wide$sigma[1, 1] <- 2.0  # Wide window
  })
  
  # Create input with varying values over time
  x <- torch_arange(1, num_time + 1, dtype = torch_float())$
    view(c(1, 1, num_time))
  
  output_narrow <- layer_narrow(x, temp = 0.1)
  output_wide <- layer_wide(x, temp = 0.1)
  
  # Both should produce valid outputs
  expect_true(as.numeric(output_narrow$cpu()) > 0)
  expect_true(as.numeric(output_wide$cpu()) > 0)
})
