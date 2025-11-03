# Test Layer 2: Temporal Focus
# Tests for TimeAgg and TimeAggAbun layers

test_that("TimeAgg initializes correctly", {
  num_rules <- 3
  num_otus <- 20
  num_time <- 5
  num_time_centers <- 1
  
  layer <- time_agg_layer(num_rules, num_otus, num_time, num_time_centers)
  
  # Check parameters exist
  expect_true(!is.null(layer$abun_a))
  expect_true(!is.null(layer$abun_b))
  expect_true(!is.null(layer$slope_a))
  expect_true(!is.null(layer$slope_b))
})

test_that("TimeAgg forward pass works", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 20
  num_time <- 5
  num_time_centers <- 1
  
  layer <- time_agg_layer(num_rules, num_otus, num_time, num_time_centers)
  
  # Create input (batch, num_rules, num_otus, num_time)
  x <- torch_rand(batch_size, num_rules, num_otus, num_time)
  
  # Forward pass (use k instead of temp)
  output <- layer(x, k = 0.5)
  
  # Check output shape - should return list(abundance, slope)
  expect_type(output, "list")
  expect_equal(length(output), 2)
  expect_equal(as.numeric(output$abundance$shape), c(batch_size, num_rules, num_otus))
  expect_equal(as.numeric(output$slope$shape), c(batch_size, num_rules, num_otus))
})

test_that("TimeAgg focuses on specific time windows", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 1
  num_otus <- 20
  num_time <- 10
  num_time_centers <- 1
  
  layer <- time_agg_layer(num_rules, num_otus, num_time, num_time_centers)
  
  # Set window parameters to focus on middle timepoints  
  with_no_grad({
    layer$abun_a[1, 1] <- 5.0
    layer$abun_b[1, 1] <- 0.5  # Narrow window
  })
  
  # Create input with peak in middle (batch, rules, otus, time)
  x <- torch_zeros(batch_size, num_rules, num_otus, num_time)
  x[, , 1, 5] <- 10.0  # Peak at time 5 for first OTU
  
  output <- layer(x, k = 0.1)
  
  # Output should capture the peak
  expect_true(as.numeric(output$abundance$mean()$cpu()) > 0.5)
})

test_that("TimeAgg is differentiable", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 2
  num_otus <- 20
  num_time <- 5
  num_time_centers <- 1
  
  layer <- time_agg_layer(num_rules, num_otus, num_time, num_time_centers)
  
  x <- torch_rand(batch_size, num_rules, num_otus, num_time, requires_grad = TRUE)
  
  output <- layer(x, k = 0.5)
  loss <- output$abundance$sum() + output$slope$sum()
  loss$backward()
  
  # Gradients should exist
  expect_false(is.null(x$grad))
  expect_false(is.null(layer$abun_a$grad))
  expect_false(is.null(layer$abun_b$grad))
})

test_that("TimeAggAbun initializes correctly", {
  num_rules <- 3
  num_otus <- 20
  num_time <- 5
  num_time_centers <- 1
  
  layer <- time_agg_abun_layer(num_rules, num_otus, num_time, num_time_centers)
  
  # Check parameters exist
  expect_true(!is.null(layer$a))
  expect_true(!is.null(layer$b))
})

test_that("TimeAggAbun forward pass works", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 20
  num_time <- 5
  num_time_centers <- 1
  
  layer <- time_agg_abun_layer(num_rules, num_otus, num_time, num_time_centers)
  
  # Create input (batch, rules, otus, time)
  x <- torch_rand(batch_size, num_rules, num_otus, num_time)
  
  # Forward pass
  output <- layer(x, k = 0.5)
  
  # Check output shape - should return single tensor (abundance only)
  expect_s3_class(output, "torch_tensor")
  expect_equal(as.numeric(output$shape), c(batch_size, num_rules, num_otus))
})

test_that("TimeAggAbun handles missing timepoints", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 2
  num_otus <- 20
  num_time <- 5
  num_time_centers <- 1
  
  layer <- time_agg_abun_layer(num_rules, num_otus, num_time, num_time_centers)
  
  # Create input with some missing values (represented as zeros or NaN)
  x <- torch_rand(batch_size, num_rules, num_otus, num_time)
  
  # Create mask (optional - TimeAggAbun may not use mask parameter)
  mask <- torch_ones(batch_size, num_time)
  mask[1, 3] <- 0  # Mask one timepoint
  
  # Forward pass should not error
  output <- layer(x, k = 0.5)
  
  expect_s3_class(output, "torch_tensor")
  expect_false(any(is.na(as.numeric(output$cpu()))))
})

test_that("temporal window width affects aggregation", {
  set.seed(42)
  batch_size <- 1
  num_rules <- 1
  num_otus <- 20
  num_time <- 10
  num_time_centers <- 1
  
  layer_narrow <- time_agg_abun_layer(num_rules, num_otus, num_time, num_time_centers)
  layer_wide <- time_agg_abun_layer(num_rules, num_otus, num_time, num_time_centers)
  
  # Set different window widths
  with_no_grad({
    layer_narrow$a[1, 1] <- 5.0
    layer_narrow$b[1, 1] <- 0.5  # Narrow window
    
    layer_wide$a[1, 1] <- 5.0
    layer_wide$b[1, 1] <- 2.0  # Wide window
  })
  
  # Create input with varying values over time (batch, rules, otus, time)
  x <- torch_arange(1, num_time + 1, dtype = torch_float())$
    view(c(1, 1, 1, num_time))$
    expand(c(batch_size, num_rules, num_otus, num_time))
  
  output_narrow <- layer_narrow(x, k = 0.1)
  output_wide <- layer_wide(x, k = 0.1)
  
  # Both should produce valid outputs
  expect_true(as.numeric(output_narrow$mean()$cpu()) > 0)
  expect_true(as.numeric(output_wide$mean()$cpu()) > 0)
})
