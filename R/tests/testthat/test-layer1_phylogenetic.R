# Test Layer 1: Phylogenetic Focus
# Tests for SpatialAgg and SpatialAggDynamic layers

test_that("SpatialAgg initializes correctly", {
  num_rules <- 3
  num_otus <- 50
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  layer <- spatial_agg(num_rules, num_otus, dist)
  
  # Check parameters exist
  expect_true(!is.null(layer$kappa))
  expect_true(!is.null(layer$eta))
  expect_equal(as.numeric(layer$kappa$shape), c(num_rules, num_otus))
  expect_equal(as.numeric(layer$eta$shape), c(num_rules, num_otus))
})

test_that("SpatialAgg forward pass works", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 50
  num_time <- 5
  
  # Create phylogenetic distances
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  layer <- spatial_agg(num_rules, num_otus, dist)
  
  # Create input
  x <- torch_rand(batch_size, num_otus, num_time)
  
  # Forward pass
  output <- layer(x, temp = 0.5)
  
  # Check output shape
  expect_equal(as.numeric(output$shape), c(batch_size, num_rules, num_time))
})

test_that("SpatialAgg output is in valid range", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 2
  num_otus <- 30
  num_time <- 3
  
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  layer <- spatial_agg(num_rules, num_otus, dist)
  x <- torch_rand(batch_size, num_otus, num_time)
  
  output <- layer(x, temp = 0.5)
  values <- as.numeric(output$cpu())
  
  # Output should be non-negative (weighted sum of abundances)
  expect_true(all(values >= 0))
})

test_that("SpatialAgg is differentiable", {
  set.seed(42)
  num_rules <- 2
  num_otus <- 20
  num_time <- 3
  
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  layer <- spatial_agg(num_rules, num_otus, dist)
  x <- torch_rand(2, num_otus, num_time, requires_grad = TRUE)
  
  output <- layer(x, temp = 0.5)
  loss <- output$sum()
  loss$backward()
  
  # Gradients should exist
  expect_false(is.null(x$grad))
  expect_false(is.null(layer$kappa$grad))
  expect_false(is.null(layer$eta$grad))
})

test_that("SpatialAggDynamic initializes correctly", {
  num_rules <- 3
  num_otus <- 50
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  layer <- spatial_agg_dynamic(num_rules, num_otus, dist)
  
  # Check parameters exist
  expect_true(!is.null(layer$kappa))
  expect_true(!is.null(layer$eta))
})

test_that("SpatialAggDynamic forward pass works", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 50
  num_time <- 5
  
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  layer <- spatial_agg_dynamic(num_rules, num_otus, dist)
  
  # Create input with slopes
  x <- torch_rand(batch_size, num_otus, num_time)
  slopes <- torch_randn(batch_size, num_otus, num_time)
  
  # Forward pass
  output <- layer(list(x, slopes), temp = 0.5)
  
  # Check output shape
  expect_equal(as.numeric(output$shape), c(batch_size, num_rules, num_time))
})

test_that("phylogenetic distance affects aggregation", {
  set.seed(42)
  num_rules <- 1
  num_otus <- 10
  num_time <- 3
  
  # Create distance matrix with clear groups
  dist_matrix <- matrix(1, num_otus, num_otus)
  dist_matrix[1:5, 1:5] <- 0.1  # Close group 1
  dist_matrix[6:10, 6:10] <- 0.1  # Close group 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  layer <- spatial_agg(num_rules, num_otus, dist)
  
  # Initialize kappa to select first OTU strongly
  with_no_grad({
    layer$kappa[1, 1] <- 5.0
    layer$kappa[1, 2:num_otus] <- -5.0
  })
  
  # Create input where first group has high abundance
  x <- torch_zeros(1, num_otus, num_time)
  x[1, 1:5, ] <- 1.0
  
  output <- layer(x, temp = 0.1)
  
  # Output should be positive (aggregating first group)
  expect_true(as.numeric(output$mean()$cpu()) > 0.1)
})
