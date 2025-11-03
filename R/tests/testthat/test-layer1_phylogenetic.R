# Test Layer 1: Phylogenetic Focus
# Tests for SpatialAgg and SpatialAggDynamic layers

test_that("SpatialAgg initializes correctly", {
  num_rules <- 3
  num_otus <- 50
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  
  layer <- spatial_agg_layer(num_rules, num_otus, dist_matrix)
  
  # Check parameters exist (only kappa, not eta - that's in SpatialAggDynamic)
  expect_true(!is.null(layer$kappa))
  expect_equal(as.numeric(layer$kappa$shape), c(num_rules, num_otus))
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
  
  layer <- spatial_agg_layer(num_rules, num_otus, dist_matrix)
  
  # Create input
  x <- torch_rand(batch_size, num_otus, num_time)
  
  # Forward pass (use k instead of temp)
  output <- layer(x, k = 0.5)
  
  # Check output shape (includes num_otus dimension)
  expect_equal(as.numeric(output$shape), c(batch_size, num_rules, num_otus, num_time))
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
  
  layer <- spatial_agg_layer(num_rules, num_otus, dist_matrix)
  x <- torch_rand(batch_size, num_otus, num_time)
  
  output <- layer(x, k = 0.5)
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
  
  layer <- spatial_agg_layer(num_rules, num_otus, dist_matrix)
  x <- torch_rand(2, num_otus, num_time, requires_grad = TRUE)
  
  output <- layer(x, k = 0.5)
  loss <- output$sum()
  loss$backward()
  
  # Gradients should exist (only kappa, not eta - that's in SpatialAggDynamic)
  expect_false(is.null(x$grad))
  expect_false(is.null(layer$kappa$grad))
})

test_that("SpatialAggDynamic initializes correctly", {
  num_rules <- 3
  num_otus <- 50
  num_otu_centers <- 5
  emb_dim <- 10
  
  # Create OTU embeddings
  otu_embeddings <- matrix(rnorm(num_otus * emb_dim), nrow = num_otus, ncol = emb_dim)
  
  layer <- spatial_agg_dynamic_layer(num_rules, num_otu_centers, otu_embeddings, emb_dim, num_otus)
  
  # Check parameters exist
  expect_true(!is.null(layer$kappa))
  expect_true(!is.null(layer$eta))
  expect_equal(as.numeric(layer$eta$shape), c(num_rules, num_otu_centers, emb_dim))
})

test_that("SpatialAggDynamic forward pass works", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 50
  num_otu_centers <- 5
  emb_dim <- 10
  num_time <- 5
  
  # Create OTU embeddings
  otu_embeddings <- matrix(rnorm(num_otus * emb_dim), nrow = num_otus, ncol = emb_dim)
  
  layer <- spatial_agg_dynamic_layer(num_rules, num_otu_centers, otu_embeddings, emb_dim, num_otus)
  
  # Create input
  x <- torch_rand(batch_size, num_otus, num_time)
  
  # Forward pass (use k instead of temp)
  output <- layer(x, k = 0.5)
  
  # Check output shape (includes num_otu_centers dimension)
  expect_equal(as.numeric(output$shape), c(batch_size, num_rules, num_otu_centers, num_time))
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
  
  layer <- spatial_agg_layer(num_rules, num_otus, dist_matrix)
  
  # Initialize kappa to select first OTU strongly
  with_no_grad({
    layer$kappa[1, 1] <- 5.0
    layer$kappa[1, 2:num_otus] <- -5.0
  })
  
  # Create input where first group has high abundance
  x <- torch_zeros(1, num_otus, num_time)
  x[1, 1:5, ] <- 1.0
  
  output <- layer(x, k = 0.1)
  
  # Output should be positive (aggregating first group)
  expect_true(as.numeric(output$mean()$cpu()) > 0.1)
})
