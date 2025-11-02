# Test Complete Models
# Tests for MDITRE and MDITREAbun models

test_that("MDITRE model initializes correctly", {
  num_rules <- 3
  num_otus <- 50
  num_time <- 5
  
  # Create phylogenetic distances
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  times <- 1:num_time
  
  model <- mditre_model(num_rules, num_otus, num_time, dist, times)
  
  # Check all layers exist
  expect_true(!is.null(model$phylo_focus))
  expect_true(!is.null(model$time_focus))
  expect_true(!is.null(model$threshold_detector))
  expect_true(!is.null(model$slope_detector))
  expect_true(!is.null(model$rules))
  expect_true(!is.null(model$classifier))
})

test_that("MDITRE forward pass works", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 50
  num_time <- 5
  
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  times <- 1:num_time
  
  model <- mditre_model(num_rules, num_otus, num_time, dist, times)
  
  # Create input
  x <- torch_rand(batch_size, num_otus, num_time)
  slopes <- torch_randn(batch_size, num_otus, num_time)
  
  # Forward pass
  output <- model(list(x, slopes))
  
  # Check output shape (binary classification)
  expect_equal(as.numeric(output$shape), c(batch_size, 2))
})

test_that("MDITRE output is valid probability", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 2
  num_otus <- 30
  num_time <- 3
  
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  model <- mditre_model(num_rules, num_otus, num_time, dist, 1:num_time)
  
  x <- torch_rand(batch_size, num_otus, num_time)
  slopes <- torch_randn(batch_size, num_otus, num_time)
  
  logits <- model(list(x, slopes))
  probs <- nnf_softmax(logits, dim = 2)
  
  # Probabilities should sum to 1
  prob_sums <- probs$sum(dim = 2)
  expect_true(all(abs(as.numeric(prob_sums$cpu()) - 1.0) < 1e-5))
})

test_that("MDITRE is differentiable end-to-end", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 2
  num_otus <- 20
  num_time <- 3
  
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  model <- mditre_model(num_rules, num_otus, num_time, dist, 1:num_time)
  
  x <- torch_rand(batch_size, num_otus, num_time, requires_grad = TRUE)
  slopes <- torch_randn(batch_size, num_otus, num_time, requires_grad = TRUE)
  
  output <- model(list(x, slopes))
  loss <- output$sum()
  loss$backward()
  
  # Gradients should exist for inputs
  expect_false(is.null(x$grad))
  expect_false(is.null(slopes$grad))
  
  # Gradients should exist for model parameters
  expect_false(is.null(model$phylo_focus$kappa$grad))
  expect_false(is.null(model$classifier$fc$weight$grad))
})

test_that("MDITREAbun model initializes correctly", {
  num_rules <- 3
  num_otus <- 50
  num_time <- 5
  
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  model <- mditre_abun_model(num_rules, num_otus, num_time, dist, 1:num_time)
  
  # Check key layers exist
  expect_true(!is.null(model$phylo_focus))
  expect_true(!is.null(model$time_focus))
  expect_true(!is.null(model$threshold_detector))
  expect_true(!is.null(model$rules))
  expect_true(!is.null(model$classifier))
})

test_that("MDITREAbun forward pass works", {
  set.seed(42)
  batch_size <- 4
  num_rules <- 3
  num_otus <- 50
  num_time <- 5
  
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  model <- mditre_abun_model(num_rules, num_otus, num_time, dist, 1:num_time)
  
  # Create input (abundance only, no slopes)
  x <- torch_rand(batch_size, num_otus, num_time)
  
  # Forward pass
  output <- model(x)
  
  # Check output shape
  expect_equal(as.numeric(output$shape), c(batch_size, 2))
})

test_that("MDITREAbun is differentiable", {
  set.seed(42)
  batch_size <- 2
  num_rules <- 2
  num_otus <- 20
  num_time <- 3
  
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  model <- mditre_abun_model(num_rules, num_otus, num_time, dist, 1:num_time)
  
  x <- torch_rand(batch_size, num_otus, num_time, requires_grad = TRUE)
  
  output <- model(x)
  loss <- output$sum()
  loss$backward()
  
  # Gradients should exist
  expect_false(is.null(x$grad))
  expect_false(is.null(model$phylo_focus$kappa$grad))
})

test_that("models can be saved and loaded", {
  skip_if_not(torch_is_installed(), "torch not properly installed")
  
  num_rules <- 2
  num_otus <- 20
  num_time <- 3
  
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  model <- mditre_model(num_rules, num_otus, num_time, dist, 1:num_time)
  
  # Save state dict
  temp_file <- tempfile(fileext = ".pt")
  torch_save(model$state_dict(), temp_file)
  
  # Create new model and load
  model2 <- mditre_model(num_rules, num_otus, num_time, dist, 1:num_time)
  model2$load_state_dict(torch_load(temp_file))
  
  # Check parameters match
  kappa1 <- as.numeric(model$phylo_focus$kappa$cpu())
  kappa2 <- as.numeric(model2$phylo_focus$kappa$cpu())
  
  expect_true(all(abs(kappa1 - kappa2) < 1e-6))
  
  # Clean up
  unlink(temp_file)
})
