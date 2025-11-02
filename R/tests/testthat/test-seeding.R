# Test Seeding Utilities
# Tests for reproducibility functions

test_that("set_mditre_seeds sets seeds correctly", {
  # Set seed
  set_mditre_seeds(42)
  
  # Generate random numbers
  r_vals1 <- runif(5)
  torch_vals1 <- as.numeric(torch_randn(5)$cpu())
  
  # Reset seed
  set_mditre_seeds(42)
  
  # Generate again
  r_vals2 <- runif(5)
  torch_vals2 <- as.numeric(torch_randn(5)$cpu())
  
  # Should be identical
  expect_equal(r_vals1, r_vals2)
  expect_equal(torch_vals1, torch_vals2)
})

test_that("different seeds produce different results", {
  # Seed 1
  set_mditre_seeds(42)
  vals1 <- as.numeric(torch_randn(10)$cpu())
  
  # Seed 2
  set_mditre_seeds(123)
  vals2 <- as.numeric(torch_randn(10)$cpu())
  
  # Should be different
  expect_false(all(vals1 == vals2))
})

test_that("seeding affects model initialization", {
  num_rules <- 2
  num_otus <- 20
  num_time <- 3
  
  dist_matrix <- matrix(runif(num_otus * num_otus), num_otus, num_otus)
  dist_matrix <- (dist_matrix + t(dist_matrix)) / 2
  diag(dist_matrix) <- 0
  dist <- torch_tensor(dist_matrix)
  
  # Model 1
  set_mditre_seeds(42)
  model1 <- mditre_model(num_rules, num_otus, num_time, dist, 1:num_time)
  kappa1 <- as.numeric(model1$phylo_focus$kappa$cpu())
  
  # Model 2 with same seed
  set_mditre_seeds(42)
  model2 <- mditre_model(num_rules, num_otus, num_time, dist, 1:num_time)
  kappa2 <- as.numeric(model2$phylo_focus$kappa$cpu())
  
  # Should be identical
  expect_equal(kappa1, kappa2)
})

test_that("seeding makes training reproducible", {
  skip("Training reproducibility test - requires full training setup")
  
  # This is a placeholder for a more comprehensive test
  # Would test that setting seed before training produces identical results
})
