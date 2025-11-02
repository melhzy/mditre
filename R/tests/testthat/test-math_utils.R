# Test Math Utilities
# Tests for binary_concrete, soft_and, soft_or functions

test_that("binary_concrete produces values in [0, 1]", {
  set.seed(42)
  logits <- torch_randn(10, 20)
  temp <- 0.5
  hard <- FALSE
  
  result <- binary_concrete(logits, temp, hard)
  
  expect_true(all(as.numeric(result$cpu()) >= 0))
  expect_true(all(as.numeric(result$cpu()) <= 1))
})

test_that("binary_concrete hard mode produces binary values", {
  set.seed(42)
  logits <- torch_randn(10, 20)
  temp <- 0.5
  hard <- TRUE
  
  result <- binary_concrete(logits, temp, hard)
  values <- as.numeric(result$cpu())
  
  # Should be approximately 0 or 1
  expect_true(all(values < 0.1 | values > 0.9))
})

test_that("binary_concrete is differentiable", {
  logits <- torch_randn(5, 10, requires_grad = TRUE)
  temp <- 0.5
  
  result <- binary_concrete(logits, temp, hard = FALSE)
  loss <- result$sum()
  loss$backward()
  
  # Gradients should exist
  expect_false(is.null(logits$grad))
  expect_true(any(as.numeric(logits$grad$cpu()) != 0))
})

test_that("soft_and produces values in [0, 1]", {
  set.seed(42)
  x <- torch_rand(10, 20)
  
  result <- soft_and(x, dim = 2)
  
  expect_true(all(as.numeric(result$cpu()) >= 0))
  expect_true(all(as.numeric(result$cpu()) <= 1))
})

test_that("soft_and approaches product for high values", {
  # When all inputs are high, soft_and should be close to product
  x <- torch_ones(5, 10) * 0.9
  
  result <- soft_and(x, dim = 2)
  expected <- torch_prod(x, dim = 2)
  
  diff <- torch_abs(result - expected)
  expect_true(all(as.numeric(diff$cpu()) < 0.1))
})

test_that("soft_or produces values in [0, 1]", {
  set.seed(42)
  x <- torch_rand(10, 20)
  
  result <- soft_or(x, dim = 2)
  
  expect_true(all(as.numeric(result$cpu()) >= 0))
  expect_true(all(as.numeric(result$cpu()) <= 1))
})

test_that("soft_or is greater than max for multiple inputs", {
  # Soft OR should be >= max of inputs
  x <- torch_rand(5, 10)
  
  result <- soft_or(x, dim = 2)
  max_vals <- torch_max(x, dim = 2)[[1]]
  
  # soft_or should be >= max (within numerical tolerance)
  diff <- result - max_vals
  expect_true(all(as.numeric(diff$cpu()) >= -1e-6))
})

test_that("temperature affects binary_concrete output", {
  set.seed(42)
  logits <- torch_randn(10, 20)
  
  result_high_temp <- binary_concrete(logits, temp = 2.0, hard = FALSE)
  result_low_temp <- binary_concrete(logits, temp = 0.1, hard = FALSE)
  
  # Low temperature should produce more extreme values (closer to 0 or 1)
  high_temp_std <- torch_std(result_high_temp)
  low_temp_std <- torch_std(result_low_temp)
  
  # Low temp should have higher variance (more extreme)
  expect_true(as.numeric(low_temp_std$cpu()) > as.numeric(high_temp_std$cpu()))
})
