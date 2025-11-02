# Layer 2: Temporal Focus - Usage Examples
# This file demonstrates how to use the temporal aggregation layers

library(mditre)
library(torch)

cat("=" %R% 80, "\n")
cat("Layer 2: Temporal Focus Examples\n")
cat("=" %R% 80, "\n\n")

# ==============================================================================
# Example 1: Basic TimeAgg Layer (with slopes)
# ==============================================================================
cat("Example 1: Basic TimeAgg Layer Usage\n")
cat("-" %R% 80, "\n")

# Create time aggregation layer
num_rules <- 5
num_otus <- 20
num_time <- 10

layer <- time_agg_layer(
  num_rules = num_rules,
  num_otus = num_otus,
  num_time = num_time,
  num_time_centers = 1,
  layer_name = "temporal_focus"
)

cat("Created TimeAgg layer:\n")
cat(sprintf("  - num_rules: %d\n", num_rules))
cat(sprintf("  - num_otus: %d\n", num_otus))
cat(sprintf("  - num_time: %d\n", num_time))

# Forward pass
batch_size <- 32
x <- torch_randn(batch_size, num_rules, num_otus, num_time)
cat(sprintf("\nInput shape: [%s]\n", paste(x$shape, collapse = ", ")))

result <- layer(x)
cat(sprintf("Output abundance shape: [%s]\n", paste(result$abundance$shape, collapse = ", ")))
cat(sprintf("Output slope shape: [%s]\n", paste(result$slope$shape, collapse = ", ")))

cat("\n")

# ==============================================================================
# Example 2: TimeAgg with Time Mask
# ==============================================================================
cat("Example 2: TimeAgg with Missing Time Points\n")
cat("-" %R% 80, "\n")

# Create mask (some time points missing for some samples)
mask <- torch_ones(batch_size, num_time)
mask[1:5, 1:3] <- 0  # First 5 samples missing first 3 time points
mask[6:10, 8:10] <- 0  # Next 5 samples missing last 3 time points

cat("Created time mask:\n")
cat(sprintf("  - Valid time points per sample vary\n"))
cat(sprintf("  - Sample 1 valid times: %d\n", mask[1, ]$sum()$item()))
cat(sprintf("  - Sample 6 valid times: %d\n", mask[6, ]$sum()$item()))

# Forward pass with mask
result <- layer(x, mask = mask)
cat(sprintf("\nOutput with mask - abundance shape: [%s]\n", 
            paste(result$abundance$shape, collapse = ", ")))
cat(sprintf("Output with mask - slope shape: [%s]\n", 
            paste(result$slope$shape, collapse = ", ")))

cat("\n")

# ==============================================================================
# Example 3: TimeAggAbun Layer (abundance only)
# ==============================================================================
cat("Example 3: TimeAggAbun Layer (no slopes)\n")
cat("-" %R% 80, "\n")

# Create abundance-only layer
layer_abun <- time_agg_abun_layer(
  num_rules = num_rules,
  num_otus = num_otus,
  num_time = num_time,
  num_time_centers = 1,
  layer_name = "temporal_focus_abun"
)

cat("Created TimeAggAbun layer (no slope computation)\n")

# Forward pass
result_abun <- layer_abun(x)
cat(sprintf("Output shape: [%s]\n", paste(result_abun$shape, collapse = ", ")))

cat("\n")

# ==============================================================================
# Example 4: Parameter Initialization
# ==============================================================================
cat("Example 4: Custom Parameter Initialization\n")
cat("-" %R% 80, "\n")

# Create new layer
layer_init <- time_agg_layer(
  num_rules = 3,
  num_otus = 10,
  num_time = 8,
  num_time_centers = 1
)

# Initialize with custom values
init_args <- list(
  abun_a_init = matrix(0.5, nrow = 3, ncol = 10),
  abun_b_init = matrix(0.3, nrow = 3, ncol = 10),
  slope_a_init = matrix(0.4, nrow = 3, ncol = 10),
  slope_b_init = matrix(0.6, nrow = 3, ncol = 10)
)

layer_init$init_params(init_args)
cat("Initialized layer with custom parameters\n")

# Get parameters
params <- layer_init$get_params()
cat(sprintf("  - abun_a mean: %.4f\n", params$abun_a$mean()$item()))
cat(sprintf("  - abun_b mean: %.4f\n", params$abun_b$mean()$item()))
cat(sprintf("  - slope_a mean: %.4f\n", params$slope_a$mean()$item()))
cat(sprintf("  - slope_b mean: %.4f\n", params$slope_b$mean()$item()))

cat("\n")

# ==============================================================================
# Example 5: Temperature Parameter (k)
# ==============================================================================
cat("Example 5: Temperature Parameter for Soft Selection\n")
cat("-" %R% 80, "\n")

x_small <- torch_randn(2, 3, 10, 8)

# Low temperature (sharper selection)
result_low <- layer_init(x_small, k = 0.5)
cat("Low temperature (k=0.5): Sharper time window selection\n")

# High temperature (softer selection)
result_high <- layer_init(x_small, k = 2.0)
cat("High temperature (k=2.0): Softer time window selection\n")

cat("\n")

# ==============================================================================
# Example 6: Inspecting Learned Time Windows
# ==============================================================================
cat("Example 6: Inspect Learned Time Windows\n")
cat("-" %R% 80, "\n")

# Do a forward pass
result <- layer_init(x_small)

# Access stored parameters
cat("Time window parameters (after forward pass):\n")
cat(sprintf("  - Window centers (mu) shape: [%s]\n", 
            paste(layer_init$m$shape, collapse = ", ")))
cat(sprintf("  - Window widths (sigma) shape: [%s]\n", 
            paste(layer_init$s_abun$shape, collapse = ", ")))
cat(sprintf("  - Slope window centers shape: [%s]\n", 
            paste(layer_init$m_slope$shape, collapse = ", ")))
cat(sprintf("  - Slope window widths shape: [%s]\n", 
            paste(layer_init$s_slope$shape, collapse = ", ")))

# Access time weights
cat(sprintf("\nTime weights shape: [%s]\n", 
            paste(layer_init$wts$shape, collapse = ", ")))
cat(sprintf("Slope time weights shape: [%s]\n", 
            paste(layer_init$wts_slope$shape, collapse = ", ")))

cat("\n")

# ==============================================================================
# Example 7: Integration with Layer 1 (Phylogenetic Focus)
# ==============================================================================
cat("Example 7: Layer 1 + Layer 2 Pipeline\n")
cat("-" %R% 80, "\n")

# Create phylogenetic distance matrix
library(ape)
set.seed(42)
tree <- rtree(20)
phylo_dist <- cophenetic.phylo(tree)

# Layer 1: Phylogenetic aggregation
layer1 <- spatial_agg_layer(
  num_rules = 5,
  num_otus = 20,
  dist = phylo_dist
)

# Layer 2: Temporal aggregation
layer2 <- time_agg_layer(
  num_rules = 5,
  num_otus = 20,
  num_time = 10,
  num_time_centers = 1
)

# Pipeline
x_input <- torch_randn(16, 20, 10)  # batch=16, otus=20, time=10
cat(sprintf("Input to pipeline: [%s]\n", paste(x_input$shape, collapse = ", ")))

x_layer1 <- layer1(x_input)
cat(sprintf("After Layer 1: [%s]\n", paste(x_layer1$shape, collapse = ", ")))

x_layer2 <- layer2(x_layer1)
cat(sprintf("After Layer 2 - abundance: [%s]\n", 
            paste(x_layer2$abundance$shape, collapse = ", ")))
cat(sprintf("After Layer 2 - slope: [%s]\n", 
            paste(x_layer2$slope$shape, collapse = ", ")))

cat("\n")

# ==============================================================================
# Example 8: Gradient Flow Through Layer
# ==============================================================================
cat("Example 8: Gradient Flow Verification\n")
cat("-" %R% 80, "\n")

# Create small test case
layer_grad <- time_agg_layer(
  num_rules = 2,
  num_otus = 5,
  num_time = 6,
  num_time_centers = 1
)

x_grad <- torch_randn(4, 2, 5, 6, requires_grad = TRUE)
result <- layer_grad(x_grad)

# Compute loss
loss <- result$abundance$sum() + result$slope$sum()
loss$backward()

cat("Gradient flow check:\n")
cat(sprintf("  - Input has gradients: %s\n", !is.null(x_grad$grad)))
cat(sprintf("  - abun_a has gradients: %s\n", !is.null(layer_grad$abun_a$grad)))
cat(sprintf("  - abun_b has gradients: %s\n", !is.null(layer_grad$abun_b$grad)))
cat(sprintf("  - slope_a has gradients: %s\n", !is.null(layer_grad$slope_a$grad)))
cat(sprintf("  - slope_b has gradients: %s\n", !is.null(layer_grad$slope_b$grad)))

cat("\n")

# ==============================================================================
# Example 9: Get and Set Parameters
# ==============================================================================
cat("Example 9: Parameter Save and Load\n")
cat("-" %R% 80, "\n")

# Get parameters
params_save <- layer_grad$get_params()
cat("Saved parameters:\n")
cat(sprintf("  - Number of parameter tensors: %d\n", length(params_save)))

# Create new layer
layer_new <- time_agg_layer(
  num_rules = 2,
  num_otus = 5,
  num_time = 6,
  num_time_centers = 1
)

# Set parameters
layer_new$set_params(params_save)
cat("Loaded parameters into new layer\n")

# Verify
params_loaded <- layer_new$get_params()
cat("Verification:\n")
cat(sprintf("  - abun_a match: %s\n", 
            all.equal(params_save$abun_a, params_loaded$abun_a)))
cat(sprintf("  - abun_b match: %s\n", 
            all.equal(params_save$abun_b, params_loaded$abun_b)))

cat("\n")

# ==============================================================================
cat("=" %R% 80, "\n")
cat("All Layer 2 examples completed successfully!\n")
cat("=" %R% 80, "\n")
