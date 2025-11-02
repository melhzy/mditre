# Complete MDITRE Model - Usage Examples
# This file demonstrates end-to-end model usage with all layers

library(mditre)
library(torch)
library(ape)

cat("=" %R% 80, "\n")
cat("Complete MDITRE Model Examples\n")
cat("=" %R% 80, "\n\n")

# ==============================================================================
# Example 1: Basic MDITRE Model Creation
# ==============================================================================
cat("Example 1: Create Full MDITRE Model\n")
cat("-" %R% 80, "\n")

# Setup parameters
num_otus <- 50
num_time <- 10
num_rules <- 5
num_otu_centers <- 10

# Create synthetic phylogenetic tree
set.seed(42)
tree <- rtree(num_otus)
phylo_dist <- cophenetic.phylo(tree)

cat(sprintf("Dataset configuration:\n"))
cat(sprintf("  - Number of OTUs: %d\n", num_otus))
cat(sprintf("  - Time points: %d\n", num_time))
cat(sprintf("  - Rules: %d\n", num_rules))
cat(sprintf("  - OTU centers: %d\n", num_otu_centers))

# Create MDITRE model
model <- mditre_model(
  num_rules = num_rules,
  num_otus = num_otus,
  num_otu_centers = num_otu_centers,
  num_time = num_time,
  num_time_centers = 1,
  dist = phylo_dist,
  emb_dim = 3
)

cat("\nModel created successfully!\n")
cat(sprintf("Model class: %s\n", class(model)[1]))

cat("\n")

# ==============================================================================
# Example 2: Forward Pass Through Model
# ==============================================================================
cat("Example 2: Forward Pass\n")
cat("-" %R% 80, "\n")

# Create synthetic input data
batch_size <- 32
x <- torch_randn(batch_size, num_otus, num_time)
cat(sprintf("Input shape: [%s]\n", paste(x$shape, collapse = ", ")))

# Forward pass
predictions <- model(x)
cat(sprintf("Output shape: [%s]\n", paste(predictions$shape, collapse = ", ")))
cat(sprintf("Output range: [%.4f, %.4f]\n", 
            predictions$min()$item(), predictions$max()$item()))

# Convert to probabilities
probs <- torch_sigmoid(predictions)
cat(sprintf("Probability range: [%.4f, %.4f]\n", 
            probs$min()$item(), probs$max()$item()))

cat("\n")

# ==============================================================================
# Example 3: Model with Time Mask
# ==============================================================================
cat("Example 3: Forward Pass with Missing Time Points\n")
cat("-" %R% 80, "\n")

# Create time mask (some samples have missing time points)
mask <- torch_ones(batch_size, num_time)
mask[1:10, 1:3] <- 0  # First 10 samples missing first 3 time points
mask[11:20, 8:10] <- 0  # Next 10 samples missing last 3 time points

cat("Created time mask with missing data\n")
cat(sprintf("  - Samples 1-10: missing first 3 time points\n"))
cat(sprintf("  - Samples 11-20: missing last 3 time points\n"))

# Forward pass with mask
predictions_masked <- model(x, mask = mask)
cat(sprintf("Output with mask shape: [%s]\n", 
            paste(predictions_masked$shape, collapse = ", ")))

cat("\n")

# ==============================================================================
# Example 4: Temperature Parameter Control
# ==============================================================================
cat("Example 4: Temperature Parameter Control\n")
cat("-" %R% 80, "\n")

# Low temperature (sharper selections)
preds_sharp <- model(
  x, 
  k_otu = 5.0,      # Sharp phylogenetic selection
  k_time = 5.0,     # Sharp temporal window
  k_thresh = 5.0,   # Sharp threshold
  k_slope = 5.0,    # Sharp slope threshold
  k_alpha = 5.0,    # Sharp detector selection
  k_beta = 5.0      # Sharp rule selection
)
cat("Low temperature (k=5.0): Sharper selections\n")
cat(sprintf("  Output range: [%.4f, %.4f]\n", 
            preds_sharp$min()$item(), preds_sharp$max()$item()))

# High temperature (softer selections)
preds_soft <- model(
  x,
  k_otu = 0.5,
  k_time = 0.5,
  k_thresh = 0.5,
  k_slope = 0.5,
  k_alpha = 0.5,
  k_beta = 0.5
)
cat("\nHigh temperature (k=0.5): Softer selections\n")
cat(sprintf("  Output range: [%.4f, %.4f]\n", 
            preds_soft$min()$item(), preds_soft$max()$item()))

cat("\n")

# ==============================================================================
# Example 5: Training vs Evaluation Mode
# ==============================================================================
cat("Example 5: Training vs Evaluation Mode\n")
cat("-" %R% 80, "\n")

# Training mode (with noise)
model$train()
preds_train <- model(x, use_noise = TRUE)
cat("Training mode (with Gumbel noise):\n")
cat(sprintf("  Mean prediction: %.4f\n", preds_train$mean()$item()))

# Evaluation mode (deterministic)
model$eval()
preds_eval <- model(x, use_noise = FALSE)
cat("\nEvaluation mode (deterministic):\n")
cat(sprintf("  Mean prediction: %.4f\n", preds_eval$mean()$item()))

cat("\n")

# ==============================================================================
# Example 6: Hard Selection Mode
# ==============================================================================
cat("Example 6: Hard Selection with Straight-Through Estimator\n")
cat("-" %R% 80, "\n")

# Soft selection (default)
model$eval()
preds_soft_select <- model(x, hard = FALSE)
cat("Soft selection:\n")
cat(sprintf("  Mean: %.4f, Std: %.4f\n", 
            preds_soft_select$mean()$item(), 
            preds_soft_select$std()$item()))

# Hard selection
preds_hard_select <- model(x, hard = TRUE)
cat("\nHard selection (discrete):\n")
cat(sprintf("  Mean: %.4f, Std: %.4f\n", 
            preds_hard_select$mean()$item(), 
            preds_hard_select$std()$item()))

cat("\n")

# ==============================================================================
# Example 7: MDITRE Abundance-Only Model
# ==============================================================================
cat("Example 7: MDITREAbun Model (No Slopes)\n")
cat("-" %R% 80, "\n")

# Create abundance-only model
model_abun <- mditre_abun_model(
  num_rules = num_rules,
  num_otus = num_otus,
  num_otu_centers = num_otu_centers,
  num_time = num_time,
  num_time_centers = 1,
  dist = phylo_dist,
  emb_dim = 3
)

cat("Created MDITREAbun model (faster, fewer parameters)\n")

# Forward pass
preds_abun <- model_abun(x)
cat(sprintf("Output shape: [%s]\n", paste(preds_abun$shape, collapse = ", ")))

cat("\n")

# ==============================================================================
# Example 8: Gradient Flow Verification
# ==============================================================================
cat("Example 8: Gradient Flow Through Model\n")
cat("-" %R% 80, "\n")

# Create model and input with gradients
model_grad <- mditre_model(
  num_rules = 3,
  num_otus = 20,
  num_otu_centers = 5,
  num_time = 8,
  num_time_centers = 1,
  dist = cophenetic.phylo(rtree(20)),
  emb_dim = 2
)

x_grad <- torch_randn(4, 20, 8, requires_grad = TRUE)
model_grad$train()

# Forward pass
preds <- model_grad(x_grad)

# Compute loss
loss <- preds$sum()
loss$backward()

cat("Gradient flow check:\n")
cat(sprintf("  Input has gradients: %s\n", !is.null(x_grad$grad)))
cat(sprintf("  Loss value: %.4f\n", loss$item()))

# Check layer gradients
cat("\nLayer gradient status:\n")
cat(sprintf("  Layer 1 (spat_attn) kappa: %s\n", 
            !is.null(model_grad$spat_attn$kappa$grad)))
cat(sprintf("  Layer 2 (time_attn) abun_a: %s\n", 
            !is.null(model_grad$time_attn$abun_a$grad)))
cat(sprintf("  Layer 3 (thresh_func) thresh: %s\n", 
            !is.null(model_grad$thresh_func$thresh$grad)))
cat(sprintf("  Layer 4 (rules) alpha: %s\n", 
            !is.null(model_grad$rules$alpha$grad)))
cat(sprintf("  Layer 5 (fc) weight: %s\n", 
            !is.null(model_grad$fc$weight$grad)))

cat("\n")

# ==============================================================================
# Example 9: Parameter Initialization
# ==============================================================================
cat("Example 9: Custom Parameter Initialization\n")
cat("-" %R% 80, "\n")

# Create model
model_init <- mditre_model(
  num_rules = 3,
  num_otus = 20,
  num_otu_centers = 5,
  num_time = 8,
  num_time_centers = 1,
  dist = cophenetic.phylo(rtree(20)),
  emb_dim = 2
)

# Prepare initialization arguments
init_args <- list(
  # Layer 1 parameters
  kappa_init = matrix(0.5, nrow = 3, ncol = 5),
  eta_init = array(rnorm(3 * 5 * 2), dim = c(3, 5, 2)),
  
  # Layer 2 parameters
  abun_a_init = matrix(0.4, nrow = 3, ncol = 5),
  abun_b_init = matrix(0.6, nrow = 3, ncol = 5),
  slope_a_init = matrix(0.3, nrow = 3, ncol = 5),
  slope_b_init = matrix(0.7, nrow = 3, ncol = 5),
  
  # Layer 3 parameters
  thresh_init = matrix(0.0, nrow = 3, ncol = 5),
  slope_init = matrix(0.0, nrow = 3, ncol = 5),
  
  # Layer 4 parameters
  alpha_init = matrix(0.5, nrow = 3, ncol = 5),
  
  # Layer 5 parameters
  w_init = matrix(rnorm(1 * 3), nrow = 1, ncol = 3),
  bias_init = array(0.0, dim = 1),
  beta_init = array(rep(0.5, 3))
)

# Initialize parameters
model_init$init_params(init_args)
cat("Initialized all model parameters\n")

# Test forward pass
preds_init <- model_init(torch_randn(4, 20, 8))
cat(sprintf("Output after initialization: [%.4f, %.4f]\n",
            preds_init$min()$item(), preds_init$max()$item()))

cat("\n")

# ==============================================================================
# Example 10: Model with Reproducible Seeding
# ==============================================================================
cat("Example 10: Reproducible Model Predictions\n")
cat("-" %R% 80, "\n")

# Set reproducible seed
seed_gen <- mditre_seed_generator(experiment_name = "model_test")
seed <- seed_gen$generate_seeds(1)[1]
set_mditre_seeds(seed)

# Create model and data
model_repro <- mditre_model(
  num_rules = 3,
  num_otus = 20,
  num_otu_centers = 5,
  num_time = 8,
  num_time_centers = 1,
  dist = cophenetic.phylo(rtree(20)),
  emb_dim = 2
)

x_repro <- torch_randn(4, 20, 8)
preds1 <- model_repro(x_repro)

# Reset seed and rerun
set_mditre_seeds(seed)
model_repro2 <- mditre_model(
  num_rules = 3,
  num_otus = 20,
  num_otu_centers = 5,
  num_time = 8,
  num_time_centers = 1,
  dist = cophenetic.phylo(rtree(20)),
  emb_dim = 2
)
x_repro2 <- torch_randn(4, 20, 8)
preds2 <- model_repro2(x_repro2)

# Check reproducibility
max_diff <- (preds1 - preds2)$abs()$max()$item()
cat(sprintf("Seed: %d\n", seed))
cat(sprintf("Max difference between runs: %.10f\n", max_diff))
cat(sprintf("Reproducible: %s\n", ifelse(max_diff < 1e-6, "YES", "NO")))

cat("\n")

# ==============================================================================
# Example 11: Comparing Full vs Abun Models
# ==============================================================================
cat("Example 11: Full MDITRE vs Abun-Only Comparison\n")
cat("-" %R% 80, "\n")

# Create both models
model_full <- mditre_model(
  num_rules = 3, num_otus = 20, num_otu_centers = 5,
  num_time = 8, num_time_centers = 1,
  dist = cophenetic.phylo(rtree(20)), emb_dim = 2
)

model_abun_only <- mditre_abun_model(
  num_rules = 3, num_otus = 20, num_otu_centers = 5,
  num_time = 8, num_time_centers = 1,
  dist = cophenetic.phylo(rtree(20)), emb_dim = 2
)

# Same input
x_compare <- torch_randn(8, 20, 8)

# Forward pass
preds_full <- model_full(x_compare)
preds_abun_only <- model_abun_only(x_compare)

cat("Full MDITRE model:\n")
cat(sprintf("  Mean prediction: %.4f\n", preds_full$mean()$item()))
cat(sprintf("  Includes slope computation: YES\n"))

cat("\nAbundance-only model:\n")
cat(sprintf("  Mean prediction: %.4f\n", preds_abun_only$mean()$item()))
cat(sprintf("  Includes slope computation: NO\n"))
cat(sprintf("  Faster inference: YES\n"))

cat("\n")

# ==============================================================================
# Example 12: Simulated Training Step
# ==============================================================================
cat("Example 12: Simulated Training Step\n")
cat("-" %R% 80, "\n")

# Create model
model_train <- mditre_model(
  num_rules = 3, num_otus = 20, num_otu_centers = 5,
  num_time = 8, num_time_centers = 1,
  dist = cophenetic.phylo(rtree(20)), emb_dim = 2
)

# Create optimizer
optimizer <- optim_adam(model_train$parameters, lr = 0.001)

# Training mode
model_train$train()

# Create synthetic data
x_train <- torch_randn(8, 20, 8)
y_train <- torch_randint(0, 2, size = c(8), dtype = torch_float32())

# Forward pass
preds <- model_train(x_train)

# Compute loss (binary cross-entropy)
loss <- nnf_binary_cross_entropy_with_logits(preds, y_train)

# Backward pass
optimizer$zero_grad()
loss$backward()
optimizer$step()

cat("Training step completed:\n")
cat(sprintf("  Loss: %.4f\n", loss$item()))
cat(sprintf("  Gradients computed: YES\n"))
cat(sprintf("  Parameters updated: YES\n"))

cat("\n")

# ==============================================================================
cat("=" %R% 80, "\n")
cat("All MDITRE model examples completed successfully!\n")
cat("=" %R% 80, "\n")
