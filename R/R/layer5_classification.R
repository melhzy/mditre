#' Dense Classification Layers
#'
#' Linear classifiers with rule selection for computing predicted outcomes.
#' Uses binary concrete to select which rules contribute to final prediction.
#'
#' @name layer5_classification
#' @rdname layer5_classification
NULL

#' Dense Classification Layer
#'
#' Linear classifier for computing predicted outcome. Combines rule responses
#' and slope information using learned weights, with binary concrete selection
#' of active rules.
#'
#' @param in_feat Number of input features (rules)
#' @param out_feat Number of output classes
#' @param layer_name Name for this layer instance (default: "dense_layer")
#' @param ... Additional arguments passed to base layer
#'
#' @return An R6 class representing a Dense classification layer
#'
#' @details
#' Architecture:
#' - Input: (x, x_slope) both (batch, num_rules)
#' - Output: (batch,) - log odds for binary classification
#'
#' The layer implements:
#' \deqn{logit(y) = W \cdot (x \odot x_{slope} \odot \beta) + b}
#'
#' where:
#' - \eqn{W} are learned classification weights
#' - \eqn{x} are rule responses (abundance)
#' - \eqn{x_{slope}} are slope rule responses
#' - \eqn{\beta} are binary selection variables (via binary concrete)
#' - \eqn{b} is the bias term
#'
#' **Binary Concrete** is used to select which rules are active, enabling
#' sparse, interpretable models.
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' # Create classification layer
#' layer <- classification_layer(
#'   in_feat = 5,   # 5 rules
#'   out_feat = 1   # binary classification
#' )
#'
#' # Forward pass
#' x <- torch_rand(32, 5)         # rule responses
#' x_slope <- torch_randn(32, 5)  # slope responses
#' output <- layer(x, x_slope)
#' print(output$shape)  # [32] - log odds
#'
#' # Convert to probabilities
#' probs <- torch_sigmoid(output)
#' }
#'
#' @export
classification_layer <- function(in_feat, out_feat,
                                 layer_name = "dense_layer", ...) {
  # Create config
  config <- list(
    in_feat = in_feat,
    out_feat = out_feat
  )
  
  # Define the module
  DenseLayerModule <- nn_module(
    classname = "DenseLayerModule",
    
    initialize = function(config, layer_name) {
      self$layer_name <- layer_name
      self$config <- config
      
      # Logistic regression coefficients
      self$weight <- nn_parameter(torch_randn(config$out_feat, config$in_feat))
      
      # Logistic regression bias
      self$bias <- nn_parameter(torch_randn(config$out_feat))
      
      # Parameter for selecting active rules
      self$beta <- nn_parameter(torch_randn(config$in_feat))
      
      # Storage for inspection
      self$sub_log_odds <- NULL
      self$log_odds <- NULL
      self$z <- NULL
    },
    
    forward = function(x, x_slope = NULL, k = 1.0, hard = FALSE, 
                      use_noise = TRUE, ...) {
      # Check that x_slope is provided
      if (is.null(x_slope)) {
        stop("DenseLayer requires x_slope argument")
      }
      
      # Binary concrete for rule selection
      if (self$training) {
        if (use_noise) {
          z <- binary_concrete(self$beta, k, hard = hard, use_noise = TRUE)
        } else {
          z <- binary_concrete(self$beta, k, hard = hard, use_noise = FALSE)
        }
      } else {
        z <- binary_concrete(self$beta, k, hard = hard, use_noise = FALSE)
      }
      
      # Store sub-components for inspection
      self$sub_log_odds <- ((x * x_slope) * 
                           ((self$weight * z$unsqueeze(1))$reshape(-1))) + self$bias
      
      # Predict the outcome
      # x_combined = x * x_slope
      x_combined <- x * x_slope
      # Apply weight with rule selection: weight * z
      weight_selected <- self$weight * z$unsqueeze(1)
      # Linear transformation: F.linear(x, weight, bias)
      x <- nnf_linear(x_combined, weight_selected, self$bias)
      
      self$z <- z
      self$log_odds <- x$squeeze(-1)
      
      return(x$squeeze(-1))
    },
    
    init_params = function(init_args) {
      # Initialize layer parameters
      # init_args should contain: w_init, bias_init, beta_init
      
      if (!is.null(init_args$w_init)) {
        self$weight$data <- torch_tensor(init_args$w_init, dtype = torch_float32())
      }
      if (!is.null(init_args$bias_init)) {
        self$bias$data <- torch_tensor(init_args$bias_init, dtype = torch_float32())
      }
      if (!is.null(init_args$beta_init)) {
        self$beta$data <- torch_tensor(init_args$beta_init, dtype = torch_float32())
      }
    },
    
    get_params = function() {
      return(list(
        weight = self$weight$detach()$cpu(),
        bias = self$bias$detach()$cpu(),
        beta = self$beta$detach()$cpu()
      ))
    },
    
    set_params = function(params) {
      if (!is.null(params$weight)) {
        self$weight$data <- params$weight$to(device = self$weight$device)
      }
      if (!is.null(params$bias)) {
        self$bias$data <- params$bias$to(device = self$bias$device)
      }
      if (!is.null(params$beta)) {
        self$beta$data <- params$beta$to(device = self$beta$device)
      }
    }
  )
  
  # Create instance
  layer <- DenseLayerModule(config, layer_name)
  
  # Register with LayerRegistry
  register_layer("layer5", "dense_layer", layer_name, layer)
  
  return(layer)
}


#' Dense Classification Layer (Abundance Only)
#'
#' Linear classifier for abundance-only models. Simplified version that only
#' uses abundance (not slope) for classification.
#'
#' @param in_feat Number of input features (rules)
#' @param out_feat Number of output classes
#' @param layer_name Name for this layer instance (default: "dense_layer_abun")
#' @param ... Additional arguments passed to base layer
#'
#' @return An R6 class representing a Dense classification layer (abundance only)
#'
#' @details
#' Architecture:
#' - Input: (batch, num_rules)
#' - Output: (batch,) - log odds for binary classification
#'
#' Simplified version of classification_layer that only uses abundance:
#' \deqn{logit(y) = W \cdot (x \odot \beta) + b}
#'
#' Used in conjunction with MDITREAbun models that don't compute slopes.
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' # Create classification layer (abundance only)
#' layer <- classification_abun_layer(
#'   in_feat = 5,
#'   out_feat = 1
#' )
#'
#' # Forward pass
#' x <- torch_rand(32, 5)  # rule responses only
#' output <- layer(x)
#' print(output$shape)  # [32]
#' }
#'
#' @export
classification_abun_layer <- function(in_feat, out_feat,
                                      layer_name = "dense_layer_abun", ...) {
  # Create config
  config <- list(
    in_feat = in_feat,
    out_feat = out_feat
  )
  
  # Define the module
  DenseLayerAbunModule <- nn_module(
    classname = "DenseLayerAbunModule",
    
    initialize = function(config, layer_name) {
      self$layer_name <- layer_name
      self$config <- config
      
      # Logistic regression coefficients
      self$weight <- nn_parameter(torch_randn(config$out_feat, config$in_feat))
      
      # Logistic regression bias
      self$bias <- nn_parameter(torch_randn(config$out_feat))
      
      # Parameter for selecting active rules
      self$beta <- nn_parameter(torch_randn(config$in_feat))
      
      # Storage for inspection
      self$sub_log_odds <- NULL
      self$log_odds <- NULL
      self$z <- NULL
    },
    
    forward = function(x, k = 1.0, hard = FALSE, use_noise = TRUE, ...) {
      # Binary concrete for rule selection
      if (self$training) {
        if (use_noise) {
          z <- binary_concrete(self$beta, k, hard = hard, use_noise = TRUE)
        } else {
          z <- binary_concrete(self$beta, k, hard = hard, use_noise = FALSE)
        }
      } else {
        z <- binary_concrete(self$beta, k, hard = hard, use_noise = FALSE)
      }
      
      # Store sub-components for inspection
      self$sub_log_odds <- (x * 
                           ((self$weight * z$unsqueeze(1))$reshape(-1))) + self$bias
      
      # Predict the outcome
      # Apply weight with rule selection: weight * z
      weight_selected <- self$weight * z$unsqueeze(1)
      # Linear transformation
      x <- nnf_linear(x, weight_selected, self$bias)
      
      self$log_odds <- x$squeeze(-1)
      self$z <- z
      
      return(x$squeeze(-1))
    },
    
    init_params = function(init_args) {
      # Initialize layer parameters
      # init_args should contain: w_init, bias_init, beta_init
      
      if (!is.null(init_args$w_init)) {
        self$weight$data <- torch_tensor(init_args$w_init, dtype = torch_float32())
      }
      if (!is.null(init_args$bias_init)) {
        self$bias$data <- torch_tensor(init_args$bias_init, dtype = torch_float32())
      }
      if (!is.null(init_args$beta_init)) {
        self$beta$data <- torch_tensor(init_args$beta_init, dtype = torch_float32())
      }
    },
    
    get_params = function() {
      return(list(
        weight = self$weight$detach()$cpu(),
        bias = self$bias$detach()$cpu(),
        beta = self$beta$detach()$cpu()
      ))
    },
    
    set_params = function(params) {
      if (!is.null(params$weight)) {
        self$weight$data <- params$weight$to(device = self$weight$device)
      }
      if (!is.null(params$bias)) {
        self$bias$data <- params$bias$to(device = self$bias$device)
      }
      if (!is.null(params$beta)) {
        self$beta$data <- params$beta$to(device = self$beta$device)
      }
    }
  )
  
  # Create instance
  layer <- DenseLayerAbunModule(config, layer_name)
  
  # Register with LayerRegistry
  register_layer("layer5", "dense_layer_abun", layer_name, layer)
  
  return(layer)
}
