#' Rule Layer
#'
#' Combines detector responses to compute approximate logical AND operations
#' as rule responses, enabling interpretable decision logic.
#'
#' @name layer4_rule
#' @rdname layer4_rule
NULL

#' Rules Layer
#'
#' Combine detector responses using approximate logical AND. Uses binary concrete
#' relaxation to select which detectors contribute to each rule, then approximates
#' AND operation via product.
#'
#' @param num_rules Number of rules
#' @param num_otus Number of OTUs (detectors per rule)
#' @param num_time_centers Number of time centers (unused, kept for compatibility)
#' @param layer_name Name for this layer instance (default: "rules")
#' @param ... Additional arguments passed to base layer
#'
#' @return An R6 class representing a Rules layer
#'
#' @details
#' Architecture:
#' - Input: (batch, num_rules, num_otus)
#' - Output: (batch, num_rules)
#'
#' The layer implements a differentiable approximation of logical AND:
#' \deqn{AND(x_1, x_2, ..., x_n) \approx \prod_i (1 - \alpha_i(1 - x_i))}
#'
#' where \eqn{\alpha_i} are learned binary selection variables (via binary concrete)
#' that determine which detectors contribute to each rule.
#'
#' **Binary Concrete**: Uses Gumbel-Softmax trick for differentiable binary selection:
#' - During training: Samples from Gumbel distribution for stochasticity
#' - During inference: Deterministic selection based on learned probabilities
#' - Optional hard mode: Straight-through estimator for discrete selection
#'
#' **Interpretability**: After training, the learned \eqn{\alpha} values indicate
#' which OTUs/microbes are important for each rule, enabling biological interpretation.
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' # Create rules layer
#' layer <- rule_layer(
#'   num_rules = 5,
#'   num_otus = 20,
#'   num_time_centers = 1
#' )
#'
#' # Forward pass
#' x <- torch_rand(32, 5, 20)  # batch=32, rules=5, otus=20
#' output <- layer(x)
#' print(output$shape)  # [32, 5]
#'
#' # Training mode with noise
#' layer$train()
#' output_train <- layer(x, k = 1.0, hard = FALSE, use_noise = TRUE)
#'
#' # Evaluation mode (deterministic)
#' layer$eval()
#' output_eval <- layer(x, k = 1.0, hard = TRUE, use_noise = FALSE)
#'
#' # Inspect learned selections
#' alpha_probs <- torch_sigmoid(layer$alpha)
#' print(alpha_probs)  # Which OTUs are selected for each rule
#' }
#'
#' @export
rule_layer <- function(num_rules, num_otus, num_time_centers,
                       layer_name = "rules", ...) {
  # Create config
  config <- list(
    num_rules = num_rules,
    num_otus = num_otus,
    num_time_centers = num_time_centers
  )
  
  # Define the module
  RulesModule <- nn_module(
    classname = "RulesModule",
    
    initialize = function(config, layer_name) {
      self$layer_name <- layer_name
      self$config <- config
      
      # Binary concrete selector variable for detectors
      self$alpha <- nn_parameter(torch_randn(config$num_rules, config$num_otus))
      
      # Storage for inspection
      self$x <- NULL
      self$z <- NULL
    },
    
    forward = function(x, k = 1.0, hard = FALSE, use_noise = TRUE) {
      # Binary concrete for detector selection
      if (self$training) {
        if (use_noise) {
          z <- binary_concrete(self$alpha, k, hard = hard, use_noise = TRUE)
        } else {
          z <- binary_concrete(self$alpha, k, hard = hard, use_noise = FALSE)
        }
      } else {
        z <- binary_concrete(self$alpha, k, hard = hard, use_noise = FALSE)
      }
      
      # Store for inspection
      self$x <- x
      self$z <- z
      
      # Approximate logical AND operation
      # AND(x1, x2, ..., xn) ≈ ∏(1 - αi(1 - xi))
      x <- (1 - z$mul(1 - x))$prod(dim = -1L)
      
      return(x)
    },
    
    init_params = function(init_args) {
      # Initialize layer parameters
      # init_args should contain: alpha_init
      
      if (!is.null(init_args$alpha_init)) {
        self$alpha$data <- torch_tensor(init_args$alpha_init, dtype = torch_float32())
      }
    },
    
    get_params = function() {
      return(list(
        alpha = self$alpha$detach()$cpu()
      ))
    },
    
    set_params = function(params) {
      if (!is.null(params$alpha)) {
        self$alpha$data <- params$alpha$to(device = self$alpha$device)
      }
    }
  )
  
  # Create instance
  layer <- RulesModule(config, layer_name)
  
  # Register with LayerRegistry
  register_layer("layer4", "rules", layer_name, layer)
  
  return(layer)
}
