#' Threshold Detector Layers
#'
#' These layers learn thresholds for abundance and slope values,
#' producing gated responses that indicate whether aggregated values
#' exceed learned thresholds.
#'
#' @name layer3_detector
#' @rdname layer3_detector
NULL

#' Threshold Detector Layer
#'
#' Learn threshold abundance for each detector. The output is a sharp but smooth
#' gated response indicating whether the aggregated abundance from previous steps
#' is above/below the learned threshold.
#'
#' @param num_rules Number of rule detectors
#' @param num_otus Number of OTUs
#' @param num_time_centers Number of time centers (unused, kept for compatibility)
#' @param layer_name Name for this layer instance (default: "threshold")
#' @param ... Additional arguments passed to base layer
#'
#' @return An R6 class representing a Threshold detector layer
#'
#' @details
#' Architecture:
#' - Input: (batch, num_rules, num_otus)
#' - Output: (batch, num_rules, num_otus)
#'
#' The layer uses sigmoid activation to create a smooth but sharp gating function:
#' \deqn{output = \sigma((x - threshold) \times k)}
#'
#' where k is a temperature parameter controlling the sharpness of the gate.
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' # Create threshold layer
#' layer <- threshold_layer(
#'   num_rules = 5,
#'   num_otus = 20,
#'   num_time_centers = 1
#' )
#'
#' # Forward pass
#' x <- torch_randn(32, 5, 20)  # batch=32, rules=5, otus=20
#' output <- layer(x)
#' print(output$shape)  # [32, 5, 20]
#'
#' # With higher temperature (sharper threshold)
#' output_sharp <- layer(x, k = 5.0)
#' }
#'
#' @export
threshold_layer <- function(num_rules, num_otus, num_time_centers,
                            layer_name = "threshold", ...) {
  # Create config
  config <- list(
    num_rules = num_rules,
    num_otus = num_otus,
    num_time_centers = num_time_centers
  )
  
  # Define the module
  ThresholdModule <- nn_module(
    classname = "ThresholdModule",
    
    initialize = function(config, layer_name) {
      self$layer_name <- layer_name
      self$config <- config
      
      # Parameter for learnable threshold abundance
      self$thresh <- nn_parameter(torch_randn(config$num_rules, config$num_otus))
    },
    
    forward = function(x, k = 1.0) {
      # Response of the detector for average abundance
      x <- torch_sigmoid((x - self$thresh) * k)
      return(x)
    },
    
    init_params = function(init_args) {
      # Initialize layer parameters
      # init_args should contain: thresh_init
      
      if (!is.null(init_args$thresh_init)) {
        self$thresh$data <- torch_tensor(init_args$thresh_init, dtype = torch_float32())
      }
    },
    
    get_params = function() {
      return(list(
        thresh = self$thresh$detach()$cpu()
      ))
    },
    
    set_params = function(params) {
      if (!is.null(params$thresh)) {
        self$thresh$data <- params$thresh$to(device = self$thresh$device)
      }
    }
  )
  
  # Create instance
  layer <- ThresholdModule(config, layer_name)
  
  # Register with LayerRegistry
  register_layer("layer3", "threshold", layer_name, layer)
  
  return(layer)
}


#' Slope Detector Layer
#'
#' Learn threshold for slope values. The output is a gated response indicating
#' whether the aggregated slope from spatial and time aggregation steps is
#' above/below the learned threshold.
#'
#' @param num_rules Number of rule detectors
#' @param num_otus Number of OTUs
#' @param num_time_centers Number of time centers (unused, kept for compatibility)
#' @param layer_name Name for this layer instance (default: "slope")
#' @param ... Additional arguments passed to base layer
#'
#' @return An R6 class representing a Slope detector layer
#'
#' @details
#' Architecture:
#' - Input: (batch, num_rules, num_otus)
#' - Output: (batch, num_rules, num_otus)
#'
#' Similar to the threshold layer, but operates on slope (rate of change) values
#' rather than abundance values. Uses sigmoid activation for smooth gating:
#' \deqn{output = \sigma((x - slope\_threshold) \times k)}
#'
#' Typically used in conjunction with time_agg_layer which computes slopes.
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' # Create slope detector layer
#' layer <- slope_layer(
#'   num_rules = 5,
#'   num_otus = 20,
#'   num_time_centers = 1
#' )
#'
#' # Forward pass (typically slope values from Layer 2)
#' x_slope <- torch_randn(32, 5, 20)
#' output <- layer(x_slope)
#' print(output$shape)  # [32, 5, 20]
#'
#' # Integration with Layer 2
#' time_layer <- time_agg_layer(5, 20, 10, 1)
#' x_temporal <- torch_randn(32, 5, 20, 10)
#' result <- time_layer(x_temporal)
#' slope_detected <- layer(result$slope)
#' }
#'
#' @export
slope_layer <- function(num_rules, num_otus, num_time_centers,
                        layer_name = "slope", ...) {
  # Create config
  config <- list(
    num_rules = num_rules,
    num_otus = num_otus,
    num_time_centers = num_time_centers
  )
  
  # Define the module
  SlopeModule <- nn_module(
    classname = "SlopeModule",
    
    initialize = function(config, layer_name) {
      self$layer_name <- layer_name
      self$config <- config
      
      # Parameter for learnable threshold slope
      self$slope <- nn_parameter(torch_randn(config$num_rules, config$num_otus))
    },
    
    forward = function(x, k = 1.0) {
      # Response of the detector for average slope
      x <- torch_sigmoid((x - self$slope) * k)
      
      # Check for NaN values
      if (self$slope$isnan()$any()$item()) {
        print(self$slope)
        stop("NaN in slope!")
      }
      
      return(x)
    },
    
    init_params = function(init_args) {
      # Initialize layer parameters
      # init_args should contain: slope_init
      
      if (!is.null(init_args$slope_init)) {
        self$slope$data <- torch_tensor(init_args$slope_init, dtype = torch_float32())
      }
    },
    
    get_params = function() {
      return(list(
        slope = self$slope$detach()$cpu()
      ))
    },
    
    set_params = function(params) {
      if (!is.null(params$slope)) {
        self$slope$data <- params$slope$to(device = self$slope$device)
      }
    }
  )
  
  # Create instance
  layer <- SlopeModule(config, layer_name)
  
  # Register with LayerRegistry
  register_layer("layer3", "slope", layer_name, layer)
  
  return(layer)
}
