#' Temporal Aggregation Layers for Time Focus
#'
#' These layers aggregate microbial time-series along the time dimension,
#' selecting contiguous time windows important for the prediction task.
#'
#' @name layer2_temporal_focus
#' @rdname layer2_temporal_focus
NULL

#' Time Aggregation Layer (with slopes)
#'
#' Aggregate time-series along the time dimension, selecting contiguous time
#' windows important for prediction using sigmoid-based importance weights.
#' Computes both average abundance and average slope within the selected time window.
#'
#' @param num_rules Number of rule detectors
#' @param num_otus Number of OTUs
#' @param num_time Number of time points
#' @param num_time_centers Number of time window centers (unused, kept for compatibility)
#' @param layer_name Name for this layer instance (default: "time_agg")
#' @param ... Additional arguments passed to base layer
#'
#' @return An R6 class representing a TimeAgg layer
#'
#' @details
#' Architecture:
#' - Input: (batch, num_rules, num_otus, time_points)
#' - Output: list(abundance, slope), both (batch, num_rules, num_otus)
#'
#' The layer learns to focus on specific time windows by parameterizing:
#' - Window center (mu): Position of the time window
#' - Window width (sigma): Width of the time window
#' - Separate parameters for abundance and slope computation
#'
#' Uses soft time window selection via unitboxcar function for differentiability.
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' # Create layer
#' layer <- time_agg_layer(
#'   num_rules = 5,
#'   num_otus = 20,
#'   num_time = 10,
#'   num_time_centers = 1
#' )
#'
#' # Forward pass
#' x <- torch_randn(32, 5, 20, 10)  # batch=32, rules=5, otus=20, time=10
#' result <- layer(x)
#' print(result$abundance$shape)  # [32, 5, 20]
#' print(result$slope$shape)      # [32, 5, 20]
#'
#' # With time mask
#' mask <- torch_ones(32, 10)
#' mask[, 1:3] <- 0  # Mask first 3 time points
#' result <- layer(x, mask = mask)
#' }
#'
#' @export
time_agg_layer <- function(num_rules, num_otus, num_time, num_time_centers,
                           layer_name = "time_agg", ...) {
  # Create config
  config <- list(
    num_rules = num_rules,
    num_otus = num_otus,
    num_time = num_time,
    num_time_centers = num_time_centers
  )
  
  # Define the module
  TimeAggModule <- nn_module(
    classname = "TimeAggModule",
    
    initialize = function(config, layer_name) {
      self$layer_name <- layer_name
      self$config <- config
      self$num_time <- config$num_time
      
      # Tensor of time points, starting from 0 to num_time - 1
      self$register_buffer("times", torch_arange(0, config$num_time - 1, dtype = torch_float32()))
      
      # Time window parameters (for abundance and slope)
      self$abun_a <- nn_parameter(torch_randn(config$num_rules, config$num_otus))
      self$slope_a <- nn_parameter(torch_randn(config$num_rules, config$num_otus))
      self$abun_b <- nn_parameter(torch_randn(config$num_rules, config$num_otus))
      self$slope_b <- nn_parameter(torch_randn(config$num_rules, config$num_otus))
      
      # Storage for inspection
      self$wts <- NULL
      self$wts_slope <- NULL
      self$m <- NULL
      self$m_slope <- NULL
      self$s_abun <- NULL
      self$s_slope <- NULL
    },
    
    forward = function(x, mask = NULL, k = 1.0) {
      # Compute unnormalized importance weights for each time point
      abun_a <- torch_sigmoid(self$abun_a)$unsqueeze(-1)
      slope_a <- torch_sigmoid(self$slope_a)$unsqueeze(-1)
      abun_b <- torch_sigmoid(self$abun_b)$unsqueeze(-1)
      slope_b <- torch_sigmoid(self$slope_b)$unsqueeze(-1)
      
      # Compute time window parameters
      sigma <- self$num_time * abun_a
      sigma_slope <- self$num_time * slope_a
      mu <- (self$num_time * abun_a / 2.0) + (1 - abun_a) * self$num_time * abun_b
      mu_slope <- (self$num_time * slope_a / 2.0) + (1 - slope_a) * self$num_time * slope_b
      
      # Compute time weights using boxcar function
      time_wts_unnorm <- unitboxcar(self$times, mu, sigma, k)
      time_wts_unnorm_slope <- unitboxcar(self$times, mu_slope, sigma_slope, k)
      
      # Mask out time points with no samples
      if (!is.null(mask)) {
        time_wts_unnorm <- time_wts_unnorm$mul(
          mask$unsqueeze(2)$unsqueeze(2)
        )
        time_wts_unnorm_slope <- time_wts_unnorm_slope$mul(
          mask$unsqueeze(2)$unsqueeze(2)
        )
      }
      
      # Store weights for inspection
      self$wts <- time_wts_unnorm
      self$wts_slope <- time_wts_unnorm_slope
      
      # Normalize importance time weights
      time_wts <- time_wts_unnorm$div(
        time_wts_unnorm$sum(dim = -1, keepdim = TRUE) + 1e-8
      )
      
      if (time_wts$isnan()$any()$item()) {
        print(time_wts_unnorm$sum(-1))
        stop("NaN in time aggregation!")
      }
      
      # Aggregation over time dimension (weighted average)
      x_abun <- x$mul(time_wts)$sum(dim = -1)
      
      # Compute approximate average slope over time window
      tau <- self$times - mu_slope
      a <- (time_wts_unnorm_slope * x)$sum(dim = -1)
      b <- (time_wts_unnorm_slope * tau)$sum(dim = -1)
      c <- time_wts_unnorm_slope$sum(dim = -1)
      d <- (time_wts_unnorm_slope * x * tau)$sum(dim = -1)
      e <- (time_wts_unnorm_slope * tau$pow(2))$sum(dim = -1)
      
      num <- (a * b) - (c * d)
      den <- (b$pow(2) - (e * c)) + 1e-8
      x_slope <- num / den
      
      if (x_slope$isnan()$any()$item()) {
        print(time_wts_unnorm_slope$sum(dim = -1))
        print(x_slope)
        stop("NaN in time aggregation!")
      }
      
      # Store parameters for inspection
      self$m <- mu
      self$m_slope <- mu_slope
      self$s_abun <- sigma
      self$s_slope <- sigma_slope
      
      return(list(abundance = x_abun, slope = x_slope))
    },
    
    init_params = function(init_args) {
      # Initialize layer parameters
      # init_args should contain: abun_a_init, abun_b_init, slope_a_init, slope_b_init
      
      if (!is.null(init_args$abun_a_init)) {
        self$abun_a$data <- torch_tensor(gtools::logit(init_args$abun_a_init), dtype = torch_float32())
      }
      if (!is.null(init_args$abun_b_init)) {
        self$abun_b$data <- torch_tensor(gtools::logit(init_args$abun_b_init), dtype = torch_float32())
      }
      if (!is.null(init_args$slope_a_init)) {
        self$slope_a$data <- torch_tensor(gtools::logit(init_args$slope_a_init), dtype = torch_float32())
      }
      if (!is.null(init_args$slope_b_init)) {
        self$slope_b$data <- torch_tensor(gtools::logit(init_args$slope_b_init), dtype = torch_float32())
      }
    },
    
    get_params = function() {
      return(list(
        abun_a = self$abun_a$detach()$cpu(),
        abun_b = self$abun_b$detach()$cpu(),
        slope_a = self$slope_a$detach()$cpu(),
        slope_b = self$slope_b$detach()$cpu()
      ))
    },
    
    set_params = function(params) {
      if (!is.null(params$abun_a)) {
        self$abun_a$data <- params$abun_a$to(device = self$abun_a$device)
      }
      if (!is.null(params$abun_b)) {
        self$abun_b$data <- params$abun_b$to(device = self$abun_b$device)
      }
      if (!is.null(params$slope_a)) {
        self$slope_a$data <- params$slope_a$to(device = self$slope_a$device)
      }
      if (!is.null(params$slope_b)) {
        self$slope_b$data <- params$slope_b$to(device = self$slope_b$device)
      }
    }
  )
  
  # Create instance
  layer <- TimeAggModule(config, layer_name)
  
  # Register with LayerRegistry
  register_layer("layer2", "time_agg", layer_name, layer)
  
  return(layer)
}


#' Time Aggregation Layer (abundance only)
#'
#' Aggregate time-series along time dimension (abundance only).
#' Simplified version of TimeAgg that only computes average abundance
#' within a selected time window, without slope calculation.
#'
#' @param num_rules Number of rule detectors
#' @param num_otus Number of OTUs
#' @param num_time Number of time points
#' @param num_time_centers Number of time window centers (unused, kept for compatibility)
#' @param layer_name Name for this layer instance (default: "time_agg_abun")
#' @param ... Additional arguments passed to base layer
#'
#' @return An R6 class representing a TimeAggAbun layer
#'
#' @details
#' Architecture:
#' - Input: (batch, num_rules, num_otus, time_points)
#' - Output: (batch, num_rules, num_otus)
#'
#' This is a simplified version of time_agg_layer that only computes
#' abundance aggregation without slope computation. Useful for models
#' that don't require rate-of-change information.
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' # Create layer
#' layer <- time_agg_abun_layer(
#'   num_rules = 5,
#'   num_otus = 20,
#'   num_time = 10,
#'   num_time_centers = 1
#' )
#'
#' # Forward pass
#' x <- torch_randn(32, 5, 20, 10)
#' result <- layer(x)
#' print(result$shape)  # [32, 5, 20]
#' }
#'
#' @export
time_agg_abun_layer <- function(num_rules, num_otus, num_time, num_time_centers,
                                layer_name = "time_agg_abun", ...) {
  # Create config
  config <- list(
    num_rules = num_rules,
    num_otus = num_otus,
    num_time = num_time,
    num_time_centers = num_time_centers
  )
  
  # Define the module
  TimeAggAbunModule <- nn_module(
    classname = "TimeAggAbunModule",
    
    initialize = function(config, layer_name) {
      self$layer_name <- layer_name
      self$config <- config
      self$num_time <- config$num_time
      
      # Tensor of time points
      self$register_buffer("times", torch_arange(0, config$num_time - 1, dtype = torch_float32()))
      
      # Time window parameters
      self$abun_a <- nn_parameter(torch_randn(config$num_rules, config$num_otus))
      self$abun_b <- nn_parameter(torch_randn(config$num_rules, config$num_otus))
      
      # Storage for inspection
      self$wts <- NULL
      self$m <- NULL
      self$s_abun <- NULL
    },
    
    forward = function(x, mask = NULL, k = 1.0) {
      # Compute unnormalized importance weights for each time point
      abun_a <- torch_sigmoid(self$abun_a)$unsqueeze(-1)
      abun_b <- torch_sigmoid(self$abun_b)$unsqueeze(-1)
      sigma <- self$num_time * abun_a
      mu <- (self$num_time * abun_a / 2.0) + (1 - abun_a) * self$num_time * abun_b
      time_wts_unnorm <- unitboxcar(self$times, mu, sigma, k)
      
      # Mask out time points with no samples
      if (!is.null(mask)) {
        time_wts_unnorm <- time_wts_unnorm$mul(
          mask$unsqueeze(2)$unsqueeze(2)
        )
      }
      
      # Store weights for inspection
      self$wts <- time_wts_unnorm
      
      # Normalize importance time weights
      time_wts <- time_wts_unnorm$div(
        time_wts_unnorm$sum(dim = -1, keepdim = TRUE) + 1e-8
      )
      
      if (time_wts$isnan()$any()$item()) {
        print(time_wts_unnorm$sum(-1))
        stop("NaN in time aggregation!")
      }
      
      # Aggregation over time dimension
      x_abun <- x$mul(time_wts)$sum(dim = -1)
      
      # Store parameters for inspection
      self$m <- mu
      self$s_abun <- sigma
      
      return(x_abun)
    },
    
    init_params = function(init_args) {
      # Initialize layer parameters
      # init_args should contain: abun_a_init, abun_b_init
      
      if (!is.null(init_args$abun_a_init)) {
        self$abun_a$data <- torch_tensor(gtools::logit(init_args$abun_a_init), dtype = torch_float32())
      }
      if (!is.null(init_args$abun_b_init)) {
        self$abun_b$data <- torch_tensor(gtools::logit(init_args$abun_b_init), dtype = torch_float32())
      }
    },
    
    get_params = function() {
      return(list(
        abun_a = self$abun_a$detach()$cpu(),
        abun_b = self$abun_b$detach()$cpu()
      ))
    },
    
    set_params = function(params) {
      if (!is.null(params$abun_a)) {
        self$abun_a$data <- params$abun_a$to(device = self$abun_a$device)
      }
      if (!is.null(params$abun_b)) {
        self$abun_b$data <- params$abun_b$to(device = self$abun_b$device)
      }
    }
  )
  
  # Create instance
  layer <- TimeAggAbunModule(config, layer_name)
  
  # Register with LayerRegistry
  register_layer("layer2", "time_agg_abun", layer_name, layer)
  
  return(layer)
}
