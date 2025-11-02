#' Base Layer for MDITRE Architecture
#'
#' @description
#' Abstract base class for all MDITRE layers. Provides common interface
#' enabling modularity and extensibility. All MDITRE layers inherit from this.
#'
#' @section Methods:
#' \describe{
#'   \item{initialize(layer_name, layer_config)}{Initialize base layer}
#'   \item{forward(...)}{Forward pass through the layer (must be implemented by subclasses)}
#'   \item{init_params(init_args)}{Initialize layer parameters}
#'   \item{get_config()}{Get layer configuration}
#'   \item{get_layer_info()}{Get detailed information about this layer}
#' }
#'
#' @param layer_name Character string with human-readable name for this layer
#' @param layer_config Named list with configuration parameters for the layer
#'
#' @export
#' @examples
#' \dontrun{
#' # Custom layer inheriting from base_layer
#' my_layer <- nn_module(
#'   "MyLayer",
#'   inherit = base_layer,
#'   
#'   initialize = function(layer_name, layer_config) {
#'     super$initialize(layer_name, layer_config)
#'     # Add custom initialization here
#'   },
#'   
#'   forward = function(x) {
#'     # Implement forward pass
#'     return(x)
#'   }
#' )
#' }
base_layer <- nn_module(
  "BaseLayer",
  
  #' @description Initialize base layer
  #' @param layer_name Human-readable name for this layer
  #' @param layer_config Configuration list for layer parameters
  initialize = function(layer_name, layer_config) {
    self$layer_name <- layer_name
    self$layer_config <- layer_config
    self$setup_complete <- FALSE
  },
  
  #' @description Forward pass through the layer (must be overridden)
  #' @param ... Arguments passed to forward method
  forward = function(...) {
    stop(sprintf("Subclass %s must implement forward() method", class(self)[1]))
  },
  
  #' @description Initialize layer parameters (must be overridden)
  #' @param init_args Named list containing initialization values
  init_params = function(init_args) {
    stop(sprintf("Subclass %s must implement init_params() method", class(self)[1]))
  },
  
  #' @description Get layer configuration
  #' @return Named list with configuration
  get_config = function() {
    return(self$layer_config)
  },
  
  #' @description Get detailed information about this layer
  #' @return Named list with layer metadata including name, type, config, and parameter counts
  get_layer_info = function() {
    # Count parameters
    total_params <- 0
    trainable_params <- 0
    
    for (param_name in names(self$parameters)) {
      param <- self$parameters[[param_name]]
      if (inherits(param, "torch_tensor")) {
        param_count <- param$numel()$item()
        total_params <- total_params + param_count
        if (param$requires_grad) {
          trainable_params <- trainable_params + param_count
        }
      }
    }
    
    return(list(
      name = self$layer_name,
      type = class(self)[1],
      config = self$get_config(),
      num_parameters = total_params,
      trainable_parameters = trainable_params
    ))
  }
)


#' Layer Registry for MDITRE
#'
#' @description
#' Registry for dynamically managing available layers. Allows for runtime
#' layer selection, easy addition of new layer implementations, and version
#' control of layer implementations.
#'
#' @export
LayerRegistry <- R6::R6Class(
  "LayerRegistry",
  
  public = list(
    #' @field registry Named list storing registered layer classes
    registry = list(),
    
    #' @description Register a new layer type
    #' @param name Character string name for the layer
    #' @param layer_class The nn_module class definition
    #' @param version Character string version (default: "1.0.0")
    register = function(name, layer_class, version = "1.0.0") {
      if (name %in% names(self$registry)) {
        warning(sprintf("Overwriting existing layer '%s'", name))
      }
      
      self$registry[[name]] <- list(
        class = layer_class,
        version = version,
        registered_at = Sys.time()
      )
      
      message(sprintf("Registered layer: %s (v%s)", name, version))
    },
    
    #' @description Get a registered layer class
    #' @param name Character string name of the layer
    #' @return The layer class definition
    get = function(name) {
      if (!name %in% names(self$registry)) {
        stop(sprintf("Layer '%s' not found in registry. Available layers: %s",
                     name, paste(names(self$registry), collapse = ", ")))
      }
      return(self$registry[[name]]$class)
    },
    
    #' @description List all registered layers
    #' @return Character vector of registered layer names
    list_layers = function() {
      return(names(self$registry))
    },
    
    #' @description Get information about a registered layer
    #' @param name Character string name of the layer
    #' @return Named list with layer metadata
    get_info = function(name) {
      if (!name %in% names(self$registry)) {
        stop(sprintf("Layer '%s' not found in registry", name))
      }
      
      layer_entry <- self$registry[[name]]
      return(list(
        name = name,
        version = layer_entry$version,
        registered_at = layer_entry$registered_at
      ))
    },
    
    #' @description Remove a layer from registry
    #' @param name Character string name of the layer
    remove = function(name) {
      if (name %in% names(self$registry)) {
        self$registry[[name]] <- NULL
        message(sprintf("Removed layer: %s", name))
      } else {
        warning(sprintf("Layer '%s' not found in registry", name))
      }
    }
  )
)

# Create global layer registry instance
#' @export
layer_registry <- LayerRegistry$new()
