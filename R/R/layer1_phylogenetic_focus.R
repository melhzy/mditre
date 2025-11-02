#' Layer 1: Phylogenetic Focus Mechanisms
#'
#' @description
#' Spatial aggregation layers that aggregate microbial time-series based on
#' phylogenetic distance, allowing the model to focus on taxonomically related
#' groups of OTUs/ASVs.
#'
#' @name layer1_phylogenetic_focus
NULL


#' Spatial Aggregation Layer (Static)
#'
#' @description
#' Aggregate time-series based on phylogenetic distance using a fixed distance matrix.
#' Uses the sigmoid function to calculate importance weights of each OTU for a detector
#' based on phylogenetic distance. OTUs within a learned kappa radius are selected with
#' higher weights.
#'
#' @section Architecture:
#' \describe{
#'   \item{Input}{(batch, num_otus, time_points)}
#'   \item{Output}{(batch, num_rules, num_otus, time_points)}
#' }
#'
#' @param num_rules Integer; number of rule detectors
#' @param num_otus Integer; number of OTUs in dataset
#' @param dist Numeric matrix; phylogenetic distance matrix (num_otus x num_otus)
#'
#' @export
#' @examples
#' \dontrun{
#' library(torch)
#' library(ape)
#' 
#' # Create example phylogenetic tree
#' tree <- rtree(20)
#' phylo_dist <- cophenetic.phylo(tree)
#' 
#' # Initialize layer
#' layer <- spatial_agg_layer(num_rules = 5, num_otus = 20, dist = phylo_dist)
#' 
#' # Forward pass
#' x <- torch_randn(32, 20, 10)  # batch=32, otus=20, time=10
#' output <- layer(x)
#' print(output$shape)  # [32, 5, 20, 10]
#' }
spatial_agg_layer <- nn_module(
  "SpatialAgg",
  inherit = base_layer,
  
  #' @description Initialize spatial aggregation layer
  initialize = function(num_rules, num_otus, dist) {
    # Call parent constructor
    config <- list(
      num_rules = num_rules,
      num_otus = num_otus,
      dist_shape = dim(dist)
    )
    super$initialize(layer_name = "spatial_agg", layer_config = config)
    
    self$num_rules <- num_rules
    self$num_otus <- num_otus
    
    # Register phylogenetic distance matrix as buffer (non-trainable)
    self$register_buffer(
      "dist",
      torch_tensor(dist, dtype = torch_float())
    )
    
    # OTU selection bandwidth
    # All OTUs within kappa radius are deemed to be selected
    self$kappa <- nn_parameter(torch_randn(num_rules, num_otus))
    
    # Storage for weights (for inspection)
    self$wts <- NULL
  },
  
  #' @description Forward pass: aggregate OTUs based on phylogenetic distance
  #' @param x Input tensor (batch, num_otus, time_points)
  #' @param k Temperature parameter for sigmoid sharpness (default: 1)
  #' @return Aggregated tensor (batch, num_rules, num_otus, time_points)
  forward = function(x, k = 1) {
    # Compute unnormalized OTU weights
    # Transform kappa to [0, max_dist] range
    dist_max <- self$dist$max()$item()
    kappa_transformed <- transf_log(self$kappa, u = dist_max, l = 0)$unsqueeze(-1L)
    
    # Compute weights: high weight if distance < kappa
    otu_wts_unnorm <- torch_sigmoid((kappa_transformed - self$dist) * k)
    
    # Store weights for inspection
    self$wts <- otu_wts_unnorm
    
    # Check for NaN
    if (torch_isnan(otu_wts_unnorm)$any()$item()) {
      print(otu_wts_unnorm$sum(dim = -1L))
      print(self$kappa)
      stop("NaN in spatial aggregation!")
    }
    
    # Aggregation of time-series along OTU dimension
    # Einstein summation: (rules, otus, otus) x (batch, otus, time) -> (batch, rules, otus, time)
    x_agg <- torch_einsum("kij,sjt->skit", list(otu_wts_unnorm, x))
    
    return(x_agg)
  },
  
  #' @description Initialize layer parameters
  #' @param init_args Named list with 'kappa_init' containing initial kappa values
  init_params = function(init_args) {
    if (!"kappa_init" %in% names(init_args)) {
      stop("init_args must contain 'kappa_init'")
    }
    
    dist_max <- self$dist$max()$item()
    kappa_init_tensor <- torch_tensor(init_args$kappa_init, dtype = torch_float())
    self$kappa$data <- inv_transf_log(kappa_init_tensor, u = dist_max, l = 0)
  }
)


#' Spatial Aggregation Layer (Dynamic)
#'
#' @description
#' Dynamic spatial aggregation based on learned OTU embeddings.
#' Instead of using a fixed distance matrix, this layer learns OTU center
#' embeddings and computes distances dynamically in embedding space.
#' This provides more flexibility for the model to discover phylogenetic
#' patterns relevant to the prediction task.
#'
#' @section Architecture:
#' \describe{
#'   \item{Input}{(batch, num_otus, time_points)}
#'   \item{Output}{(batch, num_rules, num_otu_centers, time_points)}
#' }
#'
#' @param num_rules Integer; number of rule detectors
#' @param num_otu_centers Integer; number of OTU centers to learn
#' @param otu_embeddings Numeric matrix; OTU embedding matrix (num_otus x emb_dim)
#' @param emb_dim Integer; embedding dimension
#' @param num_otus Integer; total number of OTUs
#'
#' @export
#' @examples
#' \dontrun{
#' library(torch)
#' 
#' # Create example OTU embeddings
#' otu_embeddings <- matrix(rnorm(20 * 10), nrow = 20, ncol = 10)
#' 
#' # Initialize layer
#' layer <- spatial_agg_dynamic_layer(
#'   num_rules = 5,
#'   num_otu_centers = 3,
#'   otu_embeddings = otu_embeddings,
#'   emb_dim = 10,
#'   num_otus = 20
#' )
#' 
#' # Forward pass
#' x <- torch_randn(32, 20, 10)  # batch=32, otus=20, time=10
#' output <- layer(x)
#' print(output$shape)  # [32, 5, 3, 10]
#' }
spatial_agg_dynamic_layer <- nn_module(
  "SpatialAggDynamic",
  inherit = base_layer,
  
  #' @description Initialize dynamic spatial aggregation layer
  initialize = function(num_rules, num_otu_centers, otu_embeddings, emb_dim, num_otus) {
    config <- list(
      num_rules = num_rules,
      num_otu_centers = num_otu_centers,
      emb_dim = emb_dim,
      num_otus = num_otus
    )
    super$initialize(layer_name = "spatial_agg_dynamic", layer_config = config)
    
    self$num_rules <- num_rules
    self$num_otu_centers <- num_otu_centers
    self$emb_dim <- emb_dim
    self$num_otus <- num_otus
    
    # Register OTU embeddings as buffer (non-trainable)
    self$register_buffer(
      "dist",
      torch_tensor(otu_embeddings, dtype = torch_float())
    )
    
    # Learnable OTU centers in embedding space
    self$eta <- nn_parameter(torch_randn(num_rules, num_otu_centers, emb_dim) * 0.01)
    
    # OTU selection bandwidth (in log space for positivity)
    self$kappa <- nn_parameter(torch_randn(num_rules, num_otu_centers) * 0.1)
    
    # Storage for inspection
    self$wts <- NULL
    self$kappas <- NULL
    self$emb_dist <- NULL
  },
  
  #' @description Forward pass: aggregate OTUs based on learned embedding distances
  #' @param x Input tensor (batch, num_otus, time_points)
  #' @param k Temperature parameter for sigmoid sharpness (default: 1)
  #' @return Aggregated tensor (batch, num_rules, num_otu_centers, time_points)
  forward = function(x, k = 1) {
    # Compute unnormalized OTU weights based on embedding distance
    kappa <- self$kappa$exp()$unsqueeze(-1L)  # Ensure positive
    
    # Reshape eta for broadcasting: (rules, centers, 1, emb_dim)
    eta_reshaped <- self$eta$reshape(c(self$num_rules, self$num_otu_centers, 1, self$emb_dim))
    
    # Compute Euclidean distance in embedding space
    # dist shape: (rules, centers, otus)
    dist <- (eta_reshaped - self$dist)$norm(2, dim = -1L)
    
    # Compute weights: high weight if distance < kappa
    otu_wts_unnorm <- torch_sigmoid((kappa - dist) * k)
    
    # Store for inspection
    self$wts <- otu_wts_unnorm
    self$kappas <- kappa
    self$emb_dist <- dist
    
    # Check for NaN
    if (torch_isnan(otu_wts_unnorm)$any()$item()) {
      print(otu_wts_unnorm$sum(dim = -1L))
      print(self$kappa)
      stop("NaN in spatial aggregation!")
    }
    
    # Aggregation of time-series along OTU dimension
    # Einstein summation: (rules, centers, otus) x (batch, otus, time) -> (batch, rules, centers, time)
    x_agg <- torch_einsum("kij,sjt->skit", list(otu_wts_unnorm, x))
    
    return(x_agg)
  },
  
  #' @description Initialize layer parameters
  #' @param init_args Named list with 'kappa_init' and 'eta_init'
  init_params = function(init_args) {
    if (!"kappa_init" %in% names(init_args) || !"eta_init" %in% names(init_args)) {
      stop("init_args must contain 'kappa_init' and 'eta_init'")
    }
    
    # kappa is stored in log space
    self$kappa$data <- torch_tensor(init_args$kappa_init, dtype = torch_float())$log()
    self$eta$data <- torch_tensor(init_args$eta_init, dtype = torch_float())
  }
)

# Register layers
layer_registry$register("spatial_agg", spatial_agg_layer, version = "1.0.0")
layer_registry$register("spatial_agg_dynamic", spatial_agg_dynamic_layer, version = "1.0.0")
