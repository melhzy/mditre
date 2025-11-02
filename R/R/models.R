#' MDITRE Complete Models
#'
#' Full MDITRE models that assemble all 5 layers into end-to-end architectures
#' for microbiome time-series prediction.
#'
#' @name models
#' @rdname models
NULL

#' MDITRE Model (Full - with slopes)
#'
#' Complete MDITRE model that assembles all 5 layers:
#' Layer 1 (Phylogenetic Focus) → Layer 2 (Temporal Focus) → 
#' Layer 3 (Detectors) → Layer 4 (Rules) → Layer 5 (Classification)
#'
#' @param num_rules Number of rules to learn
#' @param num_otus Number of OTUs in the dataset
#' @param num_otu_centers Number of phylogenetic centers (for dynamic aggregation)
#' @param num_time Number of time points
#' @param num_time_centers Number of time window centers (kept for compatibility)
#' @param dist Phylogenetic distance matrix or OTU embeddings
#' @param emb_dim Embedding dimension for dynamic spatial aggregation
#'
#' @return An R6 class representing the complete MDITRE model
#'
#' @details
#' **Architecture**:
#' - **Layer 1**: SpatialAggDynamic - Phylogenetic focus using learned embeddings
#' - **Layer 2**: TimeAgg - Temporal focus with abundance and slope computation
#' - **Layer 3**: Threshold + Slope detectors - Gated responses for abundance and slopes
#' - **Layer 4**: Rules (2 instances) - Soft AND for abundance and slope rules
#' - **Layer 5**: DenseLayer - Classification with rule selection
#'
#' **Input**: (batch, num_otus, num_time) - OTU abundance time-series
#' **Output**: (batch,) - Log odds for binary classification
#'
#' **Temperature Parameters**:
#' - k_otu: Phylogenetic selection sharpness
#' - k_time: Temporal window sharpness
#' - k_thresh: Abundance threshold sharpness
#' - k_slope: Slope threshold sharpness
#' - k_alpha: Rule detector selection sharpness
#' - k_beta: Final rule selection sharpness
#'
#' @examples
#' \dontrun{
#' library(torch)
#' library(ape)
#'
#' # Setup
#' num_otus <- 50
#' num_time <- 10
#' tree <- rtree(num_otus)
#' phylo_dist <- cophenetic.phylo(tree)
#'
#' # Create model
#' model <- mditre_model(
#'   num_rules = 5,
#'   num_otus = num_otus,
#'   num_otu_centers = 10,
#'   num_time = num_time,
#'   num_time_centers = 1,
#'   dist = phylo_dist,
#'   emb_dim = 3
#' )
#'
#' # Forward pass
#' x <- torch_randn(32, num_otus, num_time)
#' predictions <- model(x)
#' print(predictions$shape)  # [32]
#'
#' # Get probabilities
#' probs <- torch_sigmoid(predictions)
#' }
#'
#' @export
mditre_model <- function(num_rules, num_otus, num_otu_centers,
                         num_time, num_time_centers, dist, emb_dim) {
  # Define the complete MDITRE model
  MDITREModule <- nn_module(
    classname = "MDITREModule",
    
    initialize = function(num_rules, num_otus, num_otu_centers,
                         num_time, num_time_centers, dist, emb_dim) {
      # Layer 1: Phylogenetic Focus (Dynamic)
      self$spat_attn <- spatial_agg_dynamic_layer(
        num_rules = num_rules,
        num_otu_centers = num_otu_centers,
        dist = dist,
        emb_dim = emb_dim,
        num_otus = num_otus
      )
      
      # Layer 2: Temporal Focus
      self$time_attn <- time_agg_layer(
        num_rules = num_rules,
        num_otus = num_otu_centers,
        num_time = num_time,
        num_time_centers = num_time_centers
      )
      
      # Layer 3: Threshold Detector (for abundance)
      self$thresh_func <- threshold_layer(
        num_rules = num_rules,
        num_otus = num_otu_centers,
        num_time_centers = num_time_centers
      )
      
      # Layer 3: Slope Detector
      self$slope_func <- slope_layer(
        num_rules = num_rules,
        num_otus = num_otu_centers,
        num_time_centers = num_time_centers
      )
      
      # Layer 4: Rules (for abundance)
      self$rules <- rule_layer(
        num_rules = num_rules,
        num_otus = num_otu_centers,
        num_time_centers = num_time_centers
      )
      
      # Layer 4: Rules (for slope)
      self$rules_slope <- rule_layer(
        num_rules = num_rules,
        num_otus = num_otu_centers,
        num_time_centers = num_time_centers,
        layer_name = "rules_slope"
      )
      
      # Layer 5: Classification
      self$fc <- classification_layer(
        in_feat = num_rules,
        out_feat = 1
      )
    },
    
    forward = function(x, mask = NULL, k_alpha = 1, k_beta = 1,
                      k_otu = 1.0, k_time = 1.0, k_thresh = 1.0, k_slope = 1.0,
                      hard = FALSE, use_noise = TRUE) {
      # Layer 1: Phylogenetic aggregation
      x <- self$spat_attn(x, k = k_otu)
      
      # Layer 2: Temporal aggregation (returns list with abundance and slope)
      time_result <- self$time_attn(x, mask = mask, k = k_time)
      x <- time_result$abundance
      x_slope <- time_result$slope
      
      # Layer 3: Threshold detection (abundance)
      x <- self$thresh_func(x, k = k_thresh)
      
      # Layer 4: Rules (abundance)
      x <- self$rules(x, hard = hard, k = k_alpha, use_noise = use_noise)
      
      # Layer 3: Slope detection
      x_slope <- self$slope_func(x_slope, k = k_slope)
      
      # Layer 4: Rules (slope)
      x_slope <- self$rules_slope(x_slope, hard = hard, k = k_alpha, use_noise = use_noise)
      
      # Layer 5: Classification
      x <- self$fc(x, x_slope = x_slope, hard = hard, k = k_beta, use_noise = use_noise)
      
      return(x)
    },
    
    init_params = function(init_args) {
      # Initialize all layers that have init_params method
      self$spat_attn$init_params(init_args)
      self$time_attn$init_params(init_args)
      self$thresh_func$init_params(init_args)
      self$slope_func$init_params(init_args)
      self$rules$init_params(init_args)
      self$rules_slope$init_params(init_args)
      self$fc$init_params(init_args)
    }
  )
  
  # Create and return model instance
  model <- MDITREModule(
    num_rules, num_otus, num_otu_centers,
    num_time, num_time_centers, dist, emb_dim
  )
  
  return(model)
}


#' MDITRE Model (Abundance Only - no slopes)
#'
#' Simplified MDITRE model that only uses abundance information (no slope computation).
#' Useful when rate-of-change is not informative or data is too sparse.
#'
#' @param num_rules Number of rules to learn
#' @param num_otus Number of OTUs in the dataset
#' @param num_otu_centers Number of phylogenetic centers (for dynamic aggregation)
#' @param num_time Number of time points
#' @param num_time_centers Number of time window centers (kept for compatibility)
#' @param dist Phylogenetic distance matrix or OTU embeddings
#' @param emb_dim Embedding dimension for dynamic spatial aggregation
#'
#' @return An R6 class representing the MDITRE Abundance-only model
#'
#' @details
#' **Architecture**:
#' - **Layer 1**: SpatialAggDynamic - Phylogenetic focus using learned embeddings
#' - **Layer 2**: TimeAggAbun - Temporal focus (abundance only, no slopes)
#' - **Layer 3**: Threshold detector - Gated response for abundance only
#' - **Layer 4**: Rules (1 instance) - Soft AND for abundance rules
#' - **Layer 5**: DenseLayerAbun - Classification with rule selection (no slopes)
#'
#' **Input**: (batch, num_otus, num_time) - OTU abundance time-series
#' **Output**: (batch,) - Log odds for binary classification
#'
#' This variant is faster and has fewer parameters than the full MDITRE model.
#'
#' @examples
#' \dontrun{
#' library(torch)
#' library(ape)
#'
#' # Setup
#' num_otus <- 50
#' num_time <- 10
#' tree <- rtree(num_otus)
#' phylo_dist <- cophenetic.phylo(tree)
#'
#' # Create abundance-only model
#' model <- mditre_abun_model(
#'   num_rules = 5,
#'   num_otus = num_otus,
#'   num_otu_centers = 10,
#'   num_time = num_time,
#'   num_time_centers = 1,
#'   dist = phylo_dist,
#'   emb_dim = 3
#' )
#'
#' # Forward pass
#' x <- torch_randn(32, num_otus, num_time)
#' predictions <- model(x)
#' }
#'
#' @export
mditre_abun_model <- function(num_rules, num_otus, num_otu_centers,
                               num_time, num_time_centers, dist, emb_dim) {
  # Define the MDITRE Abundance-only model
  MDITREAbunModule <- nn_module(
    classname = "MDITREAbunModule",
    
    initialize = function(num_rules, num_otus, num_otu_centers,
                         num_time, num_time_centers, dist, emb_dim) {
      # Layer 1: Phylogenetic Focus (Dynamic)
      self$spat_attn <- spatial_agg_dynamic_layer(
        num_rules = num_rules,
        num_otu_centers = num_otu_centers,
        dist = dist,
        emb_dim = emb_dim,
        num_otus = num_otus
      )
      
      # Layer 2: Temporal Focus (Abundance only)
      self$time_attn <- time_agg_abun_layer(
        num_rules = num_rules,
        num_otus = num_otu_centers,
        num_time = num_time,
        num_time_centers = num_time_centers
      )
      
      # Layer 3: Threshold Detector (for abundance)
      self$thresh_func <- threshold_layer(
        num_rules = num_rules,
        num_otus = num_otu_centers,
        num_time_centers = num_time_centers
      )
      
      # Layer 4: Rules (for abundance)
      self$rules <- rule_layer(
        num_rules = num_rules,
        num_otus = num_otu_centers,
        num_time_centers = num_time_centers
      )
      
      # Layer 5: Classification (Abundance only)
      self$fc <- classification_abun_layer(
        in_feat = num_rules,
        out_feat = 1
      )
    },
    
    forward = function(x, mask = NULL, k_alpha = 1, k_beta = 1,
                      k_otu = 1.0, k_time = 1.0, k_thresh = 1.0, k_slope = 1.0,
                      hard = FALSE, use_noise = TRUE) {
      # Layer 1: Phylogenetic aggregation
      x <- self$spat_attn(x, k = k_otu)
      
      # Layer 2: Temporal aggregation (abundance only)
      x <- self$time_attn(x, mask = mask, k = k_time)
      
      # Layer 3: Threshold detection
      x <- self$thresh_func(x, k = k_thresh)
      
      # Layer 4: Rules
      x <- self$rules(x, hard = hard, k = k_alpha, use_noise = use_noise)
      
      # Layer 5: Classification
      x <- self$fc(x, hard = hard, k = k_beta, use_noise = use_noise)
      
      return(x)
    },
    
    init_params = function(init_args) {
      # Initialize all layers that have init_params method
      self$spat_attn$init_params(init_args)
      self$time_attn$init_params(init_args)
      self$thresh_func$init_params(init_args)
      self$rules$init_params(init_args)
      self$fc$init_params(init_args)
    }
  )
  
  # Create and return model instance
  model <- MDITREAbunModule(
    num_rules, num_otus, num_otu_centers,
    num_time, num_time_centers, dist, emb_dim
  )
  
  return(model)
}
