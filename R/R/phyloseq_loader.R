#' phyloseq Data Loader for MDITRE
#'
#' Functions to convert phyloseq objects to MDITRE-compatible tensor format.
#' Handles OTU abundance extraction, metadata processing, phylogenetic distance
#' computation, and train/test splitting.
#'
#' @name phyloseq_loader
#' @family data_loader
NULL

#' Convert phyloseq Object to MDITRE Format
#'
#' Converts a phyloseq object containing microbiome time-series data into
#' the tensor format required by MDITRE models.
#'
#' @param ps_data A phyloseq object containing OTU table, sample data, and
#'   phylogenetic tree
#' @param subject_col Character. Column name in sample_data identifying subjects
#' @param time_col Character. Column name in sample_data with time points
#' @param label_col Character. Column name in sample_data with binary labels (0/1)
#' @param normalize Logical. Whether to normalize OTU abundances (default: TRUE)
#' @param min_prevalence Numeric. Minimum prevalence threshold for OTU filtering (default: 0.1)
#' @param min_abundance Numeric. Minimum abundance threshold for OTU filtering (default: 0.0001)
#' @param log_transform Logical. Whether to log-transform abundances (default: FALSE)
#'
#' @return A list with components:
#'   \itemize{
#'     \item \code{X}: torch tensor of shape (n_subjects, n_otus, n_timepoints)
#'     \item \code{y}: torch tensor of shape (n_subjects) with binary labels
#'     \item \code{times}: torch tensor of shape (n_subjects, n_timepoints)
#'     \item \code{mask}: torch tensor indicating valid timepoints
#'     \item \code{phylo_dist}: phylogenetic distance matrix (n_otus, n_otus)
#'     \item \code{phylo_tree}: ape phylo object
#'     \item \code{metadata}: list with additional information
#'   }
#'
#' @export
#' @examples
#' \dontrun{
#' library(phyloseq)
#' library(torch)
#' 
#' # Load phyloseq data
#' data("GlobalPatterns")
#' 
#' # Convert to MDITRE format
#' mditre_data <- phyloseq_to_mditre(
#'   GlobalPatterns,
#'   subject_col = "SampleType",
#'   time_col = "Description",
#'   label_col = "Label"
#' )
#' 
#' # Access components
#' X <- mditre_data$X  # Abundance tensors
#' y <- mditre_data$y  # Labels
#' dist <- mditre_data$phylo_dist  # Distance matrix
#' }
phyloseq_to_mditre <- function(ps_data,
                               subject_col,
                               time_col,
                               label_col,
                               normalize = TRUE,
                               min_prevalence = 0.1,
                               min_abundance = 0.0001,
                               log_transform = FALSE) {
  
  # Check dependencies
  if (!requireNamespace("phyloseq", quietly = TRUE)) {
    stop("Package 'phyloseq' is required. Please install from Bioconductor.")
  }
  if (!requireNamespace("ape", quietly = TRUE)) {
    stop("Package 'ape' is required. Please install it.")
  }
  
  # Validate phyloseq object
  if (!inherits(ps_data, "phyloseq")) {
    stop("ps_data must be a phyloseq object")
  }
  
  # Extract components
  otu_table <- phyloseq::otu_table(ps_data)
  sample_data <- phyloseq::sample_data(ps_data)
  tax_table <- phyloseq::tax_table(ps_data)
  phy_tree <- phyloseq::phy_tree(ps_data)
  
  # Validate required columns
  if (!subject_col %in% colnames(sample_data)) {
    stop(sprintf("Column '%s' not found in sample_data", subject_col))
  }
  if (!time_col %in% colnames(sample_data)) {
    stop(sprintf("Column '%s' not found in sample_data", time_col))
  }
  if (!label_col %in% colnames(sample_data)) {
    stop(sprintf("Column '%s' not found in sample_data", label_col))
  }
  
  # Convert OTU table to matrix (samples x OTUs)
  if (phyloseq::taxa_are_rows(ps_data)) {
    otu_mat <- t(as.matrix(otu_table))
  } else {
    otu_mat <- as.matrix(otu_table)
  }
  
  # Filter low abundance/prevalence OTUs
  otu_mat <- filter_otus(
    otu_mat,
    min_prevalence = min_prevalence,
    min_abundance = min_abundance
  )
  
  # Normalize if requested
  if (normalize) {
    otu_mat <- normalize_abundance(otu_mat)
  }
  
  # Log transform if requested
  if (log_transform) {
    otu_mat <- log10(otu_mat + 1)
  }
  
  # Extract metadata
  subjects <- as.character(sample_data[[subject_col]])
  times <- as.numeric(sample_data[[time_col]])
  labels <- as.numeric(sample_data[[label_col]])
  
  # Organize by subject
  subject_data <- organize_by_subject(
    otu_mat = otu_mat,
    subjects = subjects,
    times = times,
    labels = labels
  )
  
  # Convert to tensors
  X_tensor <- torch_tensor(subject_data$X, dtype = torch_float32())
  y_tensor <- torch_tensor(subject_data$y, dtype = torch_float32())
  times_tensor <- torch_tensor(subject_data$times, dtype = torch_float32())
  mask_tensor <- torch_tensor(subject_data$mask, dtype = torch_float32())
  
  # Compute phylogenetic distance
  phylo_dist <- compute_phylo_distance(phy_tree, colnames(otu_mat))
  
  # Return structured list
  list(
    X = X_tensor,
    y = y_tensor,
    times = times_tensor,
    mask = mask_tensor,
    phylo_dist = phylo_dist,
    phylo_tree = phy_tree,
    metadata = list(
      n_subjects = nrow(subject_data$X),
      n_otus = ncol(subject_data$X),
      n_timepoints = dim(subject_data$X)[3],
      otu_names = colnames(otu_mat),
      subject_ids = unique(subjects),
      normalize = normalize,
      min_prevalence = min_prevalence,
      min_abundance = min_abundance,
      log_transform = log_transform
    )
  )
}


#' Filter OTUs by Prevalence and Abundance
#'
#' @param otu_mat Matrix of OTU abundances (samples x OTUs)
#' @param min_prevalence Minimum fraction of samples OTU must appear in
#' @param min_abundance Minimum mean abundance threshold
#'
#' @return Filtered OTU matrix
#' @keywords internal
filter_otus <- function(otu_mat, min_prevalence = 0.1, min_abundance = 0.0001) {
  
  n_samples <- nrow(otu_mat)
  
  # Calculate prevalence (fraction of samples with non-zero abundance)
  prevalence <- colSums(otu_mat > 0) / n_samples
  
  # Calculate mean abundance
  mean_abundance <- colMeans(otu_mat)
  
  # Filter OTUs
  keep_otus <- (prevalence >= min_prevalence) & (mean_abundance >= min_abundance)
  
  if (sum(keep_otus) == 0) {
    stop("No OTUs passed filtering criteria. Consider relaxing thresholds.")
  }
  
  otu_mat_filtered <- otu_mat[, keep_otus, drop = FALSE]
  
  message(sprintf(
    "Filtered OTUs: %d -> %d (prevalence >= %.2f, abundance >= %.4f)",
    ncol(otu_mat), ncol(otu_mat_filtered), min_prevalence, min_abundance
  ))
  
  return(otu_mat_filtered)
}


#' Normalize OTU Abundances
#'
#' Performs total sum scaling (relative abundance) normalization.
#'
#' @param otu_mat Matrix of OTU abundances (samples x OTUs)
#'
#' @return Normalized OTU matrix
#' @keywords internal
normalize_abundance <- function(otu_mat) {
  
  # Total sum scaling
  row_sums <- rowSums(otu_mat)
  
  # Avoid division by zero
  row_sums[row_sums == 0] <- 1
  
  otu_mat_norm <- otu_mat / row_sums
  
  return(otu_mat_norm)
}


#' Organize Data by Subject
#'
#' Reshapes flat sample x OTU matrix into subject x OTU x time tensor.
#'
#' @param otu_mat Matrix of OTU abundances (samples x OTUs)
#' @param subjects Vector of subject IDs for each sample
#' @param times Vector of time points for each sample
#' @param labels Vector of labels for each sample
#'
#' @return List with X (3D array), y (vector), times (matrix), mask (matrix)
#' @keywords internal
organize_by_subject <- function(otu_mat, subjects, times, labels) {
  
  unique_subjects <- unique(subjects)
  n_subjects <- length(unique_subjects)
  n_otus <- ncol(otu_mat)
  
  # Determine max time points per subject
  max_timepoints <- max(table(subjects))
  
  # Initialize arrays
  X <- array(0, dim = c(n_subjects, n_otus, max_timepoints))
  times_mat <- matrix(0, nrow = n_subjects, ncol = max_timepoints)
  mask_mat <- matrix(0, nrow = n_subjects, ncol = max_timepoints)
  y <- numeric(n_subjects)
  
  # Fill arrays
  for (i in seq_along(unique_subjects)) {
    subj <- unique_subjects[i]
    subj_idx <- which(subjects == subj)
    
    # Get data for this subject
    subj_otus <- otu_mat[subj_idx, , drop = FALSE]
    subj_times <- times[subj_idx]
    subj_label <- labels[subj_idx][1]  # Assume constant label per subject
    
    # Sort by time
    time_order <- order(subj_times)
    subj_otus <- subj_otus[time_order, , drop = FALSE]
    subj_times <- subj_times[time_order]
    
    n_tp <- length(subj_idx)
    
    # Fill arrays
    X[i, , 1:n_tp] <- t(subj_otus)
    times_mat[i, 1:n_tp] <- subj_times
    mask_mat[i, 1:n_tp] <- 1
    y[i] <- subj_label
  }
  
  list(
    X = X,
    y = y,
    times = times_mat,
    mask = mask_mat
  )
}


#' Compute Phylogenetic Distance Matrix
#'
#' Computes pairwise phylogenetic distances between OTUs using the
#' phylogenetic tree.
#'
#' @param phy_tree An ape phylo object
#' @param otu_names Character vector of OTU names to include
#'
#' @return Numeric matrix of pairwise distances (n_otus x n_otus)
#' @keywords internal
compute_phylo_distance <- function(phy_tree, otu_names) {
  
  if (!requireNamespace("ape", quietly = TRUE)) {
    stop("Package 'ape' is required")
  }
  
  # Prune tree to included OTUs
  tree_tips <- phy_tree$tip.label
  otus_in_tree <- intersect(otu_names, tree_tips)
  
  if (length(otus_in_tree) < length(otu_names)) {
    warning(sprintf(
      "%d OTUs not found in phylogenetic tree",
      length(otu_names) - length(otus_in_tree)
    ))
  }
  
  # Prune tree
  pruned_tree <- ape::keep.tip(phy_tree, otus_in_tree)
  
  # Compute cophenetic distance
  phylo_dist <- ape::cophenetic.phylo(pruned_tree)
  
  # Ensure same order as otu_names
  phylo_dist <- phylo_dist[otus_in_tree, otus_in_tree]
  
  return(phylo_dist)
}


#' Split Data into Train and Test Sets
#'
#' Splits MDITRE-formatted data into training and testing sets.
#'
#' @param mditre_data List returned from \code{phyloseq_to_mditre}
#' @param test_fraction Numeric. Fraction of subjects to use for testing (default: 0.2)
#' @param stratify Logical. Whether to stratify split by labels (default: TRUE)
#' @param seed Integer. Random seed for reproducibility (default: NULL)
#'
#' @return List with \code{train} and \code{test} components, each containing
#'   the same structure as input \code{mditre_data}
#'
#' @export
#' @examples
#' \dontrun{
#' # Split data
#' split_data <- split_train_test(mditre_data, test_fraction = 0.2, seed = 42)
#' 
#' # Access train/test sets
#' X_train <- split_data$train$X
#' X_test <- split_data$test$X
#' }
split_train_test <- function(mditre_data,
                             test_fraction = 0.2,
                             stratify = TRUE,
                             seed = NULL) {
  
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  n_subjects <- mditre_data$y$shape[1]
  
  if (stratify) {
    # Stratified split
    y_vec <- as.numeric(mditre_data$y$cpu())
    
    # Split by class
    idx_0 <- which(y_vec == 0)
    idx_1 <- which(y_vec == 1)
    
    n_test_0 <- round(length(idx_0) * test_fraction)
    n_test_1 <- round(length(idx_1) * test_fraction)
    
    test_idx_0 <- sample(idx_0, n_test_0)
    test_idx_1 <- sample(idx_1, n_test_1)
    
    test_idx <- c(test_idx_0, test_idx_1)
    train_idx <- setdiff(seq_len(n_subjects), test_idx)
    
  } else {
    # Random split
    n_test <- round(n_subjects * test_fraction)
    test_idx <- sample(seq_len(n_subjects), n_test)
    train_idx <- setdiff(seq_len(n_subjects), test_idx)
  }
  
  # Create train set
  train_data <- list(
    X = mditre_data$X[train_idx, , ],
    y = mditre_data$y[train_idx],
    times = mditre_data$times[train_idx, ],
    mask = mditre_data$mask[train_idx, ],
    phylo_dist = mditre_data$phylo_dist,
    phylo_tree = mditre_data$phylo_tree,
    metadata = mditre_data$metadata
  )
  
  # Create test set
  test_data <- list(
    X = mditre_data$X[test_idx, , ],
    y = mditre_data$y[test_idx],
    times = mditre_data$times[test_idx, ],
    mask = mditre_data$mask[test_idx, ],
    phylo_dist = mditre_data$phylo_dist,
    phylo_tree = mditre_data$phylo_tree,
    metadata = mditre_data$metadata
  )
  
  # Update metadata
  train_data$metadata$split <- "train"
  train_data$metadata$n_subjects <- length(train_idx)
  test_data$metadata$split <- "test"
  test_data$metadata$n_subjects <- length(test_idx)
  
  list(
    train = train_data,
    test = test_data
  )
}


#' Create torch DataLoader for MDITRE Data
#'
#' Wraps MDITRE-formatted data in a torch dataset and dataloader.
#'
#' @param mditre_data List returned from \code{phyloseq_to_mditre}
#' @param batch_size Integer. Batch size for training (default: 32)
#' @param shuffle Logical. Whether to shuffle data (default: TRUE)
#'
#' @return torch dataloader object
#'
#' @export
#' @examples
#' \dontrun{
#' # Create dataloader
#' train_loader <- create_dataloader(
#'   split_data$train,
#'   batch_size = 32,
#'   shuffle = TRUE
#' )
#' 
#' # Iterate through batches
#' coro::loop(for (batch in train_loader) {
#'   X_batch <- batch[[1]]
#'   y_batch <- batch[[2]]
#'   # Training step...
#' })
#' }
create_dataloader <- function(mditre_data,
                              batch_size = 32,
                              shuffle = TRUE) {
  
  # Create simple dataset
  dataset <- torch::dataset(
    name = "mditre_dataset",
    
    initialize = function(data) {
      self$X <- data$X
      self$y <- data$y
      self$times <- data$times
      self$mask <- data$mask
    },
    
    .getitem = function(idx) {
      list(
        X = self$X[idx, , ],
        y = self$y[idx],
        times = self$times[idx, ],
        mask = self$mask[idx, ]
      )
    },
    
    .length = function() {
      self$y$shape[1]
    }
  )
  
  # Create dataset instance
  ds <- dataset(mditre_data)
  
  # Create dataloader
  dataloader(
    ds,
    batch_size = batch_size,
    shuffle = shuffle
  )
}


#' Print Summary of MDITRE Data
#'
#' @param mditre_data List returned from \code{phyloseq_to_mditre}
#'
#' @export
print_mditre_data_summary <- function(mditre_data) {
  
  meta <- mditre_data$metadata
  
  cat("MDITRE Data Summary\n")
  cat("===================\n\n")
  
  cat(sprintf("Dimensions:\n"))
  cat(sprintf("  - Subjects: %d\n", meta$n_subjects))
  cat(sprintf("  - OTUs: %d\n", meta$n_otus))
  cat(sprintf("  - Max timepoints: %d\n", meta$n_timepoints))
  
  cat(sprintf("\nData shape:\n"))
  cat(sprintf("  - X: [%s]\n", paste(mditre_data$X$shape, collapse = ", ")))
  cat(sprintf("  - y: [%s]\n", paste(mditre_data$y$shape, collapse = ", ")))
  cat(sprintf("  - times: [%s]\n", paste(mditre_data$times$shape, collapse = ", ")))
  
  cat(sprintf("\nLabel distribution:\n"))
  y_vec <- as.numeric(mditre_data$y$cpu())
  cat(sprintf("  - Class 0: %d (%.1f%%)\n", 
              sum(y_vec == 0), 100 * mean(y_vec == 0)))
  cat(sprintf("  - Class 1: %d (%.1f%%)\n", 
              sum(y_vec == 1), 100 * mean(y_vec == 1)))
  
  cat(sprintf("\nPreprocessing:\n"))
  cat(sprintf("  - Normalized: %s\n", meta$normalize))
  cat(sprintf("  - Log-transformed: %s\n", meta$log_transform))
  cat(sprintf("  - Min prevalence: %.2f\n", meta$min_prevalence))
  cat(sprintf("  - Min abundance: %.4f\n", meta$min_abundance))
  
  cat(sprintf("\nPhylogenetic tree:\n"))
  cat(sprintf("  - Distance matrix: %d x %d\n", 
              nrow(mditre_data$phylo_dist), ncol(mditre_data$phylo_dist)))
  
  invisible(mditre_data)
}
