#' Core Mathematical Utilities for MDITRE
#'
#' @description
#' Mathematical operations and helper functions used across multiple MDITRE layers.
#' Includes differentiable relaxations for discrete operations.
#'
#' @name math_utils
NULL

# Constants
#' @keywords internal
EPSILON <- .Machine$double.eps


#' Binary Concrete Relaxation
#'
#' @description
#' Binary concrete (Gumbel-Softmax) relaxation for differentiable discrete selection.
#' This enables gradient-based optimization of discrete choices.
#'
#' @param x Input tensor
#' @param k Temperature parameter (higher = sharper distribution)
#' @param hard Logical; whether to use straight-through estimator for discrete output
#' @param use_noise Logical; whether to add Gumbel noise for stochasticity
#'
#' @return Tensor with relaxed binary values in [0, 1]
#'
#' @references
#' Maddison, C. J., Mnih, A., & Teh, Y. W. (2016). The concrete distribution:
#' A continuous relaxation of discrete random variables. ICLR 2017.
#'
#' @export
#' @examples
#' \dontrun{
#' library(torch)
#' x <- torch_randn(10, 5)
#' # Soft relaxation
#' z_soft <- binary_concrete(x, k = 10, hard = FALSE)
#' # Hard (discrete) with gradient
#' z_hard <- binary_concrete(x, k = 10, hard = TRUE)
#' }
binary_concrete <- function(x, k, hard = FALSE, use_noise = TRUE) {
  if (use_noise) {
    # Sample Gumbel noise: -log(-log(U))
    u <- torch_rand_like(x) + EPSILON
    logs <- torch_log(u) - torch_log1p(-u)
    z_soft <- torch_sigmoid((x + logs) * k)
  } else {
    z_soft <- torch_sigmoid(x * k)
  }
  
  # Straight-through estimator: forward = hard, backward = soft
  if (hard) {
    z_hard <- (z_soft > 0.5)$to(dtype = torch_float())
    z <- (z_hard - z_soft$detach()) + z_soft
  } else {
    z <- z_soft
  }
  
  return(z)
}


#' Unit Height Boxcar Function
#'
#' @description
#' Approximate a unit height boxcar (rectangular) function using analytic
#' approximations of the Heaviside function. Used for soft temporal windowing.
#'
#' @param x Input tensor (typically time points)
#' @param mu Center position of the boxcar
#' @param l Length of the boxcar
#' @param k Sharpness parameter (higher = sharper edges)
#'
#' @return Tensor with boxcar function values
#'
#' @details
#' The boxcar is parameterized by its center (mu) and length (l).
#' The function smoothly transitions from 0 to 1 at the edges.
#'
#' @export
#' @examples
#' \dontrun{
#' library(torch)
#' # Create time points
#' x <- torch_linspace(0, 1, 100)
#' # Boxcar centered at 0.5 with length 0.2
#' y <- unitboxcar(x, mu = 0.5, l = 0.2, k = 10)
#' plot(as_array(x), as_array(y), type = "l")
#' }
unitboxcar <- function(x, mu, l, k) {
  # Distance from center
  dist <- x - mu
  
  # Half-width of window
  window_half <- l / 2.0
  
  # Approximate boxcar using difference of sigmoids
  y <- torch_sigmoid((dist + window_half) * k) - 
       torch_sigmoid((dist - window_half) * k)
  
  return(y)
}


#' Soft AND Operation
#'
#' @description
#' Differentiable AND operation using product. Approximates logical AND
#' for continuous values in [0, 1].
#'
#' @param x Input tensor with values in [0, 1]
#' @param dim Integer; dimension along which to compute AND (default: -1)
#' @param epsilon Small constant for numerical stability
#'
#' @return Tensor with soft AND values
#'
#' @details
#' For binary inputs, product approximates AND. For continuous inputs,
#' it acts as a "soft" AND where the output is high only if all inputs are high.
#'
#' @export
#' @examples
#' \dontrun{
#' library(torch)
#' # Multiple conditions
#' x <- torch_tensor(rbind(c(0.9, 0.8, 0.95), c(0.1, 0.9, 0.8)))
#' result <- soft_and(x, dim = 2L)  # AND across columns
#' }
soft_and <- function(x, dim = -1L, epsilon = 1e-10) {
  return(torch_prod(x + epsilon, dim = dim))
}


#' Soft OR Operation
#'
#' @description
#' Differentiable OR operation. Approximates logical OR for continuous
#' values in [0, 1].
#'
#' @param x Input tensor with values in [0, 1]
#' @param dim Integer; dimension along which to compute OR (default: -1)
#' @param epsilon Small constant for numerical stability
#'
#' @return Tensor with soft OR values
#'
#' @details
#' Uses the identity: OR(a,b) = 1 - AND(1-a, 1-b) = 1 - (1-a)*(1-b)
#'
#' @export
#' @examples
#' \dontrun{
#' library(torch)
#' x <- torch_tensor(rbind(c(0.1, 0.2, 0.15), c(0.9, 0.1, 0.2)))
#' result <- soft_or(x, dim = 2L)  # OR across columns
#' }
soft_or <- function(x, dim = -1L, epsilon = 1e-10) {
  return(1 - torch_prod(1 - x + epsilon, dim = dim))
}


#' Logarithmic Transformation with Bounds
#'
#' @description
#' Transform values from unbounded space to bounded interval [l, u]
#' using sigmoid function.
#'
#' @param x Input tensor
#' @param u Upper bound
#' @param l Lower bound
#'
#' @return Transformed values in [l, u]
#'
#' @export
#' @examples
#' \dontrun{
#' library(torch)
#' x <- torch_randn(10)
#' # Transform to [0, 1]
#' y <- transf_log(x, u = 1, l = 0)
#' }
transf_log <- function(x, u, l) {
  return((u - l) * torch_sigmoid(x) + l)
}


#' Inverse Logarithmic Transformation
#'
#' @description
#' Inverse of transf_log. Transform values from [l, u] back to unbounded space.
#'
#' @param x Input tensor in [l, u]
#' @param u Upper bound
#' @param l Lower bound
#'
#' @return Transformed values (unbounded)
#'
#' @details
#' Uses logit function (inverse sigmoid): logit(p) = log(p / (1-p))
#'
#' @export
#' @examples
#' \dontrun{
#' library(torch)
#' # Forward and inverse should be identity
#' x <- torch_randn(10)
#' y <- transf_log(x, u = 1, l = 0)
#' x_recovered <- inv_transf_log(y, u = 1, l = 0)
#' }
inv_transf_log <- function(x, u, l) {
  p <- (x - l) / (u - l)
  # Clamp to avoid numerical issues
  p <- torch_clamp(p, min = EPSILON, max = 1 - EPSILON)
  return(torch_log(p) - torch_log(1 - p))  # logit
}
