"""
Core utilities and helper functions for MDITRE
Contains mathematical operations used across multiple layers
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import logit

# Lowest possible float
EPSILON = np.finfo(np.float32).tiny


def binary_concrete(x, k, hard=False, use_noise=True):
    """
    Binary concrete relaxation for differentiable discrete selection
    
    Args:
        x: Input tensor
        k: Temperature parameter (higher = sharper)
        hard: Whether to use straight-through estimator
        use_noise: Whether to add Gumbel noise
        
    Returns:
        Relaxed binary values
    """
    if use_noise:
        u = torch.zeros_like(x.data).uniform_(0, 1) + torch.tensor([EPSILON]).to(x.device)
        logs = u.log() - (-u).log1p()
        z_soft = torch.sigmoid((x + logs) * k)
    else:
        z_soft = torch.sigmoid(x * k)

    # Straight through estimator
    if hard:
        z = (z_soft > 0.5).float() - z_soft.detach() + z_soft
    else:
        z = z_soft

    return z


def unitboxcar(x, mu, l, k):
    """
    Approximate a unit height boxcar function using analytic 
    approximations of the Heaviside function
    
    Args:
        x: Input tensor
        mu: Center of the boxcar
        l: Length of the boxcar
        k: Sharpness parameter
        
    Returns:
        Boxcar function values
    """
    # Parameterize boxcar function by the center and length
    dist = x - mu
    window_half = l / 2.
    y = torch.sigmoid((dist + window_half) * k) - torch.sigmoid((dist - window_half) * k)
    return y


def transf_log(x, u, l):
    """
    Logarithmic transformation with bounds
    
    Args:
        x: Input tensor
        u: Upper bound
        l: Lower bound
        
    Returns:
        Transformed values
    """
    return (u - l) * torch.sigmoid(x) + l


def inv_transf_log(x, u, l):
    """
    Inverse logarithmic transformation
    
    Args:
        x: Input tensor
        u: Upper bound
        l: Lower bound
        
    Returns:
        Inverse transformed values
    """
    return logit((x - l) / (u - l))
