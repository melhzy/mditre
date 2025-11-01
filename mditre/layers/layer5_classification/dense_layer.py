"""
Dense Classification Layers

Linear classifiers with rule selection for computing predicted outcomes.
Uses binary concrete to select which rules contribute to final prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from ...core.base_layer import BaseLayer, LayerRegistry
from ...core.math_utils import binary_concrete


@LayerRegistry.register('layer5', 'dense_layer')
class DenseLayer(BaseLayer):
    """
    Linear classifier for computing predicted outcome.
    
    Combines rule responses and slope information using learned weights,
    with binary concrete selection of active rules.
    
    Architecture:
        Input: (x, x_slope) both (batch, num_rules)
        Output: (batch,) - log odds for binary classification
    """
    
    def __init__(self, in_feat: int, out_feat: int, 
                 layer_name: str = "dense_layer", **kwargs):
        """
        Initialize dense classification layer
        
        Args:
            in_feat: Number of input features (rules)
            out_feat: Number of output classes
            layer_name: Name for this layer instance
        """
        config = {
            'in_feat': in_feat,
            'out_feat': out_feat
        }
        super(DenseLayer, self).__init__(layer_name, config)
        
        # Logistic regression coefficients
        self.weight = nn.Parameter(torch.Tensor(out_feat, in_feat))
        
        # Logistic regression bias
        self.bias = nn.Parameter(torch.Tensor(out_feat))
        
        # Parameter for selecting active rules
        self.beta = nn.Parameter(torch.Tensor(in_feat))
        
    def forward(self, x: torch.Tensor, x_slope: Optional[torch.Tensor] = None, 
                k: float = 1., hard: bool = False, use_noise: bool = True, 
                **kwargs) -> torch.Tensor:
        """
        Forward pass: classify using selected rules
        
        Args:
            x: Rule responses (batch, num_rules)
            x_slope: Slope rule responses (batch, num_rules) - required for this layer
            k: Temperature parameter for binary concrete
            hard: Whether to use straight-through estimator
            use_noise: Whether to use Gumbel noise in training
            
        Returns:
            Log odds predictions (batch,)
        """
        if x_slope is None:
            raise ValueError("DenseLayer requires x_slope argument")
            
        if self.training:
            # Binary concrete for rule selection
            if use_noise:
                z = binary_concrete(self.beta, k, hard=hard)
            else:
                z = binary_concrete(self.beta, k, hard=hard, use_noise=False)
        else:
            # During evaluation
            z = binary_concrete(self.beta, k, hard=hard, use_noise=False)
        
        # Store sub-components for inspection
        self.sub_log_odds = ((x * x_slope) * 
                            ((self.weight * z.unsqueeze(0)).reshape(-1))) + self.bias
        
        # Predict the outcome
        x = F.linear(x * x_slope, self.weight * z.unsqueeze(0), self.bias)
        
        self.z = z
        self.log_odds = x.squeeze(-1)
        
        return x.squeeze(-1)
    
    def init_params(self, init_args: dict) -> None:
        """
        Initialize layer parameters
        
        Args:
            init_args: Dictionary with 'w_init', 'bias_init', 'beta_init' keys
        """
        self.weight.data = torch.from_numpy(init_args['w_init']).float()
        self.bias.data = torch.from_numpy(init_args['bias_init']).float()
        self.beta.data = torch.from_numpy(init_args['beta_init']).float()


@LayerRegistry.register('layer5', 'dense_layer_abun')
class DenseLayerAbun(BaseLayer):
    """
    Linear classifier for abundance-only models.
    
    Simplified version that only uses abundance (not slope) for classification.
    
    Architecture:
        Input: (batch, num_rules)
        Output: (batch,) - log odds for binary classification
    """
    
    def __init__(self, in_feat: int, out_feat: int, 
                 layer_name: str = "dense_layer_abun", **kwargs):
        """
        Initialize dense classification layer (abundance only)
        
        Args:
            in_feat: Number of input features (rules)
            out_feat: Number of output classes
            layer_name: Name for this layer instance
        """
        config = {
            'in_feat': in_feat,
            'out_feat': out_feat
        }
        super(DenseLayerAbun, self).__init__(layer_name, config)
        
        # Logistic regression coefficients
        self.weight = nn.Parameter(torch.Tensor(out_feat, in_feat))
        
        # Logistic regression bias
        self.bias = nn.Parameter(torch.Tensor(out_feat))
        
        # Parameter for selecting active rules
        self.beta = nn.Parameter(torch.Tensor(in_feat))
        
    def forward(self, x: torch.Tensor, k: float = 1., hard: bool = False, 
                use_noise: bool = True, **kwargs) -> torch.Tensor:
        """
        Forward pass: classify using selected rules (abundance only)
        
        Args:
            x: Rule responses (batch, num_rules)
            k: Temperature parameter for binary concrete
            hard: Whether to use straight-through estimator
            use_noise: Whether to use Gumbel noise in training
            
        Returns:
            Log odds predictions (batch,)
        """
        if self.training:
            # Binary concrete for rule selection
            if use_noise:
                z = binary_concrete(self.beta, k, hard=hard)
            else:
                z = binary_concrete(self.beta, k, hard=hard, use_noise=False)
        else:
            # During evaluation
            z = binary_concrete(self.beta, k, hard=hard, use_noise=False)
        
        # Store sub-components for inspection
        self.sub_log_odds = ((x) * 
                            ((self.weight * z.unsqueeze(0)).reshape(-1))) + self.bias
        
        # Predict the outcome
        x = F.linear(x, self.weight * z.unsqueeze(0), self.bias)
        
        self.log_odds = x.squeeze(-1)
        self.z = z
        
        return x.squeeze(-1)
    
    def init_params(self, init_args: dict) -> None:
        """
        Initialize layer parameters
        
        Args:
            init_args: Dictionary with 'w_init', 'bias_init', 'beta_init' keys
        """
        self.weight.data = torch.from_numpy(init_args['w_init']).float()
        self.bias.data = torch.from_numpy(init_args['bias_init']).float()
        self.beta.data = torch.from_numpy(init_args['beta_init']).float()
