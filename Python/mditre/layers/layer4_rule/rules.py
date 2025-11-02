"""
Rule Layer

Combines detector responses to compute approximate logical AND operations
as rule responses, enabling interpretable decision logic.
"""

import torch
import torch.nn as nn
import numpy as np

from ...core.base_layer import BaseLayer, LayerRegistry
from ...core.math_utils import binary_concrete


@LayerRegistry.register('layer4', 'rules')
class Rules(BaseLayer):
    """
    Combine detector responses using approximate logical AND.
    
    Uses binary concrete relaxation to select which detectors contribute
    to each rule, then approximates AND operation via product:
    AND(x1, x2, ..., xn) ≈ ∏(1 - αi(1 - xi))
    
    Architecture:
        Input: (batch, num_rules, num_otus)
        Output: (batch, num_rules)
    """
    
    def __init__(self, num_rules: int, num_otus: int, num_time_centers: int,
                 layer_name: str = "rules", **kwargs):
        """
        Initialize rule layer
        
        Args:
            num_rules: Number of rules
            num_otus: Number of OTUs (detectors per rule)
            num_time_centers: Number of time centers (unused, kept for compatibility)
            layer_name: Name for this layer instance
        """
        config = {
            'num_rules': num_rules,
            'num_otus': num_otus,
            'num_time_centers': num_time_centers
        }
        super(Rules, self).__init__(layer_name, config)
        
        # Binary concrete selector variable for detectors
        self.alpha = nn.Parameter(torch.Tensor(num_rules, num_otus))
        
    def forward(self, x: torch.Tensor, k: float = 1., hard: bool = False, 
                use_noise: bool = True) -> torch.Tensor:
        """
        Forward pass: combine detector responses into rules
        
        Args:
            x: Input tensor (batch, num_rules, num_otus)
            k: Temperature parameter for binary concrete
            hard: Whether to use straight-through estimator (hard selection)
            use_noise: Whether to use Gumbel noise in training
            
        Returns:
            Rule responses (batch, num_rules)
        """
        if self.training:
            # Binary concrete for detector selection
            if use_noise:
                z = binary_concrete(self.alpha, k, hard=hard)
            else:
                z = binary_concrete(self.alpha, k, hard=hard, use_noise=False)
        else:
            z = binary_concrete(self.alpha, k, hard=hard, use_noise=False)
        
        # Store for inspection
        self.x = x
        self.z = z
        
        # Approximate logical AND operation
        # AND(x1, x2, ..., xn) ≈ ∏(1 - αi(1 - xi))
        x = (1 - z.mul(1 - x)).prod(dim=-1)
        
        return x
    
    def init_params(self, init_args: dict) -> None:
        """
        Initialize layer parameters
        
        Args:
            init_args: Dictionary with 'alpha_init' key containing selection probabilities
        """
        self.alpha.data = torch.from_numpy(init_args['alpha_init']).float()
