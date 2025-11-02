"""
Threshold Detector Layers

These layers learn thresholds for abundance and slope values,
producing gated responses that indicate whether aggregated values
exceed learned thresholds.
"""

import torch
import torch.nn as nn
import numpy as np

from ...core.base_layer import BaseLayer, LayerRegistry


@LayerRegistry.register('layer3', 'threshold')
class Threshold(BaseLayer):
    """
    Learn threshold abundance for each detector.
    
    The output is a sharp but smooth gated response indicating whether
    the aggregated abundance from previous steps is above/below the
    learned threshold.
    
    Architecture:
        Input: (batch, num_rules, num_otus)
        Output: (batch, num_rules, num_otus)
    """
    
    def __init__(self, num_rules: int, num_otus: int, num_time_centers: int,
                 layer_name: str = "threshold", **kwargs):
        """
        Initialize threshold detector layer
        
        Args:
            num_rules: Number of rule detectors
            num_otus: Number of OTUs
            num_time_centers: Number of time centers (unused, kept for compatibility)
            layer_name: Name for this layer instance
        """
        config = {
            'num_rules': num_rules,
            'num_otus': num_otus,
            'num_time_centers': num_time_centers
        }
        super(Threshold, self).__init__(layer_name, config)
        
        # Parameter for learnable threshold abundance
        self.thresh = nn.Parameter(torch.Tensor(num_rules, num_otus))
        
    def forward(self, x: torch.Tensor, k: float = 1) -> torch.Tensor:
        """
        Forward pass: apply threshold detection
        
        Args:
            x: Input tensor (batch, num_rules, num_otus)
            k: Temperature parameter for sigmoid sharpness
            
        Returns:
            Gated response tensor (batch, num_rules, num_otus)
        """
        # Response of the detector for average abundance
        x = torch.sigmoid((x - self.thresh) * k)
        return x
    
    def init_params(self, init_args: dict) -> None:
        """
        Initialize layer parameters
        
        Args:
            init_args: Dictionary with 'thresh_init' key containing threshold values
        """
        self.thresh.data = torch.from_numpy(init_args['thresh_init']).float()


@LayerRegistry.register('layer3', 'slope')
class Slope(BaseLayer):
    """
    Learn threshold for slope values.
    
    The output is a gated response indicating whether the aggregated
    slope from spatial and time aggregation steps is above/below
    the learned threshold.
    
    Architecture:
        Input: (batch, num_rules, num_otus)
        Output: (batch, num_rules, num_otus)
    """
    
    def __init__(self, num_rules: int, num_otus: int, num_time_centers: int,
                 layer_name: str = "slope", **kwargs):
        """
        Initialize slope detector layer
        
        Args:
            num_rules: Number of rule detectors
            num_otus: Number of OTUs
            num_time_centers: Number of time centers (unused, kept for compatibility)
            layer_name: Name for this layer instance
        """
        config = {
            'num_rules': num_rules,
            'num_otus': num_otus,
            'num_time_centers': num_time_centers
        }
        super(Slope, self).__init__(layer_name, config)
        
        # Parameter for learnable threshold slope
        self.slope = nn.Parameter(torch.Tensor(num_rules, num_otus))
        
    def forward(self, x: torch.Tensor, k: float = 1) -> torch.Tensor:
        """
        Forward pass: apply slope threshold detection
        
        Args:
            x: Input tensor (batch, num_rules, num_otus)
            k: Temperature parameter for sigmoid sharpness
            
        Returns:
            Gated response tensor (batch, num_rules, num_otus)
        """
        # Response of the detector for average slope
        x = torch.sigmoid((x - self.slope) * k)
        
        if torch.isnan(self.slope).any():
            print(self.slope)
            raise ValueError('NaN in slope!')
        
        return x
    
    def init_params(self, init_args: dict) -> None:
        """
        Initialize layer parameters
        
        Args:
            init_args: Dictionary with 'slope_init' key containing slope threshold values
        """
        self.slope.data = torch.from_numpy(init_args['slope_init']).float()
