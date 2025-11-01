"""
Temporal Aggregation Layers for Time Focus

These layers aggregate microbial time-series along the time dimension,
selecting contiguous time windows important for the prediction task.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.special import logit
from typing import Optional, Tuple

from ...core.base_layer import BaseLayer, LayerRegistry
from ...core.math_utils import unitboxcar


@LayerRegistry.register('layer2', 'time_agg')
class TimeAgg(BaseLayer):
    """
    Aggregate time-series along the time dimension.
    
    Selects contiguous time windows important for prediction using sigmoid-based
    importance weights. Computes both average abundance and average slope within
    the selected time window.
    
    Architecture:
        Input: (batch, num_rules, num_otus, time_points)
        Output: (abundance, slope) both (batch, num_rules, num_otus)
    """
    
    def __init__(self, num_rules: int, num_otus: int, num_time: int, 
                 num_time_centers: int, layer_name: str = "time_agg", **kwargs):
        """
        Initialize temporal aggregation layer
        
        Args:
            num_rules: Number of rule detectors
            num_otus: Number of OTUs
            num_time: Number of time points
            num_time_centers: Number of time window centers (unused, kept for compatibility)
            layer_name: Name for this layer instance
        """
        config = {
            'num_rules': num_rules,
            'num_otus': num_otus,
            'num_time': num_time,
            'num_time_centers': num_time_centers
        }
        super(TimeAgg, self).__init__(layer_name, config)
        
        # Tensor of time points, starting from 0 to num_time - 1
        self.num_time = num_time
        self.register_buffer('times', torch.arange(num_time, dtype=torch.float32))
        
        # Time window parameters (for abundance and slope)
        self.abun_a = nn.Parameter(torch.Tensor(num_rules, num_otus))
        self.slope_a = nn.Parameter(torch.Tensor(num_rules, num_otus))
        self.abun_b = nn.Parameter(torch.Tensor(num_rules, num_otus))
        self.slope_b = nn.Parameter(torch.Tensor(num_rules, num_otus))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                k: float = 1.) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: aggregate over time dimension
        
        Args:
            x: Input tensor (batch, num_rules, num_otus, time_points)
            mask: Optional binary mask for valid time points (batch, time_points)
            k: Temperature parameter for sigmoid sharpness
            
        Returns:
            (x_abun, x_slope): Aggregated abundance and slope tensors
        """
        # Compute unnormalized importance weights for each time point
        abun_a = torch.sigmoid(self.abun_a).unsqueeze(-1)
        slope_a = torch.sigmoid(self.slope_a).unsqueeze(-1)
        abun_b = torch.sigmoid(self.abun_b).unsqueeze(-1)
        slope_b = torch.sigmoid(self.slope_b).unsqueeze(-1)
        
        # Compute time window parameters
        sigma = self.num_time * abun_a
        sigma_slope = self.num_time * slope_a
        mu = (self.num_time * abun_a / 2.) + (1 - abun_a) * self.num_time * abun_b
        mu_slope = (self.num_time * slope_a / 2.) + (1 - slope_a) * self.num_time * slope_b
        
        # Compute time weights using boxcar function
        time_wts_unnorm = unitboxcar(self.times, mu, sigma, k)
        time_wts_unnorm_slope = unitboxcar(self.times, mu_slope, sigma_slope, k)
        
        # Mask out time points with no samples
        if mask is not None:
            time_wts_unnorm = time_wts_unnorm.mul(
                mask.unsqueeze(1).unsqueeze(1))
            time_wts_unnorm_slope = time_wts_unnorm_slope.mul(
                mask.unsqueeze(1).unsqueeze(1))
        
        # Store weights for inspection
        self.wts = time_wts_unnorm
        self.wts_slope = time_wts_unnorm_slope
        
        # Normalize importance time weights
        time_wts = (time_wts_unnorm).div(
            time_wts_unnorm.sum(dim=-1, keepdims=True) + 1e-8)
        
        if torch.isnan(time_wts).any():
            print(time_wts_unnorm.sum(-1))
            raise ValueError('NaN in time aggregation!')
        
        # Aggregation over time dimension (weighted average)
        x_abun = x.mul(time_wts).sum(dim=-1)
        
        # Compute approximate average slope over time window
        tau = self.times - mu_slope
        a = (time_wts_unnorm_slope * x).sum(dim=-1)
        b = (time_wts_unnorm_slope * tau).sum(dim=-1)
        c = (time_wts_unnorm_slope).sum(dim=-1)
        d = (time_wts_unnorm_slope * x * tau).sum(dim=-1)
        e = (time_wts_unnorm_slope * (tau ** 2)).sum(dim=-1)
        num = ((a * b) - (c * d))
        den = ((b ** 2) - (e * c)) + 1e-8
        x_slope = num / den
        
        if torch.isnan(x_slope).any():
            print(time_wts_unnorm_slope.sum(dim=-1))
            print(x_slope)
            raise ValueError('NaN in time aggregation!')
        
        # Store parameters for inspection
        self.m = mu
        self.m_slope = mu_slope
        self.s_abun = sigma
        self.s_slope = sigma_slope
        
        return x_abun, x_slope
    
    def init_params(self, init_args: dict) -> None:
        """
        Initialize layer parameters
        
        Args:
            init_args: Dictionary with keys 'abun_a_init', 'abun_b_init',
                      'slope_a_init', 'slope_b_init'
        """
        self.abun_a.data = torch.from_numpy(logit(init_args['abun_a_init'])).float()
        self.abun_b.data = torch.from_numpy(logit(init_args['abun_b_init'])).float()
        self.slope_a.data = torch.from_numpy(logit(init_args['slope_a_init'])).float()
        self.slope_b.data = torch.from_numpy(logit(init_args['slope_b_init'])).float()


@LayerRegistry.register('layer2', 'time_agg_abun')
class TimeAggAbun(BaseLayer):
    """
    Aggregate time-series along time dimension (abundance only).
    
    Simplified version of TimeAgg that only computes average abundance
    within a selected time window, without slope calculation.
    
    Architecture:
        Input: (batch, num_rules, num_otus, time_points)
        Output: (batch, num_rules, num_otus)
    """
    
    def __init__(self, num_rules: int, num_otus: int, num_time: int, 
                 num_time_centers: int, layer_name: str = "time_agg_abun", **kwargs):
        """
        Initialize temporal aggregation layer (abundance only)
        
        Args:
            num_rules: Number of rule detectors
            num_otus: Number of OTUs
            num_time: Number of time points
            num_time_centers: Number of time window centers (unused, kept for compatibility)
            layer_name: Name for this layer instance
        """
        config = {
            'num_rules': num_rules,
            'num_otus': num_otus,
            'num_time': num_time,
            'num_time_centers': num_time_centers
        }
        super(TimeAggAbun, self).__init__(layer_name, config)
        
        # Tensor of time points
        self.num_time = num_time
        self.register_buffer('times', torch.arange(num_time, dtype=torch.float32))
        
        # Time window parameters
        self.abun_a = nn.Parameter(torch.Tensor(num_rules, num_otus))
        self.abun_b = nn.Parameter(torch.Tensor(num_rules, num_otus))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                k: float = 1.) -> torch.Tensor:
        """
        Forward pass: aggregate over time dimension
        
        Args:
            x: Input tensor (batch, num_rules, num_otus, time_points)
            mask: Optional binary mask for valid time points (batch, time_points)
            k: Temperature parameter for sigmoid sharpness
            
        Returns:
            x_abun: Aggregated abundance tensor (batch, num_rules, num_otus)
        """
        # Compute unnormalized importance weights for each time point
        abun_a = torch.sigmoid(self.abun_a).unsqueeze(-1)
        abun_b = torch.sigmoid(self.abun_b).unsqueeze(-1)
        sigma = self.num_time * abun_a
        mu = (self.num_time * abun_a / 2.) + (1 - abun_a) * self.num_time * abun_b
        time_wts_unnorm = unitboxcar(self.times, mu, sigma, k)
        
        # Mask out time points with no samples
        if mask is not None:
            time_wts_unnorm = time_wts_unnorm.mul(
                mask.unsqueeze(1).unsqueeze(1))
        
        # Store weights for inspection
        self.wts = time_wts_unnorm
        
        # Normalize importance time weights
        time_wts = (time_wts_unnorm).div(
            time_wts_unnorm.sum(dim=-1, keepdims=True) + 1e-8)
        
        if torch.isnan(time_wts).any():
            print(time_wts_unnorm.sum(-1))
            raise ValueError('NaN in time aggregation!')
        
        # Aggregation over time dimension
        x_abun = x.mul(time_wts).sum(dim=-1)
        
        # Store parameters for inspection
        self.m = mu
        self.s_abun = sigma
        
        return x_abun
    
    def init_params(self, init_args: dict) -> None:
        """
        Initialize layer parameters
        
        Args:
            init_args: Dictionary with keys 'abun_a_init', 'abun_b_init'
        """
        self.abun_a.data = torch.from_numpy(logit(init_args['abun_a_init'])).float()
        self.abun_b.data = torch.from_numpy(logit(init_args['abun_b_init'])).float()
