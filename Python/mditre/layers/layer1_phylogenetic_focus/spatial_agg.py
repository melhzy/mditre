"""
Spatial Aggregation Layers for Phylogenetic Focus

These layers aggregate microbial time-series based on phylogenetic distance,
allowing the model to focus on taxonomically related groups of OTUs.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.special import logit

from ...core.base_layer import BaseLayer, LayerRegistry
from ...core.math_utils import transf_log, inv_transf_log


@LayerRegistry.register('layer1', 'spatial_agg')
class SpatialAgg(BaseLayer):
    """
    Aggregate time-series based on phylogenetic distance.
    
    Uses the sigmoid function to calculate importance weights of each OTU 
    for a detector based on phylogenetic distance. OTUs within a learned 
    kappa radius are selected with higher weights.
    
    Architecture:
        Input: (batch, num_otus, time_points)
        Output: (batch, num_rules, num_otus, time_points)
    """
    
    def __init__(self, num_rules: int, num_otus: int, dist: np.ndarray, 
                 layer_name: str = "spatial_agg", **kwargs):
        """
        Initialize spatial aggregation layer
        
        Args:
            num_rules: Number of rule detectors
            num_otus: Number of OTUs in dataset
            dist: Phylogenetic distance matrix (num_otus x num_otus)
            layer_name: Name for this layer instance
        """
        config = {
            'num_rules': num_rules,
            'num_otus': num_otus,
            'dist_shape': dist.shape
        }
        super(SpatialAgg, self).__init__(layer_name, config)
        
        # Initialize phylogenetic distance matrix as a buffer (non-trainable)
        self.register_buffer('dist', torch.from_numpy(dist))
        self.dist: torch.Tensor  # Type hint for static analyzer
        
        # OTU selection bandwidth
        # All OTUs within kappa radius are deemed to be selected
        self.kappa = nn.Parameter(torch.Tensor(num_rules, num_otus))
        
    def forward(self, x: torch.Tensor, k: float = 1) -> torch.Tensor:
        """
        Forward pass: aggregate OTUs based on phylogenetic distance
        
        Args:
            x: Input tensor (batch, num_otus, time_points)
            k: Temperature parameter for sigmoid sharpness
            
        Returns:
            Aggregated tensor (batch, num_rules, num_otus, time_points)
        """
        # Compute unnormalized OTU weights
        kappa = transf_log(self.kappa, 0, self.dist.max().item()).unsqueeze(-1)
        otu_wts_unnorm = torch.sigmoid((kappa - self.dist) * k)
        
        self.wts = otu_wts_unnorm
        
        if torch.isnan(otu_wts_unnorm).any():
            print(otu_wts_unnorm.sum(dim=-1))
            print(self.kappa)
            raise ValueError('NaN in spatial aggregation!')
        
        # Aggregation of time-series along OTU dimension
        # Essentially a convolution operation
        x = torch.einsum('kij,sjt->skit', otu_wts_unnorm, x)
        
        return x
    
    def init_params(self, init_args: dict) -> None:
        """
        Initialize layer parameters
        
        Args:
            init_args: Dictionary with 'kappa_init' key containing initial kappa values
        """
        kappa_init = init_args['kappa_init']
        self.kappa.data = torch.from_numpy(
            inv_transf_log(kappa_init, 0, self.dist.max().item())
        ).float()


@LayerRegistry.register('layer1', 'spatial_agg_dynamic')
class SpatialAggDynamic(BaseLayer):
    """
    Dynamic spatial aggregation based on learned OTU embeddings.
    
    Instead of using a fixed distance matrix, this layer learns OTU center
    embeddings and computes distances dynamically in embedding space.
    This provides more flexibility for the model to discover phylogenetic
    patterns relevant to the prediction task.
    
    Architecture:
        Input: (batch, num_otus, time_points)
        Output: (batch, num_rules, num_otu_centers, time_points)
    """
    
    def __init__(self, num_rules: int, num_otu_centers: int, 
                 otu_embeddings: np.ndarray, emb_dim: int, num_otus: int,
                 layer_name: str = "spatial_agg_dynamic", **kwargs):
        """
        Initialize dynamic spatial aggregation layer
        
        Args:
            num_rules: Number of rule detectors
            num_otu_centers: Number of OTU centers to learn
            otu_embeddings: OTU embedding matrix (num_otus x emb_dim)
            emb_dim: Embedding dimension
            num_otus: Total number of OTUs
            layer_name: Name for this layer instance
        """
        config = {
            'num_rules': num_rules,
            'num_otu_centers': num_otu_centers,
            'emb_dim': emb_dim,
            'num_otus': num_otus
        }
        super(SpatialAggDynamic, self).__init__(layer_name, config)
        
        self.num_rules = num_rules
        self.num_otu_centers = num_otu_centers
        self.emb_dim = emb_dim
        
        # Register OTU embeddings as buffer
        self.register_buffer('dist', torch.from_numpy(otu_embeddings))
        self.dist: torch.Tensor  # Type hint for static analyzer
        
        # Learnable OTU centers in embedding space
        self.eta = nn.Parameter(torch.Tensor(num_rules, num_otu_centers, emb_dim))
        
        # OTU selection bandwidth
        # All OTUs within kappa radius are deemed to be selected
        self.kappa = nn.Parameter(torch.Tensor(num_rules, num_otu_centers))
        
    def forward(self, x: torch.Tensor, k: float = 1) -> torch.Tensor:
        """
        Forward pass: aggregate OTUs based on learned embedding distances
        
        Args:
            x: Input tensor (batch, num_otus, time_points)
            k: Temperature parameter for sigmoid sharpness
            
        Returns:
            Aggregated tensor (batch, num_rules, num_otu_centers, time_points)
        """
        # Compute unnormalized OTU weights based on embedding distance
        kappa = self.kappa.exp().unsqueeze(-1)
        dist = (self.eta.reshape(self.num_rules, self.num_otu_centers, 1, self.emb_dim) 
                - self.dist).norm(2, dim=-1)
        otu_wts_unnorm = torch.sigmoid((kappa - dist) * k)
        
        self.wts = otu_wts_unnorm
        
        if torch.isnan(otu_wts_unnorm).any():
            print(otu_wts_unnorm.sum(dim=-1))
            print(self.kappa)
            raise ValueError('NaN in spatial aggregation!')
        
        # Aggregation of time-series along OTU dimension
        # Essentially a convolution operation
        x = torch.einsum('kij,sjt->skit', otu_wts_unnorm, x)
        
        # Store for inspection
        self.kappas = kappa
        self.emb_dist = dist
        
        return x
    
    def init_params(self, init_args: dict) -> None:
        """
        Initialize layer parameters
        
        Args:
            init_args: Dictionary with 'kappa_init' and 'eta_init' keys
        """
        self.kappa.data = torch.from_numpy(init_args['kappa_init']).log().float()
        self.eta.data = torch.from_numpy(init_args['eta_init']).float()
