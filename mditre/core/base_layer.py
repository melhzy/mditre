"""
Base layer class for MDITRE architecture
Provides common interface for all layers to enable modularity and extensibility
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union


class BaseLayer(nn.Module, ABC):
    """
    Abstract base class for all MDITRE layers
    
    This provides a common interface that all layers must implement,
    enabling:
    - Easy layer swapping and experimentation
    - Consistent parameter initialization
    - Standardized forward pass interface
    - Future layer additions without breaking existing code
    """
    
    def __init__(self, layer_name: str, layer_config: Dict[str, Any]):
        """
        Initialize base layer
        
        Args:
            layer_name: Human-readable name for this layer
            layer_config: Configuration dictionary for layer parameters
        """
        super(BaseLayer, self).__init__()
        self.layer_name = layer_name
        self.layer_config = layer_config
        self._setup_complete = False
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the layer
        
        Args:
            *args: Positional arguments (typically input tensors)
            **kwargs: Additional layer-specific keyword arguments
            
        Returns:
            Output tensor(s) - can be a single Tensor or Tuple of Tensors
        """
        pass
    
    @abstractmethod
    def init_params(self, init_args: Dict[str, Any]) -> None:
        """
        Initialize layer parameters
        
        Args:
            init_args: Dictionary containing initialization values
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration
        
        Returns:
            Configuration dictionary
        """
        return self.layer_config.copy()
    
    def get_layer_info(self) -> Dict[str, Any]:
        """
        Get detailed information about this layer
        
        Returns:
            Dictionary with layer metadata
        """
        return {
            'name': self.layer_name,
            'type': self.__class__.__name__,
            'config': self.get_config(),
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def __repr__(self) -> str:
        """String representation of the layer"""
        return f"{self.__class__.__name__}(name='{self.layer_name}')"


class LayerRegistry:
    """
    Registry for dynamically managing available layers
    
    This allows for:
    - Runtime layer selection
    - Easy addition of new layer implementations
    - Version control of layer implementations
    """
    
    _registry = {
        'layer1': {},  # Phylogenetic Focus layers
        'layer2': {},  # Temporal Focus layers  
        'layer3': {},  # Detector layers
        'layer4': {},  # Rule layers
        'layer5': {},  # Classification layers
    }
    
    @classmethod
    def register(cls, layer_type: str, layer_name: str):
        """
        Decorator to register a layer implementation
        
        Args:
            layer_type: Type of layer (layer1, layer2, etc.)
            layer_name: Unique name for this implementation
            
        Example:
            @LayerRegistry.register('layer1', 'spatial_agg_dynamic')
            class SpatialAggDynamic(BaseLayer):
                ...
        """
        def decorator(layer_class):
            cls._registry[layer_type][layer_name] = layer_class
            return layer_class
        return decorator
    
    @classmethod
    def get_layer(cls, layer_type: str, layer_name: str):
        """
        Get a registered layer class
        
        Args:
            layer_type: Type of layer
            layer_name: Name of specific implementation
            
        Returns:
            Layer class
        """
        if layer_type not in cls._registry:
            raise ValueError(f"Unknown layer type: {layer_type}")
        if layer_name not in cls._registry[layer_type]:
            raise ValueError(f"Unknown layer name '{layer_name}' for type '{layer_type}'")
        return cls._registry[layer_type][layer_name]
    
    @classmethod
    def list_layers(cls, layer_type: Optional[str] = None) -> Dict[str, list]:
        """
        List all registered layers
        
        Args:
            layer_type: Specific layer type to list, or None for all
            
        Returns:
            Dictionary of available layers
        """
        if layer_type is not None:
            return {layer_type: list(cls._registry[layer_type].keys())}
        return {k: list(v.keys()) for k, v in cls._registry.items()}
