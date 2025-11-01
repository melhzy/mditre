"""
Example: Using MDITRE Modular Architecture

This example demonstrates how to use the new modular layer architecture
to build custom MDITRE models.
"""

import numpy as np
import torch
import torch.nn as nn

# Import from modular architecture
from mditre.layers import (
    SpatialAggDynamic,
    TimeAgg,
    Threshold,
    Slope,
    Rules,
    DenseLayer
)
from mditre.core import LayerRegistry


# Example 1: Building a custom MDITRE model using modular layers
class CustomMDITRE(nn.Module):
    """
    Custom MDITRE model using modular layers.
    
    This demonstrates how to compose MDITRE from individual layer modules,
    allowing for easy customization and experimentation.
    """
    
    def __init__(self, num_rules, num_otus, num_otu_centers,
                 num_time, num_time_centers, otu_embeddings, emb_dim):
        super(CustomMDITRE, self).__init__()
        
        # Layer 1: Phylogenetic Focus
        self.phylo_layer = SpatialAggDynamic(
            num_rules=num_rules,
            num_otu_centers=num_otu_centers,
            otu_embeddings=otu_embeddings,
            emb_dim=emb_dim,
            num_otus=num_otus,
            layer_name="phylogenetic_focus"
        )
        
        # Layer 2: Temporal Focus
        self.temporal_layer = TimeAgg(
            num_rules=num_rules,
            num_otus=num_otu_centers,
            num_time=num_time,
            num_time_centers=num_time_centers,
            layer_name="temporal_focus"
        )
        
        # Layer 3: Detectors
        self.abundance_detector = Threshold(
            num_rules=num_rules,
            num_otus=num_otu_centers,
            num_time_centers=num_time_centers,
            layer_name="abundance_threshold"
        )
        
        self.slope_detector = Slope(
            num_rules=num_rules,
            num_otus=num_otu_centers,
            num_time_centers=num_time_centers,
            layer_name="slope_threshold"
        )
        
        # Layer 4: Rules
        self.abundance_rules = Rules(
            num_rules=num_rules,
            num_otus=num_otu_centers,
            num_time_centers=num_time_centers,
            layer_name="abundance_rules"
        )
        
        self.slope_rules = Rules(
            num_rules=num_rules,
            num_otus=num_otu_centers,
            num_time_centers=num_time_centers,
            layer_name="slope_rules"
        )
        
        # Layer 5: Classification
        self.classifier = DenseLayer(
            in_feat=num_rules,
            out_feat=1,
            layer_name="classifier"
        )
    
    def forward(self, x, mask=None, k_alpha=1, k_beta=1,
                k_otu=1., k_time=1., k_thresh=1., k_slope=1.,
                hard=False, use_noise=True):
        """Forward pass through all layers"""
        
        # Layer 1: Phylogenetic aggregation
        x = self.phylo_layer(x, k=k_otu)
        
        # Layer 2: Temporal aggregation
        x_abun, x_slope = self.temporal_layer(x, mask=mask, k=k_time)
        
        # Layer 3: Threshold detection
        x_abun = self.abundance_detector(x_abun, k=k_thresh)
        x_slope = self.slope_detector(x_slope, k=k_slope)
        
        # Layer 4: Rule formation
        x_abun = self.abundance_rules(x_abun, hard=hard, k=k_alpha, use_noise=use_noise)
        x_slope = self.slope_rules(x_slope, hard=hard, k=k_alpha, use_noise=use_noise)
        
        # Layer 5: Classification
        output = self.classifier(x_abun, x_slope=x_slope, hard=hard, k=k_beta, use_noise=use_noise)
        
        return output
    
    def init_params(self, init_args):
        """Initialize all layer parameters"""
        for module in self.children():
            if hasattr(module, 'init_params'):
                module.init_params(init_args)  # type: ignore[attr-defined]
    
    def get_layer_info(self):
        """Get information about all layers"""
        info = {}
        for name, module in self.named_children():
            if hasattr(module, 'get_layer_info'):
                info[name] = module.get_layer_info()  # type: ignore[attr-defined]
        return info


# Example 2: Using LayerRegistry for dynamic model construction
def build_mditre_from_config(config):
    """
    Build MDITRE model dynamically from configuration.
    
    This demonstrates the power of LayerRegistry for runtime layer selection.
    """
    
    # Get layer classes from registry
    phylo_class = LayerRegistry.get_layer('layer1', config['phylo_type'])
    temporal_class = LayerRegistry.get_layer('layer2', config['temporal_type'])
    detector_class = LayerRegistry.get_layer('layer3', config['detector_type'])
    rule_class = LayerRegistry.get_layer('layer4', 'rules')
    classifier_class = LayerRegistry.get_layer('layer5', config['classifier_type'])
    
    print(f"Building model with:")
    print(f"  Phylogenetic: {phylo_class.__name__}")
    print(f"  Temporal: {temporal_class.__name__}")
    print(f"  Detector: {detector_class.__name__}")
    print(f"  Rule: {rule_class.__name__}")
    print(f"  Classifier: {classifier_class.__name__}")
    
    return {
        'phylo': phylo_class,
        'temporal': temporal_class,
        'detector': detector_class,
        'rule': rule_class,
        'classifier': classifier_class
    }


# Example 3: Testing the modular architecture
if __name__ == "__main__":
    print("=" * 80)
    print("MDITRE Modular Architecture Example")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Model parameters
    num_rules = 3
    num_otus = 50
    num_otu_centers = 10
    num_time = 15
    num_time_centers = 3
    emb_dim = 5
    
    # Create synthetic OTU embeddings
    otu_embeddings = np.random.randn(num_otus, emb_dim).astype(np.float32)
    
    print("\n1. Creating CustomMDITRE model...")
    model = CustomMDITRE(
        num_rules=num_rules,
        num_otus=num_otus,
        num_otu_centers=num_otu_centers,
        num_time=num_time,
        num_time_centers=num_time_centers,
        otu_embeddings=otu_embeddings,
        emb_dim=emb_dim
    )
    
    print(f"   Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize parameters properly
    print("\n   Initializing parameters...")
    init_args = {
        'kappa_init': np.random.uniform(0.5, 2.0, (num_rules, num_otu_centers)),
        'eta_init': np.random.randn(num_rules, num_otu_centers, emb_dim) * 0.1,
        'abun_a_init': np.random.uniform(0.2, 0.8, (num_rules, num_otu_centers)),
        'abun_b_init': np.random.uniform(0.2, 0.8, (num_rules, num_otu_centers)),
        'slope_a_init': np.random.uniform(0.2, 0.8, (num_rules, num_otu_centers)),
        'slope_b_init': np.random.uniform(0.2, 0.8, (num_rules, num_otu_centers)),
        'thresh_init': np.random.uniform(0.1, 0.5, (num_rules, num_otu_centers)),
        'slope_init': np.random.uniform(-0.1, 0.1, (num_rules, num_otu_centers)),
        'alpha_init': np.random.uniform(-1, 1, (num_rules, num_otu_centers)),
        'w_init': np.random.randn(1, num_rules) * 0.1,
        'bias_init': np.zeros(1),
        'beta_init': np.random.uniform(-1, 1, num_rules)
    }
    model.init_params(init_args)
    print("   [OK] Parameters initialized")
    
    # Get layer information
    print("\n2. Layer Information:")
    layer_info = model.get_layer_info()
    for layer_name, info in layer_info.items():
        print(f"   {layer_name}:")
        print(f"      Type: {info['type']}")
        print(f"      Parameters: {info['num_parameters']}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 4
    x = torch.randn(batch_size, num_otus, num_time)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output values: {output.numpy()}")
    
    # Test LayerRegistry
    print("\n4. Available layer implementations:")
    available = LayerRegistry.list_layers()
    for layer_type, implementations in available.items():
        print(f"   {layer_type}: {', '.join(implementations)}")
    
    # Example configuration
    print("\n5. Dynamic model construction example:")
    config = {
        'phylo_type': 'spatial_agg_dynamic',
        'temporal_type': 'time_agg',
        'detector_type': 'threshold',
        'classifier_type': 'dense_layer'
    }
    layer_classes = build_mditre_from_config(config)
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
