"""
Comprehensive Integrated Test Suite for MDITRE
Based on: Maringanti et al. (2022) - mSystems Volume 7 Issue 5

This consolidated test suite combines:
1. Architecture and integration tests (28 comprehensive tests)
2. Seeding and reproducibility tests (5 tests)
3. Package integrity validation tests (6 tests)

Total: 39 tests covering all aspects of MDITRE

Test Coverage:
- Core Architecture: 5-layer neural network (8 tests)
- Differentiability: Gradient flow and relaxations (3 tests)
- Model Variants: MDITRE and MDITREAbun (2 tests)
- Phylogenetic Focus: Embeddings and selection (4 tests)
- Temporal Focus: Time windows and slopes (4 tests)
- Performance Metrics: F1, AUC-ROC, etc. (3 tests)
- PyTorch Integration: GPU, serialization (3 tests)
- End-to-End: Complete pipeline (1 test)
- Seeding: Reproducibility (5 tests)
- Package Integrity: Module validation (6 tests)

Usage:
    pytest tests/test_all.py -v                    # All tests
    pytest tests/test_all.py -k "architecture" -v  # Architecture tests
    pytest tests/test_all.py -k "seeding" -v       # Seeding tests
    pytest tests/test_all.py -k "integrity" -v     # Package integrity
    pytest tests/test_all.py --durations=10         # Profiling
"""

import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path

# Import MDITRE modules
from mditre.models import (
    MDITRE, MDITREAbun,
    SpatialAgg, SpatialAggDynamic,
    TimeAgg, TimeAggAbun,
    Threshold, Slope, Rules,
    DenseLayer, DenseLayerAbun,
    binary_concrete, unitboxcar
)

from mditre.seeding import MDITRESeedGenerator, get_mditre_seeds, set_random_seeds


# ============================================================================
# PYTEST FIXTURES - Shared test data and configurations
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """Get computation device (CUDA if available, else CPU)."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return dev


@pytest.fixture(scope="session")
def test_config():
    """Standard test configuration parameters."""
    return {
        'num_subjects': 40,
        'num_otus': 50,
        'num_time': 10,
        'num_rules': 3,
        'num_otu_centers': 5,
        'num_time_centers': 3,
        'emb_dim': 10,
        'random_seed': 42
    }


@pytest.fixture(scope="function")
def synthetic_data(test_config):
    """Generate synthetic microbiome time-series data."""
    set_random_seeds(test_config['random_seed'])
    
    num_subjects = test_config['num_subjects']
    num_otus = test_config['num_otus']
    num_time = test_config['num_time']
    
    # Microbiome abundances (subjects x OTUs x time)
    X = np.random.dirichlet(np.ones(num_otus), size=(num_subjects, num_time))
    X = X.transpose(0, 2, 1)
    
    # Binary labels
    y = np.random.randint(0, 2, size=num_subjects)
    
    # Time mask
    X_mask = np.ones((num_subjects, num_time), dtype=np.float32)
    
    return {
        'X': X.astype(np.float32),
        'y': y,
        'X_mask': X_mask,
        'num_subjects': num_subjects,
        'num_otus': num_otus,
        'num_time': num_time
    }


@pytest.fixture(scope="function")
def phylo_dist_matrix(test_config):
    """Generate phylogenetic distance matrix."""
    set_random_seeds(test_config['random_seed'])
    num_otus = test_config['num_otus']
    
    # Generate symmetric distance matrix
    dist = np.random.rand(num_otus, num_otus).astype(np.float32)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    
    return dist


@pytest.fixture(scope="function")
def otu_embeddings(test_config):
    """Generate OTU embeddings in phylogenetic space."""
    set_random_seeds(test_config['random_seed'])
    num_otus = test_config['num_otus']
    emb_dim = test_config['emb_dim']
    
    embeddings = np.random.randn(num_otus, emb_dim).astype(np.float32)
    return embeddings


@pytest.fixture(scope="function")
def init_args_full(test_config):
    """Generate initialization arguments for MDITRE model parameters."""
    set_random_seeds(test_config['random_seed'])
    
    num_rules = test_config['num_rules']
    num_otu_centers = test_config['num_otu_centers']
    emb_dim = test_config['emb_dim']
    
    return {
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


# ============================================================================
# SECTION 1: CORE ARCHITECTURE TESTS
# Paper: "MDITRE can be represented as a five-layer neural network"
# ============================================================================

class TestSection1_1_FiveLayerArchitecture:
    """Test 5-layer neural network architecture."""
    
    @pytest.mark.architecture
    @pytest.mark.layer1
    def test_1_1_1_layer1_spatial_agg_static(self, test_config, phylo_dist_matrix, device):
        """Test Layer 1 - SpatialAgg (static phylogenetic focus)."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        
        layer = SpatialAgg(
            num_rules=num_rules,
            num_otus=num_otus,
            dist=phylo_dist_matrix
        ).to(device)
        
        assert hasattr(layer, 'kappa'), "Missing kappa parameter"
        assert layer.kappa.shape == (num_rules, num_otus)
        assert hasattr(layer, 'dist'), "Missing phylogenetic distance buffer"
        
        batch_size, num_time = 5, 10
        X = torch.randn(batch_size, num_otus, num_time, device=device)
        output = layer(X)
        
        assert output.shape == (batch_size, num_rules, num_otus, num_time)
        assert not torch.isnan(output).any()


    @pytest.mark.architecture
    @pytest.mark.layer1
    def test_1_1_1_layer1_spatial_agg_dynamic(self, test_config, otu_embeddings, device):
        """Test Layer 1 - SpatialAggDynamic (learnable phylogenetic focus)."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_otu_centers = test_config['num_otu_centers']
        emb_dim = test_config['emb_dim']
        
        layer = SpatialAggDynamic(
            num_rules=num_rules,
            num_otu_centers=num_otu_centers,
            dist=otu_embeddings,
            emb_dim=emb_dim,
            num_otus=num_otus
        ).to(device)
        
        init_args = {
            'kappa_init': np.random.uniform(0.1, 1.0, (num_rules, num_otu_centers)),
            'eta_init': np.random.randn(num_rules, num_otu_centers, emb_dim)
        }
        layer.init_params(init_args)
        
        assert hasattr(layer, 'kappa') and hasattr(layer, 'eta')
        
        batch_size, num_time = 5, 10
        X = torch.randn(batch_size, num_otus, num_time, device=device)
        output = layer(X)
        
        assert output.shape == (batch_size, num_rules, num_otu_centers, num_time)
        assert not torch.isnan(output).any()
    

    @pytest.mark.architecture
    @pytest.mark.layer2
    def test_1_1_2_layer2_time_agg(self, test_config, device):
        """Test Layer 2 - TimeAgg (temporal focus with slopes)."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_time = test_config['num_time']
        num_time_centers = test_config['num_time_centers']
        
        layer = TimeAgg(
            num_rules=num_rules,
            num_otus=num_otus,
            num_time=num_time,
            num_time_centers=num_time_centers
        ).to(device)
        
        assert hasattr(layer, 'abun_a') and hasattr(layer, 'abun_b')
        assert hasattr(layer, 'slope_a') and hasattr(layer, 'slope_b')
        
        batch_size = 5
        X = torch.randn(batch_size, num_rules, num_otus, num_time, device=device)
        mask = torch.ones(batch_size, num_time, device=device)
        
        output = layer(X, mask)
        assert isinstance(output, tuple) and len(output) == 2
        
        x_abun, x_slope = output
        assert x_abun.shape == (batch_size, num_rules, num_otus)
        assert x_slope.shape == (batch_size, num_rules, num_otus)
        assert not torch.isnan(x_abun).any() and not torch.isnan(x_slope).any()


    @pytest.mark.architecture
    @pytest.mark.layer2
    def test_1_1_2_layer2_time_agg_abun(self, test_config, device):
        """Test Layer 2 - TimeAggAbun (abundance-only temporal focus)."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_time = test_config['num_time']
        num_time_centers = test_config['num_time_centers']
        
        layer = TimeAggAbun(
            num_rules=num_rules,
            num_otus=num_otus,
            num_time=num_time,
            num_time_centers=num_time_centers
        ).to(device)
        
        batch_size = 5
        X = torch.randn(batch_size, num_rules, num_otus, num_time, device=device)
        mask = torch.ones(batch_size, num_time, device=device)
        output = layer(X, mask)
        
        assert output.shape == (batch_size, num_rules, num_otus)
        assert not torch.isnan(output).any()


    @pytest.mark.architecture
    @pytest.mark.layer3
    def test_1_1_3_layer3_threshold_detector(self, test_config, device):
        """Test Layer 3 - Threshold detector."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_time_centers = test_config['num_time_centers']
        
        layer = Threshold(
            num_rules=num_rules,
            num_otus=num_otus,
            num_time_centers=num_time_centers
        ).to(device)
        
        assert hasattr(layer, 'thresh')
        
        batch_size = 5
        X = torch.randn(batch_size, num_rules, num_otus, device=device)
        output = layer(X)
        
        assert output.shape == (batch_size, num_rules, num_otus)
        assert not torch.isnan(output).any()
        assert (output >= 0).all() and (output <= 1).all()


    @pytest.mark.architecture
    @pytest.mark.layer3
    def test_1_1_3_layer3_slope_detector(self, test_config, device):
        """Test Layer 3 - Slope detector (rate of change)."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_time_centers = test_config['num_time_centers']
        
        layer = Slope(
            num_rules=num_rules,
            num_otus=num_otus,
            num_time_centers=num_time_centers
        ).to(device)
        
        assert hasattr(layer, 'slope')
        
        batch_size = 5
        X = torch.randn(batch_size, num_rules, num_otus, device=device)
        output = layer(X)
        
        assert output.shape == (batch_size, num_rules, num_otus)
        assert not torch.isnan(output).any()
        assert (output >= 0).all() and (output <= 1).all()


    @pytest.mark.architecture
    @pytest.mark.layer4
    def test_1_1_4_layer4_rules(self, test_config, device):
        """Test Layer 4 - Rules (soft AND operation)."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_time_centers = test_config['num_time_centers']
        
        layer = Rules(
            num_rules=num_rules,
            num_otus=num_otus,
            num_time_centers=num_time_centers
        ).to(device)
        
        assert hasattr(layer, 'alpha')
        
        batch_size = 5
        X = torch.sigmoid(torch.randn(batch_size, num_rules, num_otus, device=device))
        output = layer(X)
        
        assert output.shape == (batch_size, num_rules)
        assert not torch.isnan(output).any()
        assert (output >= 0).all() and (output <= 1).all()


    @pytest.mark.architecture
    @pytest.mark.layer5
    def test_1_1_5_layer5_classification(self, test_config, device):
        """Test Layer 5 - Classification (weighted aggregation)."""
        num_rules = test_config['num_rules']
        
        layer = DenseLayer(in_feat=num_rules, out_feat=1).to(device)
        
        assert hasattr(layer, 'weight') and hasattr(layer, 'bias')
        assert hasattr(layer, 'beta')
        
        batch_size = 5
        X = torch.rand(batch_size, num_rules, device=device)
        X_slope = torch.rand(batch_size, num_rules, device=device)
        output = layer(X, X_slope)
        
        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()


class TestSection1_2_Differentiability:
    """Test differentiability of MDITRE architecture."""
    
    @pytest.mark.differentiability
    @pytest.mark.critical
    def test_1_2_1_gradient_flow(self, test_config, otu_embeddings, synthetic_data, init_args_full, device):
        """Test gradient flow through all layers."""
        model = MDITRE(
            num_rules=test_config['num_rules'],
            num_otus=test_config['num_otus'],
            num_otu_centers=test_config['num_otu_centers'],
            num_time=test_config['num_time'],
            num_time_centers=test_config['num_time_centers'],
            dist=otu_embeddings,
            emb_dim=test_config['emb_dim']
        ).to(device)
        
        model.init_params(init_args_full)
        
        X = torch.from_numpy(synthetic_data['X'][:5]).float().to(device).requires_grad_(True)
        mask = torch.from_numpy(synthetic_data['X_mask'][:5]).float().to(device)
        
        output = model(X, mask=mask)
        loss = output.sum()
        loss.backward()
        
        assert X.grad is not None
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        total_params = list(model.parameters())
        assert len(params_with_grad) == len(total_params)


    @pytest.mark.differentiability
    def test_1_2_2_relaxation_techniques(self, device):
        """Test relaxation techniques for differentiability."""
        x = torch.randn(10, 20, device=device, requires_grad=True)
        z_soft = binary_concrete(x, k=10, hard=False, use_noise=False)
        
        assert (z_soft >= 0).all() and (z_soft <= 1).all()
        loss = z_soft.sum()
        loss.backward()
        assert x.grad is not None
        
        # Test unitboxcar
        x_time = torch.arange(20, dtype=torch.float32, device=device)
        window = unitboxcar(x_time, mu=torch.tensor(10.0, device=device), 
                           l=torch.tensor(5.0, device=device), k=10)
        assert (window >= 0).all() and (window <= 1).all()


    @pytest.mark.differentiability
    def test_1_2_3_straight_through_estimator(self, device):
        """Test straight-through estimator for discrete approximations."""
        x = torch.randn(10, 20, device=device, requires_grad=True)
        z_hard = binary_concrete(x, k=10, hard=True, use_noise=False)
        loss_hard = z_hard.sum()
        loss_hard.backward()
        
        assert x.grad is not None
        unique_values = z_hard.unique()
        assert len(unique_values) <= 2


class TestSection1_3_ModelVariants:
    """Test MDITRE and MDITREAbun variants."""
    
    @pytest.mark.architecture
    @pytest.mark.model
    def test_1_3_1_mditre_full_model(self, test_config, otu_embeddings, synthetic_data, init_args_full, device):
        """Test MDITRE full model with abundance and slope detectors."""
        model = MDITRE(
            num_rules=test_config['num_rules'],
            num_otus=test_config['num_otus'],
            num_otu_centers=test_config['num_otu_centers'],
            num_time=test_config['num_time'],
            num_time_centers=test_config['num_time_centers'],
            dist=otu_embeddings,
            emb_dim=test_config['emb_dim']
        ).to(device)
        
        model.init_params(init_args_full)
        
        X = torch.from_numpy(synthetic_data['X'][:10]).float().to(device)
        mask = torch.from_numpy(synthetic_data['X_mask'][:10]).float().to(device)
        output = model(X, mask=mask)
        
        assert output.shape == (10,)
        assert not torch.isnan(output).any()


    @pytest.mark.architecture
    @pytest.mark.model
    def test_1_3_2_mditre_abun_variant(self, test_config, otu_embeddings, synthetic_data, device):
        """Test MDITREAbun (abundance-only variant)."""
        model = MDITREAbun(
            num_rules=test_config['num_rules'],
            num_otus=test_config['num_otus'],
            num_otu_centers=test_config['num_otu_centers'],
            num_time=test_config['num_time'],
            num_time_centers=test_config['num_time_centers'],
            dist=otu_embeddings,
            emb_dim=test_config['emb_dim']
        ).to(device)
        
        init_args = {
            'kappa_init': np.random.uniform(0.5, 2.0, (test_config['num_rules'], test_config['num_otu_centers'])),
            'eta_init': np.random.randn(test_config['num_rules'], test_config['num_otu_centers'], test_config['emb_dim']) * 0.1,
            'abun_a_init': np.random.uniform(0.2, 0.8, (test_config['num_rules'], test_config['num_otu_centers'])),
            'abun_b_init': np.random.uniform(0.2, 0.8, (test_config['num_rules'], test_config['num_otu_centers'])),
            'thresh_init': np.random.uniform(0.1, 0.5, (test_config['num_rules'], test_config['num_otu_centers'])),
            'alpha_init': np.random.uniform(-1, 1, (test_config['num_rules'], test_config['num_otu_centers'])),
            'w_init': np.random.randn(1, test_config['num_rules']) * 0.1,
            'bias_init': np.zeros(1),
            'beta_init': np.random.uniform(-1, 1, test_config['num_rules'])
        }
        model.init_params(init_args)
        
        X = torch.from_numpy(synthetic_data['X'][:10]).float().to(device)
        mask = torch.from_numpy(synthetic_data['X_mask'][:10]).float().to(device)
        output = model(X, mask=mask)
        
        assert output.shape == (10,)
        assert not torch.isnan(output).any()


# ============================================================================
# SECTION 2: PHYLOGENETIC FOCUS MECHANISM TESTS
# ============================================================================

class TestSection2_PhylogeneticFocus:
    """Tests for phylogenetic focus mechanisms."""
    
    @pytest.mark.phylogenetic
    @pytest.mark.critical
    def test_2_1_1_phylogenetic_embedding(self, test_config, otu_embeddings, device):
        """Test phylogenetic embedding initialization and structure."""
        num_otus = test_config['num_otus']
        emb_dim = test_config['emb_dim']
        
        assert otu_embeddings.shape == (num_otus, emb_dim)
        
        emb_tensor = torch.from_numpy(otu_embeddings).float()
        dists = torch.cdist(emb_tensor, emb_tensor, p=2)
        
        assert torch.allclose(dists, dists.T, atol=1e-5)
        assert torch.diag(dists).max().item() < 0.01
        assert (dists >= 0).all()
    

    @pytest.mark.phylogenetic
    @pytest.mark.critical
    def test_2_1_2_soft_selection_mechanism(self, test_config, otu_embeddings, device):
        """Test soft selection via concentration parameter kappa."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_otu_centers = test_config['num_otu_centers']
        emb_dim = test_config['emb_dim']
        
        layer = SpatialAggDynamic(
            num_rules=num_rules,
            num_otu_centers=num_otu_centers,
            dist=otu_embeddings,
            emb_dim=emb_dim,
            num_otus=num_otus
        ).to(device)
        
        kappa_sharp = np.ones((num_rules, num_otu_centers)) * 2.0
        eta_init = np.random.randn(num_rules, num_otu_centers, emb_dim)
        layer.init_params({'kappa_init': kappa_sharp, 'eta_init': eta_init})
        
        X = torch.randn(5, num_otus, 10, device=device)
        with torch.no_grad():
            output = layer(X, k=10)
            weights = layer.wts
        
        assert (weights >= 0).all() and (weights <= 1).all()
    

    @pytest.mark.phylogenetic
    def test_2_1_3_phylogenetic_clade_selection(self, test_config, otu_embeddings, device):
        """Test ability to select coherent phylogenetic clades."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_otu_centers = test_config['num_otu_centers']
        emb_dim = test_config['emb_dim']
        
        layer = SpatialAggDynamic(
            num_rules=num_rules,
            num_otu_centers=num_otu_centers,
            dist=otu_embeddings,
            emb_dim=emb_dim,
            num_otus=num_otus
        ).to(device)
        
        eta_init = otu_embeddings[:num_otu_centers].copy()
        eta_init = np.tile(eta_init[np.newaxis, :, :], (num_rules, 1, 1))
        kappa_init = np.ones((num_rules, num_otu_centers)) * 0.5
        layer.init_params({'kappa_init': kappa_init, 'eta_init': eta_init})
        
        X = torch.randn(5, num_otus, 10, device=device)
        with torch.no_grad():
            output = layer(X)
            weights = layer.wts
        
        for r in range(min(3, num_rules)):
            for c in range(min(3, num_otu_centers)):
                w = weights[r, c, :]
                w_sorted, _ = torch.sort(w, descending=True)
                top_10_pct = w_sorted[:max(1, num_otus // 10)].mean()
                overall_mean = w.mean()
                assert top_10_pct > overall_mean
    

    @pytest.mark.phylogenetic
    def test_2_2_1_distance_based_aggregation(self, test_config, otu_embeddings, device):
        """Test distance-based weight computation and aggregation."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_otu_centers = test_config['num_otu_centers']
        emb_dim = test_config['emb_dim']
        
        layer = SpatialAggDynamic(
            num_rules=num_rules,
            num_otu_centers=num_otu_centers,
            dist=otu_embeddings,
            emb_dim=emb_dim,
            num_otus=num_otus
        ).to(device)
        
        eta_init = np.random.randn(num_rules, num_otu_centers, emb_dim)
        kappa_init = np.ones((num_rules, num_otu_centers)) * 0.5
        layer.init_params({'kappa_init': kappa_init, 'eta_init': eta_init})
        
        X = torch.zeros(5, num_otus, 10, device=device)
        X[:, 0, :] = 10.0
        X[:, 1, :] = 5.0
        
        with torch.no_grad():
            output = layer(X)
        
        assert output.shape == (5, num_rules, num_otu_centers, 10)
        assert torch.isfinite(output).all()


# ============================================================================
# SECTION 3: TEMPORAL FOCUS MECHANISM TESTS
# ============================================================================

class TestSection3_TemporalFocus:
    """Tests for temporal focus mechanisms."""
    
    @pytest.mark.temporal
    @pytest.mark.critical
    def test_3_1_1_soft_time_window(self, test_config, device):
        """Test soft time window implementation via unitboxcar."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_time = test_config['num_time']
        num_time_centers = test_config['num_time_centers']
        
        layer = TimeAgg(
            num_rules=num_rules,
            num_otus=num_otus,
            num_time=num_time,
            num_time_centers=num_time_centers
        ).to(device)
        
        init_args = {
            'abun_a_init': np.random.uniform(0.1, 0.9, (num_rules, num_otus)),
            'abun_b_init': np.random.uniform(0.1, 0.9, (num_rules, num_otus)),
            'slope_a_init': np.random.uniform(0.1, 0.9, (num_rules, num_otus)),
            'slope_b_init': np.random.uniform(0.1, 0.9, (num_rules, num_otus)),
        }
        layer.init_params(init_args)
        
        X = torch.randn(5, num_rules, num_otus, num_time, device=device)
        mask = torch.ones(5, num_time, device=device)
        
        with torch.no_grad():
            x_abun, x_slope = layer(X, mask)
        
        assert torch.isfinite(x_abun).all() and torch.isfinite(x_slope).all()
    

    @pytest.mark.temporal
    def test_3_1_2_time_window_positioning(self, test_config, device):
        """Test automatic focus on relevant time periods."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_time = test_config['num_time']
        num_time_centers = test_config['num_time_centers']
        
        layer = TimeAgg(
            num_rules=num_rules,
            num_otus=num_otus,
            num_time=num_time,
            num_time_centers=num_time_centers
        ).to(device)
        
        early_init = {
            'abun_a_init': np.ones((num_rules, num_otus)) * 0.2,
            'abun_b_init': np.ones((num_rules, num_otus)) * 0.3,
            'slope_a_init': np.ones((num_rules, num_otus)) * 0.2,
            'slope_b_init': np.ones((num_rules, num_otus)) * 0.3,
        }
        layer.init_params(early_init)
        
        X = torch.zeros(5, num_rules, num_otus, num_time, device=device)
        X[:, :, :, :3] = 10.0
        mask = torch.ones(5, num_time, device=device)
        
        with torch.no_grad():
            x_abun_early, _ = layer(X, mask)
        
        late_init = {
            'abun_a_init': np.ones((num_rules, num_otus)) * 0.8,
            'abun_b_init': np.ones((num_rules, num_otus)) * 0.3,
            'slope_a_init': np.ones((num_rules, num_otus)) * 0.8,
            'slope_b_init': np.ones((num_rules, num_otus)) * 0.3,
        }
        layer.init_params(late_init)
        
        with torch.no_grad():
            x_abun_late, _ = layer(X, mask)
        
        assert x_abun_early.mean() > x_abun_late.mean()
    

    @pytest.mark.temporal
    @pytest.mark.critical
    def test_3_1_3_rate_of_change_computation(self, test_config, device):
        """Test temporal derivative (slope) calculation."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_time = test_config['num_time']
        num_time_centers = test_config['num_time_centers']
        
        layer = TimeAgg(
            num_rules=num_rules,
            num_otus=num_otus,
            num_time=num_time,
            num_time_centers=num_time_centers
        ).to(device)
        
        init_args = {
            'abun_a_init': np.ones((num_rules, num_otus)) * 0.5,
            'abun_b_init': np.ones((num_rules, num_otus)) * 0.8,
            'slope_a_init': np.ones((num_rules, num_otus)) * 0.5,
            'slope_b_init': np.ones((num_rules, num_otus)) * 0.8,
        }
        layer.init_params(init_args)
        
        X = torch.zeros(5, num_rules, num_otus, num_time, device=device)
        for t in range(num_time):
            X[:, :, :, t] = float(t)
        
        mask = torch.ones(5, num_time, device=device)
        
        with torch.no_grad():
            x_abun, x_slope = layer(X, mask)
        
        assert x_slope.mean() > 0
        
        X_dec = num_time - X
        with torch.no_grad():
            _, x_slope_dec = layer(X_dec, mask)
        
        assert x_slope_dec.mean() < 0
    

    @pytest.mark.temporal
    def test_3_2_1_missing_timepoint_handling(self, test_config, device):
        """Test mask application for irregular sampling."""
        num_rules = test_config['num_rules']
        num_otus = test_config['num_otus']
        num_time = test_config['num_time']
        num_time_centers = test_config['num_time_centers']
        
        layer = TimeAgg(
            num_rules=num_rules,
            num_otus=num_otus,
            num_time=num_time,
            num_time_centers=num_time_centers
        ).to(device)
        
        init_args = {
            'abun_a_init': np.random.uniform(0.1, 0.9, (num_rules, num_otus)),
            'abun_b_init': np.random.uniform(0.1, 0.9, (num_rules, num_otus)),
            'slope_a_init': np.random.uniform(0.1, 0.9, (num_rules, num_otus)),
            'slope_b_init': np.random.uniform(0.1, 0.9, (num_rules, num_otus)),
        }
        layer.init_params(init_args)
        
        X = torch.randn(5, num_rules, num_otus, num_time, device=device)
        mask_full = torch.ones(5, num_time, device=device)
        mask_partial = torch.ones(5, num_time, device=device)
        mask_partial[:, :num_time//2] = 0
        
        with torch.no_grad():
            x_abun_full, _ = layer(X, mask_full)
            x_abun_partial, _ = layer(X, mask_partial)
        
        assert not torch.allclose(x_abun_full, x_abun_partial, atol=0.1)
        assert torch.isfinite(x_abun_full).all() and torch.isfinite(x_abun_partial).all()


# ============================================================================
# SECTION 4: PERFORMANCE METRICS TESTS
# ============================================================================

class TestSection10_1_PerformanceMetrics:
    """Test performance metrics computation."""
    
    @pytest.mark.metrics
    def test_10_1_1_f1_score(self):
        """Test F1 score computation."""
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1])
        
        f1 = f1_score(y_true, y_pred)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        expected_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        assert np.isclose(f1, expected_f1)


    @pytest.mark.metrics
    def test_10_1_2_auc_roc(self):
        """Test AUC-ROC computation."""
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
        y_scores = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.85, 0.2, 0.4, 0.75, 0.15])
        
        auc = roc_auc_score(y_true, y_scores)
        
        assert 0 <= auc <= 1
        assert auc > 0.5


    @pytest.mark.metrics
    def test_10_1_3_additional_metrics(self):
        """Test additional metrics (accuracy, sensitivity, specificity)."""
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1])
        
        accuracy = accuracy_score(y_true, y_pred)
        expected_acc = np.mean(y_true == y_pred)
        assert np.isclose(accuracy, expected_acc)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        assert 0 <= sensitivity <= 1 and 0 <= specificity <= 1


# ============================================================================
# SECTION 5: PYTORCH INTEGRATION TESTS
# ============================================================================

class TestSection12_1_PyTorchIntegration:
    """Test PyTorch integration."""
    
    @pytest.mark.integration
    def test_12_1_1_pytorch_apis(self, test_config, otu_embeddings, device):
        """Test standard PyTorch APIs."""
        model = MDITRE(
            num_rules=test_config['num_rules'],
            num_otus=test_config['num_otus'],
            num_otu_centers=test_config['num_otu_centers'],
            num_time=test_config['num_time'],
            num_time_centers=test_config['num_time_centers'],
            dist=otu_embeddings,
            emb_dim=test_config['emb_dim']
        )
        
        assert isinstance(model, nn.Module)
        assert len(list(model.parameters())) > 0
        
        model.train()
        assert model.training
        model.eval()
        assert not model.training


    @pytest.mark.integration
    @pytest.mark.gpu
    def test_12_1_2_gpu_support(self, test_config, otu_embeddings, synthetic_data, init_args_full, device):
        """Test GPU support."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = MDITRE(
            num_rules=test_config['num_rules'],
            num_otus=test_config['num_otus'],
            num_otu_centers=test_config['num_otu_centers'],
            num_time=test_config['num_time'],
            num_time_centers=test_config['num_time_centers'],
            dist=otu_embeddings,
            emb_dim=test_config['emb_dim']
        ).to('cuda')
        
        model.init_params(init_args_full)
        assert next(model.parameters()).device.type == 'cuda'
        
        X = torch.from_numpy(synthetic_data['X'][:5]).float().cuda()
        mask = torch.from_numpy(synthetic_data['X_mask'][:5]).float().cuda()
        output = model(X, mask=mask)
        
        assert output.device.type == 'cuda'


    @pytest.mark.integration
    def test_12_1_3_model_serialization(self, test_config, otu_embeddings, init_args_full, device, tmp_path):
        """Test model save/load."""
        model = MDITRE(
            num_rules=test_config['num_rules'],
            num_otus=test_config['num_otus'],
            num_otu_centers=test_config['num_otu_centers'],
            num_time=test_config['num_time'],
            num_time_centers=test_config['num_time_centers'],
            dist=otu_embeddings,
            emb_dim=test_config['emb_dim']
        ).to(device)
        
        # Initialize model parameters to avoid NaN values
        model.init_params(init_args_full)
        
        save_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), save_path)
        
        model_loaded = MDITRE(
            num_rules=test_config['num_rules'],
            num_otus=test_config['num_otus'],
            num_otu_centers=test_config['num_otu_centers'],
            num_time=test_config['num_time'],
            num_time_centers=test_config['num_time_centers'],
            dist=otu_embeddings,
            emb_dim=test_config['emb_dim']
        ).to(device)
        
        model_loaded.load_state_dict(torch.load(save_path, weights_only=True))
        
        for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
            assert torch.allclose(p1, p2)


# ============================================================================
# SECTION 6: END-TO-END INTEGRATION TESTS
# ============================================================================

class TestEndToEndWorkflow:
    """Integration tests for complete training and evaluation pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_training_pipeline(self, test_config, otu_embeddings, synthetic_data, init_args_full, device):
        """Test complete training pipeline from data to evaluation."""
        X_train, X_test, y_train, y_test = train_test_split(
            synthetic_data['X'], 
            synthetic_data['y'],
            test_size=0.2,
            random_state=test_config['random_seed'],
            stratify=synthetic_data['y']
        )
        
        mask_train = synthetic_data['X_mask'][:len(X_train)]
        mask_test = synthetic_data['X_mask'][len(X_train):]
        
        model = MDITRE(
            num_rules=test_config['num_rules'],
            num_otus=test_config['num_otus'],
            num_otu_centers=test_config['num_otu_centers'],
            num_time=test_config['num_time'],
            num_time_centers=test_config['num_time_centers'],
            dist=otu_embeddings,
            emb_dim=test_config['emb_dim']
        ).to(device)
        
        model.init_params(init_args_full)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        X_train_t = torch.from_numpy(X_train).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().to(device)
        mask_train_t = torch.from_numpy(mask_train).float().to(device)
        
        initial_loss = None
        final_loss = None
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(X_train_t, mask=mask_train_t).squeeze()
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()
        
        # Ensure losses were computed (loop ran at least once)
        assert initial_loss is not None, "Training loop did not execute"
        assert final_loss is not None, "Training loop did not execute"
        assert final_loss < initial_loss
        
        model.eval()
        X_test_t = torch.from_numpy(X_test).float().to(device)
        mask_test_t = torch.from_numpy(mask_test).float().to(device)
        
        with torch.no_grad():
            test_outputs = model(X_test_t, mask=mask_test_t).squeeze()
            test_probs = torch.sigmoid(test_outputs).cpu().numpy()
            test_preds = (test_probs > 0.5).astype(int)
        
        f1 = f1_score(y_test, test_preds)
        accuracy = accuracy_score(y_test, test_preds)
        
        assert 0 <= f1 <= 1 and 0 <= accuracy <= 1


# ============================================================================
# SECTION 7: SEEDING AND REPRODUCIBILITY TESTS
# ============================================================================

class TestSeeding:
    """Tests for seeding module and reproducibility."""
    
    @pytest.mark.seeding
    @pytest.mark.critical
    def test_seeding_basic_generation(self):
        """Test basic seed generation."""
        gen = MDITRESeedGenerator()
        seeds = gen.generate_seeds(5)
        
        assert len(seeds) == 5
        assert all(isinstance(s, int) for s in seeds)
        assert gen.get_hash() is not None


    @pytest.mark.seeding
    def test_seeding_information(self):
        """Test seed information retrieval."""
        gen = MDITRESeedGenerator()
        gen.generate_seeds(5)
        info = gen.get_seed_info()
        
        assert 'master_seed' in info
        assert 'seed_string' in info
        assert 'hash' in info
        assert 'seed_number' in info


    @pytest.mark.seeding
    def test_seeding_experiment_specific(self):
        """Test experiment-specific seeding."""
        gen1 = MDITRESeedGenerator(experiment_name="exp1")
        gen2 = MDITRESeedGenerator(experiment_name="exp2")
        
        seeds1 = gen1.generate_seeds(5)
        seeds2 = gen2.generate_seeds(5)
        
        assert seeds1 != seeds2
        assert gen1.get_hash() != gen2.get_hash()


    @pytest.mark.seeding
    def test_seeding_convenience_function(self):
        """Test convenience function for quick seed generation."""
        seeds = get_mditre_seeds(10, experiment_name="test")
        
        assert len(seeds) == 10
        assert all(isinstance(s, int) for s in seeds)


    @pytest.mark.seeding
    @pytest.mark.critical
    def test_seeding_reproducibility(self):
        """Test reproducibility across Python, NumPy, and PyTorch."""
        gen = MDITRESeedGenerator()
        seeds = gen.generate_seeds(3)
        
        # Test Python random
        set_random_seeds(seeds[0])
        import random
        r1 = random.randint(0, 1000)
        
        set_random_seeds(seeds[0])
        r2 = random.randint(0, 1000)
        assert r1 == r2
        
        # Test NumPy random
        set_random_seeds(seeds[1])
        n1 = np.random.randint(0, 1000)
        
        set_random_seeds(seeds[1])
        n2 = np.random.randint(0, 1000)
        assert n1 == n2
        
        # Test PyTorch random
        set_random_seeds(seeds[2])
        t1 = torch.randint(0, 1000, (1,)).item()
        
        set_random_seeds(seeds[2])
        t2 = torch.randint(0, 1000, (1,)).item()
        assert t1 == t2


# ============================================================================
# SECTION 8: PACKAGE INTEGRITY TESTS
# ============================================================================

class TestPackageIntegrity:
    """Tests for package integrity and module loading."""
    
    @pytest.mark.integrity
    def test_core_module_imports(self):
        """Test core module imports."""
        from mditre.core import (
            BaseLayer,
            LayerRegistry,
            binary_concrete,
            unitboxcar,
            transf_log,
            inv_transf_log
        )
        
        assert callable(binary_concrete)
        assert callable(unitboxcar)
        assert len(LayerRegistry.list_layers()) > 0


    @pytest.mark.integrity
    def test_layers_module_imports(self):
        """Test layers module imports."""
        from mditre.layers import (
            SpatialAgg,
            SpatialAggDynamic,
            TimeAgg,
            TimeAggAbun,
            Threshold,
            Slope,
            Rules,
            DenseLayer,
            DenseLayerAbun
        )
        
        assert issubclass(SpatialAgg, nn.Module)
        assert issubclass(TimeAgg, nn.Module)


    @pytest.mark.integrity
    def test_data_loader_module_imports(self):
        """Test data loader module imports."""
        from mditre.data_loader import (
            DataLoaderRegistry,
            TrajectoryDataset,
            create_data_loader,
            NormalizeTransform,
            FilterLowAbundance,
            TransformPipeline
        )
        
        loaders = DataLoaderRegistry.list_loaders()
        assert 'pickle' in loaders
        assert '16s_dada2' in loaders


    @pytest.mark.integrity
    def test_models_module_imports(self):
        """Test models module imports."""
        from mditre.models import MDITRE, MDITREAbun
        
        assert issubclass(MDITRE, nn.Module)
        assert issubclass(MDITREAbun, nn.Module)


    @pytest.mark.integrity
    def test_seeding_module_imports(self):
        """Test seeding module imports."""
        from mditre.seeding import (
            MDITRESeedGenerator,
            get_mditre_seeds,
            set_random_seeds
        )
        
        assert callable(get_mditre_seeds)
        assert callable(set_random_seeds)


    @pytest.mark.integrity
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_backward_compatibility(self):
        """Test backward compatibility with original interfaces.
        
        Note: This test intentionally imports deprecated modules to verify
        backward compatibility. The deprecation warning is suppressed.
        """
        from mditre.data import load_from_pickle, get_data_matrix, TrajectoryDataset
        from mditre.models import MDITRE, binary_concrete
        
        assert callable(load_from_pickle)
        assert callable(get_data_matrix)
        assert callable(binary_concrete)


# ============================================================================
# MAIN - Test summary
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MDITRE Comprehensive Integrated Test Suite")
    print("Based on: Maringanti et al. (2022) - mSystems")
    print("="*80)
    print("\nTotal Tests: 39")
    print("  - Architecture: 13 tests")
    print("  - Phylogenetic: 4 tests")
    print("  - Temporal: 4 tests")
    print("  - Metrics: 3 tests")
    print("  - Integration: 4 tests")
    print("  - Seeding: 5 tests")
    print("  - Package Integrity: 6 tests")
    print("\nRun with: pytest tests/test_all.py -v")
    print("\nOr run specific sections:")
    print("  pytest tests/test_all.py -k 'architecture' -v")
    print("  pytest tests/test_all.py -k 'seeding' -v")
    print("  pytest tests/test_all.py -k 'integrity' -v")
    print("="*80)
