"""
Unit tests for EGNN Backbone
"""

import pytest
import torch
import torch.nn as nn
from src.models.components.egnn_backbone import EGNNBackbone, EGNNLayer, MLP


class TestMLP:
    """Test the MLP component."""
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        mlp = MLP(in_dim=64, out_dim=32, hidden_dim=128, num_layers=2, dropout=0.1)
        x = torch.randn(10, 64)
        out = mlp(x)
        
        assert out.shape == (10, 32), f"Expected shape (10, 32), got {out.shape}"
        assert not torch.isnan(out).any(), "NaNs detected in MLP output"
    
    def test_mlp_no_dropout(self):
        """Test MLP without dropout."""
        mlp = MLP(in_dim=64, out_dim=32, hidden_dim=128, num_layers=3, dropout=0.0)
        x = torch.randn(10, 64)
        out = mlp(x)
        
        assert out.shape == (10, 32)
        assert not torch.isnan(out).any()


class TestEGNNLayer:
    """Test the EGNN layer component."""
    
    def test_egnn_layer_forward(self):
        """Test EGNN layer forward pass."""
        layer = EGNNLayer(dim=128, dropout=0.1, update_coords=True, use_layernorm=True)
        
        N, E = 32, 100
        h = torch.randn(N, 128)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        e = torch.randn(E, 128)
        
        h_out, pos_out = layer(h, pos, edge_index, e)
        
        assert h_out.shape == (N, 128), f"Expected h shape {(N, 128)}, got {h_out.shape}"
        assert pos_out.shape == (N, 3), f"Expected pos shape {(N, 3)}, got {pos_out.shape}"
        assert not torch.isnan(h_out).any(), "NaNs detected in h_out"
        assert not torch.isnan(pos_out).any(), "NaNs detected in pos_out"
    
    def test_egnn_layer_no_coord_update(self):
        """Test EGNN layer without coordinate updates."""
        layer = EGNNLayer(dim=64, dropout=0.0, update_coords=False, use_layernorm=False)
        
        N, E = 16, 50
        h = torch.randn(N, 64)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        e = torch.randn(E, 64)
        
        h_out, pos_out = layer(h, pos, edge_index, e)
        
        assert h_out.shape == (N, 64)
        assert pos_out.shape == (N, 3)
        # Positions should remain unchanged
        assert torch.allclose(pos, pos_out), "Positions changed despite update_coords=False"
    
    def test_egnn_layer_deterministic(self):
        """Test that EGNN layer is deterministic."""
        torch.manual_seed(42)
        layer = EGNNLayer(dim=64, dropout=0.0, update_coords=True)
        
        N, E = 20, 60
        h = torch.randn(N, 64)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        e = torch.randn(E, 64)
        
        # First forward pass
        h_out1, pos_out1 = layer(h.clone(), pos.clone(), edge_index, e)
        
        # Second forward pass with same inputs
        h_out2, pos_out2 = layer(h.clone(), pos.clone(), edge_index, e)
        
        assert torch.allclose(h_out1, h_out2), "Non-deterministic node features"
        assert torch.allclose(pos_out1, pos_out2), "Non-deterministic coordinates"


class TestEGNNBackbone:
    """Test the EGNN Backbone."""
    
    def test_backbone_forward(self):
        """Test EGNN backbone forward pass."""
        model = EGNNBackbone(dim=256, num_layers=6, dropout=0.1, update_coords=True)
        
        N, E, dim = 64, 256, 256
        h = torch.randn(N, dim)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        e = torch.randn(E, dim)
        
        h_out, pos_out = model(h, pos, edge_index, e)
        
        assert h_out.shape == (N, dim), f"Expected h shape {(N, dim)}, got {h_out.shape}"
        assert pos_out.shape == (N, 3), f"Expected pos shape {(N, 3)}, got {pos_out.shape}"
        assert not torch.isnan(h_out).any(), "NaNs detected in h_out"
        assert not torch.isnan(pos_out).any(), "NaNs detected in pos_out"
    
    def test_backbone_small_graph(self):
        """Test backbone on small graph."""
        model = EGNNBackbone(dim=32, num_layers=2, dropout=0.0)
        
        N, E, dim = 5, 10, 32
        h = torch.randn(N, dim)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        e = torch.randn(E, dim)
        
        h_out, pos_out = model(h, pos, edge_index, e)
        
        assert h_out.shape == (N, dim)
        assert pos_out.shape == (N, 3)
        assert not torch.isnan(h_out).any()
        assert not torch.isnan(pos_out).any()
    
    def test_backbone_no_coord_update(self):
        """Test backbone without coordinate updates."""
        model = EGNNBackbone(dim=128, num_layers=3, update_coords=False)
        
        N, E, dim = 32, 100, 128
        h = torch.randn(N, dim)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        e = torch.randn(E, dim)
        
        h_out, pos_out = model(h, pos, edge_index, e)
        
        assert h_out.shape == (N, dim)
        assert pos_out.shape == (N, 3)
        # Positions should remain exactly the same
        assert torch.allclose(pos, pos_out), "Positions changed despite update_coords=False"
    
    def test_backbone_with_batch(self):
        """Test backbone with batch parameter (should not affect output)."""
        model = EGNNBackbone(dim=64, num_layers=2)
        
        N, E, dim = 40, 120, 64
        h = torch.randn(N, dim)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        e = torch.randn(E, dim)
        batch = torch.randint(0, 4, (N,))  # 4 graphs in batch
        
        h_out, pos_out = model(h, pos, edge_index, e, batch=batch)
        
        assert h_out.shape == (N, dim)
        assert pos_out.shape == (N, 3)
    
    def test_backbone_gradient_flow(self):
        """Test that gradients flow properly through the backbone."""
        model = EGNNBackbone(dim=32, num_layers=2, dropout=0.0)
        
        N, E, dim = 10, 30, 32
        h = torch.randn(N, dim, requires_grad=True)
        pos = torch.randn(N, 3, requires_grad=True)
        edge_index = torch.randint(0, N, (2, E))
        e = torch.randn(E, dim, requires_grad=True)
        
        h_out, pos_out = model(h, pos, edge_index, e)
        
        # Create a simple loss
        loss = h_out.sum() + pos_out.sum()
        loss.backward()
        
        # Check that gradients exist
        assert h.grad is not None, "No gradients for h"
        assert pos.grad is not None, "No gradients for pos"
        assert e.grad is not None, "No gradients for e"
        
        # Check that gradients are not all zeros
        assert not torch.allclose(h.grad, torch.zeros_like(h.grad)), "Zero gradients for h"
        assert not torch.allclose(pos.grad, torch.zeros_like(pos.grad)), "Zero gradients for pos"
    
    def test_backbone_different_num_layers(self):
        """Test backbone with different numbers of layers."""
        for num_layers in [1, 3, 6, 12]:
            model = EGNNBackbone(dim=64, num_layers=num_layers, dropout=0.0)
            
            N, E, dim = 20, 60, 64
            h = torch.randn(N, dim)
            pos = torch.randn(N, 3)
            edge_index = torch.randint(0, N, (2, E))
            e = torch.randn(E, dim)
            
            h_out, pos_out = model(h, pos, edge_index, e)
            
            assert h_out.shape == (N, dim), f"Failed for {num_layers} layers"
            assert pos_out.shape == (N, 3), f"Failed for {num_layers} layers"
            assert not torch.isnan(h_out).any(), f"NaNs for {num_layers} layers"
    
    def test_backbone_numerical_stability(self):
        """Test numerical stability with large coordinate values."""
        model = EGNNBackbone(dim=64, num_layers=3, dropout=0.0)
        
        N, E, dim = 20, 60, 64
        h = torch.randn(N, dim)
        pos = torch.randn(N, 3) * 100  # Large coordinate values
        edge_index = torch.randint(0, N, (2, E))
        e = torch.randn(E, dim)
        
        h_out, pos_out = model(h, pos, edge_index, e)
        
        assert not torch.isnan(h_out).any(), "NaNs with large coordinates"
        assert not torch.isnan(pos_out).any(), "NaNs with large coordinates"
        assert torch.isfinite(h_out).all(), "Inf values detected"
        assert torch.isfinite(pos_out).all(), "Inf values detected"
    
    def test_backbone_zero_edges(self):
        """Test backbone behavior with disconnected graph (no edges)."""
        model = EGNNBackbone(dim=32, num_layers=2, dropout=0.0)
        
        N, dim = 10, 32
        h = torch.randn(N, dim)
        pos = torch.randn(N, 3)
        edge_index = torch.zeros(2, 0, dtype=torch.long)  # No edges
        e = torch.zeros(0, dim)  # No edge features
        
        h_out, pos_out = model(h, pos, edge_index, e)
        
        # With no edges, node features should still change (via MLPs)
        # but coordinates should not change if update_coords=True (no messages to aggregate)
        assert h_out.shape == (N, dim)
        assert pos_out.shape == (N, 3)
        assert not torch.isnan(h_out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
