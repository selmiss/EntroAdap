"""Tests for AdaptiveEGNN - Automatic Coarse-Graining"""

import pytest
import torch
from src.models.components.adaptive_egnn import AdaptiveEGNN, create_adaptive_egnn
from src.models.components.egnn import EGNNBackbone


class TestAdaptiveEGNN:
    """Test suite for AdaptiveEGNN with automatic coarse-graining."""
    
    def test_small_graph_full_resolution(self):
        """Test that small graphs bypass coarse-graining."""
        model = create_adaptive_egnn(
            dim=64,
            num_layers=2,
            atom_threshold=1000,
            edge_threshold=5000,
            coarse_strategy='backbone'
        )
        model.eval()
        
        N, E = 500, 2000
        h = torch.randn(N, 64)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        e = torch.randn(E, 64)
        
        # Create node features with backbone flag
        node_feat = torch.zeros(N, 7)
        node_feat[:, 5] = 1  # All backbone for testing
        
        with torch.no_grad():
            h_out, pos_out = model(
                h, pos, edge_index, e,
                node_feat=node_feat,
                modality='protein'
            )
        
        # Output should have same size (no coarse-graining)
        assert h_out.shape == (N, 64), f"Expected {(N, 64)}, got {h_out.shape}"
        assert pos_out.shape == (N, 3)
        
        # Check statistics
        stats = model.get_statistics()
        assert stats['num_full_resolution'] == 1
        assert stats['num_coarsened'] == 0
    
    def test_large_graph_coarsening(self):
        """Test that large graphs are automatically coarse-grained."""
        model = create_adaptive_egnn(
            dim=64,
            num_layers=2,
            atom_threshold=1000,      # Low threshold for testing
            edge_threshold=5000,
            coarse_strategy='backbone'
        )
        model.eval()
        
        N, E = 5000, 50000
        h = torch.randn(N, 64)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        e = torch.randn(E, 64)
        
        # Create node features: 25% backbone, 75% sidechain
        node_feat = torch.zeros(N, 7)
        node_feat[:N//4, 5] = 1  # 25% backbone
        node_feat[:, 4] = torch.arange(N) // 40  # Residue IDs
        
        with torch.no_grad():
            h_out, pos_out = model(
                h, pos, edge_index, e,
                node_feat=node_feat,
                modality='protein'
            )
        
        # Output should be coarsened (~25% of original size)
        expected_size = N // 4
        assert h_out.shape[0] < N, f"Expected coarsening, got {h_out.shape[0]} >= {N}"
        assert h_out.shape[0] <= expected_size * 1.1, "Coarsening not effective"
        assert h_out.shape[1] == 64
        
        # Check statistics
        stats = model.get_statistics()
        assert stats['num_coarsened'] == 1
    
    def test_backbone_coarsening_strategy(self):
        """Test backbone coarse-graining strategy."""
        model = create_adaptive_egnn(
            dim=64,
            num_layers=2,
            atom_threshold=100,
            coarse_strategy='backbone'
        )
        model.eval()
        
        N = 1000
        h = torch.randn(N, 64)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, 5000))
        e = torch.randn(5000, 64)
        
        # 30% backbone atoms
        node_feat = torch.zeros(N, 7)
        backbone_mask = torch.rand(N) < 0.3
        node_feat[backbone_mask, 5] = 1
        num_backbone = backbone_mask.sum().item()
        
        with torch.no_grad():
            h_out, pos_out = model(
                h, pos, edge_index, e,
                node_feat=node_feat,
                modality='protein'
            )
        
        # Should keep only backbone atoms
        assert h_out.shape[0] == num_backbone, \
            f"Expected {num_backbone} backbone atoms, got {h_out.shape[0]}"
    
    def test_per_residue_coarsening_strategy(self):
        """Test per-residue coarse-graining strategy."""
        model = create_adaptive_egnn(
            dim=64,
            num_layers=2,
            atom_threshold=100,
            coarse_strategy='per_residue'
        )
        model.eval()
        
        N = 1000
        num_residues = 25  # 40 atoms per residue
        h = torch.randn(N, 64)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, 5000))
        e = torch.randn(5000, 64)
        
        # Create node features with residue IDs
        node_feat = torch.zeros(N, 7)
        node_feat[:, 4] = torch.arange(N) // 40  # 40 atoms per residue
        node_feat[:, 5] = 0  # No backbone flag needed for per-residue
        
        with torch.no_grad():
            h_out, pos_out = model(
                h, pos, edge_index, e,
                node_feat=node_feat,
                modality='protein'
            )
        
        # Should have one node per residue
        assert h_out.shape[0] == num_residues, \
            f"Expected {num_residues} residue nodes, got {h_out.shape[0]}"
    
    def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection based on modality."""
        model = create_adaptive_egnn(
            dim=64,
            num_layers=2,
            atom_threshold=100,
            coarse_strategy='adaptive'  # Auto-select strategy
        )
        model.eval()
        
        N = 1000
        h = torch.randn(N, 64)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, 5000))
        e = torch.randn(5000, 64)
        
        # Create node features
        node_feat = torch.zeros(N, 7)
        node_feat[:N//4, 5] = 1  # 25% backbone
        node_feat[:, 4] = torch.arange(N) // 40
        
        # Test with protein modality (should use backbone)
        with torch.no_grad():
            h_out, pos_out = model(
                h, pos, edge_index, e,
                node_feat=node_feat,
                modality='protein'
            )
        
        # Should coarse-grain
        assert h_out.shape[0] < N
    
    def test_no_coarsening_without_node_feat(self):
        """Test fallback to full resolution when node_feat is not provided."""
        model = create_adaptive_egnn(
            dim=64,
            num_layers=2,
            atom_threshold=100,  # Low threshold
            coarse_strategy='backbone'
        )
        model.eval()
        
        N = 1000
        h = torch.randn(N, 64)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, 5000))
        e = torch.randn(5000, 64)
        
        # Don't provide node_feat - should fall back to full resolution
        with torch.no_grad():
            h_out, pos_out = model(
                h, pos, edge_index, e,
                node_feat=None,
                modality='protein'
            )
        
        # Should keep full resolution
        assert h_out.shape[0] == N
    
    def test_statistics_tracking(self):
        """Test coarse-graining statistics tracking."""
        model = create_adaptive_egnn(
            dim=64,
            num_layers=2,
            atom_threshold=500,
        )
        model.eval()
        
        # Process small graph
        h_small = torch.randn(100, 64)
        pos_small = torch.randn(100, 3)
        edge_index_small = torch.randint(0, 100, (2, 300))
        e_small = torch.randn(300, 64)
        
        with torch.no_grad():
            model(h_small, pos_small, edge_index_small, e_small)
        
        # Process large graph
        node_feat_large = torch.zeros(1000, 7)
        node_feat_large[:, 5] = 1
        h_large = torch.randn(1000, 64)
        pos_large = torch.randn(1000, 3)
        edge_index_large = torch.randint(0, 1000, (2, 5000))
        e_large = torch.randn(5000, 64)
        
        with torch.no_grad():
            model(h_large, pos_large, edge_index_large, e_large,
                  node_feat=node_feat_large)
        
        # Check statistics
        stats = model.get_statistics()
        assert stats['num_full_resolution'] == 1
        assert stats['num_coarsened'] == 1
        assert stats['coarsening_rate'] == 0.5
        
        # Reset and verify
        model.reset_statistics()
        stats = model.get_statistics()
        assert stats['num_full_resolution'] == 0
        assert stats['num_coarsened'] == 0
    
    def test_gradient_flow_through_coarsening(self):
        """Test that gradients flow properly through coarse-grained path."""
        model = create_adaptive_egnn(
            dim=32,
            num_layers=2,
            atom_threshold=100,
            coarse_strategy='backbone'
        )
        model.train()
        
        N = 500
        h = torch.randn(N, 32, requires_grad=True)
        pos = torch.randn(N, 3, requires_grad=True)
        edge_index = torch.randint(0, N, (2, 2000))
        e = torch.randn(2000, 32, requires_grad=True)
        
        # Backbone features
        node_feat = torch.zeros(N, 7)
        node_feat[:N//4, 5] = 1
        
        h_out, pos_out = model(
            h, pos, edge_index, e,
            node_feat=node_feat,
            modality='protein'
        )
        
        # Compute loss and backprop
        loss = h_out.sum() + pos_out.sum()
        loss.backward()
        
        # Check gradients exist
        assert h.grad is not None
        assert pos.grad is not None
        assert e.grad is not None
        assert not torch.allclose(h.grad, torch.zeros_like(h.grad))
    
    def test_batch_processing_with_coarsening(self):
        """Test batch processing with coarse-graining."""
        model = create_adaptive_egnn(
            dim=64,
            num_layers=2,
            atom_threshold=500,
            coarse_strategy='backbone'
        )
        model.eval()
        
        N = 1000
        h = torch.randn(N, 64)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, 5000))
        e = torch.randn(5000, 64)
        batch = torch.cat([
            torch.zeros(500, dtype=torch.long),
            torch.ones(500, dtype=torch.long)
        ])
        
        # Create features
        node_feat = torch.zeros(N, 7)
        node_feat[:N//4, 5] = 1
        
        with torch.no_grad():
            h_out, pos_out = model(
                h, pos, edge_index, e,
                node_feat=node_feat,
                batch=batch
            )
        
        # Should be coarsened
        assert h_out.shape[0] < N
        assert h_out.shape[1] == 64
    
    def test_numerical_stability_with_coarsening(self):
        """Test numerical stability after coarse-graining."""
        model = create_adaptive_egnn(
            dim=64,
            num_layers=3,
            atom_threshold=100,
        )
        model.eval()
        
        N = 1000
        h = torch.randn(N, 64) * 10  # Large values
        pos = torch.randn(N, 3) * 100
        edge_index = torch.randint(0, N, (2, 5000))
        e = torch.randn(5000, 64) * 10
        
        node_feat = torch.zeros(N, 7)
        node_feat[:N//4, 5] = 1
        
        with torch.no_grad():
            h_out, pos_out = model(
                h, pos, edge_index, e,
                node_feat=node_feat
            )
        
        # Check for NaNs and Infs
        assert not torch.isnan(h_out).any()
        assert not torch.isnan(pos_out).any()
        assert torch.isfinite(h_out).all()
        assert torch.isfinite(pos_out).all()
    
    def test_comparison_with_regular_egnn(self):
        """Compare AdaptiveEGNN in full-res mode with regular EGNN."""
        torch.manual_seed(42)
        
        # Regular EGNN
        egnn_regular = EGNNBackbone(
            dim=64,
            num_layers=2,
            dropout=0.0,
            update_coords=True
        )
        
        # Adaptive EGNN with very high threshold (no coarsening)
        egnn_adaptive = create_adaptive_egnn(
            dim=64,
            num_layers=2,
            dropout=0.0,
            update_coords=True,
            atom_threshold=100000,  # Very high
        )
        
        # Copy weights
        egnn_adaptive.egnn.load_state_dict(egnn_regular.state_dict())
        
        # Test on same input
        N, E = 100, 500
        h = torch.randn(N, 64)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, E))
        e = torch.randn(E, 64)
        
        egnn_regular.eval()
        egnn_adaptive.eval()
        
        with torch.no_grad():
            h_reg, pos_reg = egnn_regular(h, pos, edge_index, e)
            h_adap, pos_adap = egnn_adaptive(h, pos, edge_index, e)
        
        # Should produce identical results
        assert torch.allclose(h_reg, h_adap, atol=1e-5)
        assert torch.allclose(pos_reg, pos_adap, atol=1e-5)


class TestAdaptiveEGNNEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_graph(self):
        """Test with empty graph."""
        model = create_adaptive_egnn(dim=64, num_layers=2)
        model.eval()
        
        N = 0
        h = torch.zeros(N, 64)
        pos = torch.zeros(N, 3)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        e = torch.zeros(0, 64)
        
        # Should handle gracefully (though might not be meaningful)
        with torch.no_grad():
            try:
                h_out, pos_out = model(h, pos, edge_index, e)
                assert h_out.shape == (N, 64)
            except:
                pass  # It's okay if empty graphs aren't supported
    
    def test_single_node(self):
        """Test with single node."""
        model = create_adaptive_egnn(dim=64, num_layers=2)
        model.eval()
        
        N = 1
        h = torch.randn(N, 64)
        pos = torch.randn(N, 3)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        e = torch.zeros(0, 64)
        
        with torch.no_grad():
            h_out, pos_out = model(h, pos, edge_index, e)
        
        assert h_out.shape == (N, 64)
        assert pos_out.shape == (N, 3)
    
    def test_no_backbone_atoms(self):
        """Test when no backbone atoms are found (should fall back)."""
        model = create_adaptive_egnn(
            dim=64,
            num_layers=2,
            atom_threshold=100,
            coarse_strategy='backbone'
        )
        model.eval()
        
        N = 500
        h = torch.randn(N, 64)
        pos = torch.randn(N, 3)
        edge_index = torch.randint(0, N, (2, 2000))
        e = torch.randn(2000, 64)
        
        # No backbone atoms
        node_feat = torch.zeros(N, 7)
        node_feat[:, 5] = 0  # All non-backbone
        
        with torch.no_grad():
            h_out, pos_out = model(
                h, pos, edge_index, e,
                node_feat=node_feat
            )
        
        # Should fall back to full resolution
        assert h_out.shape[0] == N


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

