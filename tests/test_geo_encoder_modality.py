"""
Tests for GeoEncoder with Modality-Based Input
Tests that GeoEncoder can handle different modalities through its forward method.
"""

import pytest
import torch
from src.models import GeoEncoder


def create_protein_features(N):
    """Create valid protein node features."""
    node_feat = torch.zeros(N, 7)
    node_feat[:, 0] = torch.randint(0, 46, (N,)).float()  # atom_name (0-45)
    node_feat[:, 1] = torch.randint(0, 24, (N,)).float()  # residue (0-23)
    node_feat[:, 2] = torch.randint(0, 12, (N,)).float()  # element (0-11)
    node_feat[:, 3] = torch.randint(0, 27, (N,)).float()  # chain (0-26)
    node_feat[:, 4] = torch.arange(N).float()  # residue_id (continuous)
    node_feat[:, 5] = torch.randint(0, 2, (N,)).float()   # is_backbone (0-1)
    node_feat[:, 6] = torch.randint(0, 2, (N,)).float()   # is_ca (0-1)
    return node_feat


def create_molecule_features(N):
    """Create valid molecule node features."""
    node_feat = torch.zeros(N, 9)
    node_feat[:, 0] = torch.randint(0, 119, (N,)).float()  # atomic_num (0-118)
    node_feat[:, 1] = torch.randint(0, 4, (N,)).float()    # chirality (0-3)
    node_feat[:, 2] = torch.randint(0, 12, (N,)).float()   # degree (0-11)
    node_feat[:, 3] = torch.randint(0, 12, (N,)).float()   # charge (0-11)
    node_feat[:, 4] = torch.randint(0, 10, (N,)).float()   # numH (0-9)
    node_feat[:, 5] = torch.randint(0, 6, (N,)).float()    # radical (0-5)
    node_feat[:, 6] = torch.randint(0, 6, (N,)).float()    # hybrid (0-5)
    node_feat[:, 7] = torch.randint(0, 2, (N,)).float()    # aromatic (0-1)
    node_feat[:, 8] = torch.randint(0, 2, (N,)).float()    # ring (0-1)
    return node_feat


def create_chem_edge_features(E):
    """Create valid chemical edge features."""
    edge_feat = torch.zeros(E, 3)
    edge_feat[:, 0] = torch.randint(0, 5, (E,)).float()  # bond_type (0-4)
    edge_feat[:, 1] = torch.randint(0, 6, (E,)).float()  # bond_stereo (0-5)
    edge_feat[:, 2] = torch.randint(0, 2, (E,)).float()  # conjugated (0-1)
    return edge_feat


class TestGeoEncoderModality:
    """Test suite for GeoEncoder with modality-based inputs."""
    
    @pytest.fixture
    def encoder(self):
        """Create a small encoder for testing."""
        return GeoEncoder(
            hidden_dim=64,
            num_layers=2,
            dropout=0.0,
            update_coords=True,
            use_layernorm=True,
        )
    
    def test_protein_modality_input(self, encoder):
        """Test that GeoEncoder accepts protein input with modality key."""
        N, E = 10, 20
        
        # Create protein graph data with modality wrapper
        node_feat = create_protein_features(N)
        
        data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_attr': torch.rand(E, 1) * 5.0,
                'edge_index': torch.randint(0, N, (2, E)),
                'pos': torch.randn(N, 3),
            }
        }
        
        # Forward pass
        output = encoder(data)
        
        # Check output
        assert 'node_emb' in output
        assert 'pos' in output
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
    
    def test_molecule_spatial_only(self, encoder):
        """Test molecule with only spatial edges."""
        N, E = 15, 30
        
        data = {
            'modality': 'molecule',
            'value': {
                'node_feat': create_molecule_features(N),
                'pos': torch.randn(N, 3),
                'edge_index': torch.randint(0, N, (2, E)),
                'edge_feat_dist': torch.rand(E, 1) * 5.0,
            }
        }
        
        # Forward pass
        output = encoder(data)
        
        # Check output
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
    
    def test_molecule_chemical_only(self, encoder):
        """Test molecule with only chemical edges."""
        N, E_chem = 15, 25
        
        data = {
            'modality': 'molecule',
            'value': {
                'node_feat': create_molecule_features(N),
                'pos': torch.randn(N, 3),
                'chem_edge_index': torch.randint(0, N, (2, E_chem)),
                'chem_edge_feat_cat': create_chem_edge_features(E_chem),
            }
        }
        
        # Forward pass
        output = encoder(data)
        
        # Check output
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
    
    def test_molecule_both_edge_types(self, encoder):
        """Test molecule with both chemical and spatial edges."""
        N, E_chem, E_spatial = 15, 25, 40
        
        data = {
            'modality': 'molecule',
            'value': {
                'node_feat': create_molecule_features(N),
                'pos': torch.randn(N, 3),
                'edge_index': torch.randint(0, N, (2, E_spatial)),
                'edge_feat_dist': torch.rand(E_spatial, 1) * 5.0,
                'chem_edge_index': torch.randint(0, N, (2, E_chem)),
                'chem_edge_feat_cat': create_chem_edge_features(E_chem),
            }
        }
        
        # Forward pass
        output = encoder(data)
        
        # Check output
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
    
    def test_invalid_modality(self, encoder):
        """Test error handling for invalid modality."""
        N = 10
        
        data = {
            'modality': 'invalid_type',
            'value': {
                'node_feat': torch.randn(N, 7),
                'edge_attr': torch.rand(20, 1),
                'edge_index': torch.randint(0, N, (2, 20)),
                'pos': torch.randn(N, 3),
            }
        }
        
        with pytest.raises(ValueError, match="Unknown modality"):
            encoder(data)
    
    def test_missing_modality_key(self, encoder):
        """Test error handling when modality key is missing."""
        N = 10
        
        data = {
            'value': {
                'node_feat': torch.randn(N, 7),
                'edge_attr': torch.rand(20, 1),
                'edge_index': torch.randint(0, N, (2, 20)),
                'pos': torch.randn(N, 3),
            }
        }
        
        with pytest.raises(ValueError, match="must contain 'modality' key"):
            encoder(data)
    
    def test_missing_value_key(self, encoder):
        """Test error handling when value key is missing."""
        data = {
            'modality': 'protein',
        }
        
        with pytest.raises(ValueError, match="must contain 'value' key"):
            encoder(data)
    
    def test_protein_with_batch(self, encoder):
        """Test protein encoding with batch information."""
        N, E = 20, 40
        
        # Create batched data (2 graphs of 10 nodes each)
        node_feat = create_protein_features(N)
        
        data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_attr': torch.rand(E, 1) * 5.0,
                'edge_index': torch.randint(0, N, (2, E)),
                'pos': torch.randn(N, 3),
            }
        }
        
        batch = torch.cat([
            torch.zeros(10, dtype=torch.long),
            torch.ones(10, dtype=torch.long)
        ])
        
        # Forward pass with batch
        output = encoder(data, batch=batch)
        
        # Check output
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
    
    def test_molecule_with_batch(self, encoder):
        """Test molecule encoding with batch information."""
        N, E = 30, 60
        
        data = {
            'modality': 'molecule',
            'value': {
                'node_feat': create_molecule_features(N),
                'pos': torch.randn(N, 3),
                'edge_index': torch.randint(0, N, (2, E)),
                'edge_feat_dist': torch.rand(E, 1) * 5.0,
            }
        }
        
        batch = torch.cat([
            torch.zeros(15, dtype=torch.long),
            torch.ones(15, dtype=torch.long)
        ])
        
        # Forward pass with batch
        output = encoder(data, batch=batch)
        
        # Check output
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
    
    def test_coordinates_update(self, encoder):
        """Test that coordinates are updated when update_coords=True."""
        N, E = 10, 20
        
        pos_original = torch.randn(N, 3)
        node_feat = create_protein_features(N)
        
        data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_attr': torch.rand(E, 1) * 5.0,
                'edge_index': torch.randint(0, N, (2, E)),
                'pos': pos_original.clone(),
            }
        }
        
        # Forward pass
        output = encoder(data)
        
        # Coordinates should change (if update_coords=True)
        if encoder.update_coords:
            assert not torch.allclose(pos_original, output['pos'], atol=1e-6)
    
    def test_no_coordinates_update(self):
        """Test that coordinates remain unchanged when update_coords=False."""
        encoder = GeoEncoder(
            hidden_dim=64,
            num_layers=2,
            update_coords=False,
        )
        
        N, E = 10, 20
        pos_original = torch.randn(N, 3)
        node_feat = create_protein_features(N)
        
        data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_attr': torch.rand(E, 1) * 5.0,
                'edge_index': torch.randint(0, N, (2, E)),
                'pos': pos_original.clone(),
            }
        }
        
        # Forward pass
        output = encoder(data)
        
        # Coordinates should NOT change
        assert torch.allclose(pos_original, output['pos'], atol=1e-6)
    
    def test_different_hidden_dims(self):
        """Test encoder with different hidden dimensions."""
        N, E = 10, 20
        
        for hidden_dim in [32, 128, 256]:
            encoder = GeoEncoder(hidden_dim=hidden_dim, num_layers=2)
            
            node_feat = create_protein_features(N)
            
            data = {
                'modality': 'protein',
                'value': {
                    'node_feat': node_feat,
                    'edge_attr': torch.rand(E, 1) * 5.0,
                    'edge_index': torch.randint(0, N, (2, E)),
                    'pos': torch.randn(N, 3),
                }
            }
            
            output = encoder(data)
            
            assert output['node_emb'].shape == (N, hidden_dim)
            assert output['pos'].shape == (N, 3)
    
    def test_different_num_layers(self):
        """Test encoder with different numbers of layers."""
        N, E = 10, 20
        
        for num_layers in [1, 3, 6]:
            encoder = GeoEncoder(hidden_dim=64, num_layers=num_layers)
            
            data = {
                'modality': 'molecule',
                'value': {
                    'node_feat': create_molecule_features(N),
                    'pos': torch.randn(N, 3),
                    'edge_index': torch.randint(0, N, (2, E)),
                    'edge_feat_dist': torch.rand(E, 1) * 5.0,
                }
            }
            
            output = encoder(data)
            
            assert output['node_emb'].shape == (N, 64)
            assert output['pos'].shape == (N, 3)
    
    def test_gradient_flow_protein(self, encoder):
        """Test that gradients flow through the encoder for protein input."""
        N, E = 10, 20
        
        node_feat = create_protein_features(N)
        
        data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_attr': torch.rand(E, 1) * 5.0,
                'edge_index': torch.randint(0, N, (2, E)),
                'pos': torch.randn(N, 3),
            }
        }
        
        # Forward pass
        output = encoder(data)
        loss = output['node_emb'].sum()
        loss.backward()
        
        # Check that gradients flow through the network by checking embedding layer gradients
        # (can't check input gradients for categorical features since they're discrete indices)
        has_gradients = any(p.grad is not None for p in encoder.parameters() if p.requires_grad)
        assert has_gradients, "No gradients found in model parameters"
    
    def test_gradient_flow_molecule(self, encoder):
        """Test that gradients flow through the encoder for molecule input."""
        N, E = 10, 20
        
        node_feat = create_molecule_features(N)
        
        data = {
            'modality': 'molecule',
            'value': {
                'node_feat': node_feat,
                'pos': torch.randn(N, 3),
                'edge_index': torch.randint(0, N, (2, E)),
                'edge_feat_dist': torch.rand(E, 1) * 5.0,
            }
        }
        
        # Forward pass
        output = encoder(data)
        loss = output['node_emb'].sum()
        loss.backward()
        
        # Check that gradients flow through the network by checking embedding layer gradients
        # (can't check input gradients for categorical features since they're discrete indices)
        has_gradients = any(p.grad is not None for p in encoder.parameters() if p.requires_grad)
        assert has_gradients, "No gradients found in model parameters"
    
    def test_output_consistency(self, encoder):
        """Test that running the same input twice produces the same output (deterministic)."""
        N, E = 10, 20
        
        # Set to eval mode to avoid dropout randomness
        encoder.eval()
        
        node_feat = create_protein_features(N)
        
        data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_attr': torch.rand(E, 1) * 5.0,
                'edge_index': torch.randint(0, N, (2, E)),
                'pos': torch.randn(N, 3),
            }
        }
        
        # First forward pass
        with torch.no_grad():
            output1 = encoder(data)
        
        # Second forward pass with same data
        with torch.no_grad():
            output2 = encoder(data)
        
        # Outputs should be identical
        assert torch.allclose(output1['node_emb'], output2['node_emb'], atol=1e-6)
        assert torch.allclose(output1['pos'], output2['pos'], atol=1e-6)
    
    def test_large_graph(self, encoder):
        """Test encoder on a larger graph."""
        N, E = 100, 500
        
        data = {
            'modality': 'molecule',
            'value': {
                'node_feat': create_molecule_features(N),
                'pos': torch.randn(N, 3),
                'edge_index': torch.randint(0, N, (2, E)),
                'edge_feat_dist': torch.rand(E, 1) * 5.0,
            }
        }
        
        output = encoder(data)
        
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
