"""Tests for AAEncoder - Combined Feature Embedder + EGNN Backbone"""

import pytest
import torch
from src.models import AAEncoder


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


class TestAAEncoder:
    """Test suite for AAEncoder."""
    
    @pytest.fixture
    def encoder(self):
        """Create a small encoder for testing."""
        return AAEncoder(
            hidden_dim=64,
            num_layers=2,
            dropout=0.0,
            update_coords=True,
            use_layernorm=True,
        )
    
    def test_protein_encoding(self, encoder):
        """Test encoding a protein graph."""
        N, E = 10, 20
        
        # Create protein graph data with proper feature ranges
        node_feat = create_protein_features(N)
        pos = torch.randn(N, 3)
        
        # Wrap in modality format
        data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_feat_dist': torch.rand(E, 1) * 5.0,
                'edge_index': torch.randint(0, N, (2, E)),
                'pos': pos,
            }
        }
        
        # Encode
        output = encoder(data)
        
        # Check output shapes
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
        
        # Check coordinates changed (if update_coords=True)
        if encoder.update_coords:
            assert not torch.allclose(pos, output['pos'], atol=1e-6)
    
    def test_protein_forward(self, encoder):
        """Test protein encoding via forward method with modality wrapper."""
        N, E = 10, 20
        
        # Prepare raw data with proper feature ranges
        node_feat = create_protein_features(N)
        
        data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_feat_dist': torch.rand(E, 1) * 5.0,
                'edge_index': torch.randint(0, N, (2, E)),
                'pos': torch.randn(N, 3),
            }
        }
        
        # Forward through encoder
        output = encoder(data)
        
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
    
    def test_molecule_encoding_spatial(self, encoder):
        """Test encoding a molecule with spatial edges."""
        N, E = 15, 30
        
        # Create molecule graph data with modality wrapper
        data = {
            'modality': 'molecule',
            'value': {
                'node_feat': create_molecule_features(N),
                'pos': torch.randn(N, 3),
                'edge_index': torch.randint(0, N, (2, E)),
                'edge_feat_dist': torch.rand(E, 1) * 5.0,
            }
        }
        
        # Encode
        output = encoder(data)
        
        # Check output shapes
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
    
    def test_molecule_encoding_chemical(self, encoder):
        """Test encoding a molecule with only chemical edges."""
        N, E_chem = 15, 25
        
        # Create molecule graph data with only chemical edges
        data = {
            'modality': 'molecule',
            'value': {
                'node_feat': create_molecule_features(N),
                'pos': torch.randn(N, 3),
                'chem_edge_index': torch.randint(0, N, (2, E_chem)),
                'chem_edge_feat_cat': create_chem_edge_features(E_chem),
            }
        }
        
        # Encode
        output = encoder(data)
        
        # Check output shapes
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
    
    def test_molecule_encoding_both_edges(self, encoder):
        """Test encoding a molecule with both chemical and spatial edges (should concatenate)."""
        N, E_chem, E_spatial = 15, 25, 40
        
        # Create molecule graph data with both edge types
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
        
        # Encode
        output = encoder(data)
        
        # Check output shapes (edges should be concatenated: E_chem + E_spatial)
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
    
    def test_molecule_forward(self, encoder):
        """Test molecule encoding via forward method with modality wrapper."""
        N, E = 15, 30
        
        # Prepare raw data with modality wrapper
        data = {
            'modality': 'molecule',
            'value': {
                'node_feat': create_molecule_features(N),
                'edge_feat_dist': torch.rand(E, 1) * 5.0,
                'edge_index': torch.randint(0, N, (2, E)),
                'pos': torch.randn(N, 3),
            }
        }
        
        # Forward through encoder
        output = encoder(data)
        
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
    
    def test_batch_processing(self, encoder):
        """Test with batch information."""
        N, E = 20, 40
        
        # Create batched data (2 graphs) with proper feature ranges
        node_feat = create_protein_features(N)
        batch = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
        
        data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_feat_dist': torch.rand(E, 1) * 5.0,
                'edge_index': torch.randint(0, N, (2, E)),
                'pos': torch.randn(N, 3),
            }
        }
        
        # Encode with batch info
        output = encoder(data, batch=batch)
        
        # Check output shapes
        assert output['node_emb'].shape == (N, encoder.hidden_dim)
        assert output['pos'].shape == (N, 3)
    
    def test_embedder_modality_handling(self, encoder):
        """Test that embedder correctly handles different modalities."""
        N = 10
        
        # Test protein embedding with proper feature ranges
        protein_data = {
            'node_feat': create_protein_features(N),
            'edge_feat_dist': torch.rand(20, 1),
            'edge_index': torch.randint(0, N, (2, 20)),
            'pos': torch.randn(N, 3),
        }
        protein_emb = encoder.embedder.embed_protein_graph(protein_data)
        
        assert 'node_emb' in protein_emb
        assert 'edge_emb' in protein_emb
        assert 'edge_index' in protein_emb
        assert 'pos' in protein_emb
        
        # Test molecule embedding
        mol_data = {
            'node_feat': create_molecule_features(N),
            'edge_feat_dist': torch.rand(20, 1),
            'edge_index': torch.randint(0, N, (2, 20)),
            'pos': torch.randn(N, 3),
        }
        mol_emb = encoder.embedder.embed_molecule_graph(mol_data)
        
        assert 'node_emb' in mol_emb
        assert 'edge_emb' in mol_emb
        assert 'edge_index' in mol_emb
        assert 'pos' in mol_emb
    
    def test_molecule_missing_edges(self, encoder):
        """Test error handling when molecule has no edges."""
        N = 15
        
        data = {
            'modality': 'molecule',
            'value': {
                'node_feat': create_molecule_features(N),
                'pos': torch.randn(N, 3),
            }
        }
        
        with pytest.raises(ValueError, match="must contain either chemical edges or spatial edges"):
            encoder(data)
    
    def test_no_coord_update(self):
        """Test encoder with coordinate updates disabled."""
        encoder = AAEncoder(
            hidden_dim=64,
            num_layers=2,
            update_coords=False,
        )
        
        N, E = 10, 20
        # Create protein features with proper ranges
        node_feat = create_protein_features(N)
        pos = torch.randn(N, 3)
        
        data = {
            'modality': 'protein',
            'value': {
                'node_feat': node_feat,
                'edge_feat_dist': torch.rand(E, 1) * 5.0,
                'edge_index': torch.randint(0, N, (2, E)),
                'pos': pos,
            }
        }
        
        # Encode
        output = encoder(data)
        
        # Coordinates should be unchanged when update_coords=False
        assert torch.allclose(pos, output['pos'], atol=1e-6)
    
    def test_different_hidden_dims(self):
        """Test encoder with different hidden dimensions."""
        for hidden_dim in [32, 128, 256]:
            encoder = AAEncoder(hidden_dim=hidden_dim, num_layers=2)
            
            N, E = 10, 20
            # Create protein features with proper ranges
            node_feat = create_protein_features(N)
            
            data = {
                'modality': 'protein',
                'value': {
                    'node_feat': node_feat,
                    'edge_feat_dist': torch.rand(E, 1) * 5.0,
                    'edge_index': torch.randint(0, N, (2, E)),
                    'pos': torch.randn(N, 3),
                }
            }
            
            # Encode
            output = encoder(data)
            
            assert output['node_emb'].shape == (N, hidden_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
