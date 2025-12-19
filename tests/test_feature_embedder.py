"""Tests for the feature embedder module."""

import pytest
import torch
from src.models.components.feature_embedder import FeatureEmbedder


class TestFeatureEmbedder:
    """Test the unified feature embedder."""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder for testing."""
        return FeatureEmbedder(hidden_dim=256)
    
    def test_embed_protein_graph(self, embedder):
        """Test embedding complete protein graph."""
        graph_data = {
            'node_feat': torch.tensor([
                [0, 0, 1, 0, 10, 1, 0],
                [1, 0, 1, 0, 10, 1, 1],
                [2, 0, 1, 0, 10, 1, 0],
            ], dtype=torch.long),
            'edge_attr': torch.tensor([[1.5], [2.0]], dtype=torch.float32),
            'edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            'pos': torch.randn(3, 3),
        }
        
        result = embedder.embed_protein_graph(graph_data)
        
        assert 'node_emb' in result
        assert 'edge_emb' in result
        assert 'edge_index' in result
        assert 'pos' in result
        assert result['node_emb'].shape == (3, 256)
        assert result['edge_emb'].shape == (2, 256)
        assert torch.equal(result['edge_index'], graph_data['edge_index'])
        assert torch.equal(result['pos'], graph_data['pos'])
    
    def test_embed_molecule_graph_both_edges(self, embedder):
        """Test embedding molecule graph with both edge types (concatenated)."""
        graph_data = {
            'node_feat': torch.tensor([
                [6, 0, 2, 6, 1, 0, 2, 1, 1],
                [6, 0, 2, 6, 1, 0, 2, 1, 1],
                [6, 0, 2, 6, 1, 0, 2, 1, 1],
            ], dtype=torch.long),
            'chem_edge_feat_cat': torch.tensor([
                [3, 0, 1],
                [3, 0, 1],
                [3, 0, 1],
            ], dtype=torch.long),
            'chem_edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            'edge_feat_dist': torch.tensor([[1.4], [1.4]], dtype=torch.float32),
            'edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            'pos': torch.randn(3, 3),
        }
        
        result = embedder.embed_molecule_graph(graph_data)
        
        # Check standardized output format
        assert 'node_emb' in result
        assert 'edge_emb' in result
        assert 'edge_index' in result
        assert 'pos' in result
        
        # Node embeddings
        assert result['node_emb'].shape == (3, 256)
        
        # Edges are concatenated: 3 chemical + 2 spatial = 5 total
        assert result['edge_emb'].shape == (5, 256)
        assert result['edge_index'].shape == (2, 5)
    
    def test_embed_molecule_graph_chem_only(self, embedder):
        """Test embedding molecule with only chemical edges."""
        graph_data = {
            'node_feat': torch.tensor([[6, 0, 2, 6, 1, 0, 2, 1, 1]], dtype=torch.long),
            'chem_edge_feat_cat': torch.tensor([[0, 0, 0]], dtype=torch.long),
            'chem_edge_index': torch.tensor([[0], [0]], dtype=torch.long),
            'pos': torch.randn(1, 3),
        }
        
        result = embedder.embed_molecule_graph(graph_data)
        
        assert result['node_emb'].shape == (1, 256)
        assert result['edge_emb'].shape == (1, 256)  # Only chemical edges
        assert result['edge_index'].shape == (2, 1)
    
    def test_embed_molecule_graph_spatial_only(self, embedder):
        """Test embedding molecule with only spatial edges."""
        graph_data = {
            'node_feat': torch.tensor([
                [6, 0, 2, 6, 1, 0, 2, 1, 1],
                [8, 0, 1, 6, 0, 0, 1, 0, 0],
            ], dtype=torch.long),
            'edge_feat_dist': torch.tensor([[1.5], [2.0]], dtype=torch.float32),
            'edge_index': torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            'pos': torch.randn(2, 3),
        }
        
        result = embedder.embed_molecule_graph(graph_data)
        
        assert result['node_emb'].shape == (2, 256)
        assert result['edge_emb'].shape == (2, 256)  # Only spatial edges
        assert result['edge_index'].shape == (2, 2)
    
    def test_batch_processing(self):
        """Test with larger batches."""
        embedder = FeatureEmbedder(hidden_dim=128)
        
        # Protein graph with valid ranges
        N_protein, E_protein = 100, 200
        protein_data = {
            'node_feat': torch.zeros((N_protein, 7), dtype=torch.long),
            'edge_attr': torch.rand(E_protein, 1),
            'edge_index': torch.randint(0, N_protein, (2, E_protein)),
            'pos': torch.randn(N_protein, 3),
        }
        protein_data['node_feat'][:, 0] = torch.randint(0, 46, (N_protein,))
        protein_data['node_feat'][:, 1] = torch.randint(0, 24, (N_protein,))
        protein_data['node_feat'][:, 2] = torch.randint(0, 12, (N_protein,))
        protein_data['node_feat'][:, 3] = torch.randint(0, 27, (N_protein,))
        protein_data['node_feat'][:, 4] = torch.arange(N_protein)  # residue_id
        protein_data['node_feat'][:, 5] = torch.randint(0, 2, (N_protein,))
        protein_data['node_feat'][:, 6] = torch.randint(0, 2, (N_protein,))
        
        result = embedder.embed_protein_graph(protein_data)
        assert result['node_emb'].shape == (N_protein, 128)
        assert result['edge_emb'].shape == (E_protein, 128)
        
        # Molecule graph with valid ranges
        N_mol, E_mol = 50, 100
        mol_data = {
            'node_feat': torch.zeros((N_mol, 9), dtype=torch.long),
            'edge_feat_dist': torch.rand(E_mol, 1),
            'edge_index': torch.randint(0, N_mol, (2, E_mol)),
            'pos': torch.randn(N_mol, 3),
        }
        mol_data['node_feat'][:, 0] = torch.randint(0, 119, (N_mol,))
        mol_data['node_feat'][:, 1] = torch.randint(0, 4, (N_mol,))
        mol_data['node_feat'][:, 2] = torch.randint(0, 12, (N_mol,))
        mol_data['node_feat'][:, 3] = torch.randint(0, 12, (N_mol,))
        mol_data['node_feat'][:, 4] = torch.randint(0, 10, (N_mol,))
        mol_data['node_feat'][:, 5] = torch.randint(0, 6, (N_mol,))
        mol_data['node_feat'][:, 6] = torch.randint(0, 6, (N_mol,))
        mol_data['node_feat'][:, 7] = torch.randint(0, 2, (N_mol,))
        mol_data['node_feat'][:, 8] = torch.randint(0, 2, (N_mol,))
        
        result = embedder.embed_molecule_graph(mol_data)
        assert result['node_emb'].shape == (N_mol, 128)
        assert result['edge_emb'].shape == (E_mol, 128)
    
    def test_missing_edges_error(self, embedder):
        """Test error handling when no edges provided."""
        graph_data = {
            'node_feat': torch.tensor([[6, 0, 2, 6, 1, 0, 2, 1, 1]], dtype=torch.long),
            'pos': torch.randn(1, 3),
        }
        
        with pytest.raises(ValueError, match="must contain either chemical edges or spatial edges"):
            embedder.embed_molecule_graph(graph_data)
    
    def test_different_hidden_dims(self):
        """Test with different hidden dimensions."""
        for hidden_dim in [64, 128, 256, 512]:
            embedder = FeatureEmbedder(hidden_dim=hidden_dim)
            
            protein_data = {
                'node_feat': torch.zeros((10, 7), dtype=torch.long),
                'edge_attr': torch.rand(20, 1),
                'edge_index': torch.randint(0, 10, (2, 20)),
                'pos': torch.randn(10, 3),
            }
            
            result = embedder.embed_protein_graph(protein_data)
            assert result['node_emb'].shape == (10, hidden_dim)
            assert result['edge_emb'].shape == (20, hidden_dim)
    
    def test_forward_protein_modality(self, embedder):
        """Test forward method with protein modality."""
        data = {
            'modality': 'protein',
            'value': {
                'node_feat': torch.tensor([
                    [0, 0, 1, 0, 10, 1, 0],
                    [1, 0, 1, 0, 10, 1, 1],
                    [2, 0, 1, 0, 10, 1, 0],
                ], dtype=torch.long),
                'edge_attr': torch.tensor([[1.5], [2.0]], dtype=torch.float32),
                'edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                'pos': torch.randn(3, 3),
            }
        }
        
        result = embedder(data)
        
        assert 'node_emb' in result
        assert 'edge_emb' in result
        assert 'edge_index' in result
        assert 'pos' in result
        assert result['node_emb'].shape == (3, 256)
        assert result['edge_emb'].shape == (2, 256)
    
    def test_forward_molecule_modality(self, embedder):
        """Test forward method with molecule modality."""
        data = {
            'modality': 'molecule',
            'value': {
                'node_feat': torch.tensor([
                    [6, 0, 2, 6, 1, 0, 2, 1, 1],
                    [6, 0, 2, 6, 1, 0, 2, 1, 1],
                ], dtype=torch.long),
                'edge_feat_dist': torch.tensor([[1.5]], dtype=torch.float32),
                'edge_index': torch.tensor([[0], [1]], dtype=torch.long),
                'pos': torch.randn(2, 3),
            }
        }
        
        result = embedder(data)
        
        assert 'node_emb' in result
        assert 'edge_emb' in result
        assert 'edge_index' in result
        assert 'pos' in result
        assert result['node_emb'].shape == (2, 256)
        assert result['edge_emb'].shape == (1, 256)
    
    def test_forward_invalid_modality(self, embedder):
        """Test forward method with invalid modality."""
        data = {
            'modality': 'invalid',
            'value': {
                'node_feat': torch.randn(3, 7),
                'edge_attr': torch.rand(2, 1),
                'edge_index': torch.randint(0, 3, (2, 2)),
                'pos': torch.randn(3, 3),
            }
        }
        
        with pytest.raises(ValueError, match="Unknown modality"):
            embedder(data)
    
    def test_forward_missing_modality_key(self, embedder):
        """Test forward method without modality key."""
        data = {
            'value': {
                'node_feat': torch.randn(3, 7),
                'edge_attr': torch.rand(2, 1),
                'edge_index': torch.randint(0, 3, (2, 2)),
                'pos': torch.randn(3, 3),
            }
        }
        
        with pytest.raises(ValueError, match="must contain 'modality' key"):
            embedder(data)
    
    def test_forward_missing_value_key(self, embedder):
        """Test forward method without value key."""
        data = {
            'modality': 'protein',
        }
        
        with pytest.raises(ValueError, match="must contain 'value' key"):
            embedder(data)
    
    def test_forward_molecule_both_edges(self, embedder):
        """Test forward method with molecule containing both edge types."""
        data = {
            'modality': 'molecule',
            'value': {
                'node_feat': torch.tensor([
                    [6, 0, 2, 6, 1, 0, 2, 1, 1],
                    [6, 0, 2, 6, 1, 0, 2, 1, 1],
                    [6, 0, 2, 6, 1, 0, 2, 1, 1],
                ], dtype=torch.long),
                'chem_edge_feat_cat': torch.tensor([
                    [3, 0, 1],
                    [3, 0, 1],
                ], dtype=torch.long),
                'chem_edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                'edge_feat_dist': torch.tensor([[1.4]], dtype=torch.float32),
                'edge_index': torch.tensor([[0], [1]], dtype=torch.long),
                'pos': torch.randn(3, 3),
            }
        }
        
        result = embedder(data)
        
        # Should concatenate 2 chemical + 1 spatial = 3 total edges
        assert result['edge_emb'].shape == (3, 256)
        assert result['edge_index'].shape == (2, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
