import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.data_factory.protein.pdbid_to_feature import (
    build_radius_graph,
    pdbid_to_features,
    pdbid_to_features_torch
)


class TestBuildRadiusGraph:
    """Tests for radius graph construction."""
    
    def test_empty_coordinates(self):
        """Test with empty coordinates."""
        coords = np.empty((0, 3))
        edge_index, edge_attr = build_radius_graph(coords, radius=5.0)
        
        assert edge_index.shape == (2, 0)
        assert edge_attr.shape == (0, 1)
    
    def test_single_atom(self):
        """Test with single atom (no edges)."""
        coords = np.array([[0.0, 0.0, 0.0]])
        edge_index, edge_attr = build_radius_graph(coords, radius=5.0)
        
        assert edge_index.shape == (2, 0)
        assert edge_attr.shape == (0, 1)
    
    def test_two_atoms_within_radius(self):
        """Test two atoms within radius."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ])
        edge_index, edge_attr = build_radius_graph(coords, radius=5.0)
        
        # Should have edges in both directions
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] >= 2  # At least 2 edges
        
        # Check distances
        assert np.allclose(edge_attr[0, 0], 3.0, atol=1e-5)
    
    def test_two_atoms_outside_radius(self):
        """Test two atoms outside radius."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0]
        ])
        edge_index, edge_attr = build_radius_graph(coords, radius=5.0)
        
        # No edges expected
        assert edge_index.shape == (2, 0)
        assert edge_attr.shape == (0, 1)
    
    def test_max_neighbors(self):
        """Test max_neighbors constraint."""
        # Create grid of 5 atoms in a line
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [8.0, 0.0, 0.0]
        ])
        
        # With radius 10, all atoms can reach all others
        # But with max_neighbors=2, each should have at most 2 edges
        edge_index, edge_attr = build_radius_graph(coords, radius=10.0, max_neighbors=2, sym_mode="mutual")
        
        # Count edges per node
        unique_sources = np.unique(edge_index[0])
        for source in unique_sources:
            num_edges = np.sum(edge_index[0] == source)
            assert num_edges <= 2, f"Node {source} has {num_edges} edges, expected <= 2"
    
    def test_realistic_protein_coords(self):
        """Test with realistic C-alpha coordinates."""
        # Simulate 10 C-alpha atoms in a chain (typical spacing ~3.8 Å)
        coords = np.array([
            [i * 3.8, 0.0, 0.0] for i in range(10)
        ])
        
        # Typical C-alpha radius is 8-10 Å
        edge_index, edge_attr = build_radius_graph(coords, radius=8.0)
        
        # Each interior atom should connect to ~2 neighbors
        assert edge_index.shape[1] > 0, "Should have edges"
        
        # Check edge distances are within radius
        assert np.all(edge_attr <= 8.0), "All edges should be within radius"
        assert np.all(edge_attr > 0), "All edges should have positive distance"


class TestPDBIDToFeatures:
    """Tests for PDB ID to features conversion."""
    
    def test_invalid_pdb_id(self):
        """Test with invalid PDB ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = pdbid_to_features('ZZZZ', structure_dir=tmpdir)
            # Should handle gracefully (download will fail)
            assert result is None or isinstance(result, dict)
    
    def test_empty_structure_dir(self):
        """Test creates structure directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / 'structures'
            assert not new_dir.exists()
            
            # This will try to download (may fail, but dir should be created)
            pdbid_to_features('1ABC', structure_dir=str(new_dir))
            assert new_dir.exists()
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_pdb_structure(self):
        """
        Integration test with real PDB structure.
        Uses 1AKI (small, well-characterized structure).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            result = pdbid_to_features(
                '1AKI',
                structure_dir=tmpdir,
                graph_radius=10.0,
                ca_only=False
            )
            
            if result is not None:
                # Check structure
                assert 'node_feat' in result
                assert 'coordinates' in result
                assert 'edge_index' in result
                assert 'edge_attr' in result
                assert 'num_nodes' in result
                assert 'pdb_id' in result
                assert 'atom_info' in result
                
                # Check shapes
                num_nodes = result['num_nodes']
                assert result['node_feat'].shape[0] == num_nodes
                assert result['coordinates'].shape == (num_nodes, 3)
                assert result['edge_index'].shape[0] == 2
                assert result['edge_attr'].shape[0] == result['edge_index'].shape[1]
                
                # Check PDB ID
                assert result['pdb_id'] == '1AKI'
                
                # Check coordinates are reasonable
                assert np.all(np.isfinite(result['coordinates']))
                
                print(f"\n1AKI: {num_nodes} atoms, {result['edge_index'].shape[1]} edges")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_ca_only_mode(self):
        """Test C-alpha only extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = pdbid_to_features(
                '1AKI',
                structure_dir=tmpdir,
                graph_radius=10.0,
                ca_only=True
            )
            
            if result is not None:
                # All atoms should be C-alpha
                for atom in result['atom_info']:
                    assert atom['atom_name'] == 'CA'
                
                print(f"\n1AKI C-alpha: {result['num_nodes']} atoms")


class TestPDBIDToFeaturesTorchconversion:
    """Test PyTorch tensor conversion."""
    
    def test_torch_conversion_mock(self):
        """Test that torch conversion would work with mock data."""
        # We'll just test the conversion logic without actual download
        import torch
        
        # Mock result
        mock_result = {
            'node_feat': np.random.rand(10, 7).astype(np.float32),
            'coordinates': np.random.rand(10, 3).astype(np.float32),
            'edge_index': np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
            'edge_attr': np.random.rand(3, 1).astype(np.float32),
            'num_nodes': 10,
            'pdb_id': 'TEST',
            'atom_info': []
        }
        
        # Convert manually (simulating pdbid_to_features_torch)
        converted = {
            'node_feat': torch.from_numpy(mock_result['node_feat']),
            'coordinates': torch.from_numpy(mock_result['coordinates']),
            'edge_index': torch.from_numpy(mock_result['edge_index']),
            'edge_attr': torch.from_numpy(mock_result['edge_attr']),
            'num_nodes': mock_result['num_nodes'],
            'pdb_id': mock_result['pdb_id'],
            'atom_info': mock_result['atom_info']
        }
        
        # Check types
        assert isinstance(converted['node_feat'], torch.Tensor)
        assert isinstance(converted['coordinates'], torch.Tensor)
        assert isinstance(converted['edge_index'], torch.Tensor)
        assert isinstance(converted['edge_attr'], torch.Tensor)
        
        # Check shapes preserved
        assert converted['node_feat'].shape == (10, 7)
        assert converted['coordinates'].shape == (10, 3)
        assert converted['edge_index'].shape == (2, 3)


if __name__ == '__main__':
    print("Running unit tests...")
    
    # Run fast tests
    test = TestBuildRadiusGraph()
    print("\n=== Testing radius graph construction ===")
    test.test_empty_coordinates()
    test.test_single_atom()
    test.test_two_atoms_within_radius()
    test.test_two_atoms_outside_radius()
    test.test_max_neighbors()
    test.test_realistic_protein_coords()
    print("✓ All radius graph tests passed")
    
    print("\n=== Testing PDB processing ===")
    test2 = TestPDBIDToFeatures()
    test2.test_empty_structure_dir()
    print("✓ Basic tests passed")
    
    print("\nRun with pytest for integration tests: pytest tests/test_pdbid_to_feature.py -v")
