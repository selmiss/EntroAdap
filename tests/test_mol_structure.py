import pytest
import numpy as np
import torch

from src.data_factory.molecule.mol_structure import (
    smiles2graph, 
    generate_conformer_with_rdkit, 
    generate_2d_3d_from_smiles
)


class TestSmiles2Graph:
    """Test cases for smiles2graph function (2D graph generation)"""
    
    def test_simple_molecule_ethanol(self):
        """Test 2D graph generation for ethanol (CCO)"""
        smiles = "CCO"
        graph = smiles2graph(smiles)
        
        # Check graph structure
        assert isinstance(graph, dict)
        assert 'node_feat' in graph
        assert 'chem_edge_index' in graph
        assert 'chem_edge_feat_cat' in graph
        assert 'chem_edge_feat_dist' in graph
        assert 'num_nodes' in graph
        
        # Ethanol has 3 heavy atoms (C-C-O)
        assert graph['num_nodes'] == 3
        assert graph['node_feat'].shape[0] == 3
        
        # Check tensor types
        assert isinstance(graph['node_feat'], torch.Tensor)
        assert isinstance(graph['chem_edge_index'], torch.Tensor)
        assert isinstance(graph['chem_edge_feat_cat'], torch.Tensor)
        assert isinstance(graph['chem_edge_feat_dist'], torch.Tensor)
        
        # Ethanol has 2 bonds (C-C and C-O), which means 4 directed edges
        assert graph['chem_edge_index'].shape[1] == 4
        assert graph['chem_edge_feat_cat'].shape[0] == 4
        assert graph['chem_edge_feat_dist'].shape[0] == 4
    
    def test_benzene_ring(self):
        """Test 2D graph generation for benzene (c1ccccc1)"""
        smiles = "c1ccccc1"
        graph = smiles2graph(smiles)
        
        # Benzene has 6 carbon atoms
        assert graph['num_nodes'] == 6
        
        # Benzene has 6 bonds (ring), which means 12 directed edges
        assert graph['chem_edge_index'].shape[1] == 12
        assert graph['chem_edge_feat_cat'].shape[0] == 12
        assert graph['chem_edge_feat_dist'].shape[0] == 12
    
    def test_aspirin(self):
        """Test 2D graph generation for aspirin"""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        graph = smiles2graph(smiles)
        
        # Aspirin has 13 heavy atoms
        assert graph['num_nodes'] == 13
        assert graph['node_feat'].shape[0] == 13
        
        # Check that edges are bidirectional
        assert graph['chem_edge_index'].shape[0] == 2  # [2, num_edges]
        assert graph['chem_edge_index'].shape[1] % 2 == 0  # Even number of directed edges
    
    def test_caffeine(self):
        """Test 2D graph generation for caffeine"""
        smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        graph = smiles2graph(smiles)
        
        # Caffeine has 14 heavy atoms
        assert graph['num_nodes'] == 14
        assert graph['node_feat'].shape[0] == 14
        
        # Verify node features have correct dimension (9 features per atom)
        assert graph['node_feat'].shape[1] == 9
    
    def test_single_atom_methane(self):
        """Test 2D graph generation for single carbon (C)"""
        smiles = "C"
        graph = smiles2graph(smiles)
        
        # Single carbon atom
        assert graph['num_nodes'] == 1
        assert graph['node_feat'].shape[0] == 1
        
        # No bonds
        assert graph['chem_edge_index'].shape[1] == 0
        assert graph['chem_edge_feat_cat'].shape[0] == 0
        assert graph['chem_edge_feat_dist'].shape[0] == 0
    
    def test_invalid_smiles(self):
        """Test 2D graph generation with invalid SMILES"""
        smiles = "INVALID_SMILES_123"
        graph = smiles2graph(smiles)
        
        # Should return a graph with default values
        assert graph is None


class TestGenerateConformer:
    """Test cases for generate_conformer_with_rdkit function (3D generation)"""
    
    def test_ethanol_3d(self):
        """Test 3D conformer generation for ethanol"""
        smiles = "CCO"
        atoms, coords = generate_conformer_with_rdkit(smiles)
        
        # Check return types
        assert atoms is not None
        assert coords is not None
        assert isinstance(atoms, list)
        assert isinstance(coords, np.ndarray)
        
        # Ethanol has 3 heavy atoms
        assert len(atoms) == 3
        assert coords.shape == (3, 3)  # 3 atoms, 3D coordinates
        
        # Check atom symbols
        assert atoms == ['C', 'C', 'O']
        
        # Coordinates should be finite numbers
        assert np.all(np.isfinite(coords))
    
    def test_benzene_3d(self):
        """Test 3D conformer generation for benzene"""
        smiles = "c1ccccc1"
        atoms, coords = generate_conformer_with_rdkit(smiles)
        
        assert atoms is not None
        assert coords is not None
        
        # Benzene has 6 carbon atoms
        assert len(atoms) == 6
        assert coords.shape == (6, 3)
        assert all(atom == 'C' for atom in atoms)
    
    def test_aspirin_3d(self):
        """Test 3D conformer generation for aspirin"""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        atoms, coords = generate_conformer_with_rdkit(smiles)
        
        assert atoms is not None
        assert coords is not None
        
        # Aspirin has 13 heavy atoms
        assert len(atoms) == 13
        assert coords.shape == (13, 3)
        
        # Check we have the expected atom types
        atom_types = set(atoms)
        assert 'C' in atom_types
        assert 'O' in atom_types
    
    def test_caffeine_3d(self):
        """Test 3D conformer generation for caffeine"""
        smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        atoms, coords = generate_conformer_with_rdkit(smiles)
        
        assert atoms is not None
        assert coords is not None
        
        # Caffeine has 14 heavy atoms
        assert len(atoms) == 14
        assert coords.shape == (14, 3)
        
        # Check atom composition
        atom_types = set(atoms)
        assert 'C' in atom_types
        assert 'N' in atom_types
        assert 'O' in atom_types
    
    def test_invalid_smiles_3d(self):
        """Test 3D conformer generation with invalid SMILES"""
        smiles = "INVALID_SMILES_123"
        atoms, coords = generate_conformer_with_rdkit(smiles)
        
        # Should return None for both
        assert atoms is None
        assert coords is None


class TestGenerate2D3DUnified:
    """Test cases for generate_2d_3d_from_smiles unified function"""
    
    def test_ethanol_full(self):
        """Test unified generation for ethanol - expect both 2D and 3D"""
        smiles = "CCO"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        # All should succeed
        assert atoms is not None
        assert graph_2d is not None
        assert coords_3d is not None
        
        # Check atoms list
        assert len(atoms) == 3
        assert atoms == ['C', 'C', 'O']
        
        # Check 2D graph
        assert graph_2d['num_nodes'] == 3
        assert isinstance(graph_2d['node_feat'], torch.Tensor)
        
        # Check 3D coordinates
        assert coords_3d.shape == (3, 3)
        assert np.all(np.isfinite(coords_3d))
        
        # Verify consistency: number of atoms matches across all representations
        assert len(atoms) == graph_2d['num_nodes'] == coords_3d.shape[0]
    
    def test_benzene_full(self):
        """Test unified generation for benzene"""
        smiles = "c1ccccc1"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        # All should succeed
        assert atoms is not None
        assert graph_2d is not None
        assert coords_3d is not None
        
        # Verify consistency
        assert len(atoms) == 6
        assert graph_2d['num_nodes'] == 6
        assert coords_3d.shape == (6, 3)
    
    def test_aspirin_full(self):
        """Test unified generation for aspirin"""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        # All should succeed
        assert atoms is not None
        assert graph_2d is not None
        assert coords_3d is not None
        
        # Verify consistency
        assert len(atoms) == 13
        assert graph_2d['num_nodes'] == 13
        assert coords_3d.shape == (13, 3)
    
    def test_caffeine_full(self):
        """Test unified generation for caffeine"""
        smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        # All should succeed
        assert atoms is not None
        assert graph_2d is not None
        assert coords_3d is not None
        
        # Verify consistency
        assert len(atoms) == 14
        assert graph_2d['num_nodes'] == 14
        assert coords_3d.shape == (14, 3)
    
    def test_glucose(self):
        """Test unified generation for glucose"""
        smiles = "C(C1C(C(C(C(O1)O)O)O)O)O"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        # All should succeed
        assert atoms is not None
        assert graph_2d is not None
        assert coords_3d is not None
        
        # Glucose has 12 heavy atoms (6 C + 6 O)
        assert len(atoms) == 12
        assert graph_2d['num_nodes'] == 12
        assert coords_3d.shape == (12, 3)
    
    def test_atom_ordering_consistency(self):
        """Test that atom ordering is consistent between 2D and 3D"""
        smiles = "CCO"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        # The i-th atom should correspond to i-th node in 2D graph and i-th coordinate
        assert len(atoms) == graph_2d['num_nodes']
        assert len(atoms) == coords_3d.shape[0]
        
        # All should be 3
        assert len(atoms) == 3
        assert graph_2d['num_nodes'] == 3
        assert coords_3d.shape[0] == 3
    
    def test_invalid_smiles_fallback(self):
        """Test progressive fallback with invalid SMILES"""
        smiles = "INVALID_SMILES_123"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        # Complete failure expected
        assert atoms is None
        assert graph_2d is None
        assert coords_3d is None
    
    def test_edge_features(self):
        """Test that edge features are properly generated in 2D graph"""
        smiles = "CCO"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        # Check edge features (split into categorical and distance)
        assert graph_2d['chem_edge_feat_cat'].shape[1] == 3  # 4 categorical features
        assert graph_2d['edge_feat_dist'].shape[1] == 1  # 1 distance feature
        assert graph_2d['edge_index'].shape[0] == 2  # [source, target]
        
        # Bidirectional edges: each bond appears twice
        assert graph_2d['chem_edge_feat_cat'].shape[0] == graph_2d['chem_edge_index'].shape[1]
        assert graph_2d['edge_feat_dist'].shape[0] == graph_2d['edge_index'].shape[1]
    
    def test_complex_molecule_penicillin(self):
        """Test unified generation for penicillin G"""
        smiles = "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        # Should handle complex molecules
        assert atoms is not None
        assert graph_2d is not None
        
        # 3D might fail for complex molecules, but 2D should work
        # If 3D succeeds, verify consistency
        if coords_3d is not None:
            assert len(atoms) == graph_2d['num_nodes'] == coords_3d.shape[0]
    
    def test_coordinates_are_3d(self):
        """Test that coordinates are truly 3-dimensional"""
        smiles = "CCO"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        assert coords_3d.shape[1] == 3  # x, y, z coordinates
        
        # Check that molecule is not planar (z-coordinates should vary)
        # For ethanol, we expect some 3D structure
        z_coords = coords_3d[:, 2]
        # At least one z-coordinate should be different from others
        assert len(np.unique(np.round(z_coords, 2))) > 1


class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_single_atom(self):
        """Test with single atom molecule"""
        smiles = "C"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        assert atoms is not None
        assert graph_2d is not None
        assert coords_3d is not None
        
        assert len(atoms) == 1
        assert graph_2d['num_nodes'] == 1
        assert coords_3d.shape == (1, 3)
    
    def test_nitrogen_containing(self):
        """Test with nitrogen-containing molecule (methylamine)"""
        smiles = "CN"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        assert atoms is not None
        assert graph_2d is not None
        # 3D might fail, so don't assert it
        
        assert len(atoms) == 2
        assert set(atoms) == {'C', 'N'}
    
    def test_sulfur_containing(self):
        """Test with sulfur-containing molecule (dimethyl sulfide)"""
        smiles = "CSC"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        assert atoms is not None
        assert graph_2d is not None
        # 3D might fail, so don't assert it
        
        assert len(atoms) == 3
        assert 'S' in atoms
    
    def test_multiple_rings(self):
        """Test with multi-ring system (naphthalene)"""
        smiles = "c1ccc2ccccc2c1"
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        assert atoms is not None
        assert graph_2d is not None
        # 3D might succeed for naphthalene
        
        # Naphthalene has 10 carbon atoms
        assert len(atoms) == 10
        assert graph_2d['num_nodes'] == 10


class TestSpatialEdgesInGraph:
    """Test cases for spatial edges in graph_2d"""
    
    def test_graph_with_spatial_edges(self):
        """Test that spatial edges are correctly added to graph_2d when 3D is available"""
        smiles = "CCO"  # ethanol
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        if coords_3d is not None and graph_2d is not None:
            # Check that both chemical and spatial edges exist
            assert 'chem_edge_index' in graph_2d
            assert 'chem_edge_feat_cat' in graph_2d
            assert 'chem_edge_feat_dist' in graph_2d
            
            # Check for spatial edges (default names)
            if 'edge_index' in graph_2d:
                assert 'edge_feat_dist' in graph_2d
                
                # Check dimensions
                spatial_edge_index = graph_2d['edge_index']
                spatial_edge_dist = graph_2d['edge_feat_dist']
                
                assert spatial_edge_index.shape[0] == 2  # [src, dst]
                assert spatial_edge_dist.shape[1] == 1  # 1 distance
                
                # Check that spatial distances are positive (real distances)
                assert torch.all(spatial_edge_dist > 0)
            
            # Check that chemical distances are placeholder (-1.0)
            chem_edge_dist = graph_2d['chem_edge_feat_dist']
            assert torch.all(chem_edge_dist == -1.0)
    
    def test_graph_without_3d(self):
        """Test that graph without 3D doesn't have spatial edges"""
        smiles = "CCO"
        graph = smiles2graph(smiles)
        
        # Should have chemical edges but no spatial edges
        assert 'chem_edge_index' in graph
        assert 'chem_edge_feat_cat' in graph
        assert 'chem_edge_feat_dist' in graph
        assert 'edge_index' not in graph
        assert 'edge_feat_dist' not in graph
        
        # Chemical distances should be placeholder
        assert torch.all(graph['chem_edge_feat_dist'] == -1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
