#!/usr/bin/env python
"""
Simple test script for mol_structure.py functions.
Can be run directly without pytest.
"""

import sys
import os
import torch

# Import from src package (requires running from project root or PYTHONPATH set correctly)
from src.data_factory.molecule.mol_structure import smiles2graph, generate_conformer_with_rdkit, generate_2d_3d_from_smiles

print("✓ Successfully imported all functions")


def test_smiles2graph():
    """Test 2D graph generation"""
    print("\n" + "="*60)
    print("Testing smiles2graph (2D Graph Generation)")
    print("="*60)
    
    test_molecules = [
        ("CCO", "Ethanol", 3, 4),
        ("c1ccccc1", "Benzene", 6, 12),
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin", 13, None),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine", 14, None),
    ]
    
    passed = 0
    failed = 0
    
    for smiles, name, expected_atoms, expected_edges in test_molecules:
        try:
            graph = smiles2graph(smiles)
            
            # Validate structure
            assert isinstance(graph, dict), "Graph should be a dictionary"
            assert 'node_feat' in graph, "Missing node_feat"
            assert 'chem_edge_index' in graph, "Missing chem_edge_index"
            assert 'chem_edge_feat_cat' in graph, "Missing chem_edge_feat_cat"
            assert 'chem_edge_feat_dist' in graph, "Missing chem_edge_feat_dist"
            assert 'num_nodes' in graph, "Missing num_nodes"
            
            # Check atom count
            assert graph['num_nodes'] == expected_atoms, \
                f"Expected {expected_atoms} atoms, got {graph['num_nodes']}"
            
            # Check edge count if specified
            if expected_edges is not None:
                assert graph['chem_edge_index'].shape[1] == expected_edges, \
                    f"Expected {expected_edges} edges, got {graph['chem_edge_index'].shape[1]}"
            
            print(f"✓ {name} ({smiles}): {graph['num_nodes']} atoms, {graph['chem_edge_index'].shape[1]} edges")
            passed += 1
        except Exception as e:
            print(f"✗ {name} ({smiles}): {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} tests failed"


def test_generate_conformer():
    """Test 3D conformer generation"""
    print("\n" + "="*60)
    print("Testing generate_conformer_with_rdkit (3D Generation)")
    print("="*60)
    
    test_molecules = [
        ("CCO", "Ethanol", 3, ['C', 'C', 'O']),
        ("c1ccccc1", "Benzene", 6, ['C']*6),
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin", 13, None),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine", 14, None),
    ]
    
    passed = 0
    failed = 0
    
    for smiles, name, expected_atoms, expected_symbols in test_molecules:
        try:
            atoms, coords = generate_conformer_with_rdkit(smiles)
            
            assert atoms is not None, "Atoms should not be None"
            assert coords is not None, "Coordinates should not be None"
            assert len(atoms) == expected_atoms, \
                f"Expected {expected_atoms} atoms, got {len(atoms)}"
            assert coords.shape == (expected_atoms, 3), \
                f"Expected shape ({expected_atoms}, 3), got {coords.shape}"
            
            if expected_symbols is not None:
                assert atoms == expected_symbols, \
                    f"Expected {expected_symbols}, got {atoms}"
            
            print(f"✓ {name} ({smiles}): {len(atoms)} atoms, coords shape {coords.shape}")
            passed += 1
        except Exception as e:
            print(f"✗ {name} ({smiles}): {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} tests failed"


def test_unified_generation():
    """Test unified 2D+3D generation"""
    print("\n" + "="*60)
    print("Testing generate_2d_3d_from_smiles (Unified Generation)")
    print("="*60)
    
    test_molecules = [
        ("CCO", "Ethanol", 3),
        ("c1ccccc1", "Benzene", 6),
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin", 13),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine", 14),
        ("C(C1C(C(C(C(O1)O)O)O)O)O", "Glucose", 12),
    ]
    
    passed = 0
    failed = 0
    
    for smiles, name, expected_atoms in test_molecules:
        try:
            atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
            
            # Check all components exist
            assert atoms is not None, "Atoms should not be None"
            assert graph_2d is not None, "2D graph should not be None"
            
            # Check consistency
            assert len(atoms) == expected_atoms, \
                f"Expected {expected_atoms} atom symbols, got {len(atoms)}"
            assert graph_2d['num_nodes'] == expected_atoms, \
                f"Expected {expected_atoms} atoms in 2D graph, got {graph_2d['num_nodes']}"
            
            # Check 3D if available
            if coords_3d is not None:
                assert coords_3d.shape == (expected_atoms, 3), \
                    f"Expected 3D coords shape ({expected_atoms}, 3), got {coords_3d.shape}"
                
                # Verify consistency across all representations
                assert len(atoms) == graph_2d['num_nodes'] == coords_3d.shape[0], \
                    "Atom count mismatch between atoms, 2D, and 3D"
                
                print(f"✓ {name} ({smiles}): Atoms ✓, 2D ✓, 3D ✓ - {expected_atoms} atoms consistent")
            else:
                print(f"⚠ {name} ({smiles}): Atoms ✓, 2D ✓, 3D ✗ - {expected_atoms} atoms in 2D")
            
            passed += 1
        except Exception as e:
            print(f"✗ {name} ({smiles}): {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    assert failed == 0, f"{failed} tests failed"


def test_fallback_mechanism():
    """Test progressive fallback for edge cases"""
    print("\n" + "="*60)
    print("Testing Fallback Mechanism")
    print("="*60)
    
    # Test invalid SMILES
    print("\nTest 1: Invalid SMILES (should return None, None, None)")
    try:
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles("INVALID_SMILES_123")
        assert atoms is None and graph_2d is None and coords_3d is None, \
            "Invalid SMILES should return all None"
        print("✓ Invalid SMILES handled correctly")
    except Exception as e:
        raise AssertionError(f"Invalid SMILES test failed: {e}")
    
    # Test single atom
    print("\nTest 2: Single atom (C)")
    try:
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles("C")
        assert atoms is not None and len(atoms) == 1, "Should have 1 atom"
        assert graph_2d is not None, "2D should exist for single atom"
        if coords_3d is not None:
            assert coords_3d.shape == (1, 3), "3D should have correct shape"
            print(f"✓ Single atom: atoms={atoms}, 2D nodes={graph_2d['num_nodes']}, 3D shape={coords_3d.shape}")
        else:
            print(f"⚠ Single atom: atoms={atoms}, 2D nodes={graph_2d['num_nodes']}, 3D failed")
    except Exception as e:
        raise AssertionError(f"Single atom test failed: {e}")


def test_atom_ordering():
    """Test that atom ordering is consistent between 2D and 3D"""
    print("\n" + "="*60)
    print("Testing Atom Ordering Consistency")
    print("="*60)
    
    smiles = "CCO"  # Ethanol
    try:
        atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
        
        # Check consistency
        num_atoms = len(atoms)
        num_2d = graph_2d['num_nodes']
        num_3d = coords_3d.shape[0] if coords_3d is not None else 0
        
        assert num_atoms == num_2d, f"Atoms list ({num_atoms}) != 2D nodes ({num_2d})"
        
        if coords_3d is not None:
            assert num_atoms == num_3d, f"Atoms list ({num_atoms}) != 3D coords ({num_3d})"
            assert num_2d == num_3d, f"2D nodes ({num_2d}) != 3D coords ({num_3d})"
        
        print(f"✓ Atom ordering is consistent: atoms={num_atoms}, 2D={num_2d}, 3D={num_3d}")
        print(f"  Atom sequence: {atoms}")
    except Exception as e:
        raise AssertionError(f"Atom ordering test failed: {e}")


def test_edge_features_5d():
    """Test that edge features are split into categorical (4D int) and distance (1D float)"""
    print("\n" + "="*60)
    print("Testing Split Edge Features (Categorical + Distance)")
    print("="*60)
    
    test_molecules = [
        ("CCO", "Ethanol"),
        ("c1ccccc1", "Benzene"),
        ("CC(=O)O", "Acetic acid"),
    ]
    
    for smiles, name in test_molecules:
        try:
            # Test smiles2graph (2D only)
            print(f"\nTest {name} ({smiles}) - 2D only:")
            graph_2d_only = smiles2graph(smiles)
            assert graph_2d_only is not None, f"Graph should not be None for {name}"
            
            edge_feat_cat = graph_2d_only['chem_edge_feat_cat']
            edge_feat_dist = graph_2d_only['chem_edge_feat_dist']
            print(f"  Edge feat cat shape: {edge_feat_cat.shape} (dtype: {edge_feat_cat.dtype})")
            print(f"  Edge feat dist shape: {edge_feat_dist.shape} (dtype: {edge_feat_dist.dtype})")
            
            assert edge_feat_cat.shape[1] == 3, f"Categorical features should have 3 dimensions, got {edge_feat_cat.shape[1]}"
            assert edge_feat_dist.shape[1] == 1, f"Distance features should have 1 dimension, got {edge_feat_dist.shape[1]}"
            assert edge_feat_cat.dtype == torch.int64, f"Categorical features should be int64, got {edge_feat_cat.dtype}"
            assert edge_feat_dist.dtype == torch.float32, f"Distance features should be float32, got {edge_feat_dist.dtype}"
            
            # For 2D-only graphs with the split feature format:
            # chem_edge_feat_cat has 3 dimensions: [bond_type, bond_stereo, is_conjugated]
            # No edge_type column since these are all chemical bonds
            
            # Check that distances are -1 (placeholder)
            distances = edge_feat_dist[:, 0]
            assert (distances == -1).all(), "All distances in 2D-only graph should be -1 (placeholder)"
            
            print(f"  ✓ 2D-only: {edge_feat_cat.shape[0]} edges")
            print(f"    All chemical bonds")
            print(f"    Distances: all -1 (no 3D coords)")
            
            # Test generate_2d_3d_from_smiles (with 3D if possible)
            print(f"\nTest {name} ({smiles}) - 2D+3D:")
            atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(smiles)
            
            assert graph_2d is not None, f"2D graph should not be None for {name}"
            
            # Check which keys are present - could be either combined or split format
            if 'edge_feat_cat' in graph_2d and 'edge_feat_dist' in graph_2d:
                # Combined format with edge_type in categorical features
                edge_feat_cat_3d = graph_2d['edge_feat_cat']
                edge_feat_dist_3d = graph_2d['edge_feat_dist']
                edge_index_3d = graph_2d.get('edge_index')
                
                print(f"  Edge feat cat shape: {edge_feat_cat_3d.shape} (dtype: {edge_feat_cat_3d.dtype})")
                print(f"  Edge feat dist shape: {edge_feat_dist_3d.shape} (dtype: {edge_feat_dist_3d.dtype})")
                
                assert edge_feat_cat_3d.dtype == torch.int64, "Categorical features should be int64"
                assert edge_feat_dist_3d.dtype == torch.float32, "Distance features should be float32"
                
                if coords_3d is not None:
                    print(f"  3D coords available: {coords_3d.shape}")
                    
                    # Count edge types
                    edge_types_3d = edge_feat_cat_3d[:, 3]
                    num_chemical = (edge_types_3d == 0).sum()
                    num_spatial = (edge_types_3d == 1).sum()
                    
                    print(f"  ✓ Total edges: {edge_feat_cat_3d.shape[0]}")
                    print(f"    Chemical bonds (edge_type=0): {num_chemical}")
                    print(f"    Spatial edges (edge_type=1): {num_spatial}")
                    
                    # Check that chemical bonds may have distances filled in
                    chemical_mask = edge_types_3d == 0
                    if chemical_mask.any():
                        chemical_distances = edge_feat_dist_3d[chemical_mask, 0]
                        num_with_distance = (chemical_distances >= 0).sum()
                        print(f"    Chemical bonds with distance: {num_with_distance}/{num_chemical}")
                    
                    # Check that spatial edges have positive distances
                    spatial_mask = edge_types_3d == 1
                    if spatial_mask.any():
                        spatial_distances = edge_feat_dist_3d[spatial_mask, 0]
                        assert (spatial_distances > 0).all(), "All spatial edges should have positive distances"
                        print(f"    Spatial edge distance range: [{spatial_distances.min():.2f}, {spatial_distances.max():.2f}] Å")
                        
                        # Check that spatial edges have -1 for chemical features
                        spatial_bond_type = edge_feat_cat_3d[spatial_mask, 0]
                        spatial_bond_stereo = edge_feat_cat_3d[spatial_mask, 1]
                        spatial_conjugated = edge_feat_cat_3d[spatial_mask, 2]
                        assert (spatial_bond_type == -1).all(), "Spatial edges should have bond_type=-1"
                        assert (spatial_bond_stereo == -1).all(), "Spatial edges should have bond_stereo=-1"
                        assert (spatial_conjugated == -1).all(), "Spatial edges should have is_conjugated=-1"
                else:
                    print(f"  3D coords not available")
                    # Without 3D, all edges should be chemical bonds with distance=-1
                    edge_types_3d = edge_feat_cat_3d[:, 3]
                    assert (edge_types_3d == 0).all(), "Without 3D coords, all edges should be chemical bonds"
                    distances_3d = edge_feat_dist_3d[:, 0]
                    assert (distances_3d == -1).all(), "Without 3D coords, all distances should be -1"
                    print(f"  ✓ All {edge_feat_cat_3d.shape[0]} edges are chemical bonds with distance=-1")
            else:
                # Split format with separate chem_ and spatial_ keys
                print(f"  Using split format (chem_ and spatial_ keys)")
                # For now, just verify the keys exist
                assert 'chem_edge_index' in graph_2d, "Missing chem_edge_index"
                assert 'chem_edge_feat_cat' in graph_2d, "Missing chem_edge_feat_cat"
                assert 'chem_edge_feat_dist' in graph_2d, "Missing chem_edge_feat_dist"
                print(f"  ✓ Split format validated")
            
        except Exception as e:
            raise AssertionError(f"Split edge feature test failed for {name}: {e}")
    
    print("\n✓ All split edge feature tests passed!")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" MOLECULE STRUCTURE TESTS")
    print("="*70)
    
    try:
        test_smiles2graph()
        test_generate_conformer()
        test_unified_generation()
        test_fallback_mechanism()
        test_atom_ordering()
        test_edge_features_5d()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70 + "\n")
        return 0
    except AssertionError as e:
        print("\n" + "="*70)
        print(f"✗ TESTS FAILED: {e}")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
