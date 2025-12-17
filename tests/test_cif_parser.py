import os
import pytest
import numpy as np
from pathlib import Path

from src.data_factory.protein.cif_to_cooridinates import (
    parse_cif_atoms,
    get_ca_atoms,
    get_backbone_atoms,
    coordinates_to_array,
    get_structure_info
)


@pytest.fixture
def test_cif_file():
    """Fixture to provide path to a test CIF file."""
    cif_path = Path(__file__).parent.parent / "data" / "uniprot" / "test" / "pdb_structures" / "2MIO.cif"
    if not cif_path.exists():
        pytest.skip(f"Test CIF file not found: {cif_path}")
    return str(cif_path)


@pytest.fixture
def small_cif_file():
    """Fixture to provide path to a smaller test CIF file."""
    cif_path = Path(__file__).parent.parent / "data" / "uniprot" / "test" / "pdb_structures" / "6H86.cif"
    if not cif_path.exists():
        pytest.skip(f"Test CIF file not found: {cif_path}")
    return str(cif_path)


class TestParseCifAtoms:
    """Tests for parse_cif_atoms function."""
    
    def test_parse_basic_structure(self, test_cif_file):
        """Test basic parsing returns correct structure."""
        atom_info, coords, props, failed = parse_cif_atoms(test_cif_file)
        
        assert isinstance(atom_info, list)
        assert isinstance(coords, list)
        assert isinstance(props, list)
        assert isinstance(failed, list)
        
        # All lists should have the same length
        assert len(atom_info) == len(coords)
        assert len(atom_info) == len(props)
        
        # Should have parsed some atoms
        assert len(atom_info) > 0
    
    def test_atom_info_structure(self, test_cif_file):
        """Test atom info has correct keys and types."""
        atom_info, coords, props, failed = parse_cif_atoms(test_cif_file)
        
        first_atom = atom_info[0]
        
        # Check all required keys are present
        assert 'atom_id' in first_atom
        assert 'atom_name' in first_atom
        assert 'residue_name' in first_atom
        assert 'residue_id' in first_atom
        assert 'chain' in first_atom
        assert 'element' in first_atom
        
        # Check types
        assert isinstance(first_atom['atom_id'], str)
        assert isinstance(first_atom['atom_name'], str)
        assert isinstance(first_atom['residue_name'], str)
        assert isinstance(first_atom['residue_id'], str)
        assert isinstance(first_atom['chain'], str)
        assert isinstance(first_atom['element'], str)
    
    def test_coordinates_structure(self, test_cif_file):
        """Test coordinates have correct structure."""
        atom_info, coords, props, failed = parse_cif_atoms(test_cif_file)
        
        first_coord = coords[0]
        
        # Should be a list of 3 floats [x, y, z]
        assert isinstance(first_coord, list)
        assert len(first_coord) == 3
        assert all(isinstance(c, float) for c in first_coord)
    
    def test_properties_structure(self, test_cif_file):
        """Test properties have correct structure."""
        atom_info, coords, props, failed = parse_cif_atoms(test_cif_file)
        
        first_prop = props[0]
        
        # Check required keys
        assert 'b_factor' in first_prop
        assert 'occupancy' in first_prop
        
        # Check types and valid ranges
        assert isinstance(first_prop['b_factor'], float)
        assert isinstance(first_prop['occupancy'], float)
        assert 0.0 <= first_prop['occupancy'] <= 1.0
    
    def test_failed_list_format(self, test_cif_file):
        """Test failed list has correct structure."""
        atom_info, coords, props, failed = parse_cif_atoms(test_cif_file)
        
        # Failed list should be a list (may be empty)
        assert isinstance(failed, list)
        
        # If there are failed entries, check structure
        if failed:
            assert 'error' in failed[0]
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        atom_info, coords, props, failed = parse_cif_atoms("nonexistent_file.cif")
        
        # Should return empty lists and a failed entry
        assert len(atom_info) == 0
        assert len(coords) == 0
        assert len(props) == 0
        assert len(failed) > 0
        assert 'error' in failed[0]


class TestGetCaAtoms:
    """Tests for get_ca_atoms function."""
    
    def test_ca_atoms_only(self, test_cif_file):
        """Test that only CA atoms are returned."""
        ca_info, ca_coords, ca_props, failed = get_ca_atoms(test_cif_file)
        
        # All atoms should be CA
        for atom in ca_info:
            assert atom['atom_name'] == 'CA'
    
    def test_ca_atoms_count(self, test_cif_file):
        """Test CA atoms are a subset of all atoms."""
        all_info, all_coords, all_props, _ = parse_cif_atoms(test_cif_file)
        ca_info, ca_coords, ca_props, _ = get_ca_atoms(test_cif_file)
        
        # CA atoms should be less than total atoms
        assert len(ca_info) < len(all_info)
        assert len(ca_info) > 0
    
    def test_ca_structure_consistency(self, test_cif_file):
        """Test CA atoms maintain structure consistency."""
        ca_info, ca_coords, ca_props, failed = get_ca_atoms(test_cif_file)
        
        assert len(ca_info) == len(ca_coords)
        assert len(ca_info) == len(ca_props)


class TestGetBackboneAtoms:
    """Tests for get_backbone_atoms function."""
    
    def test_backbone_atoms_only(self, test_cif_file):
        """Test that only backbone atoms (N, CA, C, O) are returned."""
        bb_info, bb_coords, bb_props, failed = get_backbone_atoms(test_cif_file)
        
        backbone_names = {'N', 'CA', 'C', 'O'}
        for atom in bb_info:
            assert atom['atom_name'] in backbone_names
    
    def test_backbone_atoms_count(self, test_cif_file):
        """Test backbone atoms are a subset of all atoms."""
        all_info, _, _, _ = parse_cif_atoms(test_cif_file)
        bb_info, _, _, _ = get_backbone_atoms(test_cif_file)
        
        # Backbone atoms should be less than total but more than CA only
        assert len(bb_info) < len(all_info)
        assert len(bb_info) > 0
    
    def test_backbone_contains_ca(self, test_cif_file):
        """Test backbone atoms include CA atoms."""
        ca_info, _, _, _ = get_ca_atoms(test_cif_file)
        bb_info, _, _, _ = get_backbone_atoms(test_cif_file)
        
        # CA count should be less than or equal to backbone count
        assert len(ca_info) <= len(bb_info)
    
    def test_backbone_structure_consistency(self, test_cif_file):
        """Test backbone atoms maintain structure consistency."""
        bb_info, bb_coords, bb_props, failed = get_backbone_atoms(test_cif_file)
        
        assert len(bb_info) == len(bb_coords)
        assert len(bb_info) == len(bb_props)


class TestCoordinatesToArray:
    """Tests for coordinates_to_array function."""
    
    def test_numpy_conversion(self, test_cif_file):
        """Test conversion to numpy array."""
        _, coords, _, _ = parse_cif_atoms(test_cif_file)
        
        coords_array = coordinates_to_array(coords)
        
        # Should return numpy array
        assert isinstance(coords_array, np.ndarray)
        
        # Shape should be (N, 3)
        assert coords_array.ndim == 2
        assert coords_array.shape[1] == 3
        assert coords_array.shape[0] == len(coords)
    
    def test_coordinate_values_preserved(self, small_cif_file):
        """Test that coordinate values are preserved after conversion."""
        _, coords, _, _ = parse_cif_atoms(small_cif_file)
        
        coords_array = coordinates_to_array(coords)
        
        # Check first and last coordinates match
        assert np.allclose(coords_array[0], coords[0])
        assert np.allclose(coords_array[-1], coords[-1])
    
    def test_empty_coordinates(self):
        """Test handling of empty coordinate list."""
        coords_array = coordinates_to_array([])
        
        assert isinstance(coords_array, np.ndarray)
        assert coords_array.shape == (0, 3) or len(coords_array) == 0


class TestGetStructureInfo:
    """Tests for get_structure_info function."""
    
    def test_structure_info_keys(self, test_cif_file):
        """Test structure info has required keys."""
        info = get_structure_info(test_cif_file)
        
        assert 'name' in info
        assert 'num_models' in info
        assert 'chains' in info
        assert 'resolution' in info
    
    def test_structure_info_types(self, test_cif_file):
        """Test structure info has correct types."""
        info = get_structure_info(test_cif_file)
        
        assert isinstance(info['name'], str)
        assert isinstance(info['num_models'], int)
        assert isinstance(info['chains'], list)
        assert info['num_models'] > 0
    
    def test_chain_info_structure(self, test_cif_file):
        """Test chain info has correct structure."""
        info = get_structure_info(test_cif_file)
        
        if info['chains']:
            chain = info['chains'][0]
            
            assert 'name' in chain
            assert 'num_residues' in chain
            assert 'num_atoms' in chain
            
            assert isinstance(chain['name'], str)
            assert isinstance(chain['num_residues'], int)
            assert isinstance(chain['num_atoms'], int)
            
            assert chain['num_residues'] > 0
            assert chain['num_atoms'] > 0
    
    def test_structure_name_matches_file(self, test_cif_file):
        """Test structure name matches expected PDB ID."""
        info = get_structure_info(test_cif_file)
        
        # For 2MIO.cif, the structure name should contain "2MIO"
        assert '2MIO' in info['name'] or info['name'] == '2MIO'


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_ca_coords_extractable(self, test_cif_file):
        """Test extracting CA coordinates as numpy array."""
        ca_info, ca_coords, ca_props, _ = get_ca_atoms(test_cif_file)
        ca_array = coordinates_to_array(ca_coords)
        
        # Should produce valid array
        assert ca_array.shape[0] == len(ca_info)
        assert ca_array.shape[1] == 3
    
    def test_coordinate_bounds_reasonable(self, test_cif_file):
        """Test that coordinate values are reasonable."""
        _, coords, _, _ = parse_cif_atoms(test_cif_file)
        coords_array = coordinates_to_array(coords)
        
        # Coordinates should be in reasonable range (typically -100 to 100 Angstroms)
        assert np.all(np.abs(coords_array) < 1000)
    
    def test_multiple_file_parsing(self, test_cif_file, small_cif_file):
        """Test parsing multiple files."""
        info1, coords1, _, _ = parse_cif_atoms(test_cif_file)
        info2, coords2, _, _ = parse_cif_atoms(small_cif_file)
        
        # Both should parse successfully
        assert len(info1) > 0
        assert len(info2) > 0
        
        # Likely different sizes
        assert len(info1) != len(info2)
    
    def test_residue_consistency(self, small_cif_file):
        """Test that atoms in same residue have consistent residue_id."""
        atom_info, _, _, _ = parse_cif_atoms(small_cif_file)
        
        # Group atoms by residue
        residues = {}
        for atom in atom_info:
            res_key = (atom['chain'], atom['residue_id'])
            if res_key not in residues:
                residues[res_key] = []
            residues[res_key].append(atom)
        
        # Each residue should have consistent residue_name
        for res_key, atoms in residues.items():
            residue_names = set(atom['residue_name'] for atom in atoms)
            assert len(residue_names) == 1, f"Inconsistent residue names in {res_key}"
    
    def test_element_matches_atom_name(self, small_cif_file):
        """Test that element symbol matches atom name."""
        atom_info, _, _, _ = parse_cif_atoms(small_cif_file)
        
        # CA atoms should have element C
        ca_atoms = [a for a in atom_info if a['atom_name'] == 'CA']
        for atom in ca_atoms:
            assert atom['element'] == 'C'
        
        # N atoms should have element N
        n_atoms = [a for a in atom_info if a['atom_name'] == 'N']
        for atom in n_atoms:
            assert atom['element'] == 'N'


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_all_functions_with_same_file(self, test_cif_file):
        """Test all functions work on the same file."""
        # Should all run without errors
        parse_cif_atoms(test_cif_file)
        get_ca_atoms(test_cif_file)
        get_backbone_atoms(test_cif_file)
        get_structure_info(test_cif_file)
    
    def test_failed_list_is_not_none(self, test_cif_file):
        """Test that failed list is always returned (not None)."""
        _, _, _, failed = parse_cif_atoms(test_cif_file)
        
        assert failed is not None
        assert isinstance(failed, list)
