import os
import pytest
import numpy as np
from pathlib import Path

from src.data_factory.protein.cif_to_cooridinates import (
    parse_cif_atoms,
    get_ca_atoms,
    get_backbone_atoms,
    coordinates_to_array,
    get_structure_info,
    atom_info_to_feature_vector,
    atom_info_list_to_features,
    get_atom_feature_dims,
    get_atomic_number,
    PROTEIN_ATOM_FEATURES
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


class TestProteinAtomFeatures:
    """Tests for PROTEIN_ATOM_FEATURES dictionary."""
    
    def test_features_dict_structure(self):
        """Test that PROTEIN_ATOM_FEATURES has all required keys."""
        assert 'atom_names' in PROTEIN_ATOM_FEATURES
        assert 'residue_names' in PROTEIN_ATOM_FEATURES
        assert 'elements' in PROTEIN_ATOM_FEATURES
        assert 'chains' in PROTEIN_ATOM_FEATURES
    
    def test_features_are_lists(self):
        """Test that all feature values are lists."""
        for key, value in PROTEIN_ATOM_FEATURES.items():
            assert isinstance(value, list), f"{key} should be a list"
            assert len(value) > 0, f"{key} should not be empty"
    
    def test_features_contain_misc(self):
        """Test that all features have 'misc' as last element."""
        assert PROTEIN_ATOM_FEATURES['atom_names'][-1] == 'misc'
        assert PROTEIN_ATOM_FEATURES['residue_names'][-1] == 'misc'
        assert PROTEIN_ATOM_FEATURES['elements'][-1] == 'misc'
        assert PROTEIN_ATOM_FEATURES['chains'][-1] == 'misc'
    
    def test_backbone_atoms_present(self):
        """Test that backbone atoms are in atom_names."""
        backbone_atoms = {'N', 'CA', 'C', 'O'}
        atom_names = PROTEIN_ATOM_FEATURES['atom_names']
        for atom in backbone_atoms:
            assert atom in atom_names
    
    def test_standard_amino_acids_present(self):
        """Test that all 20 standard amino acids are present."""
        standard_aa = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 
                      'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 
                      'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        residue_names = PROTEIN_ATOM_FEATURES['residue_names']
        for aa in standard_aa:
            assert aa in residue_names
    
    def test_common_elements_present(self):
        """Test that common protein elements are present."""
        common_elements = ['C', 'N', 'O', 'S']
        elements = PROTEIN_ATOM_FEATURES['elements']
        for elem in common_elements:
            assert elem in elements


class TestGetAtomicNumber:
    """Tests for get_atomic_number function."""
    
    def test_common_protein_elements(self):
        """Test atomic numbers for common protein elements."""
        assert get_atomic_number('H') == 1
        assert get_atomic_number('C') == 6
        assert get_atomic_number('N') == 7
        assert get_atomic_number('O') == 8
        assert get_atomic_number('S') == 16
        assert get_atomic_number('P') == 15
    
    def test_less_common_elements(self):
        """Test atomic numbers for less common elements."""
        assert get_atomic_number('Se') == 34  # Selenium
        assert get_atomic_number('Br') == 35  # Bromine
        assert get_atomic_number('I') == 53   # Iodine
        assert get_atomic_number('F') == 9    # Fluorine
        assert get_atomic_number('Cl') == 17  # Chlorine
    
    def test_unknown_element(self):
        """Test that unknown elements return 0."""
        assert get_atomic_number('XX') == 0
        assert get_atomic_number('misc') == 0
        assert get_atomic_number('UNKNOWN') == 0
        assert get_atomic_number('') == 0
    
    def test_case_sensitivity(self):
        """Test that element symbols are case-sensitive."""
        assert get_atomic_number('C') == 6
        assert get_atomic_number('Ca') == 20  # Calcium
        assert get_atomic_number('Cl') == 17  # Chlorine
        # Lowercase 'c' won't match 'C', should return 0
        assert get_atomic_number('c') == 0


class TestAtomInfoToFeatureVector:
    """Tests for atom_info_to_feature_vector function."""
    
    def test_basic_conversion_without_properties(self):
        """Test basic conversion without properties."""
        atom_info = {
            'atom_name': 'CA',
            'residue_name': 'ALA',
            'residue_id': '1',
            'chain': 'A',
            'element': 'C'
        }
        
        feature_vec = atom_info_to_feature_vector(atom_info)
        
        # Should return 7 features
        assert len(feature_vec) == 7
        assert all(isinstance(f, int) for f in feature_vec)
        
        # Check atomic number for Carbon
        assert feature_vec[0] == 6  # C has atomic number 6
    
    def test_backbone_flag_for_backbone_atoms(self):
        """Test is_backbone flag is set correctly for backbone atoms."""
        backbone_atoms = ['N', 'CA', 'C', 'O']
        
        for atom_name in backbone_atoms:
            atom_info = {
                'atom_name': atom_name,
                'residue_name': 'ALA',
                'residue_id': '1',
                'chain': 'A',
                'element': 'C' if atom_name != 'N' and atom_name != 'O' else atom_name
            }
            feature_vec = atom_info_to_feature_vector(atom_info)
            
            # Index 5 is is_backbone
            assert feature_vec[5] == 1, f"{atom_name} should be marked as backbone"
    
    def test_backbone_flag_for_sidechain_atoms(self):
        """Test is_backbone flag is set correctly for sidechain atoms."""
        sidechain_atoms = ['CB', 'CG', 'CD', 'CE', 'NZ', 'OG', 'SG']
        
        for atom_name in sidechain_atoms:
            atom_info = {
                'atom_name': atom_name,
                'residue_name': 'LYS',
                'residue_id': '1',
                'chain': 'A',
                'element': 'C'
            }
            feature_vec = atom_info_to_feature_vector(atom_info)
            
            # Index 5 is is_backbone
            assert feature_vec[5] == 0, f"{atom_name} should not be marked as backbone"
    
    def test_ca_flag_only_for_ca(self):
        """Test is_ca flag is only set for CA atoms."""
        # Test CA atom
        atom_info_ca = {
            'atom_name': 'CA',
            'residue_name': 'ALA',
            'residue_id': '1',
            'chain': 'A',
            'element': 'C'
        }
        feature_vec_ca = atom_info_to_feature_vector(atom_info_ca)
        # Index 6 is is_ca
        assert feature_vec_ca[6] == 1
        
        # Test non-CA atoms
        for atom_name in ['N', 'C', 'O', 'CB']:
            atom_info = {
                'atom_name': atom_name,
                'residue_name': 'ALA',
                'residue_id': '1',
                'chain': 'A',
                'element': 'C'
            }
            feature_vec = atom_info_to_feature_vector(atom_info)
            assert feature_vec[6] == 0, f"{atom_name} should not be marked as CA"
    
    def test_residue_id_parsing(self):
        """Test residue_id is correctly parsed."""
        atom_info = {
            'atom_name': 'CA',
            'residue_name': 'ALA',
            'residue_id': '42',
            'chain': 'A',
            'element': 'C'
        }
        feature_vec = atom_info_to_feature_vector(atom_info)
        
        # Index 4 is residue_id
        assert feature_vec[4] == 42
    
    def test_unknown_values_fallback_to_misc(self):
        """Test that unknown values map to 'misc' index or 0 for elements."""
        atom_info = {
            'atom_name': 'UNKNOWN_ATOM',
            'residue_name': 'UNK',
            'residue_id': '1',
            'chain': 'AA',  # Invalid chain (not A-Z), should map to misc
            'element': 'XX'
        }
        feature_vec = atom_info_to_feature_vector(atom_info)
        
        # Unknown element should return atomic number 0
        assert feature_vec[0] == 0  # Unknown element
        # Other unknown values should map to last index (misc)
        assert feature_vec[1] == len(PROTEIN_ATOM_FEATURES['atom_names']) - 1  # misc atom name
        assert feature_vec[2] == len(PROTEIN_ATOM_FEATURES['residue_names']) - 1  # misc residue name
        assert feature_vec[3] == len(PROTEIN_ATOM_FEATURES['chains']) - 1  # misc chain


class TestAtomInfoListToFeatures:
    """Tests for atom_info_list_to_features function."""
    
    def test_empty_list(self):
        """Test handling of empty list."""
        features = atom_info_list_to_features([])
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 0
    
    def test_single_atom_conversion(self):
        """Test conversion of single atom."""
        atom_info_list = [{
            'atom_name': 'CA',
            'residue_name': 'ALA',
            'residue_id': '1',
            'chain': 'A',
            'element': 'C'
        }]
        
        features = atom_info_list_to_features(atom_info_list)
        
        assert features.shape == (1, 7)
        assert features.dtype == np.int64
    
    def test_multiple_atoms_conversion(self):
        """Test conversion of multiple atoms."""
        atom_info_list = [
            {'atom_name': 'N', 'residue_name': 'MET', 'residue_id': '1', 'chain': 'A', 'element': 'N'},
            {'atom_name': 'CA', 'residue_name': 'MET', 'residue_id': '1', 'chain': 'A', 'element': 'C'},
            {'atom_name': 'C', 'residue_name': 'MET', 'residue_id': '1', 'chain': 'A', 'element': 'C'},
            {'atom_name': 'O', 'residue_name': 'MET', 'residue_id': '1', 'chain': 'A', 'element': 'O'},
        ]
        
        features = atom_info_list_to_features(atom_info_list)
        
        assert features.shape == (4, 7)
        assert features.dtype == np.int64
        # All backbone atoms should have is_backbone = 1
        assert np.all(features[:, 5] == 1)
        # Only second atom (CA) should have is_ca = 1
        assert features[1, 6] == 1
        assert np.sum(features[:, 6]) == 1
    
    def test_conversion_default(self):
        """Test conversion (default)."""
        atom_info_list = [
            {'atom_name': 'CA', 'residue_name': 'ALA', 'residue_id': '1', 'chain': 'A', 'element': 'C'},
        ]
        
        features = atom_info_list_to_features(atom_info_list)
        
        assert features.shape == (1, 7)
        assert features.dtype == np.int64
    
    def test_real_file_conversion(self, test_cif_file):
        """Test conversion with real CIF file data."""
        atom_info, coords, props, _ = parse_cif_atoms(test_cif_file)
        
        features = atom_info_list_to_features(atom_info)
        assert features.shape == (len(atom_info), 7)
        assert features.dtype == np.int64


class TestGetAtomFeatureDims:
    """Tests for get_atom_feature_dims function."""
    
    def test_dims(self):
        """Test feature dimensions."""
        dims = get_atom_feature_dims()
        
        assert isinstance(dims, list)
        assert len(dims) == 7
        
        # Index 0: atomic number is categorical (119: 0-118)
        assert dims[0] == 119
        
        # Indices 1-3 should be categorical (positive integers)
        assert dims[1] == len(PROTEIN_ATOM_FEATURES['atom_names'])
        assert dims[2] == len(PROTEIN_ATOM_FEATURES['residue_names'])
        assert dims[3] == len(PROTEIN_ATOM_FEATURES['chains'])
        
        # residue_id is continuous (-1)
        assert dims[4] == -1
        
        # is_backbone and is_ca are binary (2)
        assert dims[5] == 2
        assert dims[6] == 2
    
    def test_dims_match_feature_lengths(self):
        """Test that categorical dimensions match actual feature list lengths."""
        dims = get_atom_feature_dims()
        
        assert dims[0] == 119  # atomic number range
        assert dims[1] == len(PROTEIN_ATOM_FEATURES['atom_names'])
        assert dims[2] == len(PROTEIN_ATOM_FEATURES['residue_names'])
        assert dims[3] == len(PROTEIN_ATOM_FEATURES['chains'])


class TestFeatureIntegration:
    """Integration tests for feature conversion with real CIF data."""
    
    def test_ca_only_features(self, test_cif_file):
        """Test feature extraction for CA atoms only."""
        ca_info, ca_coords, ca_props, _ = get_ca_atoms(test_cif_file)
        features = atom_info_list_to_features(ca_info)
        
        # All should be CA atoms
        assert np.all(features[:, 6] == 1)  # is_ca flag
        # All should be backbone atoms
        assert np.all(features[:, 5] == 1)  # is_backbone flag
        # All CA atoms should have Carbon (atomic number 6)
        assert np.all(features[:, 0] == 6)  # atomic number for C
    
    def test_backbone_features(self, test_cif_file):
        """Test feature extraction for backbone atoms."""
        bb_info, bb_coords, bb_props, _ = get_backbone_atoms(test_cif_file)
        features = atom_info_list_to_features(bb_info)
        
        # All should be backbone atoms
        assert np.all(features[:, 5] == 1)  # is_backbone flag
        
        # Count CA atoms
        num_ca = np.sum(features[:, 6])
        assert num_ca > 0
        assert num_ca < len(features)  # CA is subset of backbone
        
        # Check that we have various atomic numbers (C, N, O)
        atomic_numbers = np.unique(features[:, 0])
        # Backbone should have at least C(6), N(7), O(8)
        assert 6 in atomic_numbers  # Carbon
        assert 7 in atomic_numbers  # Nitrogen
        assert 8 in atomic_numbers  # Oxygen
    
    def test_full_protein_features(self, small_cif_file):
        """Test feature extraction for full protein."""
        atom_info, coords, props, _ = parse_cif_atoms(small_cif_file)
        features = atom_info_list_to_features(atom_info)
        
        # Check shape
        assert features.shape[0] == len(atom_info)
        assert features.shape[1] == 7
        assert features.dtype == np.int64
        
        # Check that some atoms are backbone and some are not
        num_backbone = np.sum(features[:, 5])
        assert 0 < num_backbone < len(features)
        
        # Check that CA atoms exist
        num_ca = np.sum(features[:, 6])
        assert num_ca > 0
        
        # Check that atomic numbers are in valid range (0-118)
        atomic_numbers = features[:, 0]
        assert np.all(atomic_numbers >= 0)
        assert np.all(atomic_numbers <= 118)
        # Should have common protein elements
        unique_atomic_nums = np.unique(atomic_numbers)
        assert 6 in unique_atomic_nums  # Carbon should be present
    
    def test_feature_and_coordinate_alignment(self, test_cif_file):
        """Test that features and coordinates are properly aligned."""
        atom_info, coords, props, _ = parse_cif_atoms(test_cif_file)
        features = atom_info_list_to_features(atom_info)
        coords_array = coordinates_to_array(coords)
        
        # Should have same number of atoms
        assert features.shape[0] == coords_array.shape[0]
        assert features.shape[0] == len(atom_info)
    
    def test_residue_id_ordering(self, small_cif_file):
        """Test that residue IDs are generally increasing."""
        atom_info, _, _, _ = parse_cif_atoms(small_cif_file)
        features = atom_info_list_to_features(atom_info)
        
        residue_ids = features[:, 4]
        
        # Residue IDs should start from a positive number
        assert np.min(residue_ids) >= 0
        
        # Should have multiple residues
        unique_residues = len(np.unique(residue_ids))
        assert unique_residues > 1
    
    def test_element_to_atomic_number_consistency(self, small_cif_file):
        """Test that element strings are correctly converted to atomic numbers."""
        atom_info, _, _, _ = parse_cif_atoms(small_cif_file)
        features = atom_info_list_to_features(atom_info)
        
        # Check specific atoms
        for i, atom in enumerate(atom_info):
            element = atom['element']
            expected_atomic_num = get_atomic_number(element)
            actual_atomic_num = features[i, 0]
            assert actual_atomic_num == expected_atomic_num, \
                f"Atom {i} with element {element}: expected {expected_atomic_num}, got {actual_atomic_num}"
