import os
import gemmi
import numpy as np

# Common ions to exclude
COMMON_IONS = {
    'NA', 'CL', 'K', 'CA', 'MG', 'ZN', 'FE', 'MN', 'CU', 'NI', 'CD', 'CO',
    'BR', 'I', 'F', 'SO4', 'PO4', 'NO3', 'IOD', 'NA+', 'CL-', 'K+', 'CA2+',
    'MG2+', 'ZN2+', 'FE2+', 'FE3+', 'MN2+', 'CU2+', 'NI2+'
}

# Common solvent residues to exclude
WATER_RESIDUES = {'HOH', 'WAT', 'H2O', 'DOD', 'D2O'}

ORGANIC_SOLVENTS = {
    # Alcohols
    'EOH', 'ETH', 'ETO',  # Ethanol
    'MOH', 'MEO', 'MET',  # Methanol
    'GOL',                 # Glycerol
    'EDO', 'EGL',         # Ethylene glycol
    'PG4', 'PGE', 'PEG',  # Polyethylene glycol
    'BTN', 'BUT',         # Butanol
    'IPH', 'IPA',         # Isopropanol
    # Organic solvents
    'ACT', 'ACE', 'ACET', # Acetone
    'DMS', 'DMSO', 'MSO', # Dimethyl sulfoxide
    'ACN',                # Acetonitrile
    'DMF',                # Dimethylformamide
    'THF',                # Tetrahydrofuran
    'CHF', 'CLF',         # Chloroform
    'DCM',                # Dichloromethane
    'BEN', 'BNZ',         # Benzene
    'TOL',                # Toluene
    'HEX',                # Hexane
    'OCT',                # Octane
    'ACY',                # Acetyl
    'FOR', 'FMT',         # Formate
    'ACA', 'ACT',         # Acetate
    'TRS', 'TRIS',        # Tris buffer
    'HED', 'HEPES',       # HEPES buffer
    'MES',                # MES buffer
    'EPE',                # EPE
    'MPD',                # 2-Methyl-2,4-pentanediol
    'MLI',                # Malonate ion
}

def should_include_residue(residue, include_waters=False, include_ions=False, include_solvents=False):
    """
    Determine if a residue should be included in parsing.
    
    By default, only protein polymer atoms are included.
    
    Args:
        residue: gemmi.Residue object
        include_waters: If True, include water molecules (default: False)
        include_ions: If True, include ion residues (default: False)
        include_solvents: If True, include organic solvent molecules (default: False)
        
    Returns:
        bool: True if residue should be included
    """
    residue_name = residue.name.upper()
    
    # Filter out waters unless explicitly requested
    if not include_waters:
        if residue.is_water() or residue_name in WATER_RESIDUES:
            return False
    
    # Filter out organic solvents unless explicitly requested
    if not include_solvents:
        if residue_name in ORGANIC_SOLVENTS:
            return False
    
    # Filter out ions unless explicitly requested
    if not include_ions:
        if residue_name in COMMON_IONS:
            return False
        # Also check if it's a single-atom residue that looks like an ion
        if len(residue) == 1 and residue_name in {'NA', 'CL', 'K', 'CA', 'MG', 'ZN', 'FE', 'MN', 'CU'}:
            return False
    
    return True

def parse_cif_atoms(cif_path: str, include_waters: bool = False, include_ions: bool = False, include_solvents: bool = False, center_coordinates: bool = True):
    """
    Parse CIF file using gemmi and extract atom information with coordinates separated.
    
    By default, only protein polymer atoms are included. Waters, solvents, and ions are filtered out.
    
    Args:
        cif_path: Path to the CIF file
        include_waters: If True, include water molecules (HOH, WAT, etc.). Default: False
        include_ions: If True, include ion residues (NA, CL, MG, etc.). Default: False
        include_solvents: If True, include organic solvent molecules (ethanol, DMSO, etc.). Default: False
        center_coordinates: If True, center coordinates around the origin (subtract mean). Default: True
        
    Returns:
        tuple: (atom_info_list, coordinates_list, properties_list, failed_list)
            - atom_info_list: List of dicts with atom metadata (no coordinates)
              Keys: atom_id, atom_name, residue_name, residue_id, chain, element
            - coordinates_list: List of [x, y, z] coordinate arrays
            - properties_list: List of dicts with additional properties
              Keys: b_factor, occupancy
            - failed_list: List of dicts with failed entries
    """
    
    atom_info_list = []
    coordinates_list = []
    properties_list = []
    failed_list = []
    
    try:
        structure = gemmi.read_structure(cif_path)
        
        # Only process the first model
        if len(structure) > 0:
            model = structure[0]
            for chain in model:
                for residue in chain:
                    # Filter out waters, solvents, and ions unless explicitly requested
                    if not should_include_residue(residue, include_waters, include_ions, include_solvents):
                        continue
                    
                    for atom in residue:
                        try:
                            atom_info = {
                                'atom_id': str(atom.serial),
                                'atom_name': atom.name,
                                'residue_name': residue.name,
                                'residue_id': str(residue.seqid.num),
                                'chain': chain.name,
                                'element': atom.element.name,
                            }
                            
                            coordinates = [
                                atom.pos.x,
                                atom.pos.y,
                                atom.pos.z
                            ]
                            
                            properties = {
                                'b_factor': atom.b_iso,
                                'occupancy': atom.occ,
                            }
                            
                            atom_info_list.append(atom_info)
                            coordinates_list.append(coordinates)
                            properties_list.append(properties)
                        except Exception as e:
                            failed_list.append({
                                'atom': str(atom),
                                'residue': f"{residue.name}:{residue.seqid}",
                                'chain': chain.name,
                                'error': str(e)
                            })
                            
    except Exception as e:
        failed_list.append({
            'file': cif_path,
            'error': f"Failed to read structure: {str(e)}"
        })
    
    # Center coordinates if requested
    if center_coordinates and coordinates_list:
        coords_array = np.array(coordinates_list)
        center = np.mean(coords_array, axis=0)
        centered_coords = coords_array - center
        coordinates_list = centered_coords.tolist()
    
    return atom_info_list, coordinates_list, properties_list, failed_list


def get_ca_atoms(cif_path: str, include_waters: bool = False, include_ions: bool = False, include_solvents: bool = False):
    """
    Extract only C-alpha (CA) atoms from CIF file.
    
    Args:
        cif_path: Path to the CIF file
        include_waters: If True, include water molecules. Default: False
        include_ions: If True, include ion residues. Default: False
        include_solvents: If True, include organic solvent molecules. Default: False
        
    Returns:
        tuple: (atom_info_list, coordinates_list, properties_list, failed_list) for CA atoms only
    """
    atom_info_list, coordinates_list, properties_list, failed_list = parse_cif_atoms(
        cif_path, include_waters=include_waters, include_ions=include_ions, include_solvents=include_solvents
    )
    
    ca_atoms = []
    ca_coords = []
    ca_props = []
    
    for atom, coord, prop in zip(atom_info_list, coordinates_list, properties_list):
        if atom['atom_name'] == 'CA':
            ca_atoms.append(atom)
            ca_coords.append(coord)
            ca_props.append(prop)
    
    return ca_atoms, ca_coords, ca_props, failed_list


def get_backbone_atoms(cif_path: str, include_waters: bool = False, include_ions: bool = False, include_solvents: bool = False):
    """
    Extract backbone atoms (N, CA, C, O) from CIF file.
    
    Args:
        cif_path: Path to the CIF file
        include_waters: If True, include water molecules. Default: False
        include_ions: If True, include ion residues. Default: False
        include_solvents: If True, include organic solvent molecules. Default: False
        
    Returns:
        tuple: (atom_info_list, coordinates_list, properties_list, failed_list) for backbone atoms only
    """
    atom_info_list, coordinates_list, properties_list, failed_list = parse_cif_atoms(
        cif_path, include_waters=include_waters, include_ions=include_ions, include_solvents=include_solvents
    )
    
    backbone_names = {'N', 'CA', 'C', 'O'}
    bb_atoms = []
    bb_coords = []
    bb_props = []
    
    for atom, coord, prop in zip(atom_info_list, coordinates_list, properties_list):
        if atom['atom_name'] in backbone_names:
            bb_atoms.append(atom)
            bb_coords.append(coord)
            bb_props.append(prop)
    
    return bb_atoms, bb_coords, bb_props, failed_list


def coordinates_to_array(coordinates_list):
    """
    Convert coordinates list to numpy array if numpy is available.
    
    Args:
        coordinates_list: List of [x, y, z] coordinates
        
    Returns:
        numpy.ndarray or list: Nx3 array of coordinates
    """
    try:
        import numpy as np
        return np.array(coordinates_list)
    except ImportError:
        return coordinates_list


# Allowable features for protein atoms
PROTEIN_ATOM_FEATURES = {
    'atom_names': [
        'N', 'CA', 'C', 'O', 'CB',  # Backbone and beta carbon
        'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2',  # Gamma and delta carbons
        'CE', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3',  # Epsilon and zeta carbons
        'CH2', 'NZ', 'NH1', 'NH2', 'ND1', 'ND2', 'NE', 'NE1', 'NE2',  # Nitrogens
        'OG', 'OG1', 'OD1', 'OD2', 'OE1', 'OE2', 'OH', 'OXT',  # Oxygens
        'SG', 'SD',  # Sulfurs
        'misc'  # Unknown atom names
    ],
    'residue_names': [
        # Standard amino acids
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
        'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO',
        'SER', 'THR', 'TRP', 'TYR', 'VAL',
        # Modified/non-standard
        'MSE', 'SEC', 'PYL',
        'misc'
    ],
    'elements': [
        'H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
        'Se', 'misc'
    ],
    'chains': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['misc']
}


def safe_index(feature_list, value):
    """
    Return index of value in feature_list. If not present, return the 'misc' index (last).
    """
    try:
        return feature_list.index(value)
    except ValueError:
        return len(feature_list) - 1


def atom_info_to_feature_vector(atom_info):
    """
    Convert a single atom_info dict to a feature vector.
    
    Args:
        atom_info: Dict with keys: atom_name, residue_name, element, chain, etc.
        
    Returns:
        list: Feature vector as list of integer indices and values
    """
    atom_name = atom_info.get('atom_name', 'misc')
    
    # Determine if atom is backbone or CA
    backbone_atoms = {'N', 'CA', 'C', 'O'}
    is_backbone = 1 if atom_name in backbone_atoms else 0
    is_ca = 1 if atom_name == 'CA' else 0
    
    feature_vector = [
        safe_index(PROTEIN_ATOM_FEATURES['atom_names'], atom_name),
        safe_index(PROTEIN_ATOM_FEATURES['residue_names'], atom_info.get('residue_name', 'misc')),
        safe_index(PROTEIN_ATOM_FEATURES['elements'], atom_info.get('element', 'misc')),
        safe_index(PROTEIN_ATOM_FEATURES['chains'], atom_info.get('chain', 'misc')),
        int(atom_info.get('residue_id', 0)),  # Residue sequence number
        is_backbone,  # Binary flag for backbone atoms (0 or 1)
        is_ca,  # Binary flag for CA atoms (0 or 1)
    ]

    return feature_vector


def atom_info_list_to_features(atom_info_list):
    """
    Convert list of atom_info dicts to feature array.
    
    Args:
        atom_info_list: List of dicts with atom metadata
        
    Returns:
        numpy.ndarray: (N, 7) array where N is number of atoms
            Features: [atom_name_idx, residue_name_idx, element_idx, chain_idx, 
                      residue_id, is_backbone, is_ca] (dtype: int64)
    """
    if not atom_info_list:
        return np.array([])
    
    features = []
    for atom_info in atom_info_list:
        feature_vec = atom_info_to_feature_vector(atom_info)
        features.append(feature_vec)
    
    return np.array(features, dtype=np.int64)


def get_atom_feature_dims():
    """
    Get dimensions of each feature in the atom feature vector.
    
    Returns:
        list: Dimensions of each categorical feature, then continuous features
              -1 indicates a continuous feature, 2 indicates binary feature
    """
    dims = [
        len(PROTEIN_ATOM_FEATURES['atom_names']),  # atom name (categorical)
        len(PROTEIN_ATOM_FEATURES['residue_names']),  # residue name (categorical)
        len(PROTEIN_ATOM_FEATURES['elements']),  # element (categorical)
        len(PROTEIN_ATOM_FEATURES['chains']),  # chain (categorical)
        -1,  # residue_id (continuous)
        2,   # is_backbone (binary: 0 or 1)
        2,   # is_ca (binary: 0 or 1)
    ]
    
    return dims


def get_structure_info(cif_path: str):
    """
    Get basic structure information from CIF file.
    
    Args:
        cif_path: Path to the CIF file
        
    Returns:
        dict: Structure information including PDB ID, chains, resolution, etc.
    """
    structure = gemmi.read_structure(cif_path)
    
    info = {
        'name': structure.name,
        'num_models': len(structure),
        'chains': [],
        'resolution': None,
    }
    
    if structure:
        model = structure[0]
        for chain in model:
            chain_info = {
                'name': chain.name,
                'num_residues': len(chain),
                'num_atoms': sum(len(residue) for residue in chain)
            }
            info['chains'].append(chain_info)
    
    # Try to get resolution from metadata
    try:
        if hasattr(structure, 'resolution'):
            info['resolution'] = structure.resolution
    except:
        pass
    
    return info


if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python cif_to_cooridinates.py <cif_file_path> [--ca-only] [--backbone-only] [--info]")
        sys.exit(1)
    
    cif_file = sys.argv[1]
    
    if not os.path.exists(cif_file):
        print(f"Error: File not found: {cif_file}")
        sys.exit(1)
    
    if '--info' in sys.argv:
        info = get_structure_info(cif_file)
        print(f"\nStructure Information:")
        print(json.dumps(info, indent=2))
        sys.exit(0)
    
    if '--ca-only' in sys.argv:
        atom_info, coords, props, failed = get_ca_atoms(cif_file)
        print(f"Extracted {len(atom_info)} CA atoms")
    elif '--backbone-only' in sys.argv:
        atom_info, coords, props, failed = get_backbone_atoms(cif_file)
        print(f"Extracted {len(atom_info)} backbone atoms")
    else:
        atom_info, coords, props, failed = parse_cif_atoms(cif_file)
        print(f"Extracted {len(atom_info)} atoms")
    
    if atom_info:
        print(f"\nData structure:")
        print(f"  - Atom info: {len(atom_info)} entries")
        print(f"  - Coordinates: {len(coords)} entries")
        print(f"  - Properties: {len(props)} entries")
        print(f"  - Failed entries: {len(failed)}")
        
        if failed:
            print(f"\nFailed entries (showing first 3):")
            for fail_entry in failed[:3]:
                print(f"  {json.dumps(fail_entry, indent=4)}")
        
        print(f"\nFirst atom example:")
        print(f"  Info: {json.dumps(atom_info[0], indent=8)}")
        print(f"  Coords: {coords[0]}")
        print(f"  Props: {json.dumps(props[0], indent=8)}")
        
        print(f"\nSample of first 3 atoms:")
        for atom, coord in zip(atom_info[:3], coords[:3]):
            print(f"{atom['atom_name']:4s} {atom['residue_name']:3s} {atom['residue_id']:4s} "
                  f"({coord[0]:7.3f}, {coord[1]:7.3f}, {coord[2]:7.3f})")
        
        print(f"\nCoordinates shape: {len(coords)} x 3")
        
        # Convert to feature array
        features = atom_info_list_to_features(atom_info)
        print(f"\nFeature array shape: {features.shape}")
        print(f"Feature dimensions: {get_atom_feature_dims()}")
        print(f"\nFirst atom features: {features[0]}")
        print(f"  [atom_name_idx, residue_name_idx, element_idx, chain_idx, residue_id, is_backbone, is_ca]")
        
