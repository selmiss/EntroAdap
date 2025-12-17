import os
import gemmi

def parse_cif_atoms(cif_path: str):
    """
    Parse CIF file using gemmi and extract atom information with coordinates separated.
    
    Args:
        cif_path: Path to the CIF file
        
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
        
        for model in structure:
            for chain in model:
                for residue in chain:
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
    
    return atom_info_list, coordinates_list, properties_list, failed_list


def get_ca_atoms(cif_path: str):
    """
    Extract only C-alpha (CA) atoms from CIF file.
    
    Args:
        cif_path: Path to the CIF file
        
    Returns:
        tuple: (atom_info_list, coordinates_list, properties_list, failed_list) for CA atoms only
    """
    atom_info_list, coordinates_list, properties_list, failed_list = parse_cif_atoms(cif_path)
    
    ca_atoms = []
    ca_coords = []
    ca_props = []
    
    for atom, coord, prop in zip(atom_info_list, coordinates_list, properties_list):
        if atom['atom_name'] == 'CA':
            ca_atoms.append(atom)
            ca_coords.append(coord)
            ca_props.append(prop)
    
    return ca_atoms, ca_coords, ca_props, failed_list


def get_backbone_atoms(cif_path: str):
    """
    Extract backbone atoms (N, CA, C, O) from CIF file.
    
    Args:
        cif_path: Path to the CIF file
        
    Returns:
        tuple: (atom_info_list, coordinates_list, properties_list, failed_list) for backbone atoms only
    """
    atom_info_list, coordinates_list, properties_list, failed_list = parse_cif_atoms(cif_path)
    
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
        
