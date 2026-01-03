"""
Convert PDB ID to protein features and graph representation.
Handles downloading, parsing, and feature extraction from PDB structures.
"""

import os
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Literal

from src.data_factory.protein.map_fetch_pdb3d import download_pdb_structures
from src.data_factory.protein.cif_to_cooridinates import parse_cif_atoms, atom_info_list_to_features
from scipy.spatial import cKDTree

def build_radius_graph(
    coordinates: np.ndarray,
    radius: float = 10.0,
    max_neighbors: int = 32,
    sym_mode: Literal["union", "mutual"] = "union",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scalable radius graph with per node cap via kNN first, then radius filter.

    Properties:
      - Time and memory are O(N * max_neighbors).
      - Avoids enumerating all radius pairs, which can be O(N^2) in dense systems.

    Args:
        coordinates: (N, 3) float array
        radius: cutoff distance
        max_neighbors: number of nearest neighbors queried per node (excluding self)
        sym_mode:
            "union": keep undirected edge if either i->j or j->i appears in kNN list,
                     then output both directions
            "mutual": keep undirected edge only if both i->j and j->i appear,
                      then output both directions

    Returns:
        edge_index: (2, E) int64
        edge_attr:  (E, 1) float32 distances aligned with edge_index
    """
    if coordinates is None or len(coordinates) == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

    coords = np.ascontiguousarray(coordinates, dtype=np.float32)
    n = coords.shape[0]
    if n <= 1:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

    # Sanity check: max_neighbors cannot be None
    if max_neighbors is None:
        raise ValueError("max_neighbors cannot be None. Please provide a positive integer.")
    
    k = int(max_neighbors)
    if k <= 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

    tree = cKDTree(coords)

    k_query = min(k + 1, n)
    dists, nbrs = tree.query(coords, k=k_query)

    if k_query == 1:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

    dists = dists[:, 1:]
    nbrs = nbrs[:, 1:]

    valid = np.isfinite(dists) & (dists <= float(radius))
    if not np.any(valid):
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

    src = np.repeat(np.arange(n, dtype=np.int64), k_query - 1)[valid.reshape(-1)]
    dst = nbrs.reshape(-1).astype(np.int64, copy=False)[valid.reshape(-1)]
    dist = dists.reshape(-1).astype(np.float32, copy=False)[valid.reshape(-1)]

    in_range = (dst >= 0) & (dst < n) & (src != dst)
    src, dst, dist = src[in_range], dst[in_range], dist[in_range]
    if src.size == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

    key_dir = src.astype(np.int64) * n + dst.astype(np.int64)

    if sym_mode == "mutual":
        key_sorted = np.sort(key_dir)
        rev = dst.astype(np.int64) * n + src.astype(np.int64)
        pos = np.searchsorted(key_sorted, rev)
        ok = (pos < key_sorted.size) & (key_sorted[pos] == rev)
        src, dst, dist = src[ok], dst[ok], dist[ok]
        if src.size == 0:
            return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    key_undir = a.astype(np.int64) * n + b.astype(np.int64)

    order = np.argsort(key_undir)
    key_undir = key_undir[order]
    a = a[order]
    b = b[order]
    dist = dist[order]

    uniq_mask = np.ones_like(key_undir, dtype=bool)
    uniq_mask[1:] = key_undir[1:] != key_undir[:-1]
    a = a[uniq_mask]
    b = b[uniq_mask]
    dist = dist[uniq_mask]

    pairs = np.stack([a, b], axis=1).astype(np.int64, copy=False)

    edge_index = np.concatenate([pairs, pairs[:, [1, 0]]], axis=0).T
    edge_attr = np.concatenate([dist, dist], axis=0).reshape(-1, 1).astype(np.float32, copy=False)

    return edge_index, edge_attr

def pdbid_to_features(pdb_id: str,
                      structure_dir: str = './data/pdb_structures',
                      graph_radius: float = 6.0,
                      ca_only: bool = False,
                      max_neighbors: Optional[int] = 24,
                      sym_mode: Literal["union", "mutual"] = "union") -> Optional[Dict]:
    """
    Convert PDB ID to protein features and graph representation.
    
    This function:
    1. Searches for CIF file in target directory
    2. Downloads if not found using RCSB PDB
    3. Parses CIF file to extract atom information
    4. Generates feature vectors for atoms
    5. Builds radius graph for spatial relationships
    
    Args:
        pdb_id: 4-character PDB identifier (e.g., '1ABC')
        structure_dir: Directory to search/download CIF files (default: './pdb_structures')
        graph_radius: Distance threshold for edge creation in Angstroms (default: 10.0)
        ca_only: Extract only C-alpha atoms (default: False, extracts all atoms)
        max_neighbors: Optional limit on max neighbors per node
        sym_mode:
            "union": keep undirected edge if either i->j or j->i appears in kNN list,
                     then output both directions
            "mutual": keep undirected edge only if both i->j and j->i appear,
                      then output both directions
    Returns:
        Dictionary with keys:
            - 'node_feat': np.ndarray [num_atoms, 7] - Atom feature matrix (int64)
            - 'coordinates': np.ndarray [num_atoms, 3] - 3D atomic coordinates
            - 'edge_index': np.ndarray [2, num_edges] - Graph connectivity in COO format
            - 'edge_attr': np.ndarray [num_edges, 1] - Edge distances
            - 'num_nodes': int - Number of atoms
            - 'pdb_id': str - PDB identifier
            - 'atom_info': list - Original atom metadata (for reference)
        Returns None if processing fails
        
    Example:
        >>> data = pdbid_to_features('1ABC', graph_radius=8.0, ca_only=True)
        >>> print(f"Loaded {data['num_nodes']} atoms with {data['edge_index'].shape[1]} edges")
        
    Note:
        For C-alpha only graphs, typical radius is 8-10 Å.
        For all-atom graphs, typical radius is 5-8 Å.
    """
    # Step 1: Check if CIF file exists (case-insensitive), download if not
    os.makedirs(structure_dir, exist_ok=True)
    
    # Try to find existing file with different cases
    cif_path_upper = os.path.join(structure_dir, f"{pdb_id.upper()}.cif")
    cif_path_lower = os.path.join(structure_dir, f"{pdb_id.lower()}.cif")
    cif_path_original = os.path.join(structure_dir, f"{pdb_id}.cif")
    
    # Use whichever exists
    if os.path.exists(cif_path_upper):
        cif_path = cif_path_upper
        pdb_id = pdb_id.upper()
    elif os.path.exists(cif_path_lower):
        cif_path = cif_path_lower
        pdb_id = pdb_id.lower()
    elif os.path.exists(cif_path_original):
        cif_path = cif_path_original
    else:
        # File doesn't exist, download it (normalize to lowercase for PDB standard)
        pdb_id = pdb_id.lower()
        cif_path = cif_path_lower
        print(f"CIF file not found for {pdb_id}, downloading...")
        try:
            download_pdb_structures([pdb_id], structure_dir, file_format='cif')
            if not os.path.exists(cif_path):
                print(f"Failed to download {pdb_id}")
                return None
        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")
            return None
    
    # Step 2: Parse CIF file to extract atom information
    try:
        atom_info_list, coordinates_list, properties_list, failed_list = parse_cif_atoms(cif_path)
        
        if failed_list:
            print(f"Warning: {len(failed_list)} atoms failed to parse")
        
        if not atom_info_list:
            print(f"No atoms extracted from {pdb_id}")
            return None
        
        # Filter for C-alpha only if requested
        if ca_only:
            ca_indices = [i for i, atom in enumerate(atom_info_list) 
                         if atom['atom_name'] == 'CA']
            if not ca_indices:
                print(f"No C-alpha atoms found in {pdb_id}")
                return None
            atom_info_list = [atom_info_list[i] for i in ca_indices]
            coordinates_list = [coordinates_list[i] for i in ca_indices]
            properties_list = [properties_list[i] for i in ca_indices]
        
    except Exception as e:
        print(f"Error parsing {cif_path}: {e}")
        return None
    
    # Step 3: Convert to feature matrix
    try:
        node_feat = atom_info_list_to_features(atom_info_list)
        coordinates = np.array(coordinates_list, dtype=np.float32)
        
        if node_feat.size == 0 or coordinates.size == 0:
            print(f"Empty features or coordinates for {pdb_id}")
            return None
            
    except Exception as e:
        print(f"Error generating features for {pdb_id}: {e}")
        return None
    
    # Step 4: Build radius graph
    try:
        # Sanity check: if max_neighbors is None, use default value
        if max_neighbors is None:
            max_neighbors = 32  # Default value, same as build_radius_graph default
            
        edge_index, edge_attr = build_radius_graph(
            coordinates, 
            radius=graph_radius,
            max_neighbors=max_neighbors,
            sym_mode=sym_mode
        )
    except Exception as e:
        print(f"Error building graph for {pdb_id}: {e}")
        return None
    
    # Step 5: Package results
    result = {
        'node_feat': node_feat,
        'coordinates': coordinates,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'num_nodes': len(atom_info_list),
        'pdb_id': pdb_id,
        'atom_info': atom_info_list,  # Keep for reference
    }
    
    print(f"Loaded {pdb_id}: {result['num_nodes']} atoms, "
          f"{edge_index.shape[1]} edges (radius={graph_radius}Å)")
    
    return result


def pdbid_to_features_torch(pdb_id: str, **kwargs) -> Optional[Dict]:
    """
    Same as pdbid_to_features but returns PyTorch tensors instead of NumPy arrays.
    
    Args:
        pdb_id: 4-character PDB identifier
        **kwargs: Additional arguments passed to pdbid_to_features
        
    Returns:
        Dictionary with torch.Tensor values instead of np.ndarray
    """
    result = pdbid_to_features(pdb_id, **kwargs)
    
    if result is None:
        return None
    
    # Convert to torch tensors
    result['node_feat'] = torch.from_numpy(result['node_feat'])
    result['coordinates'] = torch.from_numpy(result['coordinates'])
    result['edge_index'] = torch.from_numpy(result['edge_index'])
    result['edge_attr'] = torch.from_numpy(result['edge_attr'])
    
    return result


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdbid_to_feature.py <pdb_id> [structure_dir] [radius]")
        print("Example: python pdbid_to_feature.py 1ABC ./pdb_structures 10.0")
        sys.exit(1)
    
    pdb_id = sys.argv[1]
    structure_dir = sys.argv[2] if len(sys.argv) > 2 else './data/pdb_structures'
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    
    print(f"\n=== Processing {pdb_id} ===")
    
    # Example 1: All atoms
    print("\n1. All atoms:")
    data = pdbid_to_features(pdb_id, structure_dir=structure_dir, graph_radius=radius)
    if data:
        print(f"  Features shape: {data['node_feat'].shape}")
        print(f"  Coordinates shape: {data['coordinates'].shape}")
        print(f"  Edges: {data['edge_index'].shape}")
    
    # Example 2: C-alpha only
    print("\n2. C-alpha only:")
    data_ca = pdbid_to_features(pdb_id, structure_dir=structure_dir, 
                                 graph_radius=radius, ca_only=True)
    if data_ca:
        print(f"  Features shape: {data_ca['node_feat'].shape}")
        print(f"  Coordinates shape: {data_ca['coordinates'].shape}")
        print(f"  Edges: {data_ca['edge_index'].shape}")
