"""
Molecular structure processing: SMILES → atoms, 2D graphs, 3D conformers.

DATA FORMATS:
1. atoms: List[str] - Atomic symbols, e.g., ['C', 'O', 'N']
2. graph_2d: dict with keys:
   - 'node_feat': torch.Tensor [num_nodes, num_node_features (9 for OGB atom features)] - OGB atom features (int64)
   - 'edge_index': torch.Tensor [2, num_spatial_edges] - Spatial edge connectivity (int64)
     Only present if 3D coordinates are available
   - 'edge_feat_dist': torch.Tensor [num_spatial_edges, 1] - Real distances (float32)
     Only present if 3D coordinates are available
   - 'chem_edge_index': torch.Tensor [2, num_chem_edges] - Chemical bond connectivity in COO format (int64)
   - 'chem_edge_feat_cat': torch.Tensor [num_chem_edges, 3] - Categorical edge features (int64)
     Features: [bond_type, bond_stereo, is_conjugated] from OGB
   - 'chem_edge_feat_dist': torch.Tensor [num_chem_edges, 1] - Distance feature (float32)
     - Placeholder value (-1.0), not real distance to avoid computational cost
   - 'num_nodes': int
3. coordinates_3d: np.ndarray [num_atoms, 3] - 3D positions in Angstroms

Note: Atom ordering is consistent across all representations.
Chemical bonds and spatial edges are stored separately for flexible downstream processing.
"""

import torch
import math

import numpy as np
import os
from src.data_factory.molecule.ogb_features import atom_to_feature_vector, bond_to_feature_vector

from rdkit import Chem
from rdkit import RDLogger
from typing import Optional, List, Tuple
from rdkit.Chem import AllChem

# Import build_radius_graph from protein module
from src.data_factory.protein.pdbid_to_feature import build_radius_graph

# Disable RDKit logging
RDLogger.DisableLog('rdApp.*')

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object.
    Note: This function only generates 2D graph without 3D coordinates,
    so distance feature is set to -1 and no spatial edges are generated.
    
    :input: SMILES string (str)
    :return: graph object with chemical bond features
             chem_edge_feat_cat: [num_edges, 3] int64 - [bond_type, bond_stereo, is_conjugated]
             chem_edge_feat_dist: [num_edges, 1] float32 - [distance]

    Adopted from OGB 
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    if mol is not None:
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype = np.int64)
    else:
        return None

    # bonds
    if mol is not None and len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_cat_features_list = []
        edge_dist_features_list = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)  # 3 features from OGB
            
            # Categorical features: [bond_type, bond_stereo, is_conjugated]
            cat_feature = [
                edge_feature[0],  # bond_type
                edge_feature[1],  # bond_stereo
                edge_feature[2],  # is_conjugated
            ]
            
            # Distance feature (placeholder, no 3D coords)
            dist_feature = [-1.0]

            # add edges in both directions
            edges_list.append((i, j))
            edge_cat_features_list.append(cat_feature)
            edge_dist_features_list.append(dist_feature)
            
            edges_list.append((j, i))
            edge_cat_features_list.append(cat_feature)
            edge_dist_features_list.append(dist_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # Categorical edge features: shape [num_edges, 3]
        edge_attr_cat = np.array(edge_cat_features_list, dtype = np.int64)
        
        # Distance edge features: shape [num_edges, 1]
        edge_attr_dist = np.array(edge_dist_features_list, dtype = np.float32)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr_cat = np.empty((0, 3), dtype = np.int64)
        edge_attr_dist = np.empty((0, 1), dtype = np.float32)

    graph = dict()
    graph['chem_edge_index'] = torch.from_numpy(edge_index)
    graph['chem_edge_feat_cat'] = torch.from_numpy(edge_attr_cat)
    graph['chem_edge_feat_dist'] = torch.from_numpy(edge_attr_dist)
    graph['node_feat'] = torch.from_numpy(x)
    graph['num_nodes'] = len(x)

    return graph 


def generate_conformer_with_rdkit(smiles: str) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    """
    Generate 3D molecular conformer from SMILES string using RDKit.
    
    This function:
    1. Parses SMILES string into molecular structure
    2. Adds hydrogens for embedding
    3. Generates 3D conformer using distance geometry
    4. Optimizes geometry using MMFF force field
    5. Removes hydrogens to get heavy atoms only
    6. Extracts atom symbols and 3D coordinates
    
    Args:
        smiles: SMILES string representing the molecule
        
    Returns:
        Tuple of (atoms, coordinates):
            - atoms: List of atomic symbols (e.g., ['C', 'O', 'N'])
            - coordinates: NumPy array of shape (N, 3) with 3D coordinates
        Returns (None, None) if conformer generation fails
        
    Note:
        Uses MMFF force field with multiple threads for optimization.
        Number of threads controlled by NUM_WORKERS environment variable (default: 12).
    """
    if Chem is None or AllChem is None:
        return None, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        num_atoms = mol.GetNumAtoms()

        mol = Chem.AddHs(mol)
        
        # Use configurable number of threads (default 16 to avoid overwhelming shared servers)
        num_threads = int(os.environ.get('RDKIT_NUM_THREADS', '16'))
        
        # Use ETKDGv3 parameters for better conformer generation
        params = AllChem.ETKDGv3()
        params.numThreads = num_threads
        # Note: maxAttempts not available in all RDKit versions
        # params.maxAttempts = 2000
        
        # Generate conformer
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=5, params=params)
        
        if len(conf_ids) == 0:
            print(f"3D conformer embedding failed for {smiles}")
            return None, None
        
        
        best_conf_id = conf_ids[0]  # Use first valid conformer ID as default
        try:
            opt_results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_threads)

            if opt_results:
                # 1) 先在“收敛”的 conformer 中找最低能量
                best_energy = float("inf")
                for idx, (not_converged, energy) in enumerate(opt_results):
                    if not_converged == 0 and energy is not None and math.isfinite(energy):
                        if energy < best_energy:
                            best_energy = energy
                            best_conf_id = conf_ids[idx]  # Use actual conformer ID

                # 2) 若没有任何收敛结果，则在“能量有效”的 conformer 中找最低能量
                if best_energy == float("inf"):
                    valid = [
                        (conf_ids[i], e) for i, (nc, e) in enumerate(opt_results)
                        if e is not None and math.isfinite(e)
                    ]
                    if valid:
                        best_conf_id = min(valid, key=lambda x: x[1])[0]
                    else:
                        best_conf_id = conf_ids[0]  # Use first valid conformer ID
        except Exception as e:
            print(f"MMFFOptimizeMoleculeConfs failed for {smiles}: {e}")
            best_conf_id = conf_ids[0]  # Use first valid conformer ID
        
        mol = Chem.RemoveHs(mol)

        if mol.GetNumConformers() == 0:
            print("GetNumConformers == 0 {}".format(smiles))
            return None, None
        if num_atoms != mol.GetNumAtoms():
            print("num_atoms != mol.GetNumAtoms() {}".format(smiles))
            return None, None

        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coordinates = np.array(mol.GetConformer(best_conf_id).GetPositions(), dtype=float)
        return atoms, coordinates
    except Exception:
        return None, None


def generate_2d_3d_from_smiles(smiles: str) -> Tuple[Optional[List[str]], Optional[dict], Optional[np.ndarray]]:
    """
    Generate both 2D graph and 3D conformer from SMILES string with progressive fallback.
    
    This unified function ensures that the atom order in the 2D graph matches the atom order
    in the 3D conformer, making them directly comparable and usable together.
    
    Fallback mechanism:
    - If 3D generation fails: Returns (atoms, graph_2d, None) - 2D graph with atom symbols
    - If 2D generation fails: Returns (atoms, None, None) - Only 1D atom information
    - If parsing SMILES fails: Returns (None, None, None) - Complete failure
    
    Args:
        smiles: SMILES string representing the molecule
        
    Returns:
        Tuple of (atoms, graph_2d, coordinates_3d):
            - atoms: List of atomic symbols (e.g., ['C', 'O', 'N'])
                Or None if SMILES parsing fails
            - graph_2d: Dictionary containing 2D graph representation with keys:
                - 'node_feat': Node feature matrix [num_nodes, num_node_features]
                - 'edge_index': Edge connectivity in COO format [2, num_edges]
                - 'edge_feat': Edge feature matrix [num_edges, num_edge_features]
                - 'num_nodes': Number of nodes in the graph
                Or None if 2D generation fails
            - coordinates_3d: NumPy array of shape (N, 3) with 3D coordinates
                Or None if 3D generation fails
        
    Note:
        The atom ordering is consistent across all representations.
        The i-th atom corresponds to the i-th node in graph_2d and i-th row in coordinates_3d.
    """
    if Chem is None or AllChem is None:
        raise ValueError("RDKit is not installed properly. Please install RDKit using `pip install rdkit`.")
    
    try:
        # Step 1: Parse SMILES and create molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"SMILES parsing failed for {smiles}, no structure generated.")
            return None, None, None
        
        num_atoms = mol.GetNumAtoms()
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        # Step 2: Generate 2D graph representation
        graph_2d = None
        try:
            # Extract atom features
            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))
            node_feat = np.array(atom_features_list, dtype=np.int64)
            
            # Extract bond features (split into categorical and distance)
            if len(mol.GetBonds()) > 0:  # mol has bonds
                edges_list = []
                edge_cat_features_list = []
                edge_dist_features_list = []
                
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    edge_feature = bond_to_feature_vector(bond)  # 3 features from OGB
                    
                    # Categorical features: [bond_type, bond_stereo, is_conjugated]
                    cat_feature = [
                        edge_feature[0],  # bond_type
                        edge_feature[1],  # bond_stereo
                        edge_feature[2],  # is_conjugated
                    ]
                    
                    # Distance feature (placeholder)
                    dist_feature = [-1.0]
                    
                    # add edges in both directions
                    edges_list.append((i, j))
                    edge_cat_features_list.append(cat_feature)
                    edge_dist_features_list.append(dist_feature)
                    
                    edges_list.append((j, i))
                    edge_cat_features_list.append(cat_feature)
                    edge_dist_features_list.append(dist_feature)
                
                edge_index = np.array(edges_list, dtype=np.int64).T
                edge_attr_cat = np.array(edge_cat_features_list, dtype=np.int64)
                edge_attr_dist = np.array(edge_dist_features_list, dtype=np.float32)
            else:  # mol has no bonds
                edge_index = np.empty((2, 0), dtype=np.int64)
                edge_attr_cat = np.empty((0, 3), dtype=np.int64)
                edge_attr_dist = np.empty((0, 1), dtype=np.float32)
            
            graph_2d = {
                'chem_edge_index': torch.from_numpy(edge_index),
                'chem_edge_feat_cat': torch.from_numpy(edge_attr_cat),
                'chem_edge_feat_dist': torch.from_numpy(edge_attr_dist),
                'node_feat': torch.from_numpy(node_feat),
                'num_nodes': num_atoms
            }
        except Exception as e:
            print(f"2D graph generation failed for {smiles}: {e}")
        
        # Step 3: Generate 3D conformer
        coordinates_3d = None
        try:
            # Add hydrogens for embedding
            mol_3d = Chem.AddHs(mol)
            
            # Configure conformer generation parameters
            num_threads = int(os.environ.get('RDKIT_NUM_THREADS', '16'))
            params = AllChem.ETKDGv2()
            params.numThreads = num_threads
            # Note: maxAttempts not available in all RDKit versions
            # params.maxAttempts = 2000
            
            # Generate conformer
            conf_ids = AllChem.EmbedMultipleConfs(mol_3d, numConfs=2, params=params)
            
            if len(conf_ids) == 0:
                print(f"3D conformer embedding failed for {smiles}")
                return atoms, graph_2d, None
            
            # Optimize geometry and select best conformer
            best_conf_id = conf_ids[0]  # Use first valid conformer ID as default
            try:
                opt_results = AllChem.MMFFOptimizeMoleculeConfs(mol_3d, numThreads=num_threads)

                if opt_results:
                    # 1) 先在"收敛"的 conformer 中找最低能量
                    best_energy = float("inf")
                    for idx, (not_converged, energy) in enumerate(opt_results):
                        if not_converged == 0 and energy is not None and math.isfinite(energy):
                            if energy < best_energy:
                                best_energy = energy
                                best_conf_id = conf_ids[idx]  # Use actual conformer ID

                    # 2) 若没有任何收敛结果，则在"能量有效"的 conformer 中找最低能量
                    if best_energy == float("inf"):
                        valid = [
                            (conf_ids[i], e) for i, (nc, e) in enumerate(opt_results)
                            if e is not None and math.isfinite(e)
                        ]
                        if valid:
                            best_conf_id = min(valid, key=lambda x: x[1])[0]
                        else:
                            best_conf_id = conf_ids[0]  # Use first valid conformer ID
            except Exception as e:
                print(f"MMFFOptimizeMoleculeConfs failed for {smiles}: {e}")
                best_conf_id = conf_ids[0]  # Use first valid conformer ID
            
            # Remove hydrogens to match 2D representation
            mol_3d = Chem.RemoveHs(mol_3d)
            
            # Validate conformer generation
            if mol_3d.GetNumConformers() == 0:
                print(f"GetNumConformers == 0 for {smiles}")
                return atoms, graph_2d, None
            
            if num_atoms != mol_3d.GetNumAtoms():
                print(f"Atom count mismatch for {smiles}: {num_atoms} != {mol_3d.GetNumAtoms()}")
                return atoms, graph_2d, None
            
            # Extract 3D coordinates
            # IMPORTANT: The atom order here matches the atom order in the 2D graph
            coordinates_3d = np.array(mol_3d.GetConformer(best_conf_id).GetPositions(), dtype=float)
            
            # Step 4: Add spatial edges if 3D coordinates are available
            if coordinates_3d is not None and graph_2d is not None:
                try:
                    # Build spatial edges using radius graph
                    # Default radius=6.0 Angstroms, max_neighbors=32
                    spatial_edge_index, spatial_edge_distances = build_radius_graph(
                        coordinates_3d,
                        radius=4.0,
                        max_neighbors=16,
                        sym_mode="union"
                    )
                    # Add spatial edges as default edge fields in graph_2d
                    E_spatial = spatial_edge_index.shape[1]
                    if E_spatial > 0:
                        graph_2d['edge_index'] = torch.from_numpy(spatial_edge_index)
                        graph_2d['edge_feat_dist'] = torch.from_numpy(spatial_edge_distances.astype(np.float32))
                    
                except Exception as e:
                    print(f"Failed to add spatial edges for {smiles}: {e}")
                    # Continue with chemical bonds only
            
        except Exception as e:
            print(f"3D conformer generation failed for {smiles}: {e}")
            # Continue with 2D only, coordinates_3d remains None
        
        return atoms, graph_2d, coordinates_3d
        
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None, None, None