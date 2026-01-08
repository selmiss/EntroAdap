"""
Build parquet dataset for EGNN masked reconstruction training.

Processes molecules (SMILES) and proteins (PDB IDs) into a unified parquet format
compatible with GraphDataset and ReconstructionTrainer.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from src.data_factory.molecule.mol_structure import generate_2d_3d_from_smiles
from src.data_factory.protein.pdbid_to_feature import pdbid_to_features


def process_molecule(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Process a single molecule SMILES into dataset format.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary with dataset fields or None if processing fails
    """
    atoms, graph_2d, coordinates_3d = generate_2d_3d_from_smiles(smiles)
    
    # Drop if structure generation failed
    if atoms is None or graph_2d is None or coordinates_3d is None:
        return None
    
    # Convert to expected format
    data = {
        'modality': 'molecule',
        'node_feat': graph_2d['node_feat'].numpy().tolist(),
        'pos': coordinates_3d.tolist(),
        'edge_index': graph_2d['edge_index'].numpy().tolist(),
        'chem_edge_index': graph_2d['chem_edge_index'].numpy().tolist(),
        'chem_edge_feat_cat': graph_2d['chem_edge_feat_cat'].numpy().tolist(),
    }
    
    # Add spatial edge distances if available
    if 'edge_feat_dist' in graph_2d:
        data['edge_feat_dist'] = graph_2d['edge_feat_dist'].numpy().tolist()
    
    return data


def process_protein(
    pdb_id: str, 
    structure_dir: str, 
    graph_radius: float = 6.0,
    ca_only: bool = False,
    max_atoms: Optional[int] = None,
    max_file_size_mb: float = 50.0,
) -> Optional[Dict[str, Any]]:
    """
    Process a single protein PDB ID into dataset format.
    
    Args:
        pdb_id: PDB identifier
        structure_dir: Directory containing CIF files
        graph_radius: Radius for edge construction (Angstroms)
        ca_only: Extract only C-alpha atoms (reduces size for large structures)
        max_atoms: Skip structures with more atoms than this limit
        max_file_size_mb: Skip CIF files larger than this (MB) before parsing
        
    Returns:
        Dictionary with dataset fields or None if processing fails
    """
    # Early check: skip very large files before parsing
    cif_path = os.path.join(structure_dir, f"{pdb_id.upper()}.cif")
    if os.path.exists(cif_path):
        file_size_mb = os.path.getsize(cif_path) / (1024 * 1024)
        if file_size_mb > max_file_size_mb:
            print(f"Skipping {pdb_id}: CIF file is {file_size_mb:.1f}MB (limit: {max_file_size_mb}MB)")
            return None
    
    result = pdbid_to_features(
        pdb_id=pdb_id,
        structure_dir=structure_dir,
        graph_radius=graph_radius,
        ca_only=ca_only,
        max_neighbors=32,
        sym_mode="union"
    )
    
    # Drop if structure processing failed
    if result is None:
        return None
    
    # Check atom count limit
    if max_atoms is not None and result['num_nodes'] > max_atoms:
        print(f"Skipping {pdb_id}: {result['num_nodes']} atoms exceeds limit of {max_atoms}")
        return None
    
    # Convert to expected format
    data = {
        'modality': 'protein',
        'node_feat': result['node_feat'].tolist(),
        'pos': result['coordinates'].tolist(),
        'edge_index': result['edge_index'].tolist(),
        'edge_attr': result['edge_attr'].tolist(),
    }
    
    return data


def build_molecule_dataset(
    smiles_file: str,
    output_path: str = './data/encoder_molecule.parquet',
) -> pd.DataFrame:
    """
    Build parquet dataset from molecule SMILES list.
    
    Args:
        smiles_file: Path to file with SMILES strings (one per line)
        output_path: Output parquet file path
        
    Returns:
        DataFrame with processed molecules
    """
    print(f"\n=== Processing Molecules ===")
    
    with open(smiles_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(smiles_list)} SMILES strings")
    
    data_list = []
    success_count = 0
    for smiles in tqdm(smiles_list, desc="Processing molecules"):
        data = process_molecule(smiles)
        if data is not None:
            data_list.append(data)
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(smiles_list)} molecules")
    
    if not data_list:
        raise ValueError("No molecules were successfully processed!")
    
    df = pd.DataFrame(data_list)
    
    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Saved to: {output_path}")
    
    return df


def build_protein_dataset(
    pdb_file: str,
    structure_dir: str = './data/pdb_structures',
    output_path: str = './data/encoder_protein.parquet',
    graph_radius: float = 6.0,
    ca_only: bool = False,
    max_atoms: Optional[int] = 50000,
    max_file_size_mb: float = 50.0,
) -> pd.DataFrame:
    """
    Build parquet dataset from protein PDB ID list.
    
    Args:
        pdb_file: Path to file with PDB IDs (one per line)
        structure_dir: Directory containing protein CIF files
        output_path: Output parquet file path
        graph_radius: Radius for protein graph edges (Angstroms)
        ca_only: Extract only C-alpha atoms (recommended for very large structures)
        max_atoms: Skip structures with more atoms than this limit (None = no limit)
        max_file_size_mb: Skip CIF files larger than this (MB) before parsing
        
    Returns:
        DataFrame with processed proteins
    """
    print(f"\n=== Processing Proteins ===")
    print(f"Settings: graph_radius={graph_radius}Å, ca_only={ca_only}")
    print(f"Limits: max_atoms={max_atoms}, max_file_size={max_file_size_mb}MB")
    
    with open(pdb_file, 'r') as f:
        pdb_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(pdb_ids)} PDB IDs")
    
    data_list = []
    success_count = 0
    
    for pdb_id in tqdm(pdb_ids, desc="Processing proteins"):
        data = process_protein(pdb_id, structure_dir, graph_radius, ca_only, max_atoms, max_file_size_mb)
        if data is not None:
            data_list.append(data)
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(pdb_ids)} proteins")
    
    if not data_list:
        raise ValueError("No proteins were successfully processed!")
    
    df = pd.DataFrame(data_list)
    
    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Saved to: {output_path}")
    
    return df


def print_first_sample_shapes(df: pd.DataFrame):
    """Print shapes of first sample to verify correctness."""
    if len(df) == 0:
        print("No data to display")
        return
    
    print(f"\n=== First Sample ({df.iloc[0]['modality']}) ===")
    first = df.iloc[0]
    
    for key, value in first.items():
        if key == 'modality':
            print(f"{key}: {value}")
        elif isinstance(value, list):
            arr = np.array(value)
            print(f"{key}: shape {arr.shape}, dtype {arr.dtype}")
        else:
            print(f"{key}: {type(value)}")


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build encoder dataset from molecules or proteins")
    parser.add_argument('--modality', type=str, required=True, choices=['molecule', 'protein'],
                       help='Type of data to process: molecule or protein')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input file (SMILES for molecule, PDB IDs for protein)')
    parser.add_argument('--structure_dir', type=str, default='./data/pdb_structures',
                       help='Directory containing protein CIF files (protein only)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output parquet file path')
    parser.add_argument('--graph_radius', type=float, default=6.0,
                       help='Radius for protein graph edges in Angstroms (protein only)')
    parser.add_argument('--ca_only', action='store_true',
                       help='Extract only C-alpha atoms for proteins (reduces size)')
    parser.add_argument('--max_atoms', type=int, default=50000,
                       help='Skip proteins with more atoms than this (0 to disable)')
    parser.add_argument('--max_file_size_mb', type=float, default=50.0,
                       help='Skip CIF files larger than this in MB (protein only)')
    
    args = parser.parse_args()
    
    # Build dataset based on modality
    if args.modality == 'molecule':
        df = build_molecule_dataset(
            smiles_file=args.input_file,
            output_path=args.output_path,
        )
    else:  # protein
        df = build_protein_dataset(
            pdb_file=args.input_file,
            structure_dir=args.structure_dir,
            output_path=args.output_path,
            graph_radius=args.graph_radius,
            ca_only=args.ca_only,
            max_atoms=args.max_atoms if args.max_atoms > 0 else None,
            max_file_size_mb=args.max_file_size_mb,
        )
    
    # Print first sample shapes
    print_first_sample_shapes(df)


if __name__ == '__main__':
    main()

