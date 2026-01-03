#!/usr/bin/env python3
"""
Build protein encoder dataset from UniProt JSON files with PDB structures.

Processes proteins from UniProt JSON (with PDB cross-references),
downloads PDB structures, and generates 3D geometry data for encoder pretraining.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Set
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data_factory.protein.pdbid_to_feature import pdbid_to_features
from src.data_factory.protein.map_fetch_pdb3d import download_pdb_structures


def extract_pdb_ids_from_uniprot_json(json_file: Path) -> List[Dict[str, Any]]:
    """
    Extract PDB IDs from UniProt JSON file.
    
    Args:
        json_file: Path to UniProt JSON file
        
    Returns:
        List of dicts with 'uniprot_id', 'pdb_id', 'method', 'resolution', 'chains'
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    pdb_records = []
    for entry in data.get('results', []):
        uniprot_id = entry.get('primaryAccession')
        if not uniprot_id:
            continue
        
        # Extract PDB cross-references
        for ref in entry.get('uniProtKBCrossReferences', []):
            if ref.get('database') == 'PDB':
                pdb_id = ref.get('id')
                if not pdb_id:
                    continue
                
                # Extract properties (method, resolution, chains)
                props = {}
                for prop in ref.get('properties', []):
                    props[prop.get('key')] = prop.get('value')
                
                pdb_records.append({
                    'uniprot_id': uniprot_id,
                    'pdb_id': pdb_id,
                    'method': props.get('Method', ''),
                    'resolution': props.get('Resolution', ''),
                    'chains': props.get('Chains', ''),
                })
    
    return pdb_records


def process_pdb_record(
    record: Dict[str, Any],
    structure_dir: Path,
    ca_only: bool = True,
    graph_radius: float = 8.0,
    max_neighbors: int = 24
) -> Optional[Dict[str, Any]]:
    """
    Process a single PDB record into dataset format.
    
    Args:
        record: Dict with 'pdb_id', 'uniprot_id', etc.
        structure_dir: Directory containing CIF files
        ca_only: Extract only C-alpha atoms (default: True)
        graph_radius: Radius for graph construction
        max_neighbors: Max neighbors per node
        
    Returns:
        Dictionary with dataset fields, or None if processing fails
    """
    pdb_id = record['pdb_id']
    
    try:
        # Generate features using existing function
        data = pdbid_to_features(
            pdb_id,
            structure_dir=str(structure_dir),
            graph_radius=graph_radius,
            ca_only=ca_only,
            max_neighbors=max_neighbors,
            sym_mode="union"
        )
        
        if data is None:
            return None
        
        # Convert to expected format (lists for parquet storage)
        result = {
            'modality': 'protein',
            'pdb_id': pdb_id,
            'uniprot_id': record.get('uniprot_id', ''),
            'method': record.get('method', ''),
            'resolution': record.get('resolution', ''),
            'chains': record.get('chains', ''),
            'num_atoms': data['num_nodes'],
            'node_feat': data['node_feat'].tolist(),
            'coordinates': data['coordinates'].tolist(),
            'edge_index': data['edge_index'].tolist(),
            'edge_attr': data['edge_attr'].tolist(),
        }
        
        return result
    
    except Exception as e:
        # Silently skip structures that fail to process
        return None


def process_batch(
    records: List[Dict[str, Any]],
    structure_dir: Path,
    ca_only: bool,
    graph_radius: float,
    max_neighbors: int
) -> List[Optional[Dict[str, Any]]]:
    """Process a batch of PDB records (for multiprocessing)."""
    results = []
    for record in records:
        result = process_pdb_record(
            record, structure_dir, ca_only, graph_radius, max_neighbors
        )
        results.append(result)
    return results


def build_protein_encoder_dataset(
    uniprot_json_dir: str,
    structure_dir: str,
    output_file: str,
    ca_only: bool = True,
    graph_radius: float = 8.0,
    max_neighbors: int = 24,
    num_workers: int = None,
    batch_size: int = 50,
    checkpoint_interval: int = 1000,
    resume: bool = True,
    verbose: bool = True,
    download_missing: bool = True,
    download_delay: float = 0.1,
    max_structures: Optional[int] = None
):
    """
    Build protein encoder dataset from UniProt JSON files.
    
    Args:
        uniprot_json_dir: Directory containing UniProt JSON files
        structure_dir: Directory to store/read PDB CIF files
        output_file: Output parquet file path
        ca_only: Extract only C-alpha atoms (default: True)
        graph_radius: Radius for graph construction (default: 8.0)
        max_neighbors: Max neighbors per node (default: 24)
        num_workers: Number of parallel workers (None = auto-detect)
        batch_size: Batch size for multiprocessing
        checkpoint_interval: Save checkpoint every N structures
        resume: Resume from existing checkpoint if available
        verbose: Print progress information
        download_missing: Download missing PDB structures
        download_delay: Delay between download requests in seconds (default: 0.1)
        max_structures: Maximum number of structures to process (for testing)
    """
    uniprot_dir = Path(uniprot_json_dir)
    struct_dir = Path(structure_dir)
    output_path = Path(output_file)
    
    if verbose:
        print("=" * 70)
        print("Building Protein Encoder Dataset")
        print("=" * 70)
        print(f"UniProt JSON directory: {uniprot_dir}")
        print(f"Structure directory: {struct_dir}")
        print(f"Output file: {output_path}")
        print(f"C-alpha only: {ca_only}")
        print(f"Graph radius: {graph_radius} Å")
        print(f"Max neighbors: {max_neighbors}")
    
    # Step 1: Extract all PDB IDs from UniProt JSON files
    if verbose:
        print("\n" + "=" * 70)
        print("Step 1: Extracting PDB IDs from UniProt JSON files")
        print("=" * 70)
    
    json_files = sorted(uniprot_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {uniprot_dir}")
    
    if verbose:
        print(f"Found {len(json_files)} JSON files:")
        for f in json_files:
            print(f"  - {f.name}")
    
    all_pdb_records = []
    for json_file in tqdm(json_files, desc="Reading JSON files", disable=not verbose):
        records = extract_pdb_ids_from_uniprot_json(json_file)
        all_pdb_records.extend(records)
    
    if verbose:
        print(f"\nTotal PDB records extracted: {len(all_pdb_records)}")
    
    # Deduplicate by PDB ID (case-insensitive, keep first occurrence)
    # Normalize all PDB IDs to lowercase for consistency
    seen_pdb_ids: Set[str] = set()
    unique_records = []
    for record in all_pdb_records:
        pdb_id_lower = record['pdb_id'].lower()
        if pdb_id_lower not in seen_pdb_ids:
            seen_pdb_ids.add(pdb_id_lower)
            # Store normalized lowercase version
            record['pdb_id'] = pdb_id_lower
            unique_records.append(record)
    
    if verbose:
        print(f"Unique PDB IDs: {len(unique_records)}")
    
    # Limit for testing if requested
    if max_structures and max_structures < len(unique_records):
        unique_records = unique_records[:max_structures]
        if verbose:
            print(f"Limited to {max_structures} structures for testing")
    
    # Step 2: Download missing PDB structures
    if download_missing:
        if verbose:
            print("\n" + "=" * 70)
            print("Step 2: Downloading missing PDB structures")
            print("=" * 70)
        
        struct_dir.mkdir(parents=True, exist_ok=True)
        
        # Check which structures are missing (check both uppercase and lowercase)
        missing_pdb_ids = []
        for record in unique_records:
            pdb_id = record['pdb_id']
            # Check for both uppercase and lowercase versions
            cif_path_lower = struct_dir / f"{pdb_id.lower()}.cif"
            cif_path_upper = struct_dir / f"{pdb_id.upper()}.cif"
            cif_path_original = struct_dir / f"{pdb_id}.cif"
            
            if not (cif_path_lower.exists() or cif_path_upper.exists() or cif_path_original.exists()):
                # Normalize to lowercase for downloading (PDB standard)
                missing_pdb_ids.append(pdb_id.lower())
        
        if verbose:
            print(f"Missing structures: {len(missing_pdb_ids)}/{len(unique_records)}")
        
        if missing_pdb_ids:
            if verbose:
                print("Downloading missing structures...")
                print(f"Rate limiting: {download_delay}s delay between requests (~{1/download_delay:.1f} req/sec)")
            # Download structures with rate limiting
            download_pdb_structures(
                missing_pdb_ids, 
                str(struct_dir), 
                file_format='cif',
                delay=download_delay,
                verbose=verbose
            )
    
    # Step 3: Check for existing checkpoint and resume
    if verbose:
        print("\n" + "=" * 70)
        print("Step 3: Processing PDB structures")
        print("=" * 70)
    
    processed_pdb_ids: Set[str] = set()
    start_idx = 0
    
    if resume and output_path.exists():
        try:
            df_existing = pd.read_parquet(output_path)
            # Normalize to lowercase for case-insensitive comparison
            processed_pdb_ids = set(pdb_id.lower() for pdb_id in df_existing['pdb_id'])
            start_idx = len(df_existing)
            if verbose:
                print(f"Found existing checkpoint: {len(df_existing)} structures already processed")
                print(f"Resuming from structure {start_idx + 1}...")
        except Exception as e:
            if verbose:
                print(f"Could not load existing checkpoint: {e}")
                print("Starting from scratch...")
    
    # Filter out already processed structures (case-insensitive)
    if processed_pdb_ids:
        unique_records = [r for r in unique_records if r['pdb_id'].lower() not in processed_pdb_ids]
        if verbose:
            print(f"Remaining structures to process: {len(unique_records)}")
    
    if len(unique_records) == 0:
        if verbose:
            print("All structures already processed!")
        return pd.read_parquet(output_path)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    if verbose:
        print(f"Using {num_workers} parallel workers with batch size {batch_size}")
        print(f"Checkpoint interval: {checkpoint_interval} structures")
    
    # Process structures with multiprocessing and checkpointing
    data_list = []
    success_count = 0
    fail_count = 0
    total_processed = start_idx
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if num_workers > 1:
        # Split into batches
        batches = [unique_records[i:i + batch_size] for i in range(0, len(unique_records), batch_size)]
        
        process_func = partial(
            process_batch,
            structure_dir=struct_dir,
            ca_only=ca_only,
            graph_radius=graph_radius,
            max_neighbors=max_neighbors
        )
        
        with Pool(num_workers) as pool:
            for batch_results in tqdm(
                pool.imap(process_func, batches),
                total=len(batches),
                desc="Processing structures",
                disable=not verbose
            ):
                for data in batch_results:
                    if data is not None:
                        data_list.append(data)
                        success_count += 1
                    else:
                        fail_count += 1
                
                # Save checkpoint every checkpoint_interval structures
                if len(data_list) >= checkpoint_interval:
                    _save_checkpoint(output_path, data_list, resume, verbose)
                    total_processed += len(data_list)
                    data_list = []
    else:
        # Single process mode
        for record in tqdm(unique_records, desc="Processing structures", disable=not verbose):
            data = process_pdb_record(
                record, struct_dir, ca_only, graph_radius, max_neighbors
            )
            if data is not None:
                data_list.append(data)
                success_count += 1
            else:
                fail_count += 1
            
            # Save checkpoint every checkpoint_interval structures
            if len(data_list) >= checkpoint_interval:
                _save_checkpoint(output_path, data_list, resume, verbose)
                total_processed += len(data_list)
                data_list = []
    
    # Save final batch if any remaining
    if data_list:
        _save_checkpoint(output_path, data_list, resume, verbose)
        total_processed += len(data_list)
    
    if verbose:
        print(f"\nSuccessfully processed: {success_count}/{len(unique_records)} new structures")
        if fail_count > 0:
            print(f"Failed to process: {fail_count} structures")
    
    # Load and return final dataset
    df_output = pd.read_parquet(output_path)
    
    if verbose:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nFinal dataset: {len(df_output)} total structures")
        print(f"Saved to: {output_path}")
        print(f"File size: {file_size_mb:.2f} MB")
        if len(unique_records) > 0:
            print(f"Success rate: {success_count/len(unique_records)*100:.1f}%")
        
        # Print summary statistics
        if len(df_output) > 0:
            print(f"\nDataset statistics:")
            print(f"  Mean atoms per structure: {df_output['num_atoms'].mean():.1f}")
            print(f"  Median atoms per structure: {df_output['num_atoms'].median():.1f}")
            print(f"  Min atoms: {df_output['num_atoms'].min()}")
            print(f"  Max atoms: {df_output['num_atoms'].max()}")
            
            # Method distribution
            method_counts = df_output['method'].value_counts()
            print(f"\n  Methods:")
            for method, count in method_counts.head(5).items():
                print(f"    {method}: {count} ({count/len(df_output)*100:.1f}%)")
    
    print("\n✅ Protein encoder dataset building complete!")
    return df_output


def _save_checkpoint(output_file: Path, new_data: List[Dict[str, Any]], append: bool, verbose: bool):
    """Save checkpoint by appending new data to existing parquet file."""
    if not new_data:
        return
    
    df_new = pd.DataFrame(new_data)
    
    if append and output_file.exists():
        # Append to existing file
        df_existing = pd.read_parquet(output_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_parquet(output_file, index=False)
        if verbose:
            print(f"  Checkpoint saved: {len(df_combined)} total structures")
    else:
        # Create new file
        df_new.to_parquet(output_file, index=False)
        if verbose:
            print(f"  Checkpoint saved: {len(df_new)} structures")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Build protein encoder dataset from UniProt JSON files with PDB structures"
    )
    parser.add_argument(
        '--uniprot_json_dir',
        type=str,
        default='data/uniprot/full',
        help='Directory containing UniProt JSON files'
    )
    parser.add_argument(
        '--structure_dir',
        type=str,
        default='data/pdb_structures',
        help='Directory to store/read PDB CIF files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/encoder/protein_pretrain.parquet',
        help='Output parquet file path'
    )
    parser.add_argument(
        '--all_atoms',
        action='store_true',
        help='Extract all atoms instead of C-alpha only'
    )
    parser.add_argument(
        '--graph_radius',
        type=float,
        default=8.0,
        help='Radius for graph construction in Angstroms (default: 8.0)'
    )
    parser.add_argument(
        '--max_neighbors',
        type=int,
        default=24,
        help='Maximum neighbors per node (default: 24)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto-detect, use 1 to disable multiprocessing)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Batch size for multiprocessing (default: 50)'
    )
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=1000,
        help='Save checkpoint every N structures (default: 1000)'
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='Do not resume from existing checkpoint (start from scratch)'
    )
    parser.add_argument(
        '--no_download',
        action='store_true',
        help='Do not download missing PDB structures'
    )
    parser.add_argument(
        '--download_delay',
        type=float,
        default=0.1,
        help='Delay between PDB download requests in seconds (default: 0.1, min recommended: 0.1)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--max_structures',
        type=int,
        default=None,
        help='Maximum number of structures to process (for testing)'
    )
    
    args = parser.parse_args()
    
    build_protein_encoder_dataset(
        uniprot_json_dir=args.uniprot_json_dir,
        structure_dir=args.structure_dir,
        output_file=args.output_file,
        ca_only=not args.all_atoms,
        graph_radius=args.graph_radius,
        max_neighbors=args.max_neighbors,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        verbose=not args.quiet,
        download_missing=not args.no_download,
        download_delay=args.download_delay,
        max_structures=args.max_structures
    )


if __name__ == '__main__':
    main()

