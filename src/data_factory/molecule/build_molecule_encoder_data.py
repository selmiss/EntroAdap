#!/usr/bin/env python3
"""
Build molecule encoder dataset with geometry from parquet files.

Processes molecules from raw parquet files (with SMILES, cid, iupac_name, brics_gid)
and generates 2D/3D geometry data for encoder pretraining, while retaining metadata.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any, List
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data_factory.molecule.mol_structure import generate_2d_3d_from_smiles


def process_molecule_with_metadata(row: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Process a single molecule with SMILES into dataset format, retaining metadata.
    
    Args:
        row: Pandas Series with 'smiles', 'cid', 'iupac_name', 'brics_gid'
        
    Returns:
        Dictionary with dataset fields and metadata, or None if processing fails
    """
    smiles = row['smiles']
    
    try:
        atoms, graph_2d, coordinates_3d = generate_2d_3d_from_smiles(smiles)
        
        # Drop if structure generation failed
        if atoms is None or graph_2d is None or coordinates_3d is None:
            return None
        
        # Convert to expected format
        data = {
            'modality': 'molecule',
            'smiles': smiles,
            'cid': row['cid'],
            'iupac_name': row['iupac_name'],
            'brics_gid': row['brics_gid'],
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
    
    except Exception as e:
        # Silently skip molecules that fail to process
        return None


def process_batch(rows: List[tuple]) -> List[Optional[Dict[str, Any]]]:
    """Process a batch of rows (for multiprocessing)."""
    return [process_molecule_with_metadata(row) for _, row in rows]


def process_parquet_split(
    input_file: Path,
    output_file: Path,
    num_workers: int = None,
    batch_size: int = 100,
    checkpoint_interval: int = 10000,
    resume: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Process a single parquet file (train/val/test split) with multiprocessing and checkpointing.
    
    Args:
        input_file: Input parquet file path
        output_file: Output parquet file path
        num_workers: Number of parallel workers (None = auto-detect)
        batch_size: Batch size for multiprocessing
        checkpoint_interval: Save checkpoint every N molecules (default: 10000)
        resume: Resume from existing checkpoint if available (default: True)
        verbose: Print progress information
        
    Returns:
        DataFrame with processed molecules
    """
    if verbose:
        print(f"\nProcessing {input_file.name}...")
    
    # Read input data
    df_input = pd.read_parquet(input_file)
    if verbose:
        print(f"Loaded {len(df_input)} molecules")
    
    # Check for existing output and resume if requested
    processed_cids = set()
    start_idx = 0
    
    if resume and output_file.exists():
        try:
            df_existing = pd.read_parquet(output_file)
            processed_cids = set(df_existing['cid'].astype(str))
            start_idx = len(df_existing)
            if verbose:
                print(f"Found existing checkpoint: {len(df_existing)} molecules already processed")
                print(f"Resuming from molecule {start_idx + 1}...")
        except Exception as e:
            if verbose:
                print(f"Could not load existing checkpoint: {e}")
                print("Starting from scratch...")
    
    # Filter out already processed molecules
    if processed_cids:
        df_input = df_input[~df_input['cid'].astype(str).isin(processed_cids)].reset_index(drop=True)
        if verbose:
            print(f"Remaining molecules to process: {len(df_input)}")
    
    if len(df_input) == 0:
        if verbose:
            print("All molecules already processed!")
        return pd.read_parquet(output_file)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    if verbose:
        print(f"Using {num_workers} parallel workers with batch size {batch_size}")
        print(f"Checkpoint interval: {checkpoint_interval} molecules")
    
    # Convert DataFrame to list of rows for processing
    rows = list(df_input.iterrows())
    
    # Process in parallel batches with checkpointing
    data_list = []
    success_count = 0
    fail_count = 0
    total_processed = start_idx
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if num_workers > 1:
        # Split into batches
        batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
        
        with Pool(num_workers) as pool:
            for batch_idx, batch_results in enumerate(tqdm(
                pool.imap(process_batch, batches),
                total=len(batches),
                desc=f"Processing {input_file.name}",
                disable=not verbose,
                initial=start_idx // batch_size
            )):
                for data in batch_results:
                    if data is not None:
                        data_list.append(data)
                        success_count += 1
                    else:
                        fail_count += 1
                
                # Save checkpoint every checkpoint_interval molecules
                if len(data_list) >= checkpoint_interval:
                    _save_checkpoint(output_file, data_list, resume, verbose)
                    total_processed += len(data_list)
                    data_list = []
    else:
        # Single process mode
        for idx, row in tqdm(rows, desc=f"Processing {input_file.name}", disable=not verbose):
            data = process_molecule_with_metadata(row)
            if data is not None:
                data_list.append(data)
                success_count += 1
            else:
                fail_count += 1
            
            # Save checkpoint every checkpoint_interval molecules
            if len(data_list) >= checkpoint_interval:
                _save_checkpoint(output_file, data_list, resume, verbose)
                total_processed += len(data_list)
                data_list = []
    
    # Save final batch if any remaining
    if data_list:
        _save_checkpoint(output_file, data_list, resume, verbose)
        total_processed += len(data_list)
    
    if verbose:
        print(f"Successfully processed: {success_count}/{len(df_input)} new molecules")
        if fail_count > 0:
            print(f"Failed to process: {fail_count} molecules")
    
    # Load and return final dataset
    df_output = pd.read_parquet(output_file)
    
    if verbose:
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"Final dataset: {len(df_output)} total molecules")
        print(f"Saved to: {output_file}")
        print(f"File size: {file_size_mb:.2f} MB")
        if len(df_input) > 0:
            print(f"Success rate: {success_count/len(df_input)*100:.1f}%")
    
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
            print(f"  Checkpoint saved: {len(df_combined)} total molecules")
    else:
        # Create new file
        df_new.to_parquet(output_file, index=False)
        if verbose:
            print(f"  Checkpoint saved: {len(df_new)} molecules")


def build_encoder_dataset(
    input_dir: str,
    output_dir: str,
    num_workers: int = None,
    batch_size: int = 100,
    checkpoint_interval: int = 10000,
    resume: bool = True,
    verbose: bool = True,
    file_pattern: str = None,
    process_all: bool = False
):
    """
    Build encoder dataset from raw parquet files with checkpointing support.
    
    Args:
        input_dir: Directory containing raw parquet files
        output_dir: Output directory for processed parquet files
        num_workers: Number of parallel workers (None = auto-detect)
        batch_size: Batch size for multiprocessing
        checkpoint_interval: Save checkpoint every N molecules (default: 10000)
        resume: Resume from existing checkpoint if available (default: True)
        verbose: Print progress information
        file_pattern: Pattern to match files (e.g., '*.parquet' or specific filenames)
        process_all: Process all .parquet files in directory (default: False)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if verbose:
        print("=" * 60)
        print("Building Molecule Encoder Dataset")
        print("=" * 60)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
    
    # Determine which files to process
    results = {}
    
    if process_all:
        # Process all .parquet files in the directory
        input_files = sorted(input_dir.glob("*.parquet"))
        if verbose:
            print(f"\nProcessing all parquet files in directory ({len(input_files)} files found)")
        
        for input_file in input_files:
            output_file = output_dir / input_file.name
            if verbose:
                print(f"\n{'='*60}")
            df = process_parquet_split(
                input_file, output_file, num_workers, batch_size, 
                checkpoint_interval, resume, verbose=verbose
            )
            results[input_file.stem] = df
    
    elif file_pattern:
        # Process files matching the pattern
        input_files = sorted(input_dir.glob(file_pattern))
        if verbose:
            print(f"\nProcessing files matching pattern '{file_pattern}' ({len(input_files)} files found)")
        
        if not input_files:
            print(f"\nWarning: No files matching pattern '{file_pattern}' found in {input_dir}")
        
        for input_file in input_files:
            output_file = output_dir / input_file.name
            if verbose:
                print(f"\n{'='*60}")
            df = process_parquet_split(
                input_file, output_file, num_workers, batch_size, 
                checkpoint_interval, resume, verbose=verbose
            )
            results[input_file.stem] = df
    
    else:
        # Default behavior: look for train/val/test splits
        splits = ['train', 'val', 'test']
        
        for split in splits:
            input_file = input_dir / f"{split}.parquet"
            output_file = output_dir / f"{split}.parquet"
            
            if not input_file.exists():
                print(f"\nWarning: {input_file} not found, skipping...")
                continue
            
            df = process_parquet_split(
                input_file, output_file, num_workers, batch_size, 
                checkpoint_interval, resume, verbose=verbose
            )
            results[split] = df
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("Dataset Summary")
        print("=" * 60)
        for split, df in results.items():
            print(f"{split.capitalize():6s}: {len(df):,} samples")
        print("=" * 60)
        
        # Print first sample info
        if results:
            first_split = list(results.keys())[0]
            df = results[first_split]
            if len(df) > 0:
                print(f"\nFirst sample from {first_split} split:")
                first = df.iloc[0]
                print(f"  SMILES: {first['smiles']}")
                print(f"  CID: {first['cid']}")
                print(f"  IUPAC name: {first['iupac_name'][:60]}...")
                print(f"  BRICS GID: {np.array(first['brics_gid'])}")
                print(f"\n  Geometry data:")
                for key in ['node_feat', 'pos', 'edge_index', 'chem_edge_index', 'chem_edge_feat_cat']:
                    if key in first:
                        arr = np.array(first[key])
                        print(f"    {key:20s}: shape {arr.shape}, dtype {arr.dtype}")
    
    print("\n✅ Encoder dataset building complete!")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Build molecule encoder dataset with geometry from raw parquet files"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/raw/mol_encode_small',
        help='Directory containing raw parquet files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/encoder/pretrain',
        help='Output directory for processed parquet files'
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
        default=100,
        help='Batch size for multiprocessing (default: 100)'
    )
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=10000,
        help='Save checkpoint every N molecules (default: 10000)'
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='Do not resume from existing checkpoint (start from scratch)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--file_pattern',
        type=str,
        default=None,
        help='Glob pattern to match specific files (e.g., "*.parquet" or "*-preprocessed.parquet")'
    )
    parser.add_argument(
        '--process_all',
        action='store_true',
        help='Process all .parquet files in the input directory'
    )
    
    args = parser.parse_args()
    
    build_encoder_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        verbose=not args.quiet,
        file_pattern=args.file_pattern,
        process_all=args.process_all
    )


if __name__ == '__main__':
    main()

