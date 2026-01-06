#!/usr/bin/env python3
"""
Build molecule SFT dataset with geometry from raw parquet files.

Processes molecules from raw SFT parquet files (with SMILES, cid, iupac_name, messages, brics_gid)
and generates 2D/3D geometry data, creating the final SFT training format.

Input format (raw):
- smiles, cid, iupac_name, messages, brics_gid

Output format (SFT ready):
- modality, smiles, cid, iupac_name, brics_gid, messages
- node_feat, pos, edge_index, chem_edge_index, chem_edge_feat_cat, edge_feat_dist
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any, List
import argparse
from multiprocessing import Pool, cpu_count

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data_factory.molecule.mol_structure import generate_2d_3d_from_smiles


def process_molecule_sft(row: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Process a single molecule with SMILES into SFT dataset format.
    
    Args:
        row: Pandas Series with 'smiles', 'cid', 'iupac_name', 'messages', 'brics_gid'
        
    Returns:
        Dictionary with all fields including geometry, or None if processing fails
    """
    smiles = row['smiles']
    
    try:
        atoms, graph_2d, coordinates_3d = generate_2d_3d_from_smiles(smiles)
        
        # Drop if structure generation failed
        if atoms is None or graph_2d is None or coordinates_3d is None:
            return None
        
        # Convert to expected format - preserve all original fields + add geometry
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
        
        # Preserve messages column (critical for SFT!)
        if 'messages' in row:
            data['messages'] = row['messages']
        
        return data
    
    except Exception as e:
        # Silently skip molecules that fail to process
        return None


def process_batch(rows: List[tuple]) -> List[Optional[Dict[str, Any]]]:
    """Process a batch of rows (for multiprocessing)."""
    return [process_molecule_sft(row) for _, row in rows]


def process_sft_parquet(
    input_file: Path,
    output_file: Path,
    num_workers: int = None,
    batch_size: int = 100,
    checkpoint_interval: int = 5000,
    resume: bool = True,
    verbose: bool = True,
    max_file_size_mb: int = 300
) -> List[pd.DataFrame]:
    """
    Process a single SFT parquet file with multiprocessing and checkpointing.
    Splits output into multiple files if size exceeds max_file_size_mb.
    
    Args:
        input_file: Input parquet file path
        output_file: Output parquet file path (base name, will add _part_N if split)
        num_workers: Number of parallel workers (None = auto-detect)
        batch_size: Batch size for multiprocessing
        checkpoint_interval: Save checkpoint every N molecules (default: 5000)
        resume: Resume from existing checkpoint if available (default: True)
        verbose: Print progress information
        max_file_size_mb: Maximum size per output file in MB (default: 300)
        
    Returns:
        List of DataFrames (one per output file)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing: {input_file.name}")
        print(f"{'='*80}")
    
    # Read input data
    df_input = pd.read_parquet(input_file)
    if verbose:
        print(f"Loaded {len(df_input):,} molecules from input file")
        print(f"Input columns: {list(df_input.columns)}")
        print(f"Input size: {input_file.stat().st_size / (1024**2):.2f} MB")
    
    # Check for existing output files and resume if requested
    processed_cids = set()
    start_idx = 0
    existing_files = []
    
    if resume:
        # Check for single file or multiple parts
        if output_file.exists():
            existing_files = [output_file]
        
        # Always check for part files (they might exist alongside the base file)
        part_files = sorted(output_file.parent.glob(f"{output_file.stem}_part_*{output_file.suffix}"))
        if part_files:
            if output_file.exists():
                # Both base file and part files exist
                existing_files = [output_file] + part_files
            else:
                existing_files = part_files
        
        if existing_files:
            try:
                for f in existing_files:
                    df_part = pd.read_parquet(f)
                    processed_cids.update(df_part['cid'].astype(str))
                    start_idx += len(df_part)
                if verbose:
                    print(f"\n✓ Found {len(existing_files)} existing file(s): {start_idx:,} molecules already processed")
                    for f in existing_files:
                        size_mb = f.stat().st_size / (1024**2)
                        print(f"    - {f.name} ({size_mb:.2f} MB)")
                    print(f"  Resuming from molecule {start_idx + 1}...")
            except Exception as e:
                if verbose:
                    print(f"\n⚠ Could not load existing checkpoints: {e}")
                    print("  Starting from scratch...")
                processed_cids = set()
                start_idx = 0
    
    # Filter out already processed molecules
    if processed_cids:
        df_input = df_input[~df_input['cid'].astype(str).isin(processed_cids)].reset_index(drop=True)
        if verbose:
            print(f"  Remaining molecules to process: {len(df_input):,}")
    
    if len(df_input) == 0:
        if verbose:
            print("\n✓ All molecules already processed!")
        # Load and return all existing files
        result_dfs = [pd.read_parquet(f) for f in existing_files]
        return result_dfs
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing Configuration")
        print(f"{'='*80}")
        print(f"Parallel workers: {num_workers}")
        print(f"Batch size: {batch_size}")
        print(f"Checkpoint interval: {checkpoint_interval:,} molecules")
        print(f"Max file size: {max_file_size_mb} MB (will split into multiple files if needed)")
        print(f"{'='*80}\n")
    
    # Convert DataFrame to list of rows for processing
    rows = list(df_input.iterrows())
    
    # Process in parallel batches with checkpointing and file splitting
    data_list = []
    success_count = 0
    fail_count = 0
    total_processed = start_idx
    current_part = len(existing_files)  # Start from next part number
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if num_workers > 1:
        # Split into batches
        batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
        
        with Pool(num_workers) as pool:
            pbar = tqdm(
                pool.imap(process_batch, batches),
                total=len(batches),
                desc=f"Processing {input_file.name}",
                disable=not verbose,
                unit="batch"
            )
            
            for batch_idx, batch_results in enumerate(pbar):
                for data in batch_results:
                    if data is not None:
                        data_list.append(data)
                        success_count += 1
                    else:
                        fail_count += 1
                
                # Update progress bar with success rate
                if verbose and (success_count + fail_count) > 0:
                    success_rate = success_count / (success_count + fail_count) * 100
                    pbar.set_postfix({
                        'success': success_count,
                        'failed': fail_count,
                        'rate': f'{success_rate:.1f}%'
                    })
                
                # Save checkpoint every checkpoint_interval molecules
                if len(data_list) >= checkpoint_interval:
                    current_part = _save_checkpoint_with_split(
                        output_file, data_list, current_part, max_file_size_mb, verbose
                    )
                    total_processed += len(data_list)
                    data_list = []
    else:
        # Single process mode
        for idx, row in tqdm(rows, desc=f"Processing {input_file.name}", disable=not verbose):
            data = process_molecule_sft(row)
            if data is not None:
                data_list.append(data)
                success_count += 1
            else:
                fail_count += 1
            
            # Save checkpoint every checkpoint_interval molecules
            if len(data_list) >= checkpoint_interval:
                current_part = _save_checkpoint_with_split(
                    output_file, data_list, current_part, max_file_size_mb, verbose
                )
                total_processed += len(data_list)
                data_list = []
    
    # Save final batch if any remaining
    if data_list:
        current_part = _save_checkpoint_with_split(
            output_file, data_list, current_part, max_file_size_mb, verbose
        )
        total_processed += len(data_list)
    
    # Load and return all output files
    output_files = _get_output_files(output_file)
    if not output_files:
        print(f"❌ Error: No output files found!")
        return []
    
    df_outputs = [pd.read_parquet(f) for f in output_files]
    
    if verbose:
        total_molecules = sum(len(df) for df in df_outputs)
        print(f"\n{'='*80}")
        print(f"Processing Complete: {input_file.name}")
        print(f"{'='*80}")
        print(f"Input molecules:  {len(df_input) + start_idx:,}")
        print(f"Successfully processed: {success_count:,} new molecules")
        if fail_count > 0:
            print(f"Failed to process: {fail_count:,} molecules ({fail_count/(success_count+fail_count)*100:.1f}%)")
        print(f"Total in output:  {total_molecules:,} molecules")
        
        print(f"\nOutput files ({len(output_files)}):")
        total_size_mb = 0
        for idx, f in enumerate(output_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            # Use cached df_outputs instead of re-reading
            num_rows = len(df_outputs[idx])
            print(f"  - {f.name:60s} {size_mb:>7.2f} MB ({num_rows:>6,} molecules)")
        print(f"  {'Total size:':60s} {total_size_mb:>7.2f} MB")
        
        print(f"\nOutput columns: {list(df_outputs[0].columns)}")
        
        if len(df_input) > 0:
            print(f"\nSuccess rate: {success_count/len(df_input)*100:.1f}%")
        print(f"{'='*80}\n")
    
    return df_outputs


def _get_output_files(base_output_file: Path) -> List[Path]:
    """Get list of all output files (single file or multiple parts)."""
    output_files = []
    
    # Check for base file
    if base_output_file.exists():
        output_files.append(base_output_file)
    
    # Check for part files
    part_files = sorted(base_output_file.parent.glob(
        f"{base_output_file.stem}_part_*{base_output_file.suffix}"
    ))
    output_files.extend(part_files)
    
    return output_files


def _get_part_filename(base_output_file: Path, part_num: int) -> Path:
    """Get filename for a specific part number."""
    if part_num == 0:
        # First file uses base name (no _part_0 suffix)
        return base_output_file
    else:
        # Subsequent files use _part_N suffix
        return base_output_file.parent / f"{base_output_file.stem}_part_{part_num}{base_output_file.suffix}"


def _save_checkpoint_with_split(
    base_output_file: Path, 
    new_data: List[Dict[str, Any]], 
    current_part: int,
    max_file_size_mb: int,
    verbose: bool
) -> int:
    """
    Save checkpoint, splitting into new file if current file exceeds size limit.
    
    Returns:
        Updated current_part number
    """
    if not new_data:
        return current_part
    
    df_new = pd.DataFrame(new_data)
    
    # Get current output file
    current_file = _get_part_filename(base_output_file, current_part)
    
    # Check if we need to start a new part
    should_split = False
    if current_file.exists():
        current_size_mb = current_file.stat().st_size / (1024 * 1024)
        
        # Estimate size of new data (rough estimate based on existing file)
        df_existing = pd.read_parquet(current_file)
        if len(df_existing) > 0:
            estimated_new_size_mb = (len(df_new) / len(df_existing)) * current_size_mb
        else:
            # Fallback: estimate based on average of 3-4 KB per molecule
            estimated_new_size_mb = len(df_new) * 0.0035  # ~3.5 KB per molecule
        
        # If adding new data would exceed limit, start new part
        if current_size_mb + estimated_new_size_mb > max_file_size_mb:
            should_split = True
            current_part += 1
            current_file = _get_part_filename(base_output_file, current_part)
            if verbose:
                print(f"  → Starting new part {current_part} (previous file: {current_size_mb:.2f} MB, estimated new: {estimated_new_size_mb:.2f} MB)")
    
    # Save data
    if current_file.exists() and not should_split:
        # Append to existing file (only if we're not splitting)
        df_existing = pd.read_parquet(current_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_parquet(current_file, index=False)
        file_size_mb = current_file.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"  ✓ Checkpoint saved to {current_file.name}: {len(df_combined):,} molecules ({file_size_mb:.2f} MB)")
    else:
        # Create new file (either first time or after split)
        df_new.to_parquet(current_file, index=False)
        file_size_mb = current_file.stat().st_size / (1024 * 1024)
        if verbose:
            action = "New file created" if not should_split else "New part started"
            print(f"  ✓ {action}: {current_file.name}: {len(df_new):,} molecules ({file_size_mb:.2f} MB)")
    
    return current_part


def build_sft_dataset(
    input_dir: str,
    output_dir: str,
    num_workers: int = None,
    batch_size: int = 100,
    checkpoint_interval: int = 5000,
    resume: bool = True,
    verbose: bool = True,
    process_all: bool = True,
    max_file_size_mb: int = 300
):
    """
    Build SFT dataset from raw parquet files with checkpointing support.
    
    Args:
        input_dir: Directory containing raw parquet files
        output_dir: Output directory for processed parquet files
        num_workers: Number of parallel workers (None = auto-detect)
        batch_size: Batch size for multiprocessing
        checkpoint_interval: Save checkpoint every N molecules (default: 5000)
        resume: Resume from existing checkpoint if available (default: True)
        verbose: Print progress information
        process_all: Process all .parquet files in directory (default: True)
        max_file_size_mb: Maximum size per output file in MB (default: 300)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if verbose:
        print("\n" + "=" * 80)
        print("MOLECULE SFT DATASET BUILDER")
        print("=" * 80)
        print(f"Input directory:  {input_dir}")
        print(f"Output directory: {output_dir}")
        print("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all .parquet files in the directory
    input_files = sorted(input_dir.glob("*.parquet"))
    
    if not input_files:
        print(f"\n❌ Error: No parquet files found in {input_dir}")
        return
    
    if verbose:
        print(f"\nFound {len(input_files)} parquet file(s) to process:")
        for f in input_files:
            size_mb = f.stat().st_size / (1024**2)
            print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    results = {}
    
    for input_file in input_files:
        output_file = output_dir / input_file.name
        
        df_list = process_sft_parquet(
            input_file, 
            output_file, 
            num_workers, 
            batch_size, 
            checkpoint_interval, 
            resume, 
            verbose=verbose,
            max_file_size_mb=max_file_size_mb
        )
        results[input_file.stem] = df_list
    
    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print("FINAL DATASET SUMMARY")
        print("=" * 80)
        total_molecules = 0
        total_files = 0
        for name, df_list in results.items():
            molecules_count = sum(len(df) for df in df_list)
            print(f"{name:50s}: {molecules_count:>8,} molecules ({len(df_list)} file(s))")
            total_molecules += molecules_count
            total_files += len(df_list)
        print("-" * 80)
        print(f"{'Total':50s}: {total_molecules:>8,} molecules ({total_files} file(s))")
        print("=" * 80)
        
        # Print first sample info
        if results:
            first_name = list(results.keys())[0]
            df_list = results[first_name]
            if df_list and len(df_list[0]) > 0:
                df = df_list[0]
                print(f"\nSample from {first_name}:")
                first = df.iloc[0]
                print(f"  SMILES: {first['smiles']}")
                print(f"  CID: {first['cid']}")
                print(f"  IUPAC: {first['iupac_name'][:60]}...")
                print(f"  BRICS GID: {np.array(first['brics_gid'])}")
                print(f"\n  Messages:")
                messages = first['messages']
                if isinstance(messages, np.ndarray):
                    messages = messages.tolist()
                for msg in messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    preview = content[:80] + '...' if len(content) > 80 else content
                    print(f"    - {role:10s}: {preview}")
                print(f"\n  Geometry data:")
                for key in ['node_feat', 'pos', 'edge_index', 'chem_edge_index', 'chem_edge_feat_cat', 'edge_feat_dist']:
                    if key in first:
                        arr = np.array(first[key])
                        print(f"    {key:20s}: shape {arr.shape}, dtype {arr.dtype}")
        
        print("\n" + "=" * 80)
        print("✅ SFT dataset building complete!")
        print("=" * 80)
        print(f"\nOutput files saved to: {output_dir}")
        
        # List all output files
        all_output_files = sorted(output_dir.glob("*.parquet"))
        if all_output_files:
            print(f"\nAll output files ({len(all_output_files)}):")
            for f in all_output_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.2f} MB)")
        
        print("\nYou can now use these files with:")
        print("  from src.data_loader.octopus_sft_dataset import MultiModalSFTDataset")
        print("  from datasets import load_dataset")
        print("")
        print("  # Option 1: Load single file")
        first_output = all_output_files[0] if all_output_files else output_dir / input_files[0].name
        print(f"  dataset = MultiModalSFTDataset(")
        print(f"      dataset_path='{first_output}',")
        print("      use_combined_parquet=True")
        print("  )")
        print("")
        print("  # Option 2: Load all files with HuggingFace datasets")
        print(f"  ds = load_dataset('parquet', data_files='{output_dir}/*.parquet')")
        print("=" * 80 + "\n")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Build molecule SFT dataset with geometry from raw parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in mol_sft_all directory
  python build_sft_data.py \\
      --input_dir data/raw/mol_sft_all \\
      --output_dir data/sft
  
  # Use fewer workers and smaller checkpoints
  python build_sft_data.py \\
      --input_dir data/raw/mol_sft_all \\
      --output_dir data/sft \\
      --num_workers 4 \\
      --checkpoint_interval 2000
  
  # Start from scratch (ignore existing checkpoints)
  python build_sft_data.py \\
      --input_dir data/raw/mol_sft_all \\
      --output_dir data/sft \\
      --no_resume
        """
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/raw/mol_sft_all',
        help='Directory containing raw SFT parquet files (default: data/raw/mol_sft_all)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/sft',
        help='Output directory for processed SFT parquet files (default: data/sft)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto-detect = CPU count - 1)'
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
        default=5000,
        help='Save checkpoint every N molecules (default: 5000)'
    )
    parser.add_argument(
        '--max_file_size_mb',
        type=int,
        default=300,
        help='Maximum size per output file in MB (default: 300). Files larger than this will be split into multiple parts.'
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
    
    args = parser.parse_args()
    
    build_sft_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        verbose=not args.quiet,
        max_file_size_mb=args.max_file_size_mb,
    )


if __name__ == '__main__':
    main()

