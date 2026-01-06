#!/usr/bin/env python3
"""
Build nucleic acid (DNA/RNA) encoder dataset from sequence files.

Processes DNA and RNA sequences from text files, generates 3D structures using X3DNA,
and creates training data for encoder pretraining.
"""

import sys
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any, List
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data_factory.nacid.seq_to_feature import sequence_to_features


# Define consistent schema for parquet files
# This ensures all datasets use regular 'list' type instead of 'large_list'
# Uses unified format: 'pos' (not 'coordinates') and 'edge_feat_dist' (not 'edge_attr')
PARQUET_SCHEMA = pa.schema([
    ('modality', pa.string()),
    ('seq_id', pa.string()),
    ('source_file', pa.string()),
    ('sequence', pa.string()),
    ('seq_length', pa.int64()),
    ('num_atoms', pa.int64()),
    ('node_feat', pa.list_(pa.list_(pa.int64()))),
    ('pos', pa.list_(pa.list_(pa.float64()))),
    ('edge_index', pa.list_(pa.list_(pa.int64()))),
    ('edge_feat_dist', pa.list_(pa.list_(pa.float64()))),
])


def process_sequence(
    seq: str,
    seq_idx: int,
    seq_type: str,
    source_file: str,
    workdir: Path,
    graph_radius: float,
    max_neighbors: int,
    max_seq_length: int,
    fiber_exe: str
) -> Optional[Dict[str, Any]]:
    """
    Process a single sequence into dataset format.
    
    Args:
        seq: Nucleic acid sequence
        seq_idx: Index of sequence in source file
        seq_type: "dna" or "rna"
        source_file: Name of source file
        workdir: Working directory for temporary files
        graph_radius: Radius for graph construction
        max_neighbors: Max neighbors per node
        max_seq_length: Maximum sequence length (truncate longer)
        fiber_exe: Path to fiber executable
        
    Returns:
        Dictionary with dataset fields, or None if processing fails
    """
    # Clean sequence
    seq = seq.strip().upper()
    if not seq:
        return None
    
    # For RNA, convert T to U (common issue: RNA sequences stored in DNA format)
    if seq_type == "rna":
        seq = seq.replace('T', 'U')
    
    # Skip sequences that are too short
    # (Long sequences will be truncated in sequence_to_features)
    if len(seq) < 10:
        return None
    
    # Create unique ID
    seq_id = f"{source_file}_{seq_idx}"
    
    # Create sequence-specific workdir
    seq_workdir = workdir / seq_id
    seq_workdir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate features
        data = sequence_to_features(
            seq=seq,
            seq_id=seq_id,
            seq_type=seq_type,
            workdir=str(seq_workdir),
            graph_radius=graph_radius,
            max_neighbors=max_neighbors,
            max_seq_length=max_seq_length,
            fiber_exe=fiber_exe
        )
        
        if data is None:
            return None
        
        # Convert arrays to lists for parquet storage
        # Use unified format: 'pos' (not 'coordinates') and 'edge_feat_dist' (not 'edge_attr')
        result = {
            'modality': data['modality'],
            'seq_id': seq_id,
            'source_file': source_file,
            'sequence': data['sequence'],
            'seq_length': data['seq_length'],
            'num_atoms': data['num_atoms'],
            'node_feat': data['node_feat'].tolist(),
            'pos': data['coordinates'].tolist(),
            'edge_index': data['edge_index'].tolist(),
            'edge_feat_dist': data['edge_attr'].tolist(),
        }
        
        return result
    
    except Exception as e:
        # Silently skip sequences that fail to process
        return None
    finally:
        # Clean up sequence-specific workdir
        if seq_workdir.exists():
            shutil.rmtree(seq_workdir, ignore_errors=True)


def process_batch(
    sequences: List[tuple],  # (seq, seq_idx, seq_type, source_file)
    workdir: Path,
    graph_radius: float,
    max_neighbors: int,
    max_seq_length: int,
    fiber_exe: str
) -> List[Optional[Dict[str, Any]]]:
    """Process a batch of sequences (for multiprocessing)."""
    results = []
    for seq, seq_idx, seq_type, source_file in sequences:
        result = process_sequence(
            seq, seq_idx, seq_type, source_file,
            workdir, graph_radius, max_neighbors, max_seq_length, fiber_exe
        )
        results.append(result)
    return results


def load_sequences_from_file(file_path: Path, seq_type: str) -> List[tuple]:
    """
    Load sequences from a text file.
    
    Args:
        file_path: Path to text file with one sequence per line
        seq_type: "dna" or "rna"
        
    Returns:
        List of tuples (seq, seq_idx, seq_type, source_file)
    """
    sequences = []
    source_file = file_path.stem  # filename without extension
    
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            seq = line.strip()
            if seq:
                sequences.append((seq, idx, seq_type, source_file))
    
    return sequences


def build_nacid_encoder_dataset(
    input_dir: str,
    output_file: str,
    seq_type: str = "dna",
    graph_radius: float = 8.0,
    max_neighbors: int = 24,
    max_seq_length: int = 500,
    num_workers: int = None,
    batch_size: int = 100,
    checkpoint_interval: int = 1000,
    resume: bool = True,
    verbose: bool = True,
    max_sequences: Optional[int] = None,
    fiber_exe: str = "fiber"
):
    """
    Build nucleic acid encoder dataset from sequence files.
    
    Args:
        input_dir: Directory containing sequence text files
        output_file: Output parquet file path
        seq_type: "dna" or "rna"
        graph_radius: Radius for graph construction (default: 8.0)
        max_neighbors: Max neighbors per node (default: 24)
        max_seq_length: Maximum sequence length to process (default: 500)
        num_workers: Number of parallel workers (None = auto-detect)
        batch_size: Batch size for multiprocessing
        checkpoint_interval: Save checkpoint every N sequences
        resume: Resume from existing checkpoint if available
        verbose: Print progress information
        max_sequences: Maximum number of sequences to process (for testing)
        fiber_exe: Path to X3DNA fiber executable
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    if verbose:
        print("=" * 70)
        print(f"Building {seq_type.upper()} Encoder Dataset")
        print("=" * 70)
        print(f"Input directory: {input_path}")
        print(f"Output file: {output_path}")
        print(f"Sequence type: {seq_type}")
        print(f"Graph radius: {graph_radius} Å")
        print(f"Max neighbors: {max_neighbors}")
        print(f"Max sequence length: {max_seq_length} nt")
        print(f"Fiber executable: {fiber_exe}")
    
    # Step 1: Load all sequences from text files
    if verbose:
        print("\n" + "=" * 70)
        print("Step 1: Loading sequences from text files")
        print("=" * 70)
    
    txt_files = sorted(input_path.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {input_path}")
    
    if verbose:
        print(f"Found {len(txt_files)} text files")
    
    all_sequences = []
    for txt_file in tqdm(txt_files, desc="Reading files", disable=not verbose):
        sequences = load_sequences_from_file(txt_file, seq_type)
        all_sequences.extend(sequences)
        if verbose:
            print(f"  {txt_file.name}: {len(sequences)} sequences")
    
    if verbose:
        print(f"\nTotal sequences loaded: {len(all_sequences)}")
    
    # Limit for testing if requested
    if max_sequences and max_sequences < len(all_sequences):
        all_sequences = all_sequences[:max_sequences]
        if verbose:
            print(f"Limited to {max_sequences} sequences for testing")
    
    # Step 2: Check for existing checkpoint and resume
    if verbose:
        print("\n" + "=" * 70)
        print("Step 2: Processing sequences")
        print("=" * 70)
    
    processed_count = 0
    if resume and output_path.exists():
        try:
            df_existing = pd.read_parquet(output_path)
            processed_count = len(df_existing)
            if verbose:
                print(f"Found existing checkpoint: {processed_count} sequences already processed")
                print(f"Resuming from sequence {processed_count + 1}...")
            
            # Skip already processed sequences
            # Note: This assumes we're processing files in the same order
            if processed_count >= len(all_sequences):
                if verbose:
                    print("All sequences already processed!")
                return pd.read_parquet(output_path)
            
            all_sequences = all_sequences[processed_count:]
        except Exception as e:
            if verbose:
                print(f"Could not load existing checkpoint: {e}")
                print("Starting from scratch...")
    
    if len(all_sequences) == 0:
        if verbose:
            print("No sequences to process!")
        return pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    if verbose:
        print(f"Using {num_workers} parallel workers with batch size {batch_size}")
        print(f"Checkpoint interval: {checkpoint_interval} sequences")
    
    # Create temporary working directory
    temp_workdir = Path(tempfile.mkdtemp(prefix="nacid_work_"))
    
    try:
        # Process sequences with multiprocessing and checkpointing
        data_list = []
        success_count = 0
        fail_count = 0
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if num_workers > 1:
            # Split into batches
            batches = [all_sequences[i:i + batch_size] for i in range(0, len(all_sequences), batch_size)]
            
            process_func = partial(
                process_batch,
                workdir=temp_workdir,
                graph_radius=graph_radius,
                max_neighbors=max_neighbors,
                max_seq_length=max_seq_length,
                fiber_exe=fiber_exe
            )
            
            with Pool(num_workers) as pool:
                for batch_results in tqdm(
                    pool.imap(process_func, batches),
                    total=len(batches),
                    desc="Processing sequences",
                    disable=not verbose
                ):
                    for data in batch_results:
                        if data is not None:
                            data_list.append(data)
                            success_count += 1
                        else:
                            fail_count += 1
                    
                    # Save checkpoint every checkpoint_interval sequences
                    if len(data_list) >= checkpoint_interval:
                        _save_checkpoint(output_path, data_list, resume, verbose)
                        data_list = []
        else:
            # Single process mode
            for seq, seq_idx, seq_type_val, source_file in tqdm(
                all_sequences,
                desc="Processing sequences",
                disable=not verbose
            ):
                data = process_sequence(
                    seq, seq_idx, seq_type_val, source_file,
                    temp_workdir, graph_radius, max_neighbors, max_seq_length, fiber_exe
                )
                if data is not None:
                    data_list.append(data)
                    success_count += 1
                else:
                    fail_count += 1
                
                # Save checkpoint every checkpoint_interval sequences
                if len(data_list) >= checkpoint_interval:
                    _save_checkpoint(output_path, data_list, resume, verbose)
                    data_list = []
        
        # Save final batch if any remaining
        if data_list:
            _save_checkpoint(output_path, data_list, resume, verbose)
    
    finally:
        # Clean up temporary working directory
        if temp_workdir.exists():
            shutil.rmtree(temp_workdir, ignore_errors=True)
    
    if verbose:
        print(f"\nSuccessfully processed: {success_count} sequences")
        if fail_count > 0:
            print(f"Failed to process: {fail_count} sequences")
    
    # Load and return final dataset
    if not output_path.exists():
        if verbose:
            print("\n⚠️  No sequences were successfully processed. Output file not created.")
        return pd.DataFrame()
    
    df_output = pd.read_parquet(output_path)
    
    if verbose:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nFinal dataset: {len(df_output)} total sequences")
        print(f"Saved to: {output_path}")
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Print summary statistics
        if len(df_output) > 0:
            print(f"\nDataset statistics:")
            print(f"  Mean sequence length: {df_output['seq_length'].mean():.1f}")
            print(f"  Median sequence length: {df_output['seq_length'].median():.1f}")
            print(f"  Min sequence length: {df_output['seq_length'].min()}")
            print(f"  Max sequence length: {df_output['seq_length'].max()}")
            print(f"  Mean atoms per sequence: {df_output['num_atoms'].mean():.1f}")
            print(f"  Median atoms per sequence: {df_output['num_atoms'].median():.1f}")
            
            # Modality distribution
            modality_counts = df_output['modality'].value_counts()
            print(f"\n  Modalities:")
            for modality, count in modality_counts.items():
                print(f"    {modality}: {count} ({count/len(df_output)*100:.1f}%)")
    
    print(f"\n✅ {seq_type.upper()} encoder dataset building complete!")
    return df_output


def _save_checkpoint(output_file: Path, new_data: List[Dict[str, Any]], append: bool, verbose: bool):
    """Save checkpoint by appending new data to existing parquet file."""
    if not new_data:
        return
    
    df_new = pd.DataFrame(new_data)
    
    # Convert to PyArrow table with explicit schema to ensure consistent types
    table_new = pa.Table.from_pandas(df_new, schema=PARQUET_SCHEMA, safe=False)
    
    if append and output_file.exists():
        # Read existing file and append
        table_existing = pq.read_table(output_file)
        table_combined = pa.concat_tables([table_existing, table_new])
        pq.write_table(table_combined, output_file)
        if verbose:
            print(f"  Checkpoint saved: {len(table_combined)} total sequences")
    else:
        # Create new file
        pq.write_table(table_new, output_file)
        if verbose:
            print(f"  Checkpoint saved: {len(table_new)} sequences")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Build nucleic acid encoder dataset from sequence files"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing sequence text files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output parquet file path'
    )
    parser.add_argument(
        '--seq_type',
        type=str,
        choices=['dna', 'rna'],
        required=True,
        help='Sequence type: dna or rna'
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
        '--max_seq_length',
        type=int,
        default=200,
        help='Maximum sequence length in nucleotides (default: 500)'
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
        default=1000,
        help='Save checkpoint every N sequences (default: 1000)'
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
        '--max_sequences',
        type=int,
        default=None,
        help='Maximum number of sequences to process (for testing)'
    )
    parser.add_argument(
        '--fiber_exe',
        type=str,
        default='fiber',
        help='Path to X3DNA fiber executable (default: fiber)'
    )
    
    args = parser.parse_args()
    
    build_nacid_encoder_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        seq_type=args.seq_type,
        graph_radius=args.graph_radius,
        max_neighbors=args.max_neighbors,
        max_seq_length=args.max_seq_length,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        verbose=not args.quiet,
        max_sequences=args.max_sequences,
        fiber_exe=args.fiber_exe
    )


if __name__ == '__main__':
    main()

