#!/usr/bin/env python3
"""
Build nucleic acid (DNA/RNA) SFT (Supervised Fine-Tuning) dataset from JSON files.

This script:
1. Reads DNA/RNA JSON files from data/nacid/raw/{dna,rna}
2. Extracts sequences and instruction exchanges
3. Generates structural features using X3DNA
4. Saves combined dataset in Parquet format
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data_factory.nacid.seq_to_feature import sequence_to_features, sequence_to_features_windowed


def extract_samples_from_json(json_file: Path, modality: str) -> List[Dict[str, Any]]:
    """
    Extract samples from DNA/RNA JSON file.
    
    Args:
        json_file: Path to JSON file
        modality: 'dna' or 'rna'
        
    Returns:
        List of dicts with sample info (sequence, messages, etc.)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    task_name = data.get('task_name', '')
    task_type = data.get('task_type', '')
    samples = data.get('samples', [])
    
    extracted_samples = []
    for sample in samples:
        # Extract sequence
        sequences = sample.get('sequences', [])
        if not sequences or len(sequences) == 0:
            continue
        sequence = sequences[0]  # Take first sequence
        
        # Extract exchanges and convert to messages format
        exchanges = sample.get('exchanges', [])
        if not exchanges or len(exchanges) == 0:
            continue
        
        # Build messages array in the same format as molecular data
        # Start with system message
        messages = [
            {
                'role': 'system',
                'content': f'You are a helpful assistant specializing in {modality.upper()} analysis and biology. The instruction that describes a task is given, paired with nucleic acid sequences. Write a response that appropriately completes the request.'
            }
        ]
        
        # Add user-assistant exchanges
        for exchange in exchanges:
            role = exchange.get('role', '').lower()
            message = exchange.get('message', '')
            
            if role == 'user':
                messages.append({
                    'role': 'user',
                    'content': message
                })
            elif role == 'assistant':
                messages.append({
                    'role': 'assistant',
                    'content': message
                })
        
        # Extract metadata (for internal processing)
        sample_id = sample.get('sample_id', 0)
        
        extracted_samples.append({
            'modality': modality,
            'sample_id': sample_id,  # Keep for internal tracking during processing
            'sequence': sequence,
            'messages': messages,
        })
    
    return extracted_samples


def process_structure(
    seq: str,
    seq_id: str,
    seq_type: str,
    workdir: Path,
    graph_radius: float = 8.0,
    max_neighbors: int = 24,
    max_seq_length: int = 500,
    fiber_exe: str = "fiber",
    use_windowing: bool = False,
    window_size: int = 500,
    window_overlap: int = 50,
    rna_single_strand: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Process a single nucleic acid sequence and extract structural features.
    
    Args:
        seq: Nucleic acid sequence
        seq_id: Unique sequence ID
        seq_type: 'dna' or 'rna'
        workdir: Working directory for temporary files
        graph_radius: Radius for graph construction
        max_neighbors: Max neighbors per node
        max_seq_length: Maximum sequence length (for non-windowed mode)
        fiber_exe: Path to fiber executable
        use_windowing: Use sliding window approach for long sequences
        window_size: Size of each window (for windowed mode)
        window_overlap: Overlap between windows (for windowed mode)
        rna_single_strand: Generate RNA as single-stranded (default: True, biologically correct)
        
    Returns:
        Dictionary with structural features, or None if processing fails
    """
    try:
        # Clean and prepare sequence
        seq = seq.strip().upper()
        
        # For RNA, convert T to U (common issue: RNA sequences stored in DNA format)
        if seq_type == "rna":
            seq = seq.replace('T', 'U')
        
        # Choose processing method based on use_windowing flag
        if use_windowing:
            # Use windowed processing for long sequences
            data = sequence_to_features_windowed(
                seq=seq,
                seq_id=seq_id,
                seq_type=seq_type,
                workdir=str(workdir),
                graph_radius=graph_radius,
                max_neighbors=max_neighbors,
                window_size=window_size,
                overlap=window_overlap,
                fiber_exe=fiber_exe,
                rna_single_strand=rna_single_strand,
            )
        else:
            # Use standard processing (truncates to max_seq_length)
            data = sequence_to_features(
                seq=seq,
                seq_id=seq_id,
                seq_type=seq_type,
                workdir=str(workdir),
                graph_radius=graph_radius,
                max_neighbors=max_neighbors,
                max_seq_length=max_seq_length,
                fiber_exe=fiber_exe,
                rna_single_strand=rna_single_strand,
            )
        
        if data is None:
            return None
        
        # Convert to list format for Parquet storage
        # Use field names that match molecular data format
        result = {
            'seq_length': data['seq_length'],
            'num_atoms': data['num_atoms'],
            'node_feat': data['node_feat'].tolist(),
            'coordinates': data['coordinates'].tolist(),  # Keep as coordinates internally, will rename when flattening
            'edge_index': data['edge_index'].tolist(),
            'edge_attr': data['edge_attr'].tolist(),  # Keep as edge_attr internally, will rename when flattening
        }
        
        return result
    
    except Exception as e:
        # Silently skip sequences that fail to process
        return None


def build_nacid_sft_dataset(
    raw_data_dir: str,
    output_dir: str,
    modality: str = "dna",
    graph_radius: float = 8.0,
    max_neighbors: int = 24,
    max_seq_length: int = 500,
    max_samples: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    max_records_per_file: int = 1000,
    fiber_exe: str = "fiber",
    use_windowing: bool = False,
    window_size: int = 500,
    window_overlap: int = 50,
    rna_single_strand: bool = True,
    verbose: bool = True,
):
    """
    Build nucleic acid SFT dataset from JSON files.
    
    Args:
        raw_data_dir: Directory containing raw JSON files (data/nacid/raw)
        output_dir: Output directory for Parquet files
        modality: 'dna' or 'rna'
        graph_radius: Radius for graph construction (default: 8.0)
        max_neighbors: Max neighbors per node (default: 24)
        max_seq_length: Maximum sequence length for non-windowed mode (default: 500)
        max_samples: Maximum number of samples to process (for testing)
        sample_fraction: Fraction of samples to randomly select (e.g., 0.05 for 1/20)
        max_records_per_file: Maximum records per Parquet file (default: 1000)
        fiber_exe: Path to X3DNA fiber executable
        use_windowing: Use sliding window approach for long sequences (default: False)
        window_size: Size of each window in nt for windowed mode (default: 500)
        window_overlap: Overlap between windows in nt for windowed mode (default: 50)
        rna_single_strand: Generate RNA as single-stranded (default: True)
        verbose: Print progress information
    """
    raw_dir = Path(raw_data_dir)
    output_path = Path(output_dir)
    
    if verbose:
        print("=" * 70)
        print(f"Building {modality.upper()} SFT Dataset")
        print("=" * 70)
        print(f"Raw data directory: {raw_dir}")
        print(f"Output directory: {output_path}")
        print(f"Modality: {modality}")
        print(f"Graph radius: {graph_radius} Å")
        print(f"Max neighbors: {max_neighbors}")
        if use_windowing:
            print(f"Processing mode: Windowed (window_size={window_size}, overlap={window_overlap})")
        else:
            print(f"Processing mode: Standard (max_seq_length={max_seq_length})")
        if modality == "rna":
            print(f"RNA single-strand: {rna_single_strand} ({'ENABLED - 50% fewer atoms' if rna_single_strand else 'DISABLED - double-stranded'})")
        print(f"Max records per file: {max_records_per_file}")
        print(f"Fiber executable: {fiber_exe}")
    
    # Step 1: Extract samples from JSON files
    if verbose:
        print("\n" + "=" * 70)
        print("Step 1: Extracting samples from JSON files")
        print("=" * 70)
    
    modality_dir = raw_dir / modality
    if not modality_dir.exists():
        raise ValueError(f"Modality directory not found: {modality_dir}")
    
    json_files = sorted(modality_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {modality_dir}")
    
    if verbose:
        print(f"Found {len(json_files)} JSON files:")
        for f in json_files[:5]:  # Show first 5
            print(f"  - {f.name}")
        if len(json_files) > 5:
            print(f"  ... and {len(json_files) - 5} more")
    
    all_samples = []
    for json_file in tqdm(json_files, desc="Reading JSON files", disable=not verbose):
        samples = extract_samples_from_json(json_file, modality)
        all_samples.extend(samples)
    
    if verbose:
        print(f"\nTotal samples extracted: {len(all_samples)}")
    
    # Random sampling if requested
    if sample_fraction is not None and 0 < sample_fraction < 1:
        np.random.seed(42)  # For reproducibility
        num_to_sample = int(len(all_samples) * sample_fraction)
        indices = np.random.choice(len(all_samples), size=num_to_sample, replace=False)
        all_samples = [all_samples[i] for i in sorted(indices)]
        if verbose:
            print(f"Randomly sampled {sample_fraction*100:.1f}% ({len(all_samples)} samples)")
    
    # Limit for testing if requested
    if max_samples and max_samples < len(all_samples):
        all_samples = all_samples[:max_samples]
        if verbose:
            print(f"Limited to {max_samples} samples for testing")
    
    if len(all_samples) == 0:
        print("No samples found!")
        return
    
    # Step 2: Process structures and add structural features
    if verbose:
        print("\n" + "=" * 70)
        print("Step 2: Processing sequences and extracting structural features")
        print("=" * 70)
    
    # Create temporary working directory
    temp_workdir = Path(tempfile.mkdtemp(prefix=f"nacid_sft_{modality}_"))
    
    samples_with_structure = []
    structure_success = 0
    structure_failed = 0
    
    try:
        for idx, sample in enumerate(tqdm(all_samples, desc="Processing structures", disable=not verbose)):
            seq = sample['sequence']
            seq_id = f"{modality}_{sample['sample_id']}_{idx}"  # Create unique ID
            
            # Create sequence-specific workdir
            seq_workdir = temp_workdir / seq_id
            seq_workdir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Extract structural features
                structure_data = process_structure(
                    seq=seq,
                    seq_id=seq_id,
                    seq_type=modality,
                    workdir=seq_workdir,
                    graph_radius=graph_radius,
                    max_neighbors=max_neighbors,
                    max_seq_length=max_seq_length,
                    fiber_exe=fiber_exe,
                    use_windowing=use_windowing,
                    window_size=window_size,
                    window_overlap=window_overlap,
                    rna_single_strand=rna_single_strand,
                )
                
                if structure_data is not None:
                    # Add structure data to sample
                    sample['structure'] = structure_data
                    samples_with_structure.append(sample)
                    structure_success += 1
                    
                    # Debug output for atom and edge counts
                    num_atoms = structure_data['num_atoms']
                    num_edges = len(structure_data['edge_index'][0]) if structure_data['edge_index'] else 0
                    if verbose:
                        tqdm.write(f"  {seq_id}: {num_atoms} atoms, {num_edges} edges (seq_len={structure_data['seq_length']})")
                else:
                    structure_failed += 1
                    if verbose:
                        tqdm.write(f"  {seq_id}: Structure processing returned None")
            except Exception as e:
                if verbose:
                    tqdm.write(f"  Error processing {seq_id}: {str(e)}")
                structure_failed += 1
            finally:
                # Clean up sequence-specific workdir
                if seq_workdir.exists():
                    shutil.rmtree(seq_workdir, ignore_errors=True)
    
    finally:
        # Clean up temporary working directory
        if temp_workdir.exists():
            shutil.rmtree(temp_workdir, ignore_errors=True)
    
    if verbose:
        print(f"\nStructure processing results:")
        print(f"  Success: {structure_success}/{len(all_samples)}")
        print(f"  Failed: {structure_failed}/{len(all_samples)}")
    
    if len(samples_with_structure) == 0:
        print("No samples with valid structure data!")
        return
    
    # Step 3: Flatten structure and prepare for saving
    if verbose:
        print("\n" + "=" * 70)
        print("Step 3: Flattening structure and preparing data")
        print("=" * 70)
    
    # Flatten structure fields to match molecular data format
    flattened_samples = []
    for sample in samples_with_structure:
        structure = sample.pop('structure')  # Remove nested structure
        
        # Create flattened sample with only fields compatible with molecular data
        flattened = {
            'modality': sample['modality'],
            'sequence': sample['sequence'],  # Keep sequence as the identifier (like SMILES for molecules)
            'messages': sample['messages'],
            # Flatten structure fields to top level
            'node_feat': structure['node_feat'],
            'pos': structure['coordinates'],  # Rename coordinates -> pos to match molecular format
            'edge_index': structure['edge_index'],
            'edge_feat_dist': structure['edge_attr'],  # Rename edge_attr -> edge_feat_dist to match molecular format
        }
        flattened_samples.append(flattened)
    
    # Step 4: Save to Parquet files with splitting
    if verbose:
        print("\n" + "=" * 70)
        print("Step 4: Saving results to Parquet format")
        print("=" * 70)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate number of files needed
    num_files = (len(flattened_samples) + max_records_per_file - 1) // max_records_per_file
    
    saved_files = []
    total_size_mb = 0
    
    for file_idx in range(num_files):
        start_idx = file_idx * max_records_per_file
        end_idx = min((file_idx + 1) * max_records_per_file, len(flattened_samples))
        batch_samples = flattened_samples[start_idx:end_idx]
        
        # Determine output filename
        if num_files == 1:
            output_file = output_path / f"{modality}_sft.parquet"
        else:
            output_file = output_path / f"{modality}_sft_part{file_idx:03d}.parquet"
        
        # Convert to DataFrame and save
        df = pd.DataFrame(batch_samples)
        df.to_parquet(output_file, index=False)
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        total_size_mb += file_size_mb
        saved_files.append(output_file)
        
        if verbose:
            print(f"  Saved {output_file.name}: {len(batch_samples)} records, {file_size_mb:.2f} MB")
    
    if verbose:
        print(f"\n✅ Dataset building complete!")
        print(f"Total records saved: {len(flattened_samples)}")
        print(f"Number of files: {len(saved_files)}")
        print(f"Output directory: {output_path}")
        print(f"Total size: {total_size_mb:.2f} MB")
        
        # Print sample statistics
        if samples_with_structure:
            seq_lengths = []
            num_atoms_list = []
            for s in samples_with_structure:
                if 'structure' in s:
                    seq_lengths.append(s['structure']['seq_length'])
                    num_atoms_list.append(s['structure']['num_atoms'])
            
            if seq_lengths and num_atoms_list:
                print(f"\nDataset statistics:")
                print(f"  Mean sequence length: {sum(seq_lengths)/len(seq_lengths):.1f} nt")
                print(f"  Min sequence length: {min(seq_lengths)} nt")
                print(f"  Max sequence length: {max(seq_lengths)} nt")
                print(f"  Mean atoms per structure: {sum(num_atoms_list)/len(num_atoms_list):.1f}")
                print(f"  Min atoms: {min(num_atoms_list)}")
                print(f"  Max atoms: {max(num_atoms_list)}")
    
    return saved_files


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Build nucleic acid SFT dataset from JSON files"
    )
    parser.add_argument(
        '--raw_data_dir',
        type=str,
        default='data/nacid/raw',
        help='Directory containing raw JSON files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/sft/nacid',
        help='Output directory for Parquet files'
    )
    parser.add_argument(
        '--modality',
        type=str,
        choices=['dna', 'rna'],
        required=True,
        help='Modality: dna or rna'
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
        default=500,
        help='Maximum sequence length in nucleotides (default: 500)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )
    parser.add_argument(
        '--sample_fraction',
        type=float,
        default=None,
        help='Fraction of samples to randomly select (e.g., 0.05 for 1/20)'
    )
    parser.add_argument(
        '--max_records_per_file',
        type=int,
        default=1000,
        help='Maximum records per Parquet file (default: 1000)'
    )
    parser.add_argument(
        '--fiber_exe',
        type=str,
        default='fiber',
        help='Path to X3DNA fiber executable (default: fiber)'
    )
    parser.add_argument(
        '--use_windowing',
        action='store_true',
        help='Use sliding window approach for long sequences (recommended for RNA)'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=500,
        help='Window size in nucleotides for windowed mode (default: 500)'
    )
    parser.add_argument(
        '--window_overlap',
        type=int,
        default=50,
        help='Overlap between windows in nucleotides (default: 50)'
    )
    parser.add_argument(
        '--rna_single_strand',
        action='store_true',
        default=True,
        help='Generate RNA as single-stranded (default: True, biologically correct)'
    )
    parser.add_argument(
        '--rna_double_strand',
        action='store_false',
        dest='rna_single_strand',
        help='Generate RNA as double-stranded (not recommended)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    build_nacid_sft_dataset(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        modality=args.modality,
        graph_radius=args.graph_radius,
        max_neighbors=args.max_neighbors,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
        sample_fraction=args.sample_fraction,
        max_records_per_file=args.max_records_per_file,
        fiber_exe=args.fiber_exe,
        use_windowing=args.use_windowing,
        window_size=args.window_size,
        window_overlap=args.window_overlap,
        rna_single_strand=args.rna_single_strand,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()

