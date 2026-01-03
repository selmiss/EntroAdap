#!/usr/bin/env python3
"""
Validate nucleic acid encoder datasets.
Checks data format, schema, and basic statistics.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def validate_dataset(parquet_file: str, verbose: bool = True):
    """
    Validate nucleic acid encoder dataset.
    
    Args:
        parquet_file: Path to parquet file
        verbose: Print detailed information
        
    Returns:
        True if valid, False otherwise
    """
    file_path = Path(parquet_file)
    
    if not file_path.exists():
        print(f"❌ File not found: {parquet_file}")
        return False
    
    if verbose:
        print(f"Validating: {parquet_file}")
        print("=" * 70)
    
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"❌ Failed to load parquet file: {e}")
        return False
    
    # Check schema
    expected_columns = [
        'modality', 'seq_id', 'source_file', 'sequence', 'seq_length',
        'num_atoms', 'node_feat', 'coordinates', 'edge_index', 'edge_attr'
    ]
    
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    
    if verbose:
        print(f"✓ Schema valid")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        print(f"  File size: {file_path.stat().st_size / (1024**2):.2f} MB")
    
    # Check each sample
    errors = []
    for idx, row in df.iterrows():
        try:
            # Check modality
            if row['modality'] not in ['dna', 'rna']:
                errors.append(f"Row {idx}: Invalid modality '{row['modality']}'")
            
            # Check node features
            node_feat = row['node_feat']
            if not isinstance(node_feat, np.ndarray):
                errors.append(f"Row {idx}: node_feat is not ndarray")
                continue
            
            node_feat_2d = np.stack(node_feat)
            num_atoms = row['num_atoms']
            
            if node_feat_2d.shape[0] != num_atoms:
                errors.append(f"Row {idx}: node_feat has {node_feat_2d.shape[0]} atoms, expected {num_atoms}")
            
            if node_feat_2d.shape[1] != 7:
                errors.append(f"Row {idx}: node_feat has {node_feat_2d.shape[1]} features, expected 7")
            
            # Check coordinates
            coords = np.stack(row['coordinates'])
            if coords.shape != (num_atoms, 3):
                errors.append(f"Row {idx}: coordinates shape {coords.shape}, expected {(num_atoms, 3)}")
            
            # Check edge data
            edge_index = np.stack(row['edge_index'])
            edge_attr = np.stack(row['edge_attr'])
            
            if edge_index.shape[0] != 2:
                errors.append(f"Row {idx}: edge_index has {edge_index.shape[0]} rows, expected 2")
            
            if edge_attr.shape[1] != 1:
                errors.append(f"Row {idx}: edge_attr has {edge_attr.shape[1]} features, expected 1")
            
            if edge_index.shape[1] != edge_attr.shape[0]:
                errors.append(f"Row {idx}: edge_index has {edge_index.shape[1]} edges, edge_attr has {edge_attr.shape[0]}")
            
            # Check feature 0 is atomic number (should be in valid range)
            atomic_nums = node_feat_2d[:, 0]
            if np.any((atomic_nums < 0) | (atomic_nums > 118)):
                errors.append(f"Row {idx}: Invalid atomic numbers")
            
        except Exception as e:
            errors.append(f"Row {idx}: {e}")
    
    if errors:
        print(f"❌ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    
    if verbose:
        print(f"✓ All {len(df)} samples valid")
        
        # Statistics
        print(f"\nDataset Statistics:")
        print(f"  Modality distribution:")
        for modality, count in df['modality'].value_counts().items():
            print(f"    {modality}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\n  Sequence length:")
        print(f"    Mean: {df['seq_length'].mean():.1f}")
        print(f"    Median: {df['seq_length'].median():.1f}")
        print(f"    Range: [{df['seq_length'].min()}, {df['seq_length'].max()}]")
        
        print(f"\n  Number of atoms:")
        print(f"    Mean: {df['num_atoms'].mean():.1f}")
        print(f"    Median: {df['num_atoms'].median():.1f}")
        print(f"    Range: [{df['num_atoms'].min()}, {df['num_atoms'].max()}]")
        
        # Sample data
        print(f"\nSample record (first):")
        sample = df.iloc[0]
        print(f"  modality: {sample['modality']}")
        print(f"  seq_id: {sample['seq_id']}")
        print(f"  sequence: {sample['sequence'][:50]}...")
        print(f"  seq_length: {sample['seq_length']}")
        print(f"  num_atoms: {sample['num_atoms']}")
        
        node_feat = np.stack(sample['node_feat'])
        coords = np.stack(sample['coordinates'])
        edge_index = np.stack(sample['edge_index'])
        edge_attr = np.stack(sample['edge_attr'])
        
        print(f"  node_feat shape: {node_feat.shape}")
        print(f"  coordinates shape: {coords.shape}")
        print(f"  edge_index shape: {edge_index.shape}")
        print(f"  edge_attr shape: {edge_attr.shape}")
        print(f"  First atom features: {node_feat[0]} (atomic_num={node_feat[0,0]})")
    
    print(f"\n✅ Dataset is valid!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate nucleic acid encoder dataset")
    parser.add_argument('parquet_file', type=str, help='Path to parquet file')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    is_valid = validate_dataset(args.parquet_file, verbose=not args.quiet)
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()

