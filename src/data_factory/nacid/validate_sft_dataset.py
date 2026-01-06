#!/usr/bin/env python3
"""
Validate and inspect DNA/RNA SFT datasets.

This script checks the format and content of generated SFT datasets.
"""

import pandas as pd
import sys
from pathlib import Path

def validate_sft_dataset(parquet_file: str, expected_modality: str):
    """Validate a single SFT dataset file."""
    print(f"\n{'='*70}")
    print(f"Validating: {parquet_file}")
    print('='*70)
    
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return False
    
    # Check required columns
    required_cols = ['modality', 'sequence', 'instruction', 'structure']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    print(f"✓ All required columns present: {required_cols}")
    
    # Check modality consistency
    unique_modalities = df['modality'].unique()
    if len(unique_modalities) != 1:
        print(f"⚠️  Warning: Multiple modalities found: {unique_modalities}")
    if expected_modality not in unique_modalities:
        print(f"❌ Expected modality '{expected_modality}' not found")
        return False
    print(f"✓ Modality is '{expected_modality}'")
    
    # Check structure format
    sample_structure = df.iloc[0]['structure']
    if not isinstance(sample_structure, dict):
        print(f"❌ Structure is not a dictionary: {type(sample_structure)}")
        return False
    
    required_structure_keys = ['num_atoms', 'node_feat', 'coordinates', 'edge_index', 'edge_attr']
    missing_keys = [key for key in required_structure_keys if key not in sample_structure]
    if missing_keys:
        print(f"❌ Structure missing required keys: {missing_keys}")
        return False
    print(f"✓ Structure has all required keys: {required_structure_keys}")
    
    # Check data types (note: parquet may convert lists to numpy arrays on load)
    import numpy as np
    if not isinstance(sample_structure['num_atoms'], (int, float, np.integer, np.floating)):
        print(f"❌ num_atoms should be numeric, got {type(sample_structure['num_atoms'])}")
        return False
    if not isinstance(sample_structure['node_feat'], (list, np.ndarray)):
        print(f"❌ node_feat should be list or array, got {type(sample_structure['node_feat'])}")
        return False
    if not isinstance(sample_structure['coordinates'], (list, np.ndarray)):
        print(f"❌ coordinates should be list or array, got {type(sample_structure['coordinates'])}")
        return False
    print(f"✓ Structure data types are correct")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Sequence lengths: min={df['sequence'].str.len().min()}, "
          f"max={df['sequence'].str.len().max()}, "
          f"mean={df['sequence'].str.len().mean():.1f}")
    
    num_atoms_list = [s['num_atoms'] for s in df['structure']]
    print(f"  Num atoms: min={min(num_atoms_list)}, "
          f"max={max(num_atoms_list)}, "
          f"mean={sum(num_atoms_list)/len(num_atoms_list):.1f}")
    
    print(f"\n✅ Validation passed for {parquet_file}")
    return True


def main():
    """Main validation function."""
    data_dir = Path('data/sft/nacid')
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)
    
    print("="*70)
    print("DNA/RNA SFT Dataset Validation")
    print("="*70)
    
    all_passed = True
    
    # Check for DNA files
    dna_files = list(data_dir.glob('dna_sft*.parquet'))
    if not dna_files:
        print("\n⚠️  No DNA SFT files found")
    else:
        print(f"\nFound {len(dna_files)} DNA SFT file(s)")
        for dna_file in sorted(dna_files):
            if not validate_sft_dataset(str(dna_file), 'dna'):
                all_passed = False
    
    # Check for RNA files
    rna_files = list(data_dir.glob('rna_sft*.parquet'))
    if not rna_files:
        print("\n⚠️  No RNA SFT files found")
    else:
        print(f"\nFound {len(rna_files)} RNA SFT file(s)")
        for rna_file in sorted(rna_files):
            if not validate_sft_dataset(str(rna_file), 'rna'):
                all_passed = False
    
    # Final summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ All validations passed!")
    else:
        print("❌ Some validations failed")
        sys.exit(1)


if __name__ == '__main__':
    main()

