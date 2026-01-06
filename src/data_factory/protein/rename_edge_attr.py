#!/usr/bin/env python
"""
Script to rename 'edge_attr' to 'edge_feat_dist' in protein SFT data files.

This script processes protein SFT parquet files and renames the 'edge_attr' key
to 'edge_feat_dist' within the 'structure' field to match the unified format
described in docs/README.md.

Usage:
    python rename_edge_attr.py --input_dir data/sft/protein --backup
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from tqdm import tqdm


def rename_edge_attr_in_structure(structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rename 'edge_attr' to 'edge_feat_dist' in a structure dictionary.
    
    Args:
        structure: Dictionary containing structure data with 'edge_attr' key
        
    Returns:
        Modified structure dictionary with 'edge_feat_dist' instead of 'edge_attr'
    """
    if 'edge_attr' in structure:
        structure['edge_feat_dist'] = structure.pop('edge_attr')
    return structure


def process_parquet_file(input_path: str, output_path: str, create_backup: bool = True):
    """
    Process a single parquet file and rename edge_attr to edge_feat_dist.
    
    Handles two formats:
    1. Nested: edge_attr inside a 'structure' dictionary column
    2. Flat: edge_attr as a direct column
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        create_backup: Whether to create a backup of the original file
    """
    print(f"Processing: {input_path}")
    
    # Create backup if requested
    if create_backup:
        backup_path = input_path + ".backup"
        if not os.path.exists(backup_path):
            print(f"  Creating backup: {backup_path}")
            shutil.copy2(input_path, backup_path)
        else:
            print(f"  Backup already exists: {backup_path}")
    
    # Load the parquet file
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")
    
    modified = False
    
    # Case 1: Check if 'edge_attr' exists as a direct column (flat format)
    if 'edge_attr' in df.columns:
        print(f"  Found 'edge_attr' as direct column (flat format)")
        if 'edge_feat_dist' in df.columns:
            print(f"  Warning: 'edge_feat_dist' already exists. Skipping to avoid duplication.")
            return
        print(f"  Renaming column 'edge_attr' to 'edge_feat_dist'...")
        df.rename(columns={'edge_attr': 'edge_feat_dist'}, inplace=True)
        modified = True
        print(f"  ✓ Successfully renamed column")
    
    # Case 2: Check if 'structure' column exists with nested edge_attr
    elif 'structure' in df.columns:
        print(f"  Found 'structure' column (nested format)")
        # Check first row to see if edge_attr exists
        if len(df) > 0:
            sample_structure = df.iloc[0]['structure']
            if isinstance(sample_structure, dict):
                if 'edge_attr' not in sample_structure:
                    print(f"  Info: 'edge_attr' not found in structure. File may already be processed.")
                    return
                if 'edge_feat_dist' in sample_structure:
                    print(f"  Info: 'edge_feat_dist' already exists. Skipping to avoid duplication.")
                    return
                
                # Process each row
                print(f"  Renaming 'edge_attr' to 'edge_feat_dist' in structure field...")
                df['structure'] = df['structure'].apply(
                    lambda s: rename_edge_attr_in_structure(s) if isinstance(s, dict) else s
                )
                modified = True
                
                # Verify the change
                sample_structure = df.iloc[0]['structure']
                if isinstance(sample_structure, dict):
                    if 'edge_feat_dist' in sample_structure and 'edge_attr' not in sample_structure:
                        print(f"  ✓ Successfully renamed edge_attr to edge_feat_dist in structure")
                    else:
                        print(f"  ✗ Error: Renaming may have failed")
                        print(f"    Keys in structure: {list(sample_structure.keys())}")
                        return
    else:
        print(f"  Warning: Neither 'edge_attr' column nor 'structure' column found. Skipping.")
        return
    
    # Save the modified dataframe if changes were made
    if modified:
        print(f"  Saving to: {output_path}")
        df.to_parquet(output_path, index=False)
        print(f"  ✓ Successfully saved {len(df)} rows")
    else:
        print(f"  No changes made")


def main():
    parser = argparse.ArgumentParser(
        description="Rename 'edge_attr' to 'edge_feat_dist' in protein SFT data files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/sft/protein",
        help="Directory containing protein SFT parquet files (default: data/sft/protein)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input_dir, modifies files in-place)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup files before modifying (adds .backup extension)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.parquet",
        help="File pattern to match (default: *.parquet)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    parquet_files = sorted(input_dir.glob(args.pattern))
    if not parquet_files:
        print(f"No files matching pattern '{args.pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(parquet_files)} file(s) to process")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Backup enabled:   {args.backup}")
    print("-" * 80)
    
    # Process each file
    for input_file in parquet_files:
        output_file = output_dir / input_file.name
        try:
            process_parquet_file(
                str(input_file),
                str(output_file),
                create_backup=args.backup
            )
            print()
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            print()
            continue
    
    print("-" * 80)
    print("Processing complete!")
    
    # Summary
    if args.backup:
        backup_files = list(input_dir.glob("*.backup"))
        if backup_files:
            print(f"\nBackup files created: {len(backup_files)}")
            print("To remove backups: rm data/sft/protein/*.backup")


if __name__ == "__main__":
    main()

