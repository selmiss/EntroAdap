#!/usr/bin/env python3
"""
Fix parquet schema inconsistencies between datasets.
Converts large_list types to regular list types for compatibility.
"""

import sys
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))


# Define consistent schema - must match build_nacid_encoder_data.py
CONSISTENT_SCHEMA = pa.schema([
    ('modality', pa.string()),
    ('seq_id', pa.string()),
    ('source_file', pa.string()),
    ('sequence', pa.string()),
    ('seq_length', pa.int64()),
    ('num_atoms', pa.int64()),
    ('node_feat', pa.list_(pa.list_(pa.int64()))),
    ('coordinates', pa.list_(pa.list_(pa.float64()))),
    ('edge_index', pa.list_(pa.list_(pa.int64()))),
    ('edge_attr', pa.list_(pa.list_(pa.float64()))),
])


def fix_parquet_schema(input_file: str, output_file: str = None, verbose: bool = True):
    """
    Fix parquet file schema to use consistent list types.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output file (if None, overwrites input)
        verbose: Print progress
    """
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file else input_path
    
    if verbose:
        print(f"Fixing schema for: {input_path}")
    
    # Read the existing table
    table = pq.read_table(input_path)
    
    if verbose:
        print(f"  Original schema:")
        for field in table.schema:
            print(f"    {field.name}: {field.type}")
    
    # Check if schema needs fixing
    needs_fixing = False
    for field in table.schema:
        if str(field.type).startswith('large_list'):
            needs_fixing = True
            break
    
    if not needs_fixing:
        if verbose:
            print("  ✓ Schema is already correct (no large_list types)")
        return
    
    # Remove unwanted columns like __index_level_0__
    columns_to_keep = [field.name for field in CONSISTENT_SCHEMA]
    columns_to_drop = [name for name in table.column_names if name not in columns_to_keep]
    
    if columns_to_drop and verbose:
        print(f"  Dropping columns: {columns_to_drop}")
    
    # Keep only the columns we want
    table_filtered = table.select(columns_to_keep)
    
    # Cast to consistent schema
    try:
        # Cast table to new schema
        table_fixed = table_filtered.cast(CONSISTENT_SCHEMA)
        
        if verbose:
            print(f"  New schema:")
            for field in table_fixed.schema:
                print(f"    {field.name}: {field.type}")
        
        # Write to output file
        pq.write_table(table_fixed, output_path)
        
        if verbose:
            print(f"  ✓ Fixed schema written to: {output_path}")
            print(f"  File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
    
    except Exception as e:
        print(f"  ✗ Error fixing schema: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Fix parquet schema inconsistencies (large_list -> list)"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input parquet file path'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output file path (default: overwrite input)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )
    
    args = parser.parse_args()
    
    fix_parquet_schema(
        input_file=args.input_file,
        output_file=args.output_file,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()

