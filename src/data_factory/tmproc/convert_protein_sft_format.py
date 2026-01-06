#!/usr/bin/env python3
"""
Convert protein SFT dataset to match DNA/RNA format.

Changes:
1. Flatten nested 'structure' dict into top-level columns
2. Rename 'coordinates' to 'pos'
3. Convert 'instruction' string to 'messages' list format
4. Remove unnecessary metadata columns (uniprot_id, pdb_id, method, resolution, chains)
5. Ensure consistent column order: modality, sequence, messages, node_feat, pos, edge_index, edge_attr

Input format (protein_sft.parquet):
- uniprot_id, pdb_id, method, resolution, chains, sequence, structure (dict), instruction, modality

Output format (protein_sft_flat.parquet):
- modality, sequence, messages, node_feat, pos, edge_index, edge_attr
(matching DNA/RNA format)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def instruction_to_messages(instruction: str) -> List[Dict[str, str]]:
    """
    Convert instruction string to messages format matching DNA/RNA data.
    
    Args:
        instruction: Long-form protein description
        
    Returns:
        List of message dicts with system, user, and assistant messages
    """
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant specializing in protein analysis and biology. The instruction that describes a task is given, paired with protein structure. Write a response that appropriately completes the request."
    }
    
    # For protein descriptions, we'll create a simple Q&A format
    user_message = {
        "role": "user",
        "content": "Could you provide a detailed description of the protein structure <STRUCTURE> and its function?"
    }
    
    assistant_message = {
        "role": "assistant",
        "content": instruction
    }
    
    return [system_message, user_message, assistant_message]


def flatten_structure_dict(row: pd.Series) -> Dict[str, Any]:
    """
    Flatten nested structure dict and rename fields to match DNA/RNA format.
    
    Args:
        row: DataFrame row with nested 'structure' column
        
    Returns:
        Dict with flattened fields
    """
    structure = row['structure']
    
    # Extract and rename fields
    result = {
        'modality': row['modality'],
        'sequence': row['sequence'],
        'messages': instruction_to_messages(row['instruction']),
        'node_feat': structure['node_feat'],
        'pos': structure['coordinates'],  # Rename coordinates -> pos
        'edge_index': structure['edge_index'],
        'edge_attr': structure['edge_attr'],
    }
    
    return result


def convert_protein_sft_format(
    input_file: str,
    output_file: str,
    verbose: bool = True
):
    """
    Convert protein SFT dataset to match DNA/RNA format.
    
    Args:
        input_file: Path to input protein_sft.parquet
        output_file: Path to output protein_sft_flat.parquet
        verbose: Print progress information
    """
    if verbose:
        print("=" * 80)
        print("PROTEIN SFT FORMAT CONVERTER")
        print("=" * 80)
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        print()
    
    # Read input file
    if verbose:
        print("Reading input file...")
    df = pd.read_parquet(input_file)
    
    if verbose:
        print(f"  Loaded {len(df)} samples")
        print(f"  Input columns: {list(df.columns)}")
        print()
    
    # Convert each row
    if verbose:
        print("Converting format...")
    
    converted_rows = []
    for idx, row in df.iterrows():
        try:
            converted = flatten_structure_dict(row)
            converted_rows.append(converted)
            
            if verbose and (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} samples", end='\r')
        except Exception as e:
            print(f"\nError processing row {idx}: {e}")
            continue
    
    if verbose:
        print(f"  Processed {len(converted_rows)}/{len(df)} samples")
        print()
    
    # Create new DataFrame
    df_converted = pd.DataFrame(converted_rows)
    
    # Reorder columns to match DNA/RNA format
    column_order = ['modality', 'sequence', 'messages', 'node_feat', 'pos', 'edge_index', 'edge_attr']
    df_converted = df_converted[column_order]
    
    if verbose:
        print("Output DataFrame:")
        print(f"  Shape: {df_converted.shape}")
        print(f"  Columns: {list(df_converted.columns)}")
        print()
    
    # Save to parquet
    if verbose:
        print(f"Saving to {output_file}...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_converted.to_parquet(
        output_file,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    if verbose:
        output_size_mb = output_path.stat().st_size / (1024**2)
        print(f"  Saved successfully ({output_size_mb:.2f} MB)")
        print()
        print("=" * 80)
        print("CONVERSION COMPLETE")
        print("=" * 80)
        print()
        print("Sample row (first entry):")
        print("-" * 80)
        for col in df_converted.columns:
            val = df_converted.iloc[0][col]
            if isinstance(val, (list, np.ndarray)) and len(str(val)) > 200:
                print(f"  {col}: {type(val).__name__} (length={len(val)})")
            else:
                print(f"  {col}: {val}")
        print("-" * 80)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python convert_protein_sft_format.py <input_file> [output_file]")
        print()
        print("Example:")
        print("  python convert_protein_sft_format.py data/sft/protein/protein_sft.parquet data/sft/protein/protein_sft_flat.parquet")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Default: add '_flat' suffix before extension
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_flat{input_path.suffix}")
    
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    convert_protein_sft_format(input_file, output_file, verbose=True)


if __name__ == "__main__":
    main()

