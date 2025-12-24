#!/usr/bin/env python3
"""
Script to extract specific keys from JSONL files and convert to parquet format.
Extracts: smiles, brics_gid, cid, iupac_name from pretrain data.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def extract_keys_from_jsonl(input_file, output_file, keys_to_keep):
    """
    Extract specific keys from JSONL file and save as parquet.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output parquet file
        keys_to_keep: List of keys to retain
    """
    print(f"Processing {input_file}...")
    
    data = []
    with open(input_file, 'r') as f:
        for line in tqdm(f, desc=f"Reading {input_file.name}"):
            try:
                record = json.loads(line.strip())
                # Extract only the keys we want
                filtered_record = {key: record.get(key) for key in keys_to_keep if key in record}
                
                # Handle brics_gids -> brics_gid rename if needed
                if 'brics_gids' in record and 'brics_gid' not in record:
                    filtered_record['brics_gid'] = record['brics_gids']
                
                data.append(filtered_record)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Print statistics
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample record:")
    if len(df) > 0:
        print(df.iloc[0].to_dict())
    
    # Save as parquet
    df.to_parquet(output_file, index=False, engine='pyarrow')
    print(f"Saved to {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract specific keys from JSONL pretrain files and convert to parquet"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing input JSONL files (train.jsonl, val.jsonl, test.jsonl)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Single input JSONL file (alternative to --input_dir)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/UWO/zjing29/proj/EntroAdap/data/raw",
        help="Directory to save output parquet files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Single output parquet file (only used with --input_file)"
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs='+',
        default=["smiles", "brics_gid", "cid", "iupac_name"],
        help="Keys to extract from JSONL records"
    )
    
    args = parser.parse_args()
    
    # Single file mode
    if args.input_file:
        input_file = Path(args.input_file)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Determine output file
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # Use same name but with .parquet extension
            output_file = output_dir / input_file.with_suffix('.parquet').name
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        extract_keys_from_jsonl(input_file, output_file, args.keys)
        print("Processing complete!")
        return
    
    # Directory mode (original behavior)
    if not args.input_dir:
        # Default to DQ-Former pretrain directory if nothing specified
        args.input_dir = "/home/UWO/zjing29/proj/DQ-Former/data/pretrain"
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = ['train', 'val', 'test']
    processed_any = False
    
    for split in splits:
        input_file = input_dir / f"{split}.jsonl"
        output_file = output_dir / f"{split}.parquet"
        
        if not input_file.exists():
            print(f"Warning: {input_file} does not exist, skipping...")
            continue
        
        extract_keys_from_jsonl(input_file, output_file, args.keys)
        processed_any = True
    
    if processed_any:
        print("All files processed successfully!")
    else:
        print(f"No split files (train.jsonl, val.jsonl, test.jsonl) found in {input_dir}")
        print("Use --input_file to process a single JSONL file instead.")


if __name__ == "__main__":
    main()

