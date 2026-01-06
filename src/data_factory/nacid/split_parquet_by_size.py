#!/usr/bin/env python3
"""
Split a large parquet file into smaller files based on target size.
Uses efficient row-by-row processing to avoid loading entire file into memory.
"""

import argparse
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm


def split_parquet_by_size(
    input_file: str,
    target_size_mb: float = 500,
    output_dir: str = None,
    verbose: bool = True
):
    """
    Split a parquet file into smaller files based on target size.
    Uses efficient streaming approach to handle large files.
    
    Args:
        input_file: Path to input parquet file
        target_size_mb: Target size for each output file in MB
        output_dir: Output directory (defaults to same as input)
        verbose: Print progress information
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get original file size
    original_size_mb = input_path.stat().st_size / (1024 * 1024)
    
    if verbose:
        print(f"=" * 70)
        print(f"Parquet File Splitter")
        print(f"=" * 70)
        print(f"Original file: {input_path.name}")
        print(f"Original size: {original_size_mb:.2f} MB")
        print(f"Target size per file: {target_size_mb} MB")
        print(f"\nAnalyzing file structure...")
    
    # Open parquet file and get metadata
    parquet_file = pq.ParquetFile(input_path)
    total_records = parquet_file.metadata.num_rows
    
    if verbose:
        print(f"Total records: {total_records}")
        print(f"Number of row groups: {parquet_file.num_row_groups}")
    
    # Estimate records per file based on size ratio
    estimated_records_per_file = max(1, int(total_records * target_size_mb / original_size_mb))
    
    if verbose:
        print(f"Estimated records per file: ~{estimated_records_per_file}")
        print(f"\n" + "=" * 70)
        print(f"Splitting file...")
        print(f"=" * 70)
    
    # Extract base name (e.g., "rna_sft" from "rna_sft.parquet")
    base_name = input_path.stem
    
    file_idx = 0
    saved_files = []
    total_saved_size_mb = 0
    
    # Process in batches
    batch_size = estimated_records_per_file
    current_batch = []
    records_processed = 0
    
    # Create progress bar
    pbar = tqdm(total=total_records, desc="Processing records", unit="rec", disable=not verbose)
    
    # Read file in batches using iter_batches
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_df = batch.to_pandas()
        current_batch.append(batch_df)
        records_processed += len(batch_df)
        pbar.update(len(batch_df))
        
        # Calculate current accumulated size (rough estimate)
        # When we reach target or end of file, write it out
        if len(current_batch) > 0:
            # Concatenate and save
            import pandas as pd
            df_to_save = pd.concat(current_batch, ignore_index=True)
            
            # Generate output filename
            output_file = output_dir / f"{base_name}_part{file_idx:03d}.parquet"
            
            # Save to parquet
            df_to_save.to_parquet(output_file, index=False)
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            saved_files.append(str(output_file))
            total_saved_size_mb += file_size_mb
            
            if verbose:
                tqdm.write(f"  ✓ Part {file_idx:03d}: {len(df_to_save)} records, {file_size_mb:.2f} MB")
            
            # Adjust batch size based on actual file size
            if file_size_mb > 0:
                # Calculate how many records we should use next time
                ratio = target_size_mb / file_size_mb
                batch_size = max(1, int(len(df_to_save) * ratio))
            
            # Reset for next file
            current_batch = []
            file_idx += 1
    
    pbar.close()
    
    if verbose:
        print(f"\n" + "=" * 70)
        print(f"✅ Splitting complete!")
        print(f"=" * 70)
        print(f"Total files created: {len(saved_files)}")
        print(f"Total records: {records_processed}")
        print(f"Total size: {total_saved_size_mb:.2f} MB")
        print(f"Average file size: {total_saved_size_mb / len(saved_files):.2f} MB")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Split a large parquet file into smaller files based on target size."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input parquet file"
    )
    parser.add_argument(
        "--target_size_mb",
        type=float,
        default=500,
        help="Target size for each output file in MB (default: 500)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to same as input file)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress information"
    )
    
    args = parser.parse_args()
    
    split_parquet_by_size(
        input_file=args.input_file,
        target_size_mb=args.target_size_mb,
        output_dir=args.output_dir,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

