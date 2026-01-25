#!/usr/bin/env python3
"""
Script to create a smaller dataset by randomly sampling from the full dataset.
Samples: 1M from train, 10k from val, 10k from test
"""

import pandas as pd
import argparse
from pathlib import Path


def create_small_dataset(input_dir, output_dir, train_samples=1000000, val_samples=10000, test_samples=10000, seed=42):
    """
    Create a smaller dataset by randomly sampling.
    
    Args:
        input_dir: Directory containing full dataset parquet files
        output_dir: Directory to save sampled dataset
        train_samples: Number of samples for train set (default: 1M)
        val_samples: Number of samples for val set (default: 10k)
        test_samples: Number of samples for test set (default: 10k)
        seed: Random seed for reproducibility
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': (input_dir / 'train.parquet', output_dir / 'train.parquet', train_samples),
        'val': (input_dir / 'val.parquet', output_dir / 'val.parquet', val_samples),
        'test': (input_dir / 'test.parquet', output_dir / 'test.parquet', test_samples)
    }
    
    for split_name, (input_file, output_file, n_samples) in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        if not input_file.exists():
            print(f"Warning: {input_file} does not exist, skipping...")
            continue
        
        # Read the data
        df = pd.read_parquet(input_file)
        original_size = len(df)
        print(f"Original size: {original_size:,} records")
        
        # Sample if dataset is larger than requested samples
        if original_size > n_samples:
            df_sampled = df.sample(n=n_samples, random_state=seed)
            print(f"Sampled {n_samples:,} records")
        else:
            df_sampled = df
            print(f"Dataset smaller than requested samples, keeping all {original_size:,} records")
        
        # Save sampled data
        df_sampled.to_parquet(output_file, index=False, engine='pyarrow')
        print(f"Saved to {output_file}")
        
        # Print file size
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Print sample record
        if len(df_sampled) > 0:
            print(f"Sample record: {df_sampled.iloc[0].to_dict()}")
    
    print("\n✅ Small dataset creation complete!")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a smaller dataset by random sampling"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="Directory containing full dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/small",
        help="Directory to save small dataset"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=1000000,
        help="Number of samples for train set (default: 1M)"
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=10000,
        help="Number of samples for val set (default: 10k)"
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=10000,
        help="Number of samples for test set (default: 10k)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    create_small_dataset(
        args.input_dir,
        args.output_dir,
        args.train_samples,
        args.val_samples,
        args.test_samples,
        args.seed
    )


if __name__ == "__main__":
    main()

