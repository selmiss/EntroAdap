#!/usr/bin/env python3
"""
Quick test script to process just the test split to verify the pipeline works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from build_molecule_encoder_data import process_parquet_split

# Process only test split for quick verification
input_file = Path('data/raw/mol_encode_small/test.parquet')
output_file = Path('data/encoder/pretrain/test.parquet')

print("Quick test: Processing test split only...")
print(f"Input: {input_file}")
print(f"Output: {output_file}")
print()

df = process_parquet_split(
    input_file=input_file,
    output_file=output_file,
    num_workers=4,  # Use 4 workers for test
    batch_size=50,
    checkpoint_interval=500,  # Checkpoint every 500 for testing
    resume=True,
    verbose=True
)

print("\n✅ Test processing complete!")
print(f"Processed {len(df)} molecules successfully")

