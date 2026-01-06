#!/bin/bash
# Test script for building protein SFT dataset with a few samples

# Change to project root
cd "$(dirname "$0")/../../.."

# Source local environment
source local_env.sh

# Run with 5 samples for testing, using sequential API (one by one)
# Using all atoms (not just C-alpha) with 30k atom threshold
python3 src/data_factory/protein_sft_v2/build_protein_sft_data.py \
    --uniprot_json_dir data/uniprot/full \
    --structure_dir data/pdb_structures \
    --output_dir data/sft/protein_demo \
    --max_samples 5 \
    --sequential_api \
    --model gpt-5-mini \
    --download_delay 0.2 \
    --max_records_per_file 1000 \
    --max_atoms 30000

echo ""
echo "Demo run complete! Check the output at: data/sft/protein_demo/"
echo ""
echo "To inspect the output:"
echo "  python3 -c \"import pandas as pd; df = pd.read_parquet('data/sft/protein_demo/protein_sft.parquet'); print(df.head()); print('\\nShape:', df.shape); print('\\nColumns:', df.columns.tolist())\""

