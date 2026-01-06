#!/bin/bash
# Production script for building full protein SFT dataset using batch API

# Change to project root
cd "$(dirname "$0")/../../.."

# Source local environment
source local_env.sh

# Run on full dataset using batch API
# This will process all proteins with single PDB from the full UniProt data
# Output will be split into multiple Parquet files (1000 records per file)
# Using all atoms (not just C-alpha) with 30k atom threshold
python3 src/data_factory/protein_sft_v2/build_protein_sft_data.py \
    --uniprot_json_dir data/uniprot/full \
    --structure_dir data/pdb_structures \
    --output_dir data/sft/protein \
    --model gpt-5-mini \
    --download_delay 0.1 \
    --max_records_per_file 2000 \
    --max_atoms 12000 \
    --max_neighbors 16

echo ""
echo "Production run complete! Check the output at: data/sft/protein/"
echo ""
echo "To inspect the output:"
echo "  python3 src/data_factory/protein_sft_v2/inspect_output.py data/sft/protein"

