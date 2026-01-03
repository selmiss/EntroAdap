#!/bin/bash
# Build protein encoder dataset from UniProt JSON files
#
# Usage:
#   bash build_encoder_dataset.sh [OPTIONS]
#
# Examples:
#   # Build full dataset with default settings (C-alpha only, 8Å radius)
#   bash build_encoder_dataset.sh
#
#   # Build with all atoms instead of C-alpha only
#   bash build_encoder_dataset.sh --all_atoms
#
#   # Test with only 100 structures
#   bash build_encoder_dataset.sh --max_structures 100
#
#   # Use more workers for faster processing
#   bash build_encoder_dataset.sh --num_workers 16

# Navigate to project root
cd "$(dirname "$0")/../../.."

# Source environment
source local_env.sh

# Default arguments
UNIPROT_JSON_DIR="data/uniprot/full"
STRUCTURE_DIR="data/pdb_structures"
OUTPUT_FILE="data/encoder/protein/protein_pretrain.parquet"

# Run the script
python src/data_factory/protein/build_protein_encoder_data.py \
    --uniprot_json_dir "$UNIPROT_JSON_DIR" \
    --structure_dir "$STRUCTURE_DIR" \
    --output_file "$OUTPUT_FILE" \
    "$@"

