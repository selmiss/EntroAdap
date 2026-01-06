#!/bin/bash
# Build protein encoder dataset from UniProt JSON files
#
# Usage:
#   bash build_encoder_dataset.sh [OPTIONS]
#
# Examples:
#   # Build full dataset with default settings (all heavy atoms, 4Å radius)
#   bash build_encoder_dataset.sh
#
#   # Build with C-alpha only instead of all heavy atoms
#   bash build_encoder_dataset.sh --ca_only
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
OUTPUT_FILE="data/encoder/protein/protein_pretrain_4A.parquet"

# Run the script
python src/data_factory/protein/build_protein_encoder_data.py \
    --uniprot_json_dir "$UNIPROT_JSON_DIR" \
    --structure_dir "$STRUCTURE_DIR" \
    --output_file "$OUTPUT_FILE" \
    --max_structures 10000 \
    --graph_radius 4.0 \
    --max_neighbors 16 \
    --num_workers 8 \
    "$@"

