#!/bin/bash
# Build molecule encoder dataset with geometry data

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Source environment if local_env.sh exists
if [ -f "local_env.sh" ]; then
    echo "Sourcing local_env.sh..."
    source local_env.sh
fi

# Run the processing script with checkpointing
# Use multiprocessing for faster processing
# Adjust --num_workers based on your CPU cores (default: auto-detect)
# Checkpoints are saved every 10,000 molecules (configurable with --checkpoint_interval)
# If interrupted, rerun this script and it will resume from the last checkpoint
#
# This script processes the standard train/val/test splits.
# For other file patterns, use:
#   --process_all: Process all .parquet files in the input directory
#   --file_pattern "pattern": Process files matching a specific pattern (e.g., "*-preprocessed.parquet")
python src/data_factory/tmproc/build_molecule_encoder_data.py \
    --input_dir data/raw/mol_encode_small \
    --output_dir data/encoder/pretrain \
    --num_workers 8 \
    --batch_size 100 \
    --checkpoint_interval 10000

echo ""
echo "✅ Encoder dataset building complete!"

