#!/bin/bash
# Build molecule encoder dataset from SFT data (mol_sft_all)
# This script processes all .parquet files in the mol_sft_all directory

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

# Run the processing script with --process_all flag
# This will process ALL .parquet files in the mol_sft_all directory:
# - detailed_structural_descriptions-preprocessed.parquet
# - comprehensive_conversations-preprocessed.parquet
python src/data_factory/tmproc/build_molecule_encoder_data.py \
    --input_dir data/raw/mol_sft_all \
    --output_dir data/encoder/sft \
    --num_workers 8 \
    --batch_size 100 \
    --checkpoint_interval 5000 \
    --process_all

echo ""
echo "✅ SFT encoder dataset building complete!"
echo "Output directory: data/encoder/sft"


