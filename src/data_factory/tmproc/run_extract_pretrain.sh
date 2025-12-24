#!/bin/bash
# Script to extract specific keys from pretrain JSONL files and convert to parquet

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Source environment if local.env.sh exists
if [ -f "local.env.sh" ]; then
    echo "Sourcing local.env.sh..."
    source local.env.sh
fi

# Run the extraction script
python src/data_factory/tmproc/extract_pretrain_keys.py \
    --input_file /home/UWO/zjing29/proj/DQ-Former/data/tmpft/comprehensive_conversations-preprocessed.jsonl \
    --output_dir data/raw/mol_sft_all \
    --keys smiles brics_gid cid iupac_name messages

echo "Extraction complete!"

