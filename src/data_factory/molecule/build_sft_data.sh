#!/bin/bash
# Build molecule SFT dataset from raw parquet files
# This script processes raw SFT data and adds molecular geometry features

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

echo ""
echo "================================================================================"
echo "MOLECULE SFT DATASET BUILDER"
echo "================================================================================"
echo ""
echo "This script will process raw SFT data and generate molecular geometry features."
echo "Large output files will be automatically split into multiple parts (max 300 MB each)."
echo ""
echo "Input:  data/raw/mol_sft_all/*.parquet"
echo "Output: data/sft/*.parquet (may create multiple part files)"
echo ""
echo "Files to process:"
ls -lh data/raw/mol_sft_all/*.parquet
echo ""
echo "================================================================================"
echo ""

# Run the processing script
python src/data_factory/molecule/build_sft_data.py \
    --input_dir data/raw/mol_sft_all \
    --output_dir data/sft/molecule_4A \
    --num_workers 8 \
    --batch_size 1 \
    --checkpoint_interval 5000 \
    --max_file_size_mb 100

echo ""
echo "================================================================================"
echo "Build complete! Check output at data/sft/molecule_4A/"
echo "================================================================================"

