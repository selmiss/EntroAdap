#!/bin/bash
# Script to create a smaller dataset by random sampling

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

# Run the sampling script
python src/data_factory/tmproc/create_small_dataset.py \
    --input_dir data/raw \
    --output_dir data/raw/small \
    --train_samples 1000000 \
    --val_samples 10000 \
    --test_samples 10000 \
    --seed 42

echo "Small dataset creation complete!"

