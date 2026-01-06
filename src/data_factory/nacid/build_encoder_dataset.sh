#!/bin/bash
# Build nucleic acid (DNA/RNA) encoder datasets from sequence files
#
# Usage: bash build_encoder_dataset.sh [--test]
# Options:
#   --test    Run in test mode (100 sequences only)

set -e

# ============================================================================
# Setup
# ============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

echo "========================================================================"
echo "Building Nucleic Acid Encoder Datasets"
echo "========================================================================"

# Source environment
cd "$PROJECT_ROOT"
[ -f "local_env.sh" ] && source local_env.sh

# Verify X3DNA fiber is available
command -v fiber &> /dev/null || {
    echo "Error: 'fiber' not found. Please install X3DNA."
    exit 1
}
echo "✓ X3DNA fiber: $(which fiber)"

# ============================================================================
# Configuration
# ============================================================================
# Paths
INPUT_DIR="$PROJECT_ROOT/data/nacid/seq_only"
OUTPUT_DIR="$PROJECT_ROOT/data/encoder/nacid"
mkdir -p "$OUTPUT_DIR"

# Parameters
GRAPH_RADIUS=4.0
MAX_NEIGHBORS=16
MAX_SEQ_LENGTH=500       # Maximum sequence length (nucleotides)
NUM_WORKERS=8
BATCH_SIZE=50
CHECKPOINT_INTERVAL=1000

# Test mode
TEST_MODE=false
TEST_SUFFIX=""
MAX_SEQ_ARG=""

if [ "$1" == "--test" ]; then
    TEST_MODE=true
    TEST_SUFFIX="_test"
    MAX_SEQ_ARG="--max_sequences 100"
    echo "Running in TEST MODE (100 sequences)"
fi

# ============================================================================
# Function to build dataset
# ============================================================================
build_dataset() {
    local seq_type=$1
    local input_dir="$INPUT_DIR/$seq_type"
    local output_file="$OUTPUT_DIR/${seq_type}_encoder${TEST_SUFFIX}.parquet"
    
    echo ""
    echo "========================================================================"
    echo "Building ${seq_type^^} Encoder Dataset"
    echo "========================================================================"
    echo "Input:  $input_dir"
    echo "Output: $output_file"
    echo ""
    
    python "$SCRIPT_DIR/build_nacid_encoder_data.py" \
        --input_dir "$input_dir" \
        --output_file "$output_file" \
        --seq_type "$seq_type" \
        --graph_radius $GRAPH_RADIUS \
        --max_neighbors $MAX_NEIGHBORS \
        --max_seq_length $MAX_SEQ_LENGTH \
        --num_workers $NUM_WORKERS \
        --batch_size $BATCH_SIZE \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        $MAX_SEQ_ARG
}

# ============================================================================
# Build datasets
# ============================================================================
build_dataset "dna"
build_dataset "rna"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================================================"
echo "✅ Dataset Building Complete!"
echo "========================================================================"
echo "Output directory: $OUTPUT_DIR"
echo ""
ls -lh "$OUTPUT_DIR"/*${TEST_SUFFIX}.parquet 2>/dev/null || echo "No output files found"

