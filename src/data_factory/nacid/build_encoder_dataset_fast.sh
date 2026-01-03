#!/bin/bash
# OPTIMIZED version - processes sequences 2-3x faster
#
# Optimizations:
#   - Shorter sequences (200 nt vs 500 nt) = 2.5x fewer atoms
#   - Smaller radius (6.0Å vs 8.0Å) = fewer edges
#   - Fewer neighbors (16 vs 24) = faster graph building
#
# Usage: bash build_encoder_dataset_fast.sh [--test]

set -e

# ============================================================================
# Setup
# ============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

echo "========================================================================"
echo "Building Nucleic Acid Encoder Datasets (FAST MODE)"
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
# Configuration - OPTIMIZED for speed
# ============================================================================
# Paths
INPUT_DIR="$PROJECT_ROOT/data/nacid/seq_only"
OUTPUT_DIR="$PROJECT_ROOT/data/encoder/nacid"
mkdir -p "$OUTPUT_DIR"

# OPTIMIZED Parameters (2-3x faster)
GRAPH_RADIUS=4.0         # Reduced from 8.0 (fewer edges)
MAX_NEIGHBORS=16         # Reduced from 24 (faster graph build)
MAX_SEQ_LENGTH=200       # Reduced from 500 (faster processing)
MAX_SEQUENCES=3000    # Limit number of sequences (default: 1M for DNA, comment out for all)
NUM_WORKERS=1        # Increased from 8 (more parallelism)
BATCH_SIZE=10           # Increased from 50
CHECKPOINT_INTERVAL=200  # More frequent checkpoints

echo "⚡ Optimizations enabled:"
echo "  • Reduced graph radius: $GRAPH_RADIUS Å (was 8.0)"
echo "  • Reduced max neighbors: $MAX_NEIGHBORS (was 24)"
echo "  • Reduced max sequence length: $MAX_SEQ_LENGTH nt (was 500)"
echo "  • Max sequences to process: ${MAX_SEQUENCES:-all}"
echo "  • Increased workers: $NUM_WORKERS (was 8)"

# Test mode
TEST_MODE=false
TEST_SUFFIX=""
MAX_SEQ_ARG=""

if [ "$1" == "--test" ]; then
    TEST_MODE=true
    TEST_SUFFIX="_fast_test"
    MAX_SEQ_ARG="--max_sequences 100"
    echo "Running in TEST MODE (100 sequences)"
else
    TEST_SUFFIX="_fast"
    # Use MAX_SEQUENCES if set (not in test mode)
    if [ -n "$MAX_SEQUENCES" ]; then
        MAX_SEQ_ARG="--max_sequences $MAX_SEQUENCES"
    fi
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
# build_dataset "dna"
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

