#!/bin/bash
# Demo script to build nucleic acid SFT dataset
# Process a small sample to verify the pipeline works

set -e  # Exit on error

echo "========================================================================"
echo "DNA/RNA SFT Dataset Builder - Demo"
echo "========================================================================"

# Source environment
source local_env.sh

# Test with a small number of samples first
MAX_SAMPLES=10

echo ""
echo "Testing DNA SFT dataset building (${MAX_SAMPLES} samples)..."
echo "------------------------------------------------------------------------"
python3 src/data_factory/nacid/build_nacid_sft_data.py \
    --raw_data_dir data/nacid/raw \
    --output_dir data/sft/nacid \
    --modality dna \
    --graph_radius 4.0 \
    --max_neighbors 16 \
    --max_seq_length 500 \
    --max_samples ${MAX_SAMPLES} \
    --max_records_per_file 1000 \
    --fiber_exe fiber

echo ""
echo "Testing RNA SFT dataset building with windowing (${MAX_SAMPLES} samples)..."
echo "------------------------------------------------------------------------"
python3 src/data_factory/nacid/build_nacid_sft_data.py \
    --raw_data_dir data/nacid/raw \
    --output_dir data/sft/nacid \
    --modality rna \
    --graph_radius 4.0 \
    --max_neighbors 16 \
    --max_samples ${MAX_SAMPLES} \
    --max_records_per_file 1000 \
    --use_windowing \
    --window_size 500 \
    --window_overlap 50 \
    --fiber_exe fiber

echo ""
echo "========================================================================"
echo "Demo complete! Check data/sft/nacid/ for output files"
echo "========================================================================"

