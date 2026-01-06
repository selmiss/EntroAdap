#!/bin/bash
# Build full nucleic acid SFT datasets for DNA and RNA
# This processes all available data (can take a long time)

set -e  # Exit on error

echo "========================================================================"
echo "DNA/RNA SFT Dataset Builder - Full Pipeline"
echo "========================================================================"

# Source environment
source local_env.sh

# Configuration
MAX_SEQ_LENGTH_DNA=500
MAX_SEQ_LENGTH_RNA=500  # Process RNA sequences up to 500nt (no windowing)
GRAPH_RADIUS=4.0
MAX_NEIGHBORS=16
MAX_RECORDS_PER_FILE=2000
SAMPLE_FRACTION=0.4  # Process 1% (1/100) of the data

echo ""
echo "Configuration:"
echo "  DNA max sequence length: ${MAX_SEQ_LENGTH_DNA} nt"
echo "  RNA max sequence length: ${MAX_SEQ_LENGTH_RNA} nt (no windowing)"
echo "  Graph radius: ${GRAPH_RADIUS} Å"
echo "  Max neighbors: ${MAX_NEIGHBORS}"
echo "  Records per file: ${MAX_RECORDS_PER_FILE}"
echo "  Sample fraction: ${SAMPLE_FRACTION} (1/100 of data)"
echo ""

# Build DNA dataset
# echo "========================================================================"
# echo "Building DNA SFT Dataset (Full)"
# echo "========================================================================"
# python3 src/data_factory/nacid/build_nacid_sft_data.py \
#     --raw_data_dir data/nacid/raw \
#     --output_dir data/sft/nacid \
#     --modality dna \
#     --graph_radius ${GRAPH_RADIUS} \
#     --max_neighbors ${MAX_NEIGHBORS} \
#     --max_seq_length ${MAX_SEQ_LENGTH} \
#     --max_records_per_file ${MAX_RECORDS_PER_FILE} \
#     --sample_fraction ${SAMPLE_FRACTION} \
#     --fiber_exe fiber

echo ""
echo "========================================================================"
echo "Building RNA SFT Dataset (Full) - No Windowing, Single-Stranded"
echo "========================================================================"
python3 src/data_factory/nacid/build_nacid_sft_data.py \
    --raw_data_dir data/nacid/raw \
    --output_dir data/sft/nacid \
    --modality rna \
    --graph_radius ${GRAPH_RADIUS} \
    --max_neighbors ${MAX_NEIGHBORS} \
    --max_seq_length ${MAX_SEQ_LENGTH_RNA} \
    --max_records_per_file ${MAX_RECORDS_PER_FILE} \
    --sample_fraction ${SAMPLE_FRACTION} \
    --rna_single_strand \
    --fiber_exe fiber

echo ""
echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Output files are in: data/sft/nacid/"
echo ""
ls -lh data/sft/nacid/
