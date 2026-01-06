#!/bin/bash
# Convert protein SFT dataset to match DNA/RNA format
#
# This script flattens the nested 'structure' dict and converts
# the instruction format to messages format for consistency.

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Project root: $PROJECT_ROOT"
echo

# Input and output paths
INPUT_FILE="$PROJECT_ROOT/data/sft/protein/protein_sft.parquet"
OUTPUT_FILE="$PROJECT_ROOT/data/sft/protein/protein_sft_flat.parquet"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

echo "Converting protein SFT format..."
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo

# Run conversion script
python "$SCRIPT_DIR/convert_protein_sft_format.py" "$INPUT_FILE" "$OUTPUT_FILE"

echo
echo "Done! Converted file saved to:"
echo "  $OUTPUT_FILE"

