#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT"

python src/data_factory/benchmark/mol_qa/build_mol_qa_data.py \
    --max_workers 8 \
    --input_dir dq_data/mol_instructions_processed/reagent_prediction \
    --output_dir data/benchmark/reagent_prediction

