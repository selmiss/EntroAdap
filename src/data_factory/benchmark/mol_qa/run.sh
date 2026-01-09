#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT"

[ -f "local_env.sh" ] && source local_env.sh

python src/data_factory/benchmark/mol_qa/build_mol_qa_data.py \
    --max_workers 8 \
    --input_dir /home/UWO/zjing29/proj/DQ-Former/data/mol_instructions_processed/open_question \
    --output_dir data/benchmark/open_question