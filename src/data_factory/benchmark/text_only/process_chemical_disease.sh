#!/bin/bash

cd "$(dirname "$0")/../../.."
source local_env.sh

python src/data_factory/benchmark/process_text_only_sft.py \
    /home/UWO/zjing29/proj/DQ-Former/data/Biomolecular_Text_Instructions/chemical_entity_recognition.json \
    --output_dir data/benchmark \
    --dataset_name chemical_disease_interaction_extraction \
    --system_prompt "You are a helpful assistant specialized in extracting chemical-disease interactions from biomedical text."

