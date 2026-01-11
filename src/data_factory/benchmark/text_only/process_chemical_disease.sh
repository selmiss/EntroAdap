#!/bin/bash


source local_env.sh

python src/data_factory/benchmark/text_only/process_text_only_sft.py \
    dq_data/Biomolecular_Text_Instructions/chemical_entity_recognition.json \
    --output_dir data/benchmark \
    --dataset_name chemical_entity_recognition \
    --system_prompt ""

# python src/data_factory/benchmark/text_only/process_text_only_sft.py \
#     dq_data/Biomolecular_Text_Instructions/chemical_protein_interaction_extraction.json \
#     --output_dir data/benchmark \
#     --dataset_name chemical_protein_interaction_extraction \
#     --system_prompt ""

# python src/data_factory/benchmark/text_only/process_text_only_sft.py \
#     dq_data/Biomolecular_Text_Instructions/multi_choice_question.json \
#     --output_dir data/benchmark \
#     --dataset_name multi_choice_question \
#     --system_prompt ""

# python src/data_factory/benchmark/text_only/process_text_only_sft.py \
#     dq_data/Biomolecular_Text_Instructions/true_or_false_question.json \
#     --output_dir data/benchmark \
#     --dataset_name true_or_false_question \
#     --system_prompt ""
