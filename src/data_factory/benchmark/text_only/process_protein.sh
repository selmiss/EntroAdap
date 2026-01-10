#!/bin/bash

source local_env.sh

python src/data_factory/benchmark/text_only/process_text_only_sft.py \
    dq_data/Protein-oriented_Instructions/protein_design.json \
    --output_dir data/benchmark/protein \
    --dataset_name protein_design \
    --system_prompt ""

python src/data_factory/benchmark/text_only/process_text_only_sft.py \
    dq_data/Protein-oriented_Instructions/protein_function.json \
    --output_dir data/benchmark/protein \
    --dataset_name protein_function \
    --system_prompt ""

python src/data_factory/benchmark/text_only/process_text_only_sft.py \
    dq_data/Protein-oriented_Instructions/domain_motif.json \
    --output_dir data/benchmark/protein \
    --dataset_name domain_motif \
    --system_prompt ""

python src/data_factory/benchmark/text_only/process_text_only_sft.py \
    dq_data/Protein-oriented_Instructions/general_function.json \
    --output_dir data/benchmark/protein \
    --dataset_name general_function \
    --system_prompt ""

python src/data_factory/benchmark/text_only/process_text_only_sft.py \
    dq_data/Protein-oriented_Instructions/catalytic_activity.json \
    --output_dir data/benchmark/protein \
    --dataset_name catalytic_activity \
    --system_prompt ""

