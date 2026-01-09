#!/bin/bash
# Octopus training script with custom fusion blocks
#
# Features:
# 1. Modality statistics: Automatically prints data amount for each modality before training
# 2. Wandb logging: Configure via config YAML (wandb_project, wandb_run_name)
# 3. Dataset max samples: Control max samples per dataset via config YAML (dataset_max_samples)

: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"
: "${CHECKPOINT_DIR:?Environment variable CHECKPOINT_DIR not set}"

GPU_IDS=4
export MASTER_PORT=29504

# Load configuration from YAML file
# CONFIG_FILE="configs/sft/octopus/octopus_8B_v2.yaml"

# deepspeed --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} src/runner/sft.py \
#     --config ${CONFIG_FILE}

# Load configuration from YAML file
CONFIG_FILE="configs/sft/octopus/octopus_8B_s3_v2.yaml"

deepspeed --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} src/runner/sft.py \
    --config ${CONFIG_FILE}

