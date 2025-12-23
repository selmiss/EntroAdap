#!/bin/bash
# Octopus training script with custom fusion blocks

: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"
: "${CHECKPOINT_DIR:?Environment variable CHECKPOINT_DIR not set}"

GPU_IDS=7
export MASTER_PORT=29500

# Load configuration from YAML file
CONFIG_FILE="configs/sft/octopus/octopus_test_tiny.yaml"

deepspeed --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} src/runner/sft.py \
    --config ${CONFIG_FILE}

