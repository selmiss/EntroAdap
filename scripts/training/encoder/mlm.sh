#!/bin/bash
# Masked reconstruction pre-training script

# Source environment
cd "$(dirname "$0")/../../.." || exit
source local_env.sh

GPU_IDS=${GPU_IDS:-7}
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export MASTER_PORT=${MASTER_PORT:-29500}

# Load configuration from YAML file
CONFIG_FILE="${1:-configs/aa_encoder/all_modality.yaml}"

python src/runner/aa_encoder.py --config ${CONFIG_FILE}
